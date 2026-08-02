#!/usr/bin/env python3
"""Train and diagnose the standalone geometric model on public data."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch

# When this file is executed by path, Python places tools/ rather than the
# repository root first on sys.path.  Kaggle also preinstalls an unrelated
# package named benchmark, so make the local package unambiguous.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import OptimizerSpec
from benchmark.batches import prepare_batch
from benchmark.runner import _loss_and_accuracy, _make_model_spec, _resolve_batch_sizes
from benchmark.manifest import load_manifest
from data import make_dataloaders


DEFAULT_SUBMISSION = ROOT / "submissions/geometric_navigation/submission.py"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("diagnostic_submission", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_decimal_tokens(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    digit_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
    all_digits = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
    for position in range(tokens.shape[1]):
        active = valid[:, position]
        token = tokens[:, position]
        is_digit = token.ge(digit_offset) & token.lt(digit_offset + 10)
        all_digits = all_digits & (~active | is_digit)
        digit = (token - digit_offset).clamp(0, 9)
        values = torch.where(active & is_digit, values * 10 + digit, values)
    return values, all_digits & valid.any(dim=1)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def gpu_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def train_model(
    model,
    submission,
    dataloader,
    manifest,
    device: torch.device,
    seconds: float,
    maximum_steps: int,
) -> dict:
    bundle = submission.build_optimizer(
        model, OptimizerSpec(seconds, device.type)
    )
    if seconds <= 0 or maximum_steps <= 0:
        return {"steps": 0, "seconds": 0.0, "steps_per_second": 0.0}
    model.train()
    iterator = iter(dataloader)
    started = time.monotonic()
    deadline = started + seconds
    steps = 0
    while steps < maximum_steps and time.monotonic() < deadline:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        bundle.optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = _loss_and_accuracy(
            model,
            batch,
            manifest,
            device,
            training_loss=submission.training_loss,
            token_training_loss=submission.token_training_loss,
        )
        loss.backward()
        if manifest.runtime.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), manifest.runtime.grad_clip
            )
        bundle.optimizer.step()
        if bundle.scheduler is not None:
            bundle.scheduler.step()
        steps += 1
    elapsed = time.monotonic() - started
    return {
        "steps": steps,
        "seconds": elapsed,
        "steps_per_second": steps / elapsed if elapsed else 0.0,
    }


def evaluate_model(
    model,
    dataloader,
    device: torch.device,
    digit_offset: int,
    number_landmarks: int,
    maximum_examples: int,
) -> tuple[dict, float]:
    model.eval()
    started = time.monotonic()
    examples = tokens = exact = correct_tokens = 0
    landmark_correct = landmark_targets = invalid = 0
    correct_probability_sum = final_entropy_sum = 0.0
    grouped: dict[int, list[int]] = {}
    step_entropy_sum = step_norm_sum = step_movement_sum = None
    trajectory_examples = 0
    with torch.no_grad():
        for raw_batch in dataloader:
            input_ids, targets, attention_mask, target_positions = prepare_batch(
                raw_batch, device
            )
            logits, auxiliary = model(input_ids, attention_mask=attention_mask)
            if target_positions is None:
                raise ValueError("diagnostics require separate E1 outputs")
            batch_indices = torch.arange(logits.shape[0], device=device).reshape(-1, 1)
            token_logits = logits[
                batch_indices, target_positions.clamp_min(0)
            ]
            valid = targets.ne(-100)
            predictions = token_logits.argmax(dim=-1)
            row_exact = ((predictions == targets) | ~valid).all(dim=1)
            batch_examples = int(valid.any(dim=1).sum().item())
            examples += batch_examples
            exact += int(row_exact.sum().item())
            tokens += int(valid.sum().item())
            correct_tokens += int(((predictions == targets) & valid).sum().item())

            target_values, target_valid = parse_decimal_tokens(
                targets, valid, digit_offset
            )
            predicted_values, predicted_valid = parse_decimal_tokens(
                predictions, valid, digit_offset
            )
            valid_prediction = predicted_valid & predicted_values.lt(number_landmarks)
            invalid += int((~valid_prediction).sum().item())

            probabilities = auxiliary["final_landmark_probabilities"].float()
            landmark_predictions = probabilities.argmax(dim=-1)
            target_in_range = target_valid & target_values.lt(number_landmarks)
            landmark_targets += int(target_in_range.sum().item())
            landmark_correct += int(
                ((landmark_predictions == target_values) & target_in_range).sum().item()
            )
            correct_probability = probabilities.gather(
                1, target_values.clamp(0, number_landmarks - 1).unsqueeze(1)
            ).squeeze(1)
            correct_probability_sum += float(
                (correct_probability * target_in_range).sum().item()
            )
            final_entropy = -(
                probabilities * probabilities.clamp_min(1.0e-8).log()
            ).sum(dim=-1)
            final_entropy_sum += float(final_entropy.sum().item())

            time_steps = auxiliary["parsed_time_steps"]
            for depth in time_steps.unique().tolist():
                selected = time_steps.eq(depth)
                pair = grouped.setdefault(int(depth), [0, 0])
                pair[0] += int(row_exact[selected].sum().item())
                pair[1] += int(selected.sum().item())

            entropy_history = auxiliary["entropy_history"].float()
            norm_history = auxiliary["state_norm_history"].float()
            movement_history = auxiliary["movement_history"].float()
            entropy_sum = entropy_history.sum(dim=1)
            norm_sum = norm_history.sum(dim=1)
            movement_sum = movement_history.sum(dim=1)
            if step_entropy_sum is None:
                step_entropy_sum = entropy_sum
                step_norm_sum = norm_sum
                step_movement_sum = movement_sum
            else:
                step_entropy_sum += entropy_sum
                step_norm_sum += norm_sum
                step_movement_sum += movement_sum
            trajectory_examples += input_ids.shape[0]
            if examples >= maximum_examples:
                break
    elapsed = time.monotonic() - started
    denominator = max(1, examples)
    target_denominator = max(1, landmark_targets)
    metrics = {
        "exact_accuracy": exact / denominator,
        "token_accuracy": correct_tokens / max(1, tokens),
        "accuracy_by_t": {
            str(depth): values[0] / max(1, values[1])
            for depth, values in sorted(grouped.items())
        },
        "final_landmark_accuracy": landmark_correct / target_denominator,
        "correct_landmark_probability": correct_probability_sum / target_denominator,
        "final_landmark_entropy": final_entropy_sum / denominator,
        "landmark_entropy_by_step": (
            (step_entropy_sum / trajectory_examples).tolist()
            if step_entropy_sum is not None
            else []
        ),
        "state_norm_by_step": (
            (step_norm_sum / trajectory_examples).tolist()
            if step_norm_sum is not None
            else []
        ),
        "cosine_movement_distance_by_step": (
            (step_movement_sum / trajectory_examples).tolist()
            if step_movement_sum is not None
            else []
        ),
        "invalid_prediction_percentage": 100.0 * invalid / denominator,
        "examples": examples,
    }
    return metrics, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--training-seconds", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--max-eval-examples", type=int, default=512)
    parser.add_argument("--variant", default="geometric_v0")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    device = torch.device(manifest.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA manifest requested but CUDA is unavailable")
    seed = manifest.runtime.seeds[0]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device.index or 0)

    module = load_module(args.submission.resolve())
    submission = module.SUBMISSION
    batch_size, eval_batch_size = _resolve_batch_sizes(submission, manifest)
    loaders = make_dataloaders(
        replace(
            manifest.data,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            seed=seed,
        ),
        device=device,
    )
    if args.split not in loaders:
        raise ValueError(f"split {args.split!r} not available: {sorted(loaders)}")
    spec = _make_model_spec(manifest)
    model = submission.build_model(spec).to(device=device, dtype=torch.float32)
    training = train_model(
        model,
        submission,
        loaders["train"],
        manifest,
        device,
        args.training_seconds,
        args.max_steps,
    )
    metrics, evaluation_seconds = evaluate_model(
        model,
        loaders[args.split],
        device,
        module.DIGIT_OFFSET,
        module.NUM_LANDMARKS,
        args.max_eval_examples,
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device.index or 0))
        if device.type == "cuda"
        else 0
    )
    submission_bytes = args.submission.read_bytes()
    result = {
        "commit": git_commit(),
        "submission_sha256": hashlib.sha256(submission_bytes).hexdigest(),
        "variant": args.variant,
        "dataset": "e1",
        "seed": seed,
        "gpu": gpu_name(device),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "config": {
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "training_time_seconds": args.training_seconds,
            "dtype": manifest.runtime.dtype,
            "amp": manifest.runtime.amp,
            "compile": manifest.runtime.compile,
        },
        "runtime": {
            "peak_gpu_memory_bytes": peak_memory,
            "optimizer_steps": training["steps"],
            "optimizer_steps_per_second": training["steps_per_second"],
            "training_duration_seconds": training["seconds"],
            "evaluation_duration_seconds": evaluation_seconds,
            "examples_per_second": metrics["examples"] / max(evaluation_seconds, 1.0e-9),
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
