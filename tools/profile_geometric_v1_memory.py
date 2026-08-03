#!/usr/bin/env python3
"""Run a maximum-Easy-N V1 CUDA forward/backward memory preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch, count_model_state_elements
from data.squaring_mod import (
    VOCAB_SIZE,
    collate_squaring_mod,
    tokenize_squaring_mod_with_result,
)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("v1_memory_submission", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_batch(batch_size: int) -> dict[str, torch.Tensor]:
    rows = []
    for index in range(batch_size):
        input_ids, labels = tokenize_squaring_mod_with_result(
            4095,
            index % 4095,
            4,
            1,
            separate_input_output=True,
        )
        rows.append({"input_ids": input_ids, "labels": labels})
    return collate_squaring_mod(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the maximum-N memory preflight requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    module = load_module(args.submission.resolve())
    spec = ModelSpec(VOCAB_SIZE, 16, 500_000_000)
    model = module.SUBMISSION.build_model(spec).to(device=device, dtype=torch.float32)
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in make_batch(args.batch_size).items()
    }
    logits, auxiliary = model(batch["input_ids"], batch["attention_mask"])
    batch_indices = torch.arange(args.batch_size, device=device).reshape(-1, 1)
    token_logits = logits[batch_indices, batch["target_positions"].clamp_min(0)]
    loss = module.SUBMISSION.token_training_loss(
        TokenLossBatch(
            logits=token_logits,
            labels=batch["labels"],
            valid_mask=batch["labels"].ne(-100),
            target_positions=batch["target_positions"],
            auxiliary=auxiliary,
        )
    )
    loss.backward()
    bundle = module.SUBMISSION.build_optimizer(model, OptimizerSpec(300.0, "cuda"))
    bundle.optimizer.step()
    optimizer_state_elements = sum(
        value.numel()
        for state in bundle.optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    non_trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    source = args.submission.read_bytes()
    result = {
        "submission_sha256": hashlib.sha256(source).hexdigest(),
        "variant": getattr(module, "VARIANT", "unknown"),
        "batch_size": args.batch_size,
        "modulus": 4095,
        "time_steps": 4,
        "dtype": "float32",
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
        "model_state_elements": count_model_state_elements(model),
        "optimizer_state_elements": optimizer_state_elements,
        "loss": float(loss.detach()),
        "finite_logits": bool(torch.isfinite(logits).all()),
        "finite_loss": bool(torch.isfinite(loss)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
