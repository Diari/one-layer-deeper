#!/usr/bin/env python3
"""Evaluate a matched control/full Easy gate from benchmark RESULT_JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rung(result: dict, family: str, time_steps: int) -> float | None:
    key = "rungs" if family == "seen" else "ood_n_rungs"
    for value in result["seeds"][0]["depth_profile"].get(key, []):
        if value.get("time_steps") == time_steps:
            return value.get("exact_accuracy")
    return None


def measurements(result: dict) -> dict:
    seed = result["seeds"][0]
    evaluation = seed["evaluation"]
    return {
        "manifest": result["manifest"],
        "seed": seed["seed"],
        "batch_size": seed["training_batch_size"],
        "eval_batch_size": seed["evaluation_batch_size"],
        "training_seconds": seed["training_seconds"],
        "optimizer_steps": seed["completed_training_steps"],
        "model_state_elements": seed["model_state_elements"],
        "test_exact_accuracy": evaluation["test"]["exact_accuracy"],
        "ood_exact_accuracy": evaluation.get("ood", {}).get("exact_accuracy"),
        "seen_t1_accuracy": rung(result, "seen", 1),
        "ood_n_t1_accuracy": rung(result, "ood_n", 1),
    }


def compare_pair(dataset: str, control: dict, full: dict) -> dict:
    left = measurements(control)
    right = measurements(full)
    matched_fields = ("manifest", "seed", "batch_size", "eval_batch_size")
    mismatches = {
        field: [left[field], right[field]]
        for field in matched_fields
        if left[field] != right[field]
    }
    training_delta = abs(left["training_seconds"] - right["training_seconds"])
    if training_delta > 1.0:
        mismatches["training_seconds"] = [
            left["training_seconds"],
            right["training_seconds"],
        ]
    accuracy_delta = right["test_exact_accuracy"] - left["test_exact_accuracy"]
    requirements = {
        "matched": not mismatches,
        "test_delta_at_least_0_10": accuracy_delta >= 0.10,
    }
    if dataset == "e1":
        requirements["full_test_accuracy_at_least_0_70"] = (
            right["test_exact_accuracy"] >= 0.70
        )
    if dataset == "e5":
        requirements["seen_t1_improved"] = (
            right["seen_t1_accuracy"] is not None
            and left["seen_t1_accuracy"] is not None
            and right["seen_t1_accuracy"] > left["seen_t1_accuracy"]
        )
        requirements["ood_n_t1_improved"] = (
            right["ood_n_t1_accuracy"] is not None
            and left["ood_n_t1_accuracy"] is not None
            and right["ood_n_t1_accuracy"] > left["ood_n_t1_accuracy"]
        )
    gated = dataset in ("e1", "e2", "e5")
    return {
        "dataset": dataset,
        "gated": gated,
        "matched": not mismatches,
        "mismatches": mismatches,
        "control": left,
        "full": right,
        "test_exact_accuracy_delta": accuracy_delta,
        "requirements": requirements,
        "passed": all(requirements.values()) if gated else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("e1", "e2", "e3", "e4", "e5"), required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--repeat-control", type=Path)
    parser.add_argument("--repeat-full", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "primary": compare_pair(args.dataset, load(args.control), load(args.full))
    }
    if (args.repeat_control is None) != (args.repeat_full is None):
        raise ValueError("both repeat files are required together")
    if args.repeat_control is not None and args.repeat_full is not None:
        result["repeat"] = compare_pair(
            args.dataset, load(args.repeat_control), load(args.repeat_full)
        )
    result["passed"] = result["primary"]["passed"]
    if args.dataset == "e5":
        result["passed"] = bool(
            result["primary"]["passed"]
            and result.get("repeat", {}).get("passed", False)
        )
        result["repeat_required"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
