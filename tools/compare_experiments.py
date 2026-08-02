#!/usr/bin/env python3
"""Compare matched structured experiment results without overclaiming."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MATCH_FIELDS = (
    "dataset",
    "seed",
    "gpu",
)
MATCH_CONFIG_FIELDS = (
    "batch_size",
    "eval_batch_size",
    "training_time_seconds",
    "dtype",
    "amp",
    "compile",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    mismatches = {}
    for field in MATCH_FIELDS:
        if baseline.get(field) != candidate.get(field):
            mismatches[field] = [baseline.get(field), candidate.get(field)]
    for field in MATCH_CONFIG_FIELDS:
        left = baseline.get("config", {}).get(field)
        right = candidate.get("config", {}).get(field)
        if left != right:
            mismatches[f"config.{field}"] = [left, right]

    baseline_accuracy = baseline.get("metrics", {}).get("exact_accuracy")
    candidate_accuracy = candidate.get("metrics", {}).get("exact_accuracy")
    delta = None
    if baseline_accuracy is not None and candidate_accuracy is not None:
        delta = candidate_accuracy - baseline_accuracy
    result = {
        "matched": not mismatches,
        "mismatches": mismatches,
        "baseline_variant": baseline.get("variant"),
        "candidate_variant": candidate.get("variant"),
        "exact_accuracy_delta": delta if not mismatches else None,
        "conclusion": (
            "matched comparison; inspect uncertainty and repeated seeds"
            if not mismatches
            else "not comparable until mismatched controls are aligned"
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
