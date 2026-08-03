#!/usr/bin/env python3
"""Generate a local Easy manifest without changing evaluator-owned manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MANIFESTS = {
    dataset: ROOT / f"benchmark/manifests/h100_easy_{dataset}.json"
    for dataset in ("e1", "e2", "e3", "e4", "e5")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=tuple(OFFICIAL_MANIFESTS), default="e1"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--training-duration", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def generate_manifest(args: argparse.Namespace) -> dict:
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.training_duration <= 0:
        raise ValueError("training duration must be positive")
    official = OFFICIAL_MANIFESTS[args.dataset]
    payload = json.loads(official.read_text(encoding="utf-8"))
    payload["name"] = (
        f"squaring-mod-easy-{args.dataset}-kaggle-geometric-navigation"
    )
    payload["data"]["batch_size"] = args.batch_size
    payload["data"]["eval_batch_size"] = args.eval_batch_size
    payload["runtime"].update(
        {
            "device": args.device,
            "dtype": args.dtype,
            "amp": args.amp,
            "compile": args.compile,
            "total_training_time_seconds": args.training_duration,
            "seeds": [args.seed],
        }
    )
    if args.device == "cpu":
        payload["data"]["pin_memory"] = False
        payload["data"]["num_workers"] = 0
    return payload


def main() -> None:
    args = parse_args()
    payload = generate_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Parse the generated file through the evaluator's strict schema.
    load_manifest(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
