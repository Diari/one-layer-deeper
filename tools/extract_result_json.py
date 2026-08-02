#!/usr/bin/env python3
"""Extract the last evaluator RESULT_JSON record from a runner log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFIX = "RESULT_JSON="


def extract(path: Path) -> dict:
    matches = [
        line[len(PREFIX) :]
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith(PREFIX)
    ]
    if not matches:
        raise ValueError(f"no {PREFIX} line in {path}")
    return json.loads(matches[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = extract(args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
