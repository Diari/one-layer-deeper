#!/usr/bin/env python3
"""Generate a self-contained matched V1 ablation from the source of truth."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions/geometric_navigation_v1/submission.py"
VARIANTS = ("control", "fourier", "snap_no_landmark_loss", "full")
SOURCE_LINE = 'VARIANT = "full"'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = SOURCE.read_text(encoding="utf-8")
    if source.count(SOURCE_LINE) != 1:
        raise ValueError("source variant marker is missing or ambiguous")
    rendered = source.replace(SOURCE_LINE, f'VARIANT = "{args.variant}"', 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    print(f"variant={args.variant} sha256={digest} output={args.output}")


if __name__ == "__main__":
    main()
