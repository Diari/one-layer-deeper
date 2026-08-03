#!/usr/bin/env python3
"""Generate one official public Easy dataset with the documented semantics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.squaring_mod import SquaringModGenerationConfig, generate_squaring_mod_dataset


OUTPUTS = {
    "e1": "squaring_mod_new11_easy_bidirectional_fixed_n_323_t123",
    "e2": "squaring_mod_new11_easy_bidirectional_fixed_n_899_t124",
    "e3": "squaring_mod_new11_easy_bidirectional_fixed_t_b1011_t2",
    "e4": "squaring_mod_new11_easy_bidirectional_fixed_t_b1112_t2",
    "e5": "squaring_mod_new11_easy_bidirectional_variable_b1011_t123",
}


def config(dataset: str, root: Path) -> SquaringModGenerationConfig:
    common = dict(
        output_dir=str(root / OUTPUTS[dataset]),
        train_fraction=0.8,
        test_fraction=0.2,
        split_group="prompt",
        seed=45,
        separate_input_output=True,
        depth_evaluation_time_steps=[1, 2, 4, 8, 16, 32, 64],
        ood_n_depth_evaluation_examples_per_setting=256,
    )
    if dataset == "e1":
        return SquaringModGenerationConfig(
            **common,
            fixed_p=17,
            fixed_q=19,
            time_steps=[1, 2, 3],
            ood_time_steps=[6],
            examples_per_setting=250,
            ood_examples_per_setting=100,
            depth_evaluation_exhaustive_x=True,
            ood_n_depth_evaluation_modulus_bits=[10, 11],
        )
    if dataset == "e2":
        return SquaringModGenerationConfig(
            **common,
            fixed_p=29,
            fixed_q=31,
            time_steps=[1, 2, 4],
            ood_time_steps=[7],
            examples_per_setting=800,
            ood_examples_per_setting=300,
            depth_evaluation_exhaustive_x=True,
            ood_n_depth_evaluation_modulus_bits=[11, 12],
        )
    if dataset == "e3":
        return SquaringModGenerationConfig(
            **common,
            modulus_bits=[10, 11],
            fixed_time_steps=2,
            ood_time_steps=[4],
            examples_per_setting=2000,
            ood_examples_per_setting=400,
            depth_evaluation_examples_per_setting=256,
            ood_n_depth_evaluation_modulus_bits=[12, 13],
        )
    if dataset == "e4":
        return SquaringModGenerationConfig(
            **common,
            modulus_bits=[11, 12],
            fixed_time_steps=2,
            ood_time_steps=[4],
            examples_per_setting=4000,
            ood_examples_per_setting=600,
            depth_evaluation_examples_per_setting=256,
            ood_n_depth_evaluation_modulus_bits=[13, 14],
        )
    return SquaringModGenerationConfig(
        **common,
        modulus_bits=[10, 11],
        time_steps=[1, 2, 3],
        ood_time_steps=[6],
        examples_per_setting=1000,
        ood_examples_per_setting=300,
        depth_evaluation_examples_per_setting=256,
        ood_n_depth_evaluation_modulus_bits=[12, 13],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(OUTPUTS), required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "data/generated")
    args = parser.parse_args()
    generation_config = config(args.dataset, args.root)
    output = Path(generation_config.output_dir)
    if (output / "dataset_config.json").exists():
        print(f"using existing {args.dataset}: {output}")
        return
    result = generate_squaring_mod_dataset(generation_config)
    print(f"generated {args.dataset}: {result['num_examples']} examples at {output}")


if __name__ == "__main__":
    main()
