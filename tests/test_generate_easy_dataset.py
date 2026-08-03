from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    path = ROOT / "tools/generate_easy_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_easy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_easy_configs_match_official_public_ladder(tmp_path):
    module = load_tool()
    configs = {dataset: module.config(dataset, tmp_path) for dataset in module.OUTPUTS}
    assert configs["e1"].fixed_p == 17 and configs["e1"].time_steps == [1, 2, 3]
    assert configs["e2"].fixed_p == 29 and configs["e2"].time_steps == [1, 2, 4]
    assert configs["e3"].modulus_bits == [10, 11]
    assert configs["e3"].fixed_time_steps == 2
    assert configs["e4"].modulus_bits == [11, 12]
    assert configs["e4"].examples_per_setting == 4000
    assert configs["e5"].modulus_bits == [10, 11]
    assert configs["e5"].time_steps == [1, 2, 3]
    for value in configs.values():
        assert value.separate_input_output
        assert value.depth_evaluation_time_steps == [1, 2, 4, 8, 16, 32, 64]
