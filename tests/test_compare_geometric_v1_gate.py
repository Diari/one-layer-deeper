from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    path = ROOT / "tools/compare_geometric_v1_gate.py"
    spec = importlib.util.spec_from_file_location("compare_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(test_accuracy, seen_t1, ood_t1):
    return {
        "manifest": "squaring-mod-easy-e5-kaggle-geometric-navigation",
        "seeds": [
            {
                "seed": 74,
                "training_batch_size": 64,
                "evaluation_batch_size": 128,
                "training_seconds": 300.0,
                "completed_training_steps": 100,
                "model_state_elements": 1000,
                "evaluation": {
                    "test": {"exact_accuracy": test_accuracy},
                    "ood": {"exact_accuracy": test_accuracy},
                },
                "depth_profile": {
                    "rungs": [{"time_steps": 1, "exact_accuracy": seen_t1}],
                    "ood_n_rungs": [
                        {"time_steps": 1, "exact_accuracy": ood_t1}
                    ],
                },
            }
        ],
    }


def test_e5_gate_requires_delta_and_both_t1_improvements():
    module = load_tool()
    passed = module.compare_pair("e5", result(0.2, 0.1, 0.1), result(0.31, 0.2, 0.2))
    assert passed["passed"]
    failed = module.compare_pair("e5", result(0.2, 0.1, 0.1), result(0.31, 0.1, 0.2))
    assert not failed["passed"]


def test_e1_gate_requires_absolute_accuracy():
    module = load_tool()
    comparison = module.compare_pair("e1", result(0.1, 0.1, 0.1), result(0.69, 0.2, 0.2))
    assert not comparison["passed"]
