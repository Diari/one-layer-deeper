from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from benchmark.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/make_kaggle_manifest.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("make_kaggle_manifest", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dataset", ["e1", "e2", "e3", "e4", "e5"])
def test_generated_easy_manifests_preserve_dataset_semantics(dataset, tmp_path):
    module = load_tool()
    args = argparse.Namespace(
        output=tmp_path / f"{dataset}.json",
        dataset=dataset,
        batch_size=64,
        eval_batch_size=128,
        training_duration=300.0,
        seed=74,
        device="cuda:0",
        dtype="float32",
        amp=False,
        compile=False,
    )
    payload = module.generate_manifest(args)
    args.output.write_text(module.json.dumps(payload), encoding="utf-8")
    parsed = load_manifest(args.output)
    official = load_manifest(module.OFFICIAL_MANIFESTS[dataset])
    assert parsed.data.data_root == official.data.data_root
    assert parsed.data.batch_size == 64
    assert parsed.data.eval_batch_size == 128
    assert parsed.runtime.total_training_time_seconds == 300.0
    assert parsed.runtime.seeds == (74,)
    assert parsed.runtime.dtype == "float32"
    assert not parsed.runtime.amp
    assert not parsed.runtime.compile


def test_cpu_manifest_disables_worker_pinning(tmp_path):
    module = load_tool()
    args = argparse.Namespace(
        output=tmp_path / "cpu.json",
        dataset="e5",
        batch_size=2,
        eval_batch_size=2,
        training_duration=1.0,
        seed=74,
        device="cpu",
        dtype="float32",
        amp=False,
        compile=False,
    )
    payload = module.generate_manifest(args)
    assert payload["data"]["num_workers"] == 0
    assert payload["data"]["pin_memory"] is False
