"""Non-interactive Kaggle P100 workflow for matched variable-N V1 gates."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback


REPOSITORY_URL = "https://github.com/Diari/one-layer-deeper.git"
GIT_COMMIT = "e5945d7ab5e89c86964540ada0a67cfe128f23bb"
EXPERIMENT_SEQUENCE = ("e1", "e2", "e5", "e3", "e4")
STOP_AFTER_DATASET = "e1"
PRIMARY_VARIANTS = ("control", "full")
RUN_FAILURE_ABLATIONS = True
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
TRAINING_SECONDS = 300
SEED = 74
P100_TORCH_VERSION = "2.10.0+cu126"
P100_TORCH_INDEX = "https://download.pytorch.org/whl/cu126"

WORKING = Path("/kaggle/working")
REPOSITORY = WORKING / "one-layer-deeper"
ARTIFACTS = WORKING / "artifacts"
RUNNER_LOG = ARTIFACTS / "runner.log"
TESTS_LOG = ARTIFACTS / "tests.log"
SUMMARY = ARTIFACTS / "summary.json"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    log: Path | None = None,
    append: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    rendered = " ".join(command)
    print(f"$ {rendered}", flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="", flush=True)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with log.open(mode, encoding="utf-8") as handle:
            handle.write(f"$ {rendered}\n")
            handle.write(completed.stdout)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, command, output=completed.stdout
        )
    return completed


def archive() -> None:
    destination = WORKING / "artifacts.zip"
    if destination.exists():
        destination.unlink()
    shutil.make_archive(str(destination.with_suffix("")), "zip", ARTIFACTS)


def remove_cloned_repository() -> None:
    """Keep Kaggle output limited to the requested artifact contract."""

    if REPOSITORY.exists():
        shutil.rmtree(REPOSITORY)


def environment_report() -> dict:
    import torch

    available = torch.cuda.is_available()
    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "configured_git_commit": GIT_COMMIT,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "gpu_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if available else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": available,
        "bf16_supported": torch.cuda.is_bf16_supported() if available else False,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def selected_datasets() -> tuple[str, ...]:
    if STOP_AFTER_DATASET is None:
        return EXPERIMENT_SEQUENCE
    if STOP_AFTER_DATASET not in EXPERIMENT_SEQUENCE:
        raise ValueError("STOP_AFTER_DATASET must be in EXPERIMENT_SEQUENCE or None")
    index = EXPERIMENT_SEQUENCE.index(STOP_AFTER_DATASET)
    return EXPERIMENT_SEQUENCE[: index + 1]


def run_benchmark(submission: Path, manifest: Path, result: Path) -> None:
    marker = f"===== runner: {manifest.stem} {submission.parent.name} =====\n"
    with RUNNER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(marker)
    run(
        [
            sys.executable,
            "-m",
            "benchmark.runner",
            "--manifest",
            str(manifest),
            "--submission-file",
            str(submission),
            "--include-structured-metrics",
        ],
        cwd=REPOSITORY,
        log=RUNNER_LOG,
        append=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )
    run(
        [
            sys.executable,
            "tools/extract_result_json.py",
            str(RUNNER_LOG),
            "--output",
            str(result),
        ],
        cwd=REPOSITORY,
        log=RUNNER_LOG,
        append=True,
    )


def run_variant(dataset: str, variant: str, manifest: Path) -> dict:
    directory = ARTIFACTS / dataset / variant
    directory.mkdir(parents=True, exist_ok=True)
    submission = directory / "submission.py"
    run(
        [
            sys.executable,
            "tools/make_geometric_v1_variant.py",
            "--variant",
            variant,
            "--output",
            str(submission),
        ],
        cwd=REPOSITORY,
        log=RUNNER_LOG,
        append=True,
    )
    digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    (directory / "submission.sha256").write_text(
        f"{digest}  submission.py\n", encoding="utf-8"
    )
    result = directory / "result.json"
    run_benchmark(submission, manifest, result)
    run(
        [
            sys.executable,
            "tools/diagnose_geometric_navigation.py",
            "--manifest",
            str(manifest),
            "--submission",
            str(submission),
            "--output",
            str(directory / "diagnostics.json"),
            "--training-seconds",
            str(TRAINING_SECONDS),
            "--variant",
            f"geometric_v1_{variant}",
        ],
        cwd=REPOSITORY,
        log=RUNNER_LOG,
        append=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )
    payload = json.loads(result.read_text(encoding="utf-8"))
    seed_result = payload["seeds"][0]
    return {
        "dataset": dataset,
        "variant": variant,
        "commit": GIT_COMMIT,
        "submission_sha256": digest,
        "result": str(result.relative_to(ARTIFACTS)),
        "diagnostics": str((directory / "diagnostics.json").relative_to(ARTIFACTS)),
        "model_state_elements": seed_result["model_state_elements"],
        "optimizer_steps": seed_result["completed_training_steps"],
        "test_exact_accuracy": seed_result["evaluation"]["test"]["exact_accuracy"],
        "training_seconds": seed_result["training_seconds"],
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stages: dict[str, str] = {}
    runs: list[dict] = []
    environment = {}
    resolved = "unknown"
    try:
        print(datetime.now(timezone.utc).isoformat(), flush=True)
        nvidia = shutil.which("nvidia-smi")
        nvidia_output = run([nvidia]).stdout if nvidia else "nvidia-smi unavailable\n"
        (ARTIFACTS / "nvidia-smi.txt").write_text(nvidia_output, encoding="utf-8")
        stages["nvidia_smi"] = "passed" if nvidia else "unavailable"
        if "REPLACE_" in GIT_COMMIT or "USERNAME" in REPOSITORY_URL:
            raise ValueError("edit REPOSITORY_URL and GIT_COMMIT near the top")
        if REPOSITORY.exists():
            shutil.rmtree(REPOSITORY)
        run(["git", "clone", REPOSITORY_URL, str(REPOSITORY)])
        run(["git", "checkout", "--detach", GIT_COMMIT], cwd=REPOSITORY)
        resolved = run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY).stdout.strip()
        if resolved != GIT_COMMIT:
            raise RuntimeError(f"checkout mismatch: {resolved} != {GIT_COMMIT}")
        stages["checkout"] = "passed"

        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                f"torch=={P100_TORCH_VERSION}",
                "--index-url",
                P100_TORCH_INDEX,
            ],
            cwd=REPOSITORY,
            log=TESTS_LOG,
        )
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                ".",
                "--no-deps",
                "--ignore-requires-python",
            ],
            cwd=REPOSITORY,
            log=TESTS_LOG,
            append=True,
        )
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "jsonargparse==4.49.0",
                "modal>=1.1.0",
                "psycopg[binary]>=3.2,<4",
                "pytest>=8",
            ],
            cwd=REPOSITORY,
            log=TESTS_LOG,
            append=True,
        )
        run(
            [
                sys.executable,
                "-c",
                "import torch; x=torch.ones(1,device='cuda:0'); "
                "print(torch.__version__,torch.version.cuda,"
                "torch.cuda.get_device_name(0),(x+x).item())",
            ],
            cwd=REPOSITORY,
            log=TESTS_LOG,
            append=True,
        )
        environment = environment_report()
        (ARTIFACTS / "environment.txt").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stages["dependencies"] = "passed"

        run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=REPOSITORY,
            log=TESTS_LOG,
            append=True,
        )
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_geometric_navigation_v1.py",
                "tests/test_kaggle_manifest.py",
                "tests/test_compare_geometric_v1_gate.py",
                "tests/test_generate_easy_dataset.py",
            ],
            cwd=REPOSITORY,
            log=TESTS_LOG,
            append=True,
        )
        stages["tests"] = "passed"

        memory_submission = ARTIFACTS / "memory_preflight_submission.py"
        run(
            [
                sys.executable,
                "tools/make_geometric_v1_variant.py",
                "--variant",
                "full",
                "--output",
                str(memory_submission),
            ],
            cwd=REPOSITORY,
            log=RUNNER_LOG,
            append=True,
        )
        run(
            [
                sys.executable,
                "tools/profile_geometric_v1_memory.py",
                "--submission",
                str(memory_submission),
                "--output",
                str(ARTIFACTS / "maximum_n_memory.json"),
                "--batch-size",
                str(BATCH_SIZE),
            ],
            cwd=REPOSITORY,
            log=RUNNER_LOG,
            append=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        stages["maximum_n_memory"] = "passed"

        for dataset in selected_datasets():
            run(
                [sys.executable, "tools/generate_easy_dataset.py", "--dataset", dataset],
                cwd=REPOSITORY,
                log=RUNNER_LOG,
                append=True,
            )
            manifest = ARTIFACTS / "manifests" / f"{dataset}.json"
            run(
                [
                    sys.executable,
                    "tools/make_kaggle_manifest.py",
                    "--dataset",
                    dataset,
                    "--output",
                    str(manifest),
                    "--batch-size",
                    str(BATCH_SIZE),
                    "--eval-batch-size",
                    str(EVAL_BATCH_SIZE),
                    "--training-duration",
                    str(TRAINING_SECONDS),
                    "--seed",
                    str(SEED),
                    "--device",
                    "cuda:0",
                    "--dtype",
                    "float32",
                    "--no-amp",
                    "--no-compile",
                ],
                cwd=REPOSITORY,
                log=RUNNER_LOG,
                append=True,
            )
            stages[f"{dataset}_data_manifest"] = "passed"
            by_variant = {}
            for variant in PRIMARY_VARIANTS:
                run_summary = run_variant(dataset, variant, manifest)
                runs.append(run_summary)
                by_variant[variant] = ARTIFACTS / run_summary["result"]
                stages[f"{dataset}_{variant}"] = "passed"
            gate_path = ARTIFACTS / dataset / "gate.json"
            run(
                [
                    sys.executable,
                    "tools/compare_geometric_v1_gate.py",
                    "--dataset",
                    dataset,
                    "--control",
                    str(by_variant["control"]),
                    "--full",
                    str(by_variant["full"]),
                    "--output",
                    str(gate_path),
                ],
                cwd=REPOSITORY,
                log=RUNNER_LOG,
                append=True,
            )
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            primary_passed = gate["primary"]["passed"]
            if dataset == "e5" and primary_passed:
                stages[f"{dataset}_gate"] = "primary_passed_repeat_required"
            else:
                stages[f"{dataset}_gate"] = (
                    "passed" if primary_passed else "failed_or_informational"
                )
            if primary_passed is False and RUN_FAILURE_ABLATIONS:
                for variant in ("fourier", "snap_no_landmark_loss"):
                    runs.append(run_variant(dataset, variant, manifest))
                    stages[f"{dataset}_{variant}"] = "passed"
            if primary_passed is False and dataset in ("e1", "e2", "e5"):
                print(f"Gate {dataset} failed; stopping the sequence.", flush=True)
                break
            if dataset == "e5" and primary_passed:
                print("E5 primary gate passed; repeat it before promotion.", flush=True)
                break

        write_json(
            SUMMARY,
            {
                "passed": True,
                "commit": resolved,
                "experiment_sequence": list(selected_datasets()),
                "runs": runs,
                "stages": stages,
                "environment": environment,
            },
        )
        remove_cloned_repository()
        archive()
    except Exception as exc:
        stages["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(
            SUMMARY,
            {
                "passed": False,
                "commit": resolved,
                "runs": runs,
                "stages": stages,
                "environment": environment,
                "traceback": traceback.format_exc(),
            },
        )
        remove_cloned_repository()
        archive()
        raise


if __name__ == "__main__":
    main()
