"""Non-interactive Kaggle GPU entry point for the fixed-N E1 prototype."""

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
GIT_COMMIT = "85a4bce0bd541f28282db356b4d66c5e49972c74"
RUN_BASELINE = True
RUN_GEOMETRIC = True
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
        mode = "a" if append else "w"
        with log.open(mode, encoding="utf-8") as handle:
            handle.write(f"$ {rendered}\n")
            handle.write(completed.stdout)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, command, output=completed.stdout
        )
    return completed


def create_archive() -> None:
    archive = WORKING / "artifacts.zip"
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", ARTIFACTS)


def environment_report() -> dict:
    import torch

    gpu = None
    memory = None
    bf16 = False
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory
        bf16 = torch.cuda.is_bf16_supported()
    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "configured_git_commit": GIT_COMMIT,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu_name": gpu,
        "gpu_memory_bytes": memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": bf16,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def generate_e1_dataset() -> None:
    output = (
        REPOSITORY
        / "data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123"
    )
    if (output / "config.json").exists():
        print(f"Using existing E1 dataset: {output}", flush=True)
        return
    run(
        [
            sys.executable, "-m", "data.squaring_mod",
            "--output_dir", str(output),
            "--fixed_p", "17", "--fixed_q", "19",
            "--time_steps", "[1,2,3]", "--ood_time_steps", "[6]",
            "--examples_per_setting", "250",
            "--ood_examples_per_setting", "100",
            "--depth_evaluation_time_steps", "[1,2,4,8,16,32,64]",
            "--depth_evaluation_exhaustive_x", "true",
            "--ood_n_depth_evaluation_modulus_bits", "[10,11]",
            "--ood_n_depth_evaluation_examples_per_setting", "256",
            "--train_fraction", "0.8", "--test_fraction", "0.2",
            "--split_group", "prompt", "--seed", "45",
            "--separate_input_output", "true",
        ],
        cwd=REPOSITORY, log=RUNNER_LOG, append=True,
    )


def runner(submission: Path, result: Path) -> None:
    manifest = ARTIFACTS / "generated_manifest.json"
    with RUNNER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"===== runner: {submission} =====\n")
    run(
        [
            sys.executable, "-m", "benchmark.runner",
            "--manifest", str(manifest),
            "--submission-file", str(submission),
            "--include-structured-metrics",
        ],
        cwd=REPOSITORY, log=RUNNER_LOG, append=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )
    run(
        [
            sys.executable, "tools/extract_result_json.py", str(RUNNER_LOG),
            "--output", str(result),
        ],
        cwd=REPOSITORY, log=RUNNER_LOG, append=True,
    )


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stages: dict[str, str] = {}
    environment = {}
    try:
        print(datetime.now(timezone.utc).isoformat(), flush=True)
        nvidia_path = shutil.which("nvidia-smi")
        if nvidia_path is None:
            nvidia_output = "nvidia-smi is not available on PATH\n"
            print(nvidia_output, end="", flush=True)
            stages["nvidia_smi"] = "unavailable"
        else:
            nvidia_output = run([nvidia_path]).stdout
            stages["nvidia_smi"] = "passed"
        (ARTIFACTS / "nvidia-smi.txt").write_text(
            nvidia_output, encoding="utf-8"
        )
        environment = environment_report()
        (ARTIFACTS / "environment.txt").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stages["environment"] = "passed"

        if "USERNAME" in REPOSITORY_URL or GIT_COMMIT.startswith("REPLACE_"):
            raise ValueError("edit REPOSITORY_URL and GIT_COMMIT near the top of the kernel")
        if REPOSITORY.exists():
            shutil.rmtree(REPOSITORY)
        run(["git", "clone", REPOSITORY_URL, str(REPOSITORY)])
        run(["git", "checkout", "--detach", GIT_COMMIT], cwd=REPOSITORY)
        resolved = run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY).stdout.strip()
        if resolved != GIT_COMMIT:
            raise RuntimeError(f"checkout mismatch: {resolved} != {GIT_COMMIT}")
        print(f"Git commit: {resolved}", flush=True)
        stages["checkout"] = "passed"

        # Kaggle currently ships a CUDA 12.8 wheel that omits Pascal (sm_60),
        # while its P100 accelerator is Pascal.  The official CUDA 12.6 wheel
        # retains sm_60.  Install it before importing any benchmark modules.
        run(
            [
                sys.executable, "-m", "pip", "install", "--upgrade",
                "--force-reinstall", f"torch=={P100_TORCH_VERSION}",
                "--index-url", P100_TORCH_INDEX,
            ],
            cwd=REPOSITORY, log=TESTS_LOG,
        )
        # Kaggle is on Python 3.12, whereas the official evaluator pins 3.13.5.
        # Installing the source tree without dependency resolution lets this
        # architecture-validation run use Kaggle's interpreter without changing
        # the repository metadata or evaluator behavior.
        run(
            [
                sys.executable, "-m", "pip", "install", "-e", ".",
                "--no-deps", "--ignore-requires-python",
            ],
            cwd=REPOSITORY, log=TESTS_LOG, append=True,
        )
        run(
            [
                sys.executable, "-m", "pip", "install",
                "jsonargparse==4.49.0", "modal>=1.1.0",
                "psycopg[binary]>=3.2,<4",
            ],
            cwd=REPOSITORY, log=TESTS_LOG, append=True,
        )
        # Fail immediately if the replacement wheel cannot execute on the P100.
        run(
            [
                sys.executable, "-c",
                "import torch; x=torch.ones(1,device='cuda:0'); "
                "print(torch.__version__, torch.version.cuda, "
                "torch.cuda.get_device_name(0), (x+x).item())",
            ],
            cwd=REPOSITORY, log=TESTS_LOG, append=True,
        )
        stages["dependencies"] = "passed"
        run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=REPOSITORY, log=TESTS_LOG, append=True,
        )
        stages["repository_tests"] = "passed"
        run(
            [
                sys.executable, "-m", "unittest", "discover",
                "-s", "tests", "-p", "test_geometric_navigation.py", "-v",
            ],
            cwd=REPOSITORY, log=TESTS_LOG, append=True,
        )
        stages["geometric_tests"] = "passed"

        generate_e1_dataset()
        stages["dataset"] = "passed"
        run(
            [
                sys.executable, "tools/make_kaggle_manifest.py",
                "--output", str(ARTIFACTS / "generated_manifest.json"),
                "--batch-size", str(BATCH_SIZE),
                "--eval-batch-size", str(EVAL_BATCH_SIZE),
                "--training-duration", str(TRAINING_SECONDS),
                "--seed", str(SEED), "--device", "cuda:0",
                "--dtype", "float32", "--no-amp", "--no-compile",
            ],
            cwd=REPOSITORY, log=RUNNER_LOG, append=True,
        )
        stages["manifest"] = "passed"

        if RUN_BASELINE:
            runner(
                REPOSITORY / "submissions/baseline_adamw/submission.py",
                ARTIFACTS / "baseline_result.json",
            )
            stages["baseline"] = "passed"
        else:
            stages["baseline"] = "not_run"
        if RUN_GEOMETRIC:
            runner(
                REPOSITORY / "submissions/geometric_navigation/submission.py",
                ARTIFACTS / "geometric_result.json",
            )
            stages["geometric"] = "passed"
        else:
            stages["geometric"] = "not_run"

        run(
            [
                sys.executable, "tools/diagnose_geometric_navigation.py",
                "--manifest", str(ARTIFACTS / "generated_manifest.json"),
                "--submission", "submissions/geometric_navigation/submission.py",
                "--output", str(ARTIFACTS / "diagnostics.json"),
                "--training-seconds", str(TRAINING_SECONDS),
            ],
            cwd=REPOSITORY, log=RUNNER_LOG, append=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        stages["diagnostics"] = "passed"

        submission_source = REPOSITORY / "submissions/geometric_navigation/submission.py"
        copied_submission = ARTIFACTS / "submission.py"
        shutil.copy2(submission_source, copied_submission)
        digest = hashlib.sha256(copied_submission.read_bytes()).hexdigest()
        (ARTIFACTS / "submission.sha256").write_text(
            f"{digest}  submission.py\n", encoding="utf-8"
        )
        stages["hash"] = "passed"
        write_json(
            SUMMARY,
            {
                "passed": True, "commit": resolved,
                "submission_sha256": digest, "stages": stages,
                "environment": environment,
            },
        )
        create_archive()
    except Exception as exc:
        stages["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(
            SUMMARY,
            {
                "passed": False, "stages": stages, "environment": environment,
                "traceback": traceback.format_exc(),
            },
        )
        create_archive()
        raise


if __name__ == "__main__":
    main()
