# Kaggle P100 workflow for geometric V1

The script runs without notebook interaction. Its checked-in first-run setting
stops after the matched E1 `control` and `full` comparison. A failed gate also
runs the two diagnostic ablations. Later gates must retain this order:
E1 → E2 → E5 → E3 → E4.

The checked-in first run is already pinned to the tested implementation commit
and the `diaris` Kaggle account. For another fork, edit exactly two fields:

1. Replace `diaris` in `kernel-metadata.json` with the Kaggle account slug.
2. Set `GIT_COMMIT` near the top of `geometric_navigation_v1.py` to a pushed
   commit containing V1. Do not use a branch name.

To advance after a passing gate, change `STOP_AFTER_DATASET` to the next value.
Use `None` only when deliberately running the complete remaining sequence.
E5 promotion requires a second matched run; keep both artifact archives and
pass both pairs to `tools/compare_geometric_v1_gate.py`.

Install the Kaggle client and credentials locally (never print or commit the
credential):

```bash
pip install kaggle
mkdir -p ~/.kaggle
# place the downloaded token at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Push and retrieve the first P100 run:

```bash
kaggle kernels push \
  -p kaggle/geometric-navigation-v1 \
  --accelerator NvidiaTeslaP100

kaggle kernels status USERNAME/geometric-navigation-v1-easy

mkdir -p experiments/geometric_navigation_v1/results/latest

kaggle kernels output \
  USERNAME/geometric-navigation-v1-easy \
  -p experiments/geometric_navigation_v1/results/latest \
  --force
```

The downloaded archive contains `artifacts/summary.json`, environment and test
logs, the generated manifest, maximum-N memory preflight, and per-dataset,
per-variant directories containing `result.json`, `diagnostics.json`, the exact
standalone submission, and its SHA-256. Each dataset also has `gate.json`.

Internet is enabled only because the kernel clones the pinned repository and
installs the P100-compatible PyTorch wheel. The kernel remains private and uses
one GPU.
