# Kaggle P100 workflow for geometric V1/V1.1

The script runs without notebook interaction. Its checked-in setting performs
the isolated E5 V1.1 comparison: `control`, unchanged `full`, and
`relative_full`. The gate compares `relative_full` against `control`, while
`full_vs_relative.json` records the one-variable architectural comparison.
Failure ablations are disabled because the earlier E5 run already measured
`fourier` and `snap_no_landmark_loss`.

The checked-in first run is already pinned to the tested implementation commit
and the `diaris` Kaggle account. For another fork, edit exactly two fields:

1. Replace `diaris` in `kernel-metadata.json` with the Kaggle account slug.
2. Set `GIT_COMMIT` near the top of `geometric_navigation_v1.py` to a pushed
   commit containing V1. Do not use a branch name.

V1.1 must improve both seen-N and OOD-N T=1 accuracy and beat control by ten
test-accuracy points before a repeat is justified. The diagnostics also report
correct-landmark mean rank, reciprocal rank, and Recall@1/8/16/32/64/128 on
both test and OOD splits. Do not begin V2 based only on a throughput or
fixed-modulus result.

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
standalone submission, and its SHA-256. E5 also has `gate.json` and
`full_vs_relative.json`.

Internet is enabled only because the kernel clones the pinned repository and
installs the P100-compatible PyTorch wheel. The kernel remains private and uses
one GPU.
