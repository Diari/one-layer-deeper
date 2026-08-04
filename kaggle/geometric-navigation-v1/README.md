# Kaggle P100 workflow for the V1 snapping ablation

The script runs without notebook interaction. Its checked-in setting performs
one isolated E5 comparison: unchanged `full` V1 against
`no_snap_landmark_loss`. The candidate retains the full landmark bank, masked
similarity distribution, landmark CE, entropy, optimizer, seed, batches, and
training duration. It removes only the recurrent landmark-weighted state
replacement. The run stops after this matched pair.

The checked-in first run is already pinned to the tested implementation commit
and the `diaris` Kaggle account. For another fork, edit exactly two fields:

1. Replace `diaris` in `kernel-metadata.json` with the Kaggle account slug.
2. Set `GIT_COMMIT` near the top of `geometric_navigation_v1.py` to a pushed
   commit containing V1. Do not use a branch name.

This completes the landmark-supervision/snapping 2x2. If the candidate retains
the full model's gain, landmark supervision rather than recurrent projection is
the operative mechanism. Repeat a material effect before promotion.

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

kaggle kernels status USERNAME/geometric-navigation-no-snap-e5

mkdir -p experiments/geometric_navigation_v1/results/latest

kaggle kernels output \
  USERNAME/geometric-navigation-no-snap-e5 \
  -p experiments/geometric_navigation_v1/results/latest \
  --force
```

The downloaded archive contains `artifacts/summary.json`, environment and test
logs, the generated manifest, maximum-N memory preflight, and per-dataset,
per-variant directories containing `result.json`, `diagnostics.json`, the exact
standalone submission, and its SHA-256. E5 also has
`full_vs_no_snap_landmark_loss.json`.

Internet is enabled only because the kernel clones the pinned repository and
installs the P100-compatible PyTorch wheel. The kernel remains private and uses
one GPU.
