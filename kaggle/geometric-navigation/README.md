# Kaggle P100 workflow

The checked-in configuration targets the private kernel
`diaris/geometric-navigation-e1-p100` and the public fork at
`github.com/Diari/one-layer-deeper`. No metadata edit is required for that
account. To reuse this workflow under another account:

1. Push this branch to a GitHub repository Kaggle can clone.
2. In `geometric_navigation.py`, set `REPOSITORY_URL` and
   `GIT_COMMIT` to the full 40-character commit containing this prototype.
3. In `kernel-metadata.json`, replace `diaris` in the `id` field with your
   Kaggle username. Keep the slug `geometric-navigation-e1-p100`.

The kernel is private, GPU-enabled, and internet-enabled so it can clone the
exact commit. No credential is stored in the kernel. Install the Kaggle API
credential separately at `~/.kaggle/kaggle.json`; never print or commit it.

```bash
pip install kaggle

kaggle kernels push \
  -p kaggle/geometric-navigation \
  --accelerator NvidiaTeslaP100

kaggle kernels status diaris/geometric-navigation-e1-p100

mkdir -p experiments/geometric_navigation/results/latest

kaggle kernels output \
  diaris/geometric-navigation-e1-p100 \
  -p experiments/geometric_navigation/results/latest \
  --force
```

For another account, replace `diaris` in the status and output commands too.
The downloaded output contains `artifacts.zip`; extracted artifacts include `summary.json`, runner
results, diagnostics, logs, environment and GPU reports, the generated
manifest, and a SHA-256-addressed copy of the standalone submission. If
`RUN_BASELINE` is false, `baseline_result.json` is cleanly absent and the
summary marks that stage `not_run`.
