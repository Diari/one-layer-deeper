# Variable-N Geometric Navigation V1

This directory contains the Easy-only variable-modulus successor to the fixed
N=323 V0. `submission.py` is the standalone source of truth. It supports
candidate residues 0 through 4095 and masks candidates at or above each
example's parsed modulus.

The checked-in submission is the `full` variant. Matched standalone ablations
are generated without hand-editing it:

```bash
python tools/make_geometric_v1_variant.py \
  --variant control --output /tmp/geometric-v1-control/submission.py
```

Available variants are:

- `control`: learned residue start embedding and recurrent model, no geometry.
- `digit_x`: ordered decimal x-digit start encoder replacing only the control's
  absolute residue lookup; modulus encoding and decoder remain unchanged.
- `fourier`: dynamic Fourier landmark start state, no projection.
- `snap_no_landmark_loss`: Fourier landmarks and snapping, no landmark CE.
- `full`: Fourier landmarks, snapping, landmark CE, and entropy regularization.
- `relative_full`: identical to `full` except that landmarks contain no learned
  absolute residue embedding; this is the isolated V1.1 experiment.

Run the focused CPU checks with:

```bash
uv run pytest -q tests/test_geometric_navigation_v1.py
uv run python -m benchmark.runner \
  --manifest experiments/geometric_navigation/configs/smoke_cpu_geometric.json \
  --submission-file submissions/geometric_navigation_v1/submission.py
uv run python tools/make_kaggle_manifest.py \
  --dataset e1 --device cpu --training-duration 1 \
  --output /tmp/geometric-v1-e1-manifest-check.json
```

For a matched local GPU E1 run:

```bash
uv run python tools/generate_easy_dataset.py --dataset e1
uv run python tools/make_kaggle_manifest.py \
  --dataset e1 \
  --output experiments/geometric_navigation_v1/configs/e1_local_gpu.json
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmark.runner \
  --manifest experiments/geometric_navigation_v1/configs/e1_local_gpu.json \
  --submission-file submissions/geometric_navigation_v1/submission.py \
  --include-structured-metrics
```

After a P100 gate passes, official H100 calibration requires hosted
authentication and separate control/full submissions:

```bash
one-layer login
one-layer submit /tmp/geometric-v1-control/submission.py \
  --tier easy --dataset e1 --wait
one-layer submit submissions/geometric_navigation_v1/submission.py \
  --tier easy --dataset e1 --wait
```

Generate the control immediately beforehand with the variant tool so its hash
is recorded. Replace `e1` only after the preceding gate passes.

The P100 workflow and artifact contract are documented under
`kaggle/geometric-navigation-v1/`. The model is intentionally capped at 4096
landmarks and four decimal output digits. It is not a Medium/Hard or scalable
variable-N design.
