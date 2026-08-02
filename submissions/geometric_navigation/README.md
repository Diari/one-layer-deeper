# Geometric Navigation E1

`submission.py` is the standalone source of truth for the fixed-modulus
geometric recurrent prototype. It parses the public prompt markers on-device,
starts from a learned landmark for `x`, applies one shared transition exactly
`T` active times, softly projects after each step, and right-aligns a learned
decimal-token decoder with the evaluator's target positions.

## Local checks

```bash
uv run python -m unittest discover -s tests
uv run python -m unittest discover -s tests -p test_geometric_navigation.py -v
uv run python -m client.cli validate submissions/geometric_navigation/submission.py
uv run python -m benchmark.runner \
  --manifest experiments/geometric_navigation/configs/smoke_cpu_geometric.json \
  --submission-file submissions/geometric_navigation/submission.py
```

The repository's official `smoke_cpu.json` allows only 0.05 seconds for final
evaluation. A required 64-step evaluation loop cannot meet that deadline on a
CPU, so the separate smoke manifest changes only the local time/step controls;
it does not replace or edit the evaluator-owned manifest.

For a local GPU E1 run, first generate the public datasets, then generate a
local manifest. The official manifest is never edited.

```bash
bash scripts/generate_datasets.sh
uv run python tools/make_kaggle_manifest.py \
  --output experiments/geometric_navigation/configs/kaggle_e1_p100.json
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmark.runner \
  --manifest experiments/geometric_navigation/configs/kaggle_e1_p100.json \
  --submission-file submissions/geometric_navigation/submission.py
```

## Kaggle

Edit the username and repository commit described in
`kaggle/geometric-navigation/README.md`, then push the kernel. A successful run
writes `summary.json`, results, logs, environment details, the generated
manifest, and a hashed copy of the submission under
`/kaggle/working/artifacts`, plus `/kaggle/working/artifacts.zip`.

## Known limitations

- The landmark bank and circular features are fixed to `N = 323`.
- Training executes at most the E1 training depths 1--3; evaluation executes a
  fixed 64-step loop with per-example active masking.
- The first matched P100 run is recorded in
  `experiments/geometric_navigation/EXPERIMENT_LOG.md`; it does not establish
  success on the official H100 competition environment.
- The full 323-way projection at every active or masked recurrent iteration is
  the expected throughput bottleneck.
