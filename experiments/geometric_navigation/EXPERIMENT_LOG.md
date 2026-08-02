# Geometric Navigation Experiment Log

Do not claim an improvement without a matched baseline. Change one variable per
candidate and attach the structured JSON result. H100 and P100 results are not
directly comparable.

| Experiment ID | Date | Git commit | Submission SHA-256 | Variant | Dataset | GPU | Train batch | Eval batch | Duration | Seed | Parameters | Model-state elements | Peak memory | Optimizer steps | Exact accuracy | Accuracy by T | Landmark accuracy | Entropy | Main conclusion | Next single change |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| `e1-p100-000` | 2026-08-02 | `81b254a5b6d0bb05d8ac7699b1f459046ed7f6f1` | `57119c3fac3e2d9b4a300e84955b1bda178c34c2b74c4576fc2b2e8e121106f5` | `geometric_v0` | E1 | Tesla P100 16 GB | 64 | 128 | 300 s | 74 | 672,393 | 677,561 | 54,469,632 B | 9,091 diagnostics (9,075 evaluator) | 82.67% test; 91.33% evaluator mean | T1 54%; T2 96%; T3 98% | 82.67% | 0.01293 | Under matched P100 conditions, V0 mean exact accuracy was 91.33% versus 2.50% for the supplied baseline; neither model certified a depth rung. | Test only `ENTROPY_WEIGHT=0.005` against this matched V0 baseline. |

For each run, preserve `summary.json`, evaluator result JSON, diagnostics JSON,
the generated manifest, logs, and the exact hashed `submission.py`.
