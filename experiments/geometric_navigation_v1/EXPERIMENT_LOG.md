# Variable-N V1 Experiment Log

Do not promote V1 or claim geometric improvement until a matched E5 pair clears
the defined ten-point and T=1 gates twice. P100 and official H100 measurements
belong in separate rows.

| Experiment ID | Date | Commit | Submission SHA-256 | Variant | Dataset | GPU | Train/Eval batch | Duration | Seed | Parameters | State elements | Peak memory | Steps | Test exact | Seen T=1 | OOD-N T=1 | Landmark accuracy | Entropy | Conclusion | Next single change |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `v1-e1-p100-001-control` | pending | pending | pending | `control` | E1 | P100 | 64/128 | 300 s | 74 | pending | pending | pending | pending | pending | pending | pending | n/a | n/a | No result yet. | Run matched full V1. |
| `v1-e1-p100-001-full` | pending | pending | pending | `full` | E1 | P100 | 64/128 | 300 s | 74 | pending | pending | pending | pending | pending | pending | pending | pending | pending | No result yet. | Apply E1 gate without tuning. |

For every row, retain evaluator `result.json`, `diagnostics.json`, generated
manifest, exact standalone submission, SHA-256, commit, logs, environment, and
the dataset gate JSON. Change one experimental variable at a time.
