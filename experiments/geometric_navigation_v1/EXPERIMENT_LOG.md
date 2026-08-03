# Variable-N V1 Experiment Log

Do not promote V1 or claim geometric improvement until a matched E5 pair clears
the defined ten-point and T=1 gates twice. P100 and official H100 measurements
belong in separate rows.

| Experiment ID | Date | Commit | Submission SHA-256 | Variant | Dataset | GPU | Train/Eval batch | Duration | Seed | Parameters | State elements | Peak memory | Steps | Test exact | Seen T=1 | OOD-N T=1 | Landmark accuracy | Entropy | Conclusion | Next single change |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `v1-e1-p100-001-control` | 2026-08-03 | `d2428733734c6621718fcb73cde6b54229d0faab` | `0d6c94534e3a62746bc45e91ba8eb843668db7ae9fafbf4666549baa8061536b` | `control` | E1 | Tesla P100 | 64/128 | 300 s | 74 | 1,152,776 | 1,152,776 | 41,183,232 B | 10,304 | 54.67% | 0.00% | 1.56% | n/a | n/a | Matched recurrent control established. | Compare unchanged full V1. |
| `v1-e1-p100-001-full` | 2026-08-03 | `d2428733734c6621718fcb73cde6b54229d0faab` | `60c8d57a75570b63c273b12b9c5488a5468c52b49cf65c8ce45a35a42d353c3e` | `full` | E1 | Tesla P100 | 64/128 | 300 s | 74 | 1,171,977 | 1,176,073 | 1,424,055,808 B | 4,545 | 80.67% | 23.68% | 1.76% | 80.67% | 0.01234 | Passed E1: +26.00 test points over control and above 70%; maximum-N batch-64 preflight peaked at 1,962,463,232 B. This is not yet evidence for variable-N success. | Run unchanged matched E2. |

For every row, retain evaluator `result.json`, `diagnostics.json`, generated
manifest, exact standalone submission, SHA-256, commit, logs, environment, and
the dataset gate JSON. Change one experimental variable at a time.
