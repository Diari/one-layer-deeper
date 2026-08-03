# Geometric Navigation Prototype

- Build a standalone geometric recurrent submission.
- Do not modify evaluator scoring or dataset semantics.
- Do not overwrite official manifests; generate separate local copies.
- Never hard-code the answer-producing arithmetic.
- Never access evaluator-owned files from `submission.py`.
- Never commit Kaggle credentials or API tokens.
- Add tests before or alongside architectural changes.
- Run unit tests and the smoke evaluator after model changes.
- Save experiments as structured JSON.
- Record Git commit hashes and submission SHA-256 hashes.
- Change one experimental variable at a time.
- Do not claim an improvement without a matched baseline.
- Keep fixed-N and variable-N work separate.
- Preserve fixed-N V0 while variable-N V1 is developed separately.
- Stop after the Easy-only 4096-landmark V1 gates are ready for Kaggle.
- Do not begin Medium, Hard, or scalable landmarks until V1 clears E5.
