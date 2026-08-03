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
- Current scope is the single-variable V1.1 E5 experiment that removes the
  absolute residue embedding while retaining V1's relative geometry.
- Do not begin toroidal V2, Medium, Hard, or scalable landmarks unless V1.1
  provides new E5 generalization evidence.
