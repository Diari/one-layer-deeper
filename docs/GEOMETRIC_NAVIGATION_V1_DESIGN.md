# Variable-N Geometric Navigation V1

## Research question

V1 asks whether the geometric advantage observed in fixed-N E1 persists when
the modulus changes. It is deliberately an Easy-only intermediate model. A
positive result requires more than memorizing E1: on E5, full geometry must
beat a matched non-geometric recurrence by at least ten test-accuracy points
and improve both seen-N and OOD-N depth-1 accuracy. The winning comparison must
then repeat.

No result has yet established that claim.

## Dynamic landmarks

Each example has 4096 candidate residue identities. Candidate `r` is valid only
when `r < N`; masked candidates receive exactly zero projection probability.
The landmark at a valid position combines:

- one learned global residue embedding;
- `r/N` and normalized `log(1+N)`;
- eight pairs of `sin(2 pi k r/N)` and `cos(2 pi k r/N)`;
- a separate projection of the learned modulus context.

The modulus context encodes its decimal digits and normalized scalar magnitude.
Circular coordinates preserve the topology of residues for each supplied
modulus, while global embeddings let frequently useful numeric identities share
parameters between examples. The complete `[batch, 4096, 128]` landmark bank is
built once per forward pass and reused at every recurrent step.

This is geometry, not answer arithmetic. The coordinates contain no transition
target, factors, generated labels, or modular-operation implementation.

## Recurrent navigation and snapping

One gated residual transition consumes the current state and modulus context.
The same object and parameters are reused at every depth. Training executes four
fixed loop iterations and evaluation executes 64. Expensive transition and
projection operations select only active GPU rows, while `torch.where` preserves
inactive states. Empty active selections keep the fixed loop device-local and
avoid a host synchronization.

For snapping variants, normalized candidate-to-landmark similarities are
masked, temperature-scaled, and converted to probabilities. The next state is
their weighted landmark sum. Applying this after each step may prevent small
errors from accumulating away from valid numeric states.

## Decoding and training signals

Learned heads predict four right-aligned decimal token positions for the answer
and for reconstruction of prompt `x`. Full V1 combines answer cross-entropy,
final landmark supervision parsed only from evaluator-supplied labels, starting
reconstruction, and a weak entropy penalty with weights 1.0, 0.5, 0.1, and
0.01. It never converts a selected landmark index into output digits.

## Matched variants

The standalone source conditionally constructs only modules that are used:

| Variant | Fourier landmark start | Recurrent snapping | Landmark CE | Entropy penalty |
|---|---:|---:|---:|---:|
| `control` | no | no | no | no |
| `fourier` | yes | no | no | no |
| `snap_no_landmark_loss` | yes | yes | no | yes |
| `full` | yes | yes | yes | yes |

`control` retains the same parser, modulus encoder, transition, answer decoder,
optimizer, and reconstruction objective. Its start state uses a learned residue
embedding. Generated variants change only the checked-in `VARIANT` constant and
remain standalone submission files.

## Experiment ladder and gates

P100 screening uses float32, AMP and compilation disabled, batches 64/128,
300 seconds, and seed 74. The order is E1, E2, E5, E3, then E4. E1 additionally
requires full test exact accuracy of at least 70%. E1 and E2 require a ten-point
full-over-control test advantage. E5 requires that advantage plus better seen-N
and OOD-N T=1 accuracy, followed by a repeated matched comparison. Failed gates
trigger the two diagnostic ablations without changing optimizer or width.

P100 timing does not calibrate H100 performance. A passed P100 gate must be
repeated with the official 60-second H100 manifest before promotion.

## Initial P100 evidence

The matched seed-74, 300-second P100 ladder completed on 2026-08-03. Full V1
passed the fixed-modulus screens on E1 (80.67% versus 54.67% control) and E2
(78.13% versus 58.33%). It did not pass the decisive variable-N E5 gate:
10.08% versus 1.17% is an 8.92-point advantage, below the required ten points,
and OOD-N T=1 accuracy was lower than control. The failed-gate ablations scored
1.25% (`fourier`) and 1.42% (`snap_no_landmark_loss`), showing that supplied-label
landmark supervision accounts for nearly all of the E5 gain.

E3 and E4 scaling diagnostics were at floor. Full V1 scored 2.13% on E3 and
0.44% on E4, only 0.13 points above control on each, while completing roughly
5,000 optimizer steps versus 13,000 for control. This rejects promotion of the
current 4096-way V1. The next isolated architectural test should remove the
absolute global residue embedding while retaining N-relative coordinates; it
must be evaluated as a new matched experiment rather than presented as part of
these V1 results.

## V1.1 relative-only falsification test

V1.1 changes exactly one architectural variable. `relative_full` removes the
learned global `residue_embedding[r]`; every landmark is instead constructed
only from `r/N`, normalized `log(N)`, eight-frequency circular coordinates,
and the learned modulus context. Snapping, landmark CE, entropy, reconstruction,
transition, decoders, optimizer, batches, seed, and training duration remain
matched to full V1.

The first E5 run contains `control`, unchanged `full`, and `relative_full` on
the same P100. In addition to the original gate, diagnostics record the rank of
the supplied target landmark and Recall@1/8/16/32/64/128 on test and OOD data.
V2 proposal work is justified only if V1.1 improves variable-N/OOD behavior or
shows high small-k target recall; removing the only useful V1 mechanism without
that evidence would not be a supported scaling strategy.

## Scope and risks

The 4096-wide bank covers scored Easy examples through the 12-bit E4 boundary
and four digits cover residues through 4095. E4's auxiliary OOD-N depth profile
can contain larger moduli; V1 remains safe but cannot faithfully represent
starting residues at or above 4096. That limitation must be reported rather
than mistaken for a scalable variable-N result.

The dominant costs are the batch-sized landmark tensor and repeated
candidate-to-landmark similarities. The P100 preflight measures maximum-N
forward/backward memory at batch 64. If it exceeds 16 GB, reductions must occur
in the prescribed order: train batch, evaluation batch, transition width,
model width, then Fourier frequencies. Snapping remains part of the full test.

Source safety tests cover prohibited I/O, network/process use, checkpoint
loading, host tensor transfers, answer exponentiation, factor helpers, and a
direct square-remainder pattern. The evaluator remains authoritative.

V1 does not begin Medium, Hard, factor geometry, explicit modular algorithms,
or scalable landmark construction. Evidence against the hypothesis includes
failure to clear matched gates, diffuse or wrongly collapsed landmarks,
throughput losses that erase convergence gains, or sharper degradation with
depth and changing N than the control.
