# Geometric Navigation for Easy E1

## Hypothesis

Repeated modular computation may be easier to learn when each valid numerical
state is a location and one application of the unknown operation is movement
between locations. The E1 prototype therefore maps `x` to a landmark, reuses
one learned transition `T` times, softly snaps each candidate back to the
landmark manifold, and decodes only after the final step.

This is an empirical architecture hypothesis, not an implementation of the
public data generator's arithmetic. The model has no answer-producing formula,
transition table, factors, generated labels, or precomputed answers.

## Representation

E1 fixes `N = 323`, so V0 allocates one learned 128-dimensional embedding for
each remainder from 0 through 322. Each landmark also receives eight circular
Fourier pairs:

`sin(2 pi k r / N), cos(2 pi k r / N)` for `k = 1..8`.

Circular coordinates make remainder 0 adjacent to the end of the residue
interval instead of treating the interval as a line with unrelated endpoints.
The features only encode positions. A learned projection combines them with
the landmark embeddings, allowing the model to retain or ignore different
frequencies.

Landmarks provide a learned discrete-state manifold. Without projection, small
transition errors can accumulate off-manifold over recurrence. Soft projection
computes normalized candidate-to-landmark similarities, applies a learned
temperature and softmax, and uses the probability-weighted landmark sum as the
next state. Evaluation can lower the temperature without introducing a hard
training decision.

## Prompt parsing and context

The standalone file uses the repository's public IDs: markers `N=2`, `X=3`,
`T=4`, padding `0`, and decimal digits `7..16`. Device-local masked decimal
folds parse multi-digit `N`, `x`, and `T`. No input-dependent value is moved to
the host.

The modulus context combines learned embeddings of the supplied decimal digits
with normalized scalar magnitude features. This makes `N` a real model input,
but it is not variable-N support: all state landmarks and circular buffers are
still defined for 323. Development prompts with another modulus remain
well-formed and do not crash, but their predictions have no variable-modulus
semantics.

## Recurrent transition

Exactly one transition module is registered and reused. At every step it
concatenates state, modulus context, their elementwise product, and their
difference. Layer normalization, two projected branches, SiLU gating, a learned
output projection, a residual connection, and final normalization produce the
candidate state.

Training runs the fixed E1 ceiling of three iterations; evaluation runs 64.
Per-example masks apply an update only while `T > step`, so mixed depths share a
batch and inactive examples preserve their state and landmark probabilities.
All examples use the same transition parameters at every depth.

## Learned decoders and losses

The answer decoder predicts four right-aligned decimal token positions from the
final state. The evaluator gathers the last one, two, or three prompt positions
according to the supplied target length, so right alignment matches variable
answer widths. The decoder does not receive a predicted landmark index and does
not convert an index to digits.

Training uses only prompt values and evaluator-supplied labels:

- Answer cross-entropy trains the final decimal-token prediction.
- Starting-state reconstruction predicts the decimal `x` tokens already in the
  prompt, preserving input identity in the geometric start state.
- Final landmark negative log likelihood parses the supplied answer labels and
  encourages probability mass at that labeled remainder.
- A small positive entropy penalty discourages completely diffuse landmark
  distributions. It is intentionally weak to avoid premature hard snapping.

The weights are 1.0, 0.5, 0.1, and 0.01 respectively. Soft projection preserves
the gradient path through the transition, temperature, landmark bank, geometric
projection, modulus context, and learned decoders.

## Fixed-N scope and ablations

V0's 323 landmarks, Fourier buffer, start-index range, and landmark-supervision
range are fixed-N shortcuts. No E5 or scalable landmark work is included. The
standalone constants expose four future matched variants: disable Fourier
features; disable snapping; disable landmark supervision; or use the full V0.
Only one switch should change per matched experiment.

E5 would require modulus-conditioned or dynamically generated landmarks,
valid-state masking per example, circular features constructed for each
supplied modulus, safe handling of wider `x` and answers, and evidence that a
shared transition generalizes across modulus identities. Those changes are
deliberately out of scope here.

## Competition-rule review

The submission is one UTF-8 Python file, imports only PyTorch and the public
benchmark API, calls `assert_model_state`, returns full-sequence vocabulary
logits, and registers fixed Fourier features as persistent state. Parsing,
recurrence, projection, and decoding stay on the input device. The optimizer
covers every trainable parameter once. Source tests target checkpoint loading,
filesystem and network access, host transfers, process launch, direct answer
arithmetic, and factor/prime helper patterns. Static scanning cannot prove a
negative, so evaluator validation and behavioral review remain necessary.

## Evidence standard

Support for the hypothesis requires full V0 to beat a small recurrent baseline
or relevant ablation under the same dataset, GPU, batch sizes, duration, seed,
precision, AMP, and compilation settings. Useful secondary evidence is higher
final-landmark accuracy, increasing correct-landmark probability, controlled
entropy, and stable state norms over depth. Repeated seeds are needed before a
strong conclusion.

The hypothesis is weakened if snapping does not improve matched accuracy or
long-depth stability, landmark probabilities remain diffuse or collapse
incorrectly, throughput loss erases learning gains in the fixed time budget, or
the learned transition fails immediately outside depths 1--3. No success claim
is warranted until the first Kaggle result exists.
