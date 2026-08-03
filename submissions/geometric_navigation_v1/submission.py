"""Variable-modulus geometric recurrent prototype for the Easy datasets.

This standalone submission parses the public prompt grammar on-device, builds
one masked landmark bank per example, reuses a single transition for every
requested step, and decodes decimal tokens from the final state.  It contains
no arithmetic answer solver or precomputed transition data.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
)


PAD_TOKEN = 0
N_TOKEN = 2
X_TOKEN = 3
T_TOKEN = 4
DIGIT_OFFSET = 7
NUM_DIGITS = 10

MAX_VALUE = 4096
WIDTH = 128
TRANSITION_WIDTH = 512
NUM_FOURIER_FREQUENCIES = 8
MAX_TRAIN_T = 4
MAX_EVAL_T = 64
MAX_OUTPUT_DIGITS = 4

ANSWER_WEIGHT = 1.0
LANDMARK_WEIGHT = 0.5
RECONSTRUCTION_WEIGHT = 0.1
ENTROPY_WEIGHT = 0.01

LEARNING_RATE = 1.0e-3
WARMUP_STEPS = 10
SCHEDULE_STEPS = 100_000
EVAL_TEMPERATURE_SCALE = 0.60

# tools/make_geometric_v1_variant.py changes this exact line when producing a
# matched standalone ablation.  The checked-in source of truth is full V1.
VARIANT = "full"
VALID_VARIANTS = (
    "control",
    "fourier",
    "snap_no_landmark_loss",
    "full",
    "relative_full",
)


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


def _variant_flags(variant: str) -> tuple[bool, bool, bool, bool]:
    if variant not in VALID_VARIANTS:
        raise ValueError("unknown geometric V1 variant")
    uses_geometry = variant != "control"
    uses_snapping = variant in ("snap_no_landmark_loss", "full", "relative_full")
    uses_landmark_loss = variant in ("full", "relative_full")
    uses_entropy_loss = variant in (
        "snap_no_landmark_loss",
        "full",
        "relative_full",
    )
    return uses_geometry, uses_snapping, uses_landmark_loss, uses_entropy_loss


def _valid_token_mask(input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
    if attention_mask is None:
        return input_ids.ne(PAD_TOKEN)
    if attention_mask.ndim == 2:
        return attention_mask.to(dtype=torch.bool)
    if attention_mask.ndim == 3:
        return attention_mask.to(dtype=torch.bool).any(dim=1)
    raise ValueError("invalid attention_mask shape")


def _marker_position(input_ids: Tensor, marker: int) -> Tensor:
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    marked = torch.where(
        input_ids.eq(marker),
        positions.reshape(1, -1),
        torch.full_like(input_ids, -1),
    )
    return marked.amax(dim=1)


def _parse_decimal_field(
    input_ids: Tensor,
    valid_mask: Tensor,
    start_marker: int,
    end_marker: int | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Parse a decimal field using only vectorized, device-local tensors."""

    batch, length = input_ids.shape
    positions = torch.arange(length, device=input_ids.device).reshape(1, -1)
    start = _marker_position(input_ids, start_marker)
    if end_marker is None:
        end = valid_mask.long().sum(dim=1)
    else:
        end = _marker_position(input_ids, end_marker)
    digit_mask = (
        valid_mask
        & positions.gt(start.reshape(-1, 1))
        & positions.lt(end.reshape(-1, 1))
        & input_ids.ge(DIGIT_OFFSET)
        & input_ids.lt(DIGIT_OFFSET + NUM_DIGITS)
    )

    value = torch.zeros(batch, dtype=torch.long, device=input_ids.device)
    right_aligned_tokens = torch.full(
        (batch, MAX_OUTPUT_DIGITS),
        -100,
        dtype=torch.long,
        device=input_ids.device,
    )
    for position in range(length):
        active = digit_mask[:, position]
        digit = input_ids[:, position] - DIGIT_OFFSET
        value = torch.where(active, value * 10 + digit, value)
        shifted = torch.cat(
            (right_aligned_tokens[:, 1:], input_ids[:, position : position + 1]),
            dim=1,
        )
        right_aligned_tokens = torch.where(
            active.reshape(-1, 1), shifted, right_aligned_tokens
        )
    return value, right_aligned_tokens, digit_mask


def parse_prompt_tokens(
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Parse mixed N, x, and T values according to the public token markers."""

    valid_mask = _valid_token_mask(input_ids, attention_mask)
    modulus, modulus_digits, modulus_digit_mask = _parse_decimal_field(
        input_ids, valid_mask, N_TOKEN, X_TOKEN
    )
    starting_value, starting_digits, starting_digit_mask = _parse_decimal_field(
        input_ids, valid_mask, X_TOKEN, T_TOKEN
    )
    time_steps, time_digits, time_digit_mask = _parse_decimal_field(
        input_ids, valid_mask, T_TOKEN, None
    )
    return {
        "modulus": modulus,
        "starting_value": starting_value,
        "time_steps": time_steps,
        "modulus_digits": modulus_digits,
        "starting_digits": starting_digits,
        "time_digits": time_digits,
        "modulus_digit_mask": modulus_digit_mask,
        "starting_digit_mask": starting_digit_mask,
        "time_digit_mask": time_digit_mask,
        "valid_mask": valid_mask,
    }


def make_variable_fourier_features(values: Tensor, modulus: Tensor) -> Tensor:
    """Create circular coordinates for every value/modulus pair, not targets."""

    values = values.to(dtype=torch.float32).reshape(1, -1, 1)
    modulus = modulus.to(dtype=torch.float32).clamp_min(1.0).reshape(-1, 1, 1)
    frequencies = torch.arange(
        1,
        NUM_FOURIER_FREQUENCIES + 1,
        device=values.device,
        dtype=values.dtype,
    ).reshape(1, 1, -1)
    angles = values * frequencies * (2.0 * math.pi) / modulus
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


def make_landmark_coordinate_features(values: Tensor, modulus: Tensor) -> Tensor:
    """Return r/N, normalized log(N), and circular Fourier coordinates."""

    values = values.to(dtype=torch.float32).reshape(1, -1)
    safe_modulus = modulus.to(dtype=torch.float32).clamp_min(1.0).reshape(-1, 1)
    ratio = values / safe_modulus
    normalized_log = (
        torch.log1p(safe_modulus) / math.log1p(float(MAX_VALUE))
    ).expand(-1, values.shape[1])
    circular = make_variable_fourier_features(values.reshape(-1), modulus)
    return torch.cat(
        (ratio.unsqueeze(-1), normalized_log.unsqueeze(-1), circular), dim=-1
    )


class ModulusEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.digit_embedding = nn.Embedding(NUM_DIGITS, WIDTH)
        self.scalar_projection = nn.Sequential(
            nn.Linear(2, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, WIDTH),
        )
        self.output_norm = nn.LayerNorm(WIDTH)

    def forward(
        self,
        input_ids: Tensor,
        digit_mask: Tensor,
        modulus: Tensor,
    ) -> Tensor:
        safe_digits = (input_ids - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
        embedded = self.digit_embedding(safe_digits)
        weights = digit_mask.to(dtype=embedded.dtype).unsqueeze(-1)
        digit_context = (embedded * weights).sum(dim=1)
        digit_context = digit_context / weights.sum(dim=1).clamp_min(1.0)
        numeric = modulus.to(dtype=embedded.dtype).clamp_min(1.0)
        scalar_features = torch.stack(
            (
                numeric / float(MAX_VALUE),
                torch.log1p(numeric) / math.log1p(float(MAX_VALUE)),
            ),
            dim=-1,
        )
        return self.output_norm(
            digit_context + self.scalar_projection(scalar_features)
        )


class DynamicLandmarkGeometry(nn.Module):
    """Build one variable-N landmark bank and optionally snap into it."""

    def __init__(
        self,
        uses_snapping: bool,
        uses_absolute_residue_embedding: bool,
    ) -> None:
        super().__init__()
        self.uses_snapping = uses_snapping
        self.uses_absolute_residue_embedding = uses_absolute_residue_embedding
        if uses_absolute_residue_embedding:
            self.residue_embedding = nn.Embedding(MAX_VALUE, WIDTH)
        self.coordinate_projection = nn.Linear(
            2 + 2 * NUM_FOURIER_FREQUENCIES, WIDTH, bias=False
        )
        self.modulus_projection = nn.Linear(WIDTH, WIDTH, bias=False)
        self.landmark_norm = nn.LayerNorm(WIDTH)
        self.register_buffer(
            "residue_values", torch.arange(MAX_VALUE, dtype=torch.float32)
        )
        if uses_snapping:
            self.candidate_norm = nn.LayerNorm(WIDTH)
            self.log_temperature = nn.Parameter(torch.tensor(-1.5))

    def landmarks(
        self,
        modulus: Tensor,
        modulus_context: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        coordinates = make_landmark_coordinate_features(
            self.residue_values, modulus
        ).to(dtype=modulus_context.dtype)
        values = (
            self.coordinate_projection(coordinates)
            + self.modulus_projection(modulus_context).unsqueeze(1)
        )
        if self.uses_absolute_residue_embedding:
            values = values + self.residue_embedding.weight.unsqueeze(0)
        safe_modulus = modulus.clamp(1, MAX_VALUE)
        valid_mask = self.residue_values.reshape(1, -1).lt(
            safe_modulus.reshape(-1, 1)
        )
        return self.landmark_norm(values), valid_mask, coordinates

    def project(
        self,
        candidate: Tensor,
        landmarks: Tensor,
        valid_mask: Tensor,
        *,
        sharpen: bool,
    ) -> tuple[Tensor, Tensor]:
        if not self.uses_snapping:
            raise RuntimeError("projection is unavailable for this variant")
        normalized_candidate = F.normalize(self.candidate_norm(candidate), dim=-1)
        normalized_landmarks = F.normalize(landmarks, dim=-1)
        scores = torch.bmm(
            normalized_landmarks, normalized_candidate.unsqueeze(-1)
        ).squeeze(-1)
        temperature = self.log_temperature.exp().clamp(0.03, 2.0)
        if sharpen:
            temperature = temperature * EVAL_TEMPERATURE_SCALE
        scaled_scores = scores / temperature
        scaled_scores = scaled_scores.masked_fill(
            ~valid_mask, torch.finfo(scaled_scores.dtype).min
        )
        probabilities = F.softmax(scaled_scores, dim=-1)
        projected = torch.bmm(probabilities.unsqueeze(1), landmarks).squeeze(1)
        return projected, probabilities


class SharedGeometricTransition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        feature_width = 4 * WIDTH
        self.input_norm = nn.LayerNorm(feature_width)
        self.left = nn.Linear(feature_width, TRANSITION_WIDTH)
        self.right = nn.Linear(feature_width, TRANSITION_WIDTH)
        self.output = nn.Linear(TRANSITION_WIDTH, WIDTH)
        self.output_norm = nn.LayerNorm(WIDTH)

    def forward(self, state: Tensor, modulus_context: Tensor) -> Tensor:
        features = torch.cat(
            (
                state,
                modulus_context,
                state * modulus_context,
                state - modulus_context,
            ),
            dim=-1,
        )
        features = self.input_norm(features)
        hidden = F.silu(self.left(features)) * self.right(features)
        return self.output_norm(state + self.output(hidden))


class VariableGeometricNavigationModel(nn.Module):
    num_loops = MAX_EVAL_T

    def __init__(self, spec: ModelSpec, variant: str = VARIANT) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.variant = variant
        (
            self.uses_geometry,
            self.uses_snapping,
            self.uses_landmark_loss,
            self.uses_entropy_loss,
        ) = _variant_flags(variant)
        self.uses_absolute_residue_embedding = variant != "relative_full"
        self.modulus_encoder = ModulusEncoder()
        if self.uses_geometry:
            self.geometry = DynamicLandmarkGeometry(
                self.uses_snapping,
                self.uses_absolute_residue_embedding,
            )
        else:
            self.start_embedding = nn.Embedding(MAX_VALUE, WIDTH)
        self.start_norm = nn.LayerNorm(WIDTH)
        self.transition = SharedGeometricTransition()
        self.answer_decoder = nn.Linear(
            WIDTH, MAX_OUTPUT_DIGITS * spec.vocab_size
        )
        self.reconstruction_decoder = nn.Linear(
            WIDTH, MAX_OUTPUT_DIGITS * spec.vocab_size
        )

    def _position_answer_logits(
        self,
        slot_logits: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        batch, length = valid_mask.shape
        positions = torch.arange(length, device=slot_logits.device).reshape(1, -1)
        valid_lengths = valid_mask.long().sum(dim=1, keepdim=True)
        slot_indices = (MAX_OUTPUT_DIGITS - valid_lengths + positions).clamp(
            0, MAX_OUTPUT_DIGITS - 1
        )
        batch_indices = torch.arange(batch, device=slot_logits.device).reshape(-1, 1)
        return slot_logits[batch_indices, slot_indices]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        parsed = parse_prompt_tokens(input_ids, attention_mask)
        modulus_context = self.modulus_encoder(
            input_ids,
            parsed["modulus_digit_mask"],
            parsed["modulus"],
        )
        start_indices = parsed["starting_value"].clamp(0, MAX_VALUE - 1)
        if self.uses_geometry:
            landmarks, landmark_mask, coordinate_features = self.geometry.landmarks(
                parsed["modulus"], modulus_context
            )
            batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
            start_values = landmarks[batch_indices, start_indices]
        else:
            landmarks = modulus_context.new_empty((input_ids.shape[0], 0, WIDTH))
            landmark_mask = torch.empty(
                (input_ids.shape[0], 0), dtype=torch.bool, device=input_ids.device
            )
            coordinate_features = modulus_context.new_empty((input_ids.shape[0], 0, 0))
            start_values = self.start_embedding(start_indices)

        starting_state = self.start_norm(start_values + modulus_context)
        state = starting_state
        if self.uses_snapping:
            probabilities = F.one_hot(
                start_indices, num_classes=MAX_VALUE
            ).to(dtype=state.dtype)
            probabilities = probabilities * landmark_mask.to(dtype=state.dtype)
        else:
            probabilities = state.new_empty((input_ids.shape[0], 0))

        state_history = [state]
        probability_history = [probabilities]
        entropy_history = []
        state_norm_history = []
        movement_history = []
        active_history = []
        maximum_steps = MAX_TRAIN_T if self.training else MAX_EVAL_T
        for step in range(maximum_steps):
            active = parsed["time_steps"].gt(step)
            # Keep the loop depth fixed while restricting expensive input-
            # dependent work to active GPU rows.  Empty active tensors are
            # valid, so this needs no device synchronization or Python branch.
            active_indices = active.nonzero(as_tuple=False).reshape(-1)
            active_candidate = self.transition(
                state[active], modulus_context[active]
            )
            if self.uses_snapping:
                active_next, active_probabilities = self.geometry.project(
                    active_candidate,
                    landmarks[active],
                    landmark_mask[active],
                    sharpen=not self.training,
                )
                scattered_probabilities = probabilities.index_copy(
                    0, active_indices, active_probabilities
                )
            else:
                active_next = active_candidate
                scattered_probabilities = probabilities
            scattered_state = state.index_copy(0, active_indices, active_next)
            next_state = torch.where(active.unsqueeze(-1), scattered_state, state)
            if self.uses_snapping:
                probabilities = torch.where(
                    active.unsqueeze(-1), scattered_probabilities, probabilities
                )
                entropy = -(
                    probabilities * probabilities.clamp_min(1.0e-8).log()
                ).sum(dim=-1)
            else:
                entropy = state.new_zeros(state.shape[0])
            movement = 1.0 - F.cosine_similarity(state, next_state, dim=-1)
            state = next_state
            state_history.append(state)
            probability_history.append(probabilities)
            entropy_history.append(entropy)
            state_norm_history.append(state.norm(dim=-1))
            movement_history.append(movement)
            active_history.append(active)

        slot_logits = self.answer_decoder(state).reshape(
            input_ids.shape[0], MAX_OUTPUT_DIGITS, self.config.vocab_size
        )
        logits = self._position_answer_logits(slot_logits, parsed["valid_mask"])
        reconstruction_logits = self.reconstruction_decoder(starting_state).reshape(
            input_ids.shape[0], MAX_OUTPUT_DIGITS, self.config.vocab_size
        )
        auxiliary = {
            "parsed_modulus": parsed["modulus"],
            "parsed_starting_value": parsed["starting_value"],
            "parsed_time_steps": parsed["time_steps"],
            "starting_digit_labels": parsed["starting_digits"],
            "starting_state": starting_state,
            "final_state": state,
            "landmark_mask": landmark_mask,
            "landmark_coordinate_features": coordinate_features,
            "final_landmark_probabilities": probabilities,
            "state_history": torch.stack(state_history, dim=0),
            "landmark_probability_history": torch.stack(
                probability_history, dim=0
            ),
            "entropy_history": torch.stack(entropy_history, dim=0),
            "state_norm_history": torch.stack(state_norm_history, dim=0),
            "movement_history": torch.stack(movement_history, dim=0),
            "active_history": torch.stack(active_history, dim=0),
            "reconstruction_logits": reconstruction_logits,
        }
        return logits, auxiliary


def _parse_supplied_answer(batch: TokenLossBatch) -> tuple[Tensor, Tensor]:
    labels = batch.labels
    valid = batch.valid_mask
    answer = torch.zeros(labels.shape[0], dtype=torch.long, device=labels.device)
    all_digits = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
    for position in range(labels.shape[1]):
        token = labels[:, position]
        active = valid[:, position]
        is_digit = token.ge(DIGIT_OFFSET) & token.lt(DIGIT_OFFSET + NUM_DIGITS)
        all_digits = all_digits & (~active | is_digit)
        digit = (token - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
        answer = torch.where(active & is_digit, answer * 10 + digit, answer)
    usable = valid.any(dim=1) & all_digits & answer.lt(MAX_VALUE)
    return answer, usable


def geometric_v1_token_training_loss(batch: TokenLossBatch) -> Tensor:
    valid = batch.valid_mask
    answer_loss = F.cross_entropy(batch.logits[valid].float(), batch.labels[valid])
    auxiliary = batch.auxiliary
    _, uses_snapping, uses_landmark_loss, uses_entropy_loss = _variant_flags(VARIANT)

    reconstruction_logits = auxiliary["reconstruction_logits"].float()
    reconstruction_labels = auxiliary["starting_digit_labels"]
    reconstruction_loss = F.cross_entropy(
        reconstruction_logits.reshape(-1, reconstruction_logits.shape[-1]),
        reconstruction_labels.reshape(-1),
        ignore_index=-100,
    )
    total = ANSWER_WEIGHT * answer_loss + RECONSTRUCTION_WEIGHT * reconstruction_loss

    if uses_snapping and (uses_landmark_loss or uses_entropy_loss):
        final_probabilities = auxiliary["final_landmark_probabilities"].float()
        if uses_landmark_loss:
            target_landmark, usable = _parse_supplied_answer(batch)
            usable = usable & target_landmark.lt(auxiliary["parsed_modulus"])
            selected = final_probabilities.gather(
                1, target_landmark.clamp(0, MAX_VALUE - 1).unsqueeze(1)
            ).squeeze(1)
            terms = -selected.clamp_min(1.0e-8).log()
            landmark_loss = (terms * usable).sum() / usable.sum().clamp_min(1)
            total = total + LANDMARK_WEIGHT * landmark_loss
        if uses_entropy_loss:
            entropy = -(
                final_probabilities
                * final_probabilities.clamp_min(1.0e-8).log()
            ).sum(dim=-1).mean()
            total = total + ENTROPY_WEIGHT * entropy
    return total


def build_model(spec: ModelSpec) -> VariableGeometricNavigationModel:
    model = VariableGeometricNavigationModel(spec, VARIANT)
    assert_model_state(model, spec)
    return model


def build_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    )

    def schedule(step: int) -> float:
        if step < WARMUP_STEPS:
            return float(step + 1) / float(WARMUP_STEPS)
        progress = min(
            1.0,
            float(step - WARMUP_STEPS)
            / float(max(1, SCHEDULE_STEPS - WARMUP_STEPS)),
        )
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return OptimizerBundle(optimizer=optimizer, scheduler=scheduler)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=geometric_v1_token_training_loss,
    batch_size=64,
    eval_batch_size=128,
)
