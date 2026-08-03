from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re
from unittest import mock

import pytest
import torch

from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch, count_model_state_elements
from benchmark.validation import validate_optimizer
from data.squaring_mod import (
    DIGIT_OFFSET,
    VOCAB_SIZE,
    collate_squaring_mod,
    tokenize_squaring_mod_with_result,
)


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_PATH = ROOT / "submissions/geometric_navigation_v1/submission.py"
DIAGNOSTIC_PATH = ROOT / "tools/diagnose_geometric_navigation.py"


def load_submission_module():
    spec = importlib.util.spec_from_file_location(
        "geometric_navigation_v1_submission", SUBMISSION_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_diagnostic_module():
    spec = importlib.util.spec_from_file_location(
        "geometric_navigation_diagnostic", DIAGNOSTIC_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(modulus: int, x: int, time_steps: int, result: int = 1):
    input_ids, labels = tokenize_squaring_mod_with_result(
        modulus, x, time_steps, result, separate_input_output=True
    )
    return {"input_ids": input_ids, "labels": labels}


def prompt_batch(*rows):
    return collate_squaring_mod(list(rows))


@pytest.fixture(scope="module")
def module():
    return load_submission_module()


@pytest.fixture(scope="module")
def model_spec():
    return ModelSpec(
        vocab_size=VOCAB_SIZE,
        max_seq_len=16,
        maximum_model_state_elements=500_000_000,
    )


def build_model(module, model_spec, variant="full"):
    torch.manual_seed(74)
    return module.VariableGeometricNavigationModel(model_spec, variant)


def loss_batch(module, model, batch):
    logits, auxiliary = model(batch["input_ids"], batch["attention_mask"])
    batch_indices = torch.arange(logits.shape[0]).reshape(-1, 1)
    token_logits = logits[batch_indices, batch["target_positions"].clamp_min(0)]
    return logits, auxiliary, TokenLossBatch(
        logits=token_logits,
        labels=batch["labels"],
        valid_mask=batch["labels"].ne(-100),
        target_positions=batch["target_positions"],
        auxiliary=auxiliary,
    )


def test_interface_state_optimizer_and_forward(module, model_spec):
    model = module.SUBMISSION.build_model(model_spec)
    batch = prompt_batch(row(323, 5, 2), row(899, 302, 4))
    logits, auxiliary = model(batch["input_ids"], batch["attention_mask"])
    assert model.config.vocab_size == VOCAB_SIZE
    assert model.config.max_seq_len == 16
    assert logits.shape == (*batch["input_ids"].shape, VOCAB_SIZE)
    assert isinstance(auxiliary, dict)
    assert count_model_state_elements(model) <= 500_000_000

    bundle = module.build_optimizer(model, OptimizerSpec(300.0, "cpu"))
    validate_optimizer(bundle, model, torch.device("cpu"))
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    actual = [
        parameter
        for group in bundle.optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(actual) == len(expected)
    assert len({id(parameter) for parameter in actual}) == len(actual)


def test_parser_handles_mixed_moduli_lengths_padding_and_t(module):
    batch = prompt_batch(
        row(323, 5, 1),
        row(899, 302, 2),
        row(1024, 1001, 3),
        row(4095, 4094, 12),
    )
    parsed = module.parse_prompt_tokens(batch["input_ids"], batch["attention_mask"])
    assert parsed["modulus"].tolist() == [323, 899, 1024, 4095]
    assert parsed["starting_value"].tolist() == [5, 302, 1001, 4094]
    assert parsed["time_steps"].tolist() == [1, 2, 3, 12]
    assert parsed["starting_digits"].shape == (4, 4)
    assert parsed["starting_digits"][-1].tolist() == [11, 7, 16, 11]
    assert (batch["input_ids"] == 0).any()


def test_variable_fourier_features_are_periodic_finite_and_fixed(module):
    modulus = torch.tensor([323, 899, 4095])
    for value in modulus.tolist():
        features = module.make_variable_fourier_features(
            torch.tensor([0.0, float(value)]), torch.tensor([value])
        )
        assert features.shape == (1, 2, 16)
        assert torch.isfinite(features).all()
        assert torch.allclose(features[:, 0], features[:, 1], atol=2.0e-4)
        assert not features.requires_grad
    coordinates = module.make_landmark_coordinate_features(
        torch.arange(8), modulus
    )
    assert coordinates.shape == (3, 8, 18)
    assert torch.isfinite(coordinates).all()
    assert not coordinates.requires_grad


def test_correct_landmark_rank_and_recall_inputs_are_masked():
    diagnostic = load_diagnostic_module()
    probabilities = torch.tensor(
        [
            [0.50, 0.30, 0.20, 0.90],
            [0.10, 0.40, 0.30, 0.20],
        ]
    )
    targets = torch.tensor([2, 1])
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    ranks = diagnostic.correct_landmark_ranks(probabilities, targets, mask)
    assert ranks.tolist() == [3, 1]


def test_dynamic_masks_projection_and_single_bank_build(module, model_spec):
    model = build_model(module, model_spec)
    batch = prompt_batch(row(323, 5, 1), row(899, 302, 2), row(4095, 4000, 3))
    calls = []

    def record_bank(geometry, inputs, output):
        calls.append(id(geometry))

    handle = model.geometry.register_forward_hook(record_bank)
    # landmarks() is invoked directly, so hook that bound method with a wrapper.
    original = model.geometry.landmarks

    def counted(*args, **kwargs):
        calls.append(id(model.geometry))
        return original(*args, **kwargs)

    with mock.patch.object(model.geometry, "landmarks", counted):
        _, auxiliary = model(batch["input_ids"], batch["attention_mask"])
    handle.remove()
    assert calls == [id(model.geometry)]
    mask = auxiliary["landmark_mask"]
    probabilities = auxiliary["final_landmark_probabilities"]
    assert mask.shape == (3, 4096)
    assert mask.sum(dim=1).tolist() == [323, 899, 4095]
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1.0e-5)
    assert torch.count_nonzero(probabilities.masked_select(~mask)) == 0
    assert torch.isfinite(probabilities).all()
    assert torch.isfinite(auxiliary["final_state"]).all()


def test_shared_transition_active_counts_and_inactive_state(module, model_spec):
    model = build_model(module, model_spec)
    batch = prompt_batch(
        row(323, 5, 1), row(899, 6, 2), row(1024, 7, 3), row(4095, 8, 4)
    )
    transition_ids = []
    handle = model.transition.register_forward_hook(
        lambda layer, inputs, output: transition_ids.append(id(layer))
    )
    _, auxiliary = model(batch["input_ids"], batch["attention_mask"])
    handle.remove()
    assert transition_ids == [id(model.transition)] * 4
    assert auxiliary["active_history"].sum(dim=0).tolist() == [1, 2, 3, 4]
    history = auxiliary["state_history"]
    assert torch.equal(history[1, 0], history[2, 0])
    assert torch.equal(history[2, 1], history[3, 1])
    assert torch.equal(history[3, 2], history[4, 2])
    transition_names = [name for name, _ in model.named_parameters() if "transition" in name]
    assert not any(re.search(r"transition[s]?\.[0-9]", name) for name in transition_names)


def test_eval_t64_uses_same_transition_and_is_finite(module, model_spec):
    model = build_model(module, model_spec).eval()
    batch = prompt_batch(row(323, 5, 64))
    calls = []

    def identity_transition(state, context):
        return state

    def cheap_projection(candidate, landmarks, valid_mask, *, sharpen):
        probabilities = valid_mask.to(candidate.dtype)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        return candidate, probabilities

    handle = model.transition.register_forward_hook(
        lambda layer, inputs, output: calls.append(id(layer))
    )
    with (
        mock.patch.object(model.transition, "forward", identity_transition),
        mock.patch.object(model.geometry, "project", cheap_projection),
        torch.no_grad(),
    ):
        logits, auxiliary = model(batch["input_ids"], batch["attention_mask"])
    handle.remove()
    assert calls == [id(model.transition)] * 64
    assert auxiliary["active_history"].sum().item() == 64
    assert torch.isfinite(logits).all()
    for key in (
        "final_state",
        "final_landmark_probabilities",
        "entropy_history",
        "state_norm_history",
        "movement_history",
    ):
        assert torch.isfinite(auxiliary[key]).all(), key


def test_full_loss_backward_reaches_geometric_components(module, model_spec):
    model = build_model(module, model_spec)
    batch = prompt_batch(row(323, 5, 2, 25), row(899, 302, 4, 81))
    _, _, token_batch = loss_batch(module, model, batch)
    loss = module.geometric_v1_token_training_loss(token_batch)
    assert torch.isfinite(loss)
    loss.backward()
    components = {
        "residue_embedding": model.geometry.residue_embedding.weight,
        "coordinate_projection": model.geometry.coordinate_projection.weight,
        "landmark_modulus_projection": model.geometry.modulus_projection.weight,
        "modulus_encoder": model.modulus_encoder.scalar_projection[0].weight,
        "transition": model.transition.left.weight,
        "temperature": model.geometry.log_temperature,
        "answer_decoder": model.answer_decoder.weight,
        "reconstruction_decoder": model.reconstruction_decoder.weight,
    }
    for name, parameter in components.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_relative_full_removes_only_absolute_residue_parameters(module, model_spec):
    full = build_model(module, model_spec, "full")
    relative = build_model(module, model_spec, "relative_full")
    assert hasattr(full.geometry, "residue_embedding")
    assert not hasattr(relative.geometry, "residue_embedding")
    full_names = {name for name, _ in full.named_parameters()}
    relative_names = {name for name, _ in relative.named_parameters()}
    assert full_names - relative_names == {"geometry.residue_embedding.weight"}
    assert not (relative_names - full_names)

    original_variant = module.VARIANT
    module.VARIANT = "relative_full"
    try:
        batch = prompt_batch(row(323, 5, 2, 25), row(899, 302, 4, 81))
        _, _, token_batch = loss_batch(module, relative, batch)
        loss = module.geometric_v1_token_training_loss(token_batch)
        assert torch.isfinite(loss)
        loss.backward()
        components = {
            "coordinate_projection": relative.geometry.coordinate_projection.weight,
            "landmark_modulus_projection": relative.geometry.modulus_projection.weight,
            "modulus_encoder": relative.modulus_encoder.scalar_projection[0].weight,
            "transition": relative.transition.left.weight,
            "temperature": relative.geometry.log_temperature,
            "answer_decoder": relative.answer_decoder.weight,
            "reconstruction_decoder": relative.reconstruction_decoder.weight,
        }
        for name, parameter in components.items():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        for name, parameter in relative.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    finally:
        module.VARIANT = original_variant


def test_digit_x_uses_ordered_digits_and_all_parameters_receive_gradients(
    module, model_spec
):
    model = build_model(module, model_spec, "digit_x")
    assert not hasattr(model, "geometry")
    assert not hasattr(model, "start_embedding")
    assert hasattr(model, "start_digit_encoder")

    ordered = prompt_batch(row(899, 302, 2, 25), row(899, 320, 2, 81))
    parsed = module.parse_prompt_tokens(
        ordered["input_ids"], ordered["attention_mask"]
    )
    encoded = model.start_digit_encoder(
        ordered["input_ids"], parsed["starting_digit_mask"]
    )
    assert encoded.shape == (2, module.WIDTH)
    assert torch.isfinite(encoded).all()
    assert not torch.allclose(encoded[0], encoded[1])

    original_variant = module.VARIANT
    module.VARIANT = "digit_x"
    try:
        _, _, token_batch = loss_batch(module, model, ordered)
        loss = module.geometric_v1_token_training_loss(token_batch)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
    finally:
        module.VARIANT = original_variant


def test_digit_xn_adds_only_ordered_modulus_encoding(module, model_spec):
    digit_x = build_model(module, model_spec, "digit_x")
    digit_xn = build_model(module, model_spec, "digit_xn")
    assert isinstance(digit_x.modulus_encoder, module.ModulusEncoder)
    assert isinstance(digit_xn.modulus_encoder, module.OrderedDigitEncoder)
    assert isinstance(digit_xn.start_digit_encoder, module.OrderedDigitEncoder)
    assert not hasattr(digit_xn, "geometry")
    assert not hasattr(digit_xn, "start_embedding")

    batch = prompt_batch(row(323, 25, 2, 81), row(332, 25, 2, 81))
    parsed = module.parse_prompt_tokens(batch["input_ids"], batch["attention_mask"])
    encoded = digit_xn.modulus_encoder(
        batch["input_ids"], parsed["modulus_digit_mask"]
    )
    assert encoded.shape == (2, module.WIDTH)
    assert torch.isfinite(encoded).all()
    assert not torch.allclose(encoded[0], encoded[1])

    original_variant = module.VARIANT
    module.VARIANT = "digit_xn"
    try:
        _, _, token_batch = loss_batch(module, digit_xn, batch)
        loss = module.geometric_v1_token_training_loss(token_batch)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in digit_xn.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
    finally:
        module.VARIANT = original_variant


def test_control_has_no_geometric_parameters_and_no_unused_parameters(
    module, model_spec
):
    original_variant = module.VARIANT
    module.VARIANT = "control"
    try:
        model = module.build_model(model_spec)
        names = [name for name, _ in model.named_parameters()]
        assert not hasattr(model, "geometry")
        assert not any(
            word in name
            for name in names
            for word in ("coordinate_projection", "log_temperature", "landmark")
        )
        batch = prompt_batch(row(323, 5, 1, 25), row(899, 302, 4, 81))
        _, _, token_batch = loss_batch(module, model, batch)
        loss = module.geometric_v1_token_training_loss(token_batch)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    finally:
        module.VARIANT = original_variant


@pytest.mark.parametrize(
    ("variant", "has_geometry", "has_snapping"),
    [
        ("control", False, False),
        ("digit_x", False, False),
        ("digit_xn", False, False),
        ("fourier", True, False),
        ("snap_no_landmark_loss", True, True),
        ("full", True, True),
        ("relative_full", True, True),
    ],
)
def test_variants_conditionally_construct_modules(
    module, model_spec, variant, has_geometry, has_snapping
):
    model = build_model(module, model_spec, variant)
    assert hasattr(model, "geometry") is has_geometry
    if has_geometry:
        assert hasattr(model.geometry, "log_temperature") is has_snapping
        assert hasattr(model.geometry, "residue_embedding") is (
            variant != "relative_full"
        )
    else:
        assert hasattr(model, "start_digit_encoder") is (
            variant in ("digit_x", "digit_xn")
        )
        assert hasattr(model, "start_embedding") is (variant == "control")


def test_evaluation_is_deterministic(module, model_spec):
    model = build_model(module, model_spec).eval()
    batch = prompt_batch(row(323, 5, 2), row(899, 302, 4))
    with mock.patch.object(module, "MAX_EVAL_T", 4), torch.no_grad():
        first, first_aux = model(batch["input_ids"], batch["attention_mask"])
        second, second_aux = model(batch["input_ids"], batch["attention_mask"])
    assert torch.equal(first, second)
    assert torch.equal(
        first_aux["final_landmark_probabilities"],
        second_aux["final_landmark_probabilities"],
    )


def test_source_safety_standalone_and_variant_generation(module, tmp_path):
    source = SUBMISSION_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    forbidden_patterns = {
        "checkpoint loading": r"torch\s*\.\s*load\s*\(",
        "process launch": r"\bsubprocess\b|os\s*\.\s*system\s*\(",
        "network": r"\b(?:socket|requests|urllib)\b",
        "filesystem access": r"(?<![A-Za-z_])open\s*\(",
        "host transfer": r"\.\s*(?:cpu|numpy)\s*\(",
        "answer exponentiation": r"(?<![A-Za-z_])pow\s*\(",
        "factor helpers": r"\b(?:factor|prime)[A-Za-z_]*\s*\(",
        "explicit square remainder": r"x\s*\*\s*x\s*%",
    }
    for name, pattern in forbidden_patterns.items():
        assert re.search(pattern, source, flags=re.IGNORECASE) is None, name
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imports <= {"__future__", "math", "torch", "benchmark"}

    generated = source.replace('VARIANT = "full"', 'VARIANT = "control"', 1)
    assert generated.count('VARIANT = "control"') == 1
    ast.parse(generated)
