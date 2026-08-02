from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re
from unittest import mock
import unittest

import torch

from benchmark import (
    ModelSpec,
    OptimizerSpec,
    TokenLossBatch,
    count_model_state_elements,
)
from benchmark.validation import validate_optimizer
from data.squaring_mod import (
    DIGIT_OFFSET,
    TOKEN_IDS,
    VOCAB_SIZE,
    collate_squaring_mod,
    tokenize_squaring_mod_with_result,
)


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_PATH = ROOT / "submissions/geometric_navigation/submission.py"


def load_submission_module():
    spec = importlib.util.spec_from_file_location(
        "geometric_navigation_submission", SUBMISSION_PATH
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


class GeometricNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_submission_module()
        cls.spec = ModelSpec(
            vocab_size=VOCAB_SIZE,
            max_seq_len=12,
            maximum_model_state_elements=500_000_000,
        )

    def build_model(self):
        torch.manual_seed(74)
        return self.module.SUBMISSION.build_model(self.spec)

    def test_interface_state_and_optimizer_contract(self) -> None:
        model = self.build_model()
        self.assertEqual(model.config.vocab_size, VOCAB_SIZE)
        self.assertEqual(model.config.max_seq_len, 12)
        batch = prompt_batch(row(323, 5, 2), row(323, 302, 3))
        logits, auxiliary = model(
            batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        self.assertEqual(
            tuple(logits.shape),
            (*batch["input_ids"].shape, VOCAB_SIZE),
        )
        self.assertIsInstance(auxiliary, dict)
        self.assertLessEqual(count_model_state_elements(model), 500_000_000)

        bundle = self.module.SUBMISSION.build_optimizer(
            model, OptimizerSpec(300.0, "cpu")
        )
        validate_optimizer(bundle, model, torch.device("cpu"))
        expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
        actual = [
            parameter
            for group in bundle.optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(len({id(parameter) for parameter in actual}), len(actual))

    def test_parse_known_prompts_padding_lengths_and_mixed_t(self) -> None:
        batch = prompt_batch(
            row(323, 5, 2),
            row(323, 302, 3),
            row(323, 17, 12),
        )
        parsed = self.module.parse_prompt_tokens(
            batch["input_ids"], batch["attention_mask"]
        )
        self.assertEqual(parsed["modulus"].tolist(), [323, 323, 323])
        self.assertEqual(parsed["starting_value"].tolist(), [5, 302, 17])
        self.assertEqual(parsed["time_steps"].tolist(), [2, 3, 12])
        self.assertEqual(tuple(parsed["starting_digits"].shape), (3, 4))
        self.assertEqual(
            parsed["starting_digits"][1, -3:].tolist(),
            [DIGIT_OFFSET + 3, DIGIT_OFFSET, DIGIT_OFFSET + 2],
        )
        self.assertTrue((batch["input_ids"] == TOKEN_IDS["PAD"]).any())

    def test_development_modulus_does_not_crash_parser_or_forward(self) -> None:
        model = self.build_model()
        batch = prompt_batch(row(143, 42, 2))
        logits, auxiliary = model(batch["input_ids"], batch["attention_mask"])
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(auxiliary["parsed_modulus"].item(), 143)
        self.assertFalse(auxiliary["fixed_modulus_match"].item())

    def test_fourier_geometry_is_finite_and_periodic(self) -> None:
        values = torch.tensor([0.0, 1.0, 322.0, 323.0])
        features = self.module.make_circular_fourier_features(values, 323)
        self.assertEqual(tuple(features.shape), (4, 16))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.allclose(features[0], features[3], atol=2.0e-5))
        self.assertFalse(features.requires_grad)
        self.assertTrue(torch.allclose(features[0, :8], torch.zeros(8)))

    def test_landmark_bank_and_projection(self) -> None:
        model = self.build_model()
        geometry = model.geometry
        landmarks = geometry.landmarks()
        self.assertEqual(tuple(landmarks.shape), (323, 128))
        projected, probabilities = geometry.project(landmarks[:3], sharpen=False)
        self.assertEqual(tuple(projected.shape), (3, 128))
        self.assertEqual(tuple(probabilities.shape), (3, 323))
        self.assertTrue(torch.allclose(probabilities.sum(dim=-1), torch.ones(3)))
        self.assertTrue(torch.isfinite(projected).all())
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertFalse(geometry.fourier_features.requires_grad)

    def test_recurrence_uses_one_shared_transition_and_independent_masks(self) -> None:
        model = self.build_model()
        batch = prompt_batch(
            row(323, 5, 1), row(323, 6, 2), row(323, 7, 3)
        )
        transition_ids = []

        def record_transition(module, inputs, output):
            transition_ids.append(id(module))

        handle = model.transition.register_forward_hook(record_transition)
        _, auxiliary = model(batch["input_ids"], batch["attention_mask"])
        handle.remove()
        self.assertEqual(transition_ids, [id(model.transition)] * 3)
        self.assertEqual(
            auxiliary["active_history"].sum(dim=0).tolist(), [1, 2, 3]
        )
        history = auxiliary["state_history"]
        self.assertTrue(torch.equal(history[1, 0], history[2, 0]))
        self.assertTrue(torch.equal(history[2, 0], history[3, 0]))
        self.assertTrue(torch.equal(history[2, 1], history[3, 1]))
        transition_parameter_names = [
            name for name, _ in model.named_parameters() if "transition" in name
        ]
        self.assertFalse(any(re.search(r"transition[s]?\.[0-9]", name) for name in transition_parameter_names))

    def test_evaluation_is_deterministic(self) -> None:
        model = self.build_model().eval()
        batch = prompt_batch(row(323, 5, 2), row(323, 302, 3))
        with mock.patch.object(self.module, "MAX_EVAL_T", 3), torch.no_grad():
            first, first_aux = model(batch["input_ids"], batch["attention_mask"])
            second, second_aux = model(batch["input_ids"], batch["attention_mask"])
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(
            torch.equal(
                first_aux["final_landmark_probabilities"],
                second_aux["final_landmark_probabilities"],
            )
        )

    def test_loss_backward_reaches_all_major_components(self) -> None:
        model = self.build_model()
        batch = prompt_batch(row(323, 5, 2, 25), row(323, 302, 3, 81))
        logits, auxiliary = model(batch["input_ids"], batch["attention_mask"])
        batch_indices = torch.arange(logits.shape[0]).reshape(-1, 1)
        token_logits = logits[
            batch_indices, batch["target_positions"].clamp_min(0)
        ]
        loss = self.module.geometric_token_training_loss(
            TokenLossBatch(
                logits=token_logits,
                labels=batch["labels"],
                valid_mask=batch["labels"].ne(-100),
                target_positions=batch["target_positions"],
                auxiliary=auxiliary,
            )
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        component_parameters = {
            "landmarks": model.geometry.remainder_embedding.weight,
            "fourier_projection": model.geometry.fourier_projection.weight,
            "modulus_encoder": model.modulus_encoder.scalar_projection[0].weight,
            "transition": model.transition.left.weight,
            "temperature": model.geometry.log_temperature,
            "answer_decoder": model.answer_decoder.weight,
            "reconstruction_decoder": model.reconstruction_decoder.weight,
        }
        for name, parameter in component_parameters.items():
            with self.subTest(component=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_extreme_supported_t_is_numerically_stable(self) -> None:
        model = self.build_model().eval()
        batch = prompt_batch(row(323, 5, 64))

        def identity_transition(state, context):
            return state

        def cheap_projection(candidate, *, sharpen):
            probabilities = torch.full(
                (candidate.shape[0], 323),
                1.0 / 323.0,
                dtype=candidate.dtype,
                device=candidate.device,
            )
            return candidate, probabilities

        with (
            mock.patch.object(model.transition, "forward", identity_transition),
            mock.patch.object(model.geometry, "project", cheap_projection),
            torch.no_grad(),
        ):
            logits, auxiliary = model(batch["input_ids"], batch["attention_mask"])
        self.assertEqual(auxiliary["active_history"].sum().item(), 64)
        for key in (
            "final_state",
            "final_landmark_probabilities",
            "entropy_history",
            "state_norm_history",
            "movement_history",
        ):
            self.assertTrue(torch.isfinite(auxiliary[key]).all(), key)
        self.assertTrue(torch.isfinite(logits).all())

    def test_source_rule_safety_and_standalone_imports(self) -> None:
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
            with self.subTest(rule=name):
                self.assertIsNone(re.search(pattern, source, flags=re.IGNORECASE))
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
        self.assertLessEqual(imports, {"__future__", "math", "torch", "benchmark"})


if __name__ == "__main__":
    unittest.main()
