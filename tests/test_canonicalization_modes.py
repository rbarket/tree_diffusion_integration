from __future__ import annotations

import random
import unittest
from unittest.mock import patch

from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.dataset import pairs_from_prefix_rows
from src.tree_diffusion.observation import compute_symbolic_residual
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.training_examples import generate_training_example


class CanonicalizationModeTests(unittest.TestCase):
    def test_integrand_and_observation_canonicalization_keep_additive_constants(self) -> None:
        expr = parse_prefix_string("add x INT+ 1")

        integrand = canonicalize(expr, strip_additive_constants=False)
        observation_expr = canonicalize(expr, strip_additive_constants=False)

        self.assertEqual(serialize_prefix_string(integrand), "add INT+ 1 x")
        self.assertEqual(serialize_prefix_string(observation_expr), "add INT+ 1 x")

    def test_integral_canonicalization_strips_top_level_additive_constants(self) -> None:
        expr = parse_prefix_string("add sin x INT+ 7")

        integral = canonicalize(expr)

        self.assertEqual(serialize_prefix_string(integral), "sin x")

    def test_symbolic_residual_preserves_constant_residual(self) -> None:
        residual = compute_symbolic_residual(
            current_derivative=parse_prefix_string("add x INT+ 1"),
            target_integrand=parse_prefix_string("x"),
        )

        self.assertEqual(serialize_prefix_string(residual), "INT+ 1")

    def test_dataset_and_training_input_preserve_integrand_additive_constant(self) -> None:
        pair = pairs_from_prefix_rows(
            [
                {
                    "integrand_prefix": "add x INT+ 1",
                    "integral_prefix": "add div pow x INT+ 2 INT+ 2 x",
                }
            ]
        )[0]

        self.assertEqual(serialize_prefix_string(pair.target_integrand), "add INT+ 1 x")

        with patch(
            "src.tree_diffusion.training_examples.generate_current_candidate",
            return_value=(parse_prefix_string("x"), 1, False),
        ):
            example = generate_training_example(
                pair.target_integrand,
                pair.target_antiderivative,
                tokenizer=TreeDiffusionTokenizer(max_positions=64),
                rng=random.Random(0),
                residual_mode="none",
                sigma_small=3,
            )

        integrand_tokens = _section(example.input_tokens, "<F>", "</F>")
        self.assertEqual(integrand_tokens, ["add", "INT+", "1", "x"])


def _section(tokens: list[str], start_token: str, end_token: str) -> list[str]:
    start = tokens.index(start_token) + 1
    end = tokens.index(end_token, start)
    return tokens[start:end]


if __name__ == "__main__":
    unittest.main()
