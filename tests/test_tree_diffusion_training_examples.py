from __future__ import annotations

import random
import unittest
from unittest.mock import patch

from src.mathlang.ast import Expr
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string, parse_prefix_tokens
from src.mathlang.serializer import serialize_prefix_tokens
from src.tree_diffusion.edit_path import EditTarget, structural_distance
from src.tree_diffusion.observation import Observation
from src.tree_diffusion.positions import index_tree_positions
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.training_examples import (
    TreeDiffusionTrainingExample,
    generate_current_candidate,
    generate_training_example,
)


class TreeDiffusionTrainingExamplesTests(unittest.TestCase):
    def test_generate_current_candidate_validates_arguments(self) -> None:
        target = parse_prefix_string("pow x INT+ 3")

        invalid_cases = (
            {"rho": -0.1},
            {"rho": 1.1},
            {"sigma_small": 0},
            {"smax": 0},
            {"max_attempts": 0},
        )

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    generate_current_candidate(target, **kwargs)

    def test_mutation_mode_candidate_differs_from_target(self) -> None:
        target = parse_prefix_string("pow x INT+ 3")

        current, num_mutations, used_random_init = generate_current_candidate(
            target,
            rng=random.Random(0),
            rho=0.0,
            smax=3,
            sigma_small=2,
        )

        self.assertFalse(used_random_init)
        self.assertGreaterEqual(num_mutations, 1)
        self.assertNotEqual(canonicalize(current), canonicalize(target))
        reparsed = parse_prefix_tokens(serialize_prefix_tokens(current))
        self.assertEqual(canonicalize(current), canonicalize(reparsed))

    def test_random_init_candidate_path(self) -> None:
        target = parse_prefix_string("pow x INT+ 3")

        current, num_mutations, used_random_init = generate_current_candidate(
            target,
            rng=random.Random(1),
            rho=1.0,
            sigma_small=2,
            max_random_size=3,
        )

        self.assertTrue(used_random_init)
        self.assertEqual(num_mutations, 0)
        self.assertIsInstance(current, Expr)
        reparsed = parse_prefix_tokens(serialize_prefix_tokens(current))
        self.assertEqual(canonicalize(current), canonicalize(reparsed))
        self.assertNotEqual(canonicalize(current), canonicalize(target))

    def test_generated_training_example_has_expected_token_structure(self) -> None:
        example = _generate_simple_example(seed=2)

        self.assertIsInstance(example, TreeDiffusionTrainingExample)
        self.assertIsInstance(example.observation, Observation)
        self.assertIsInstance(example.edit_target, EditTarget)
        self.assertEqual(example.input_tokens[-1], "<EDIT>")
        self.assertTrue(example.target_tokens[0].startswith("<POS_"))
        self.assertEqual(example.target_tokens[-1], "<eos>")
        self.assertIsNone(example.input_ids)
        self.assertIsNone(example.target_ids)
        self.assertFalse(example.used_random_init)
        self.assertGreaterEqual(example.num_mutations, 1)

    def test_target_position_is_valid_in_current_tree(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=64)
        example = _generate_simple_example(seed=3, tokenizer=tokenizer)
        index = index_tree_positions(example.current_antiderivative)

        position = tokenizer.token_to_position(example.target_tokens[0])

        self.assertIn(position, index.node_id_to_node)

    def test_target_replacement_matches_edit_target(self) -> None:
        example = _generate_simple_example(seed=4)

        self.assertEqual(
            example.target_tokens[1:-1],
            serialize_prefix_tokens(example.edit_target.replacement_subtree),
        )

    def test_edit_moves_structurally_closer_or_reaches_target(self) -> None:
        example = _generate_simple_example(seed=5)

        before = structural_distance(
            example.current_antiderivative,
            example.target_antiderivative,
        )
        after = structural_distance(
            example.edit_target.resulting_tree,
            example.target_antiderivative,
        )

        self.assertLessEqual(after, before)
        self.assertLess(after, before)

    def test_observation_does_not_contain_target_antiderivative_dedicated_field(self) -> None:
        current = parse_prefix_string("sin x")
        target_integrand = parse_prefix_string("sin x")
        target_antiderivative = parse_prefix_string("mul INT- 1 cos x")

        with patch(
            "src.tree_diffusion.training_examples.generate_current_candidate",
            return_value=(current, 1, False),
        ):
            example = generate_training_example(
                target_integrand,
                target_antiderivative,
                tokenizer=TreeDiffusionTokenizer(max_positions=64),
                rng=random.Random(6),
                rho=0.0,
                smax=1,
                sigma_small=2,
            )

        for token in ("<F>", "</F>", "<CUR>", "</CUR>", "<DER>", "</DER>", "<RES>", "</RES>", "<NUM>", "</NUM>", "<EDIT>"):
            self.assertIn(token, example.input_tokens)

        for forbidden in ("<TARGET>", "<GOLD>", "<ANTIDERIVATIVE_TARGET>", "<I_STAR>"):
            self.assertNotIn(forbidden, example.input_tokens)

    def test_encode_true_populates_ids_and_roundtrips(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=64)
        example = _generate_simple_example(
            seed=7,
            tokenizer=tokenizer,
            encode=True,
            max_input_length=256,
            max_target_length=64,
        )

        self.assertIsNotNone(example.input_ids)
        self.assertIsNotNone(example.target_ids)
        assert example.input_ids is not None
        assert example.target_ids is not None
        self.assertEqual(len(example.input_ids), 256)
        self.assertEqual(len(example.target_ids), 64)
        self.assertEqual(tokenizer.decode_ids(example.input_ids, strip_pad=True), example.input_tokens)
        self.assertEqual(tokenizer.decode_ids(example.target_ids, strip_pad=True), example.target_tokens)

    def test_deterministic_with_seeded_rng(self) -> None:
        first = _generate_simple_example(seed=8)
        second = _generate_simple_example(seed=8)

        self.assertEqual(
            serialize_prefix_tokens(first.current_antiderivative),
            serialize_prefix_tokens(second.current_antiderivative),
        )
        self.assertEqual(first.input_tokens, second.input_tokens)
        self.assertEqual(first.target_tokens, second.target_tokens)

    def test_residual_modes(self) -> None:
        current = parse_prefix_string("pow x INT+ 5")
        target_integrand = parse_prefix_string("pow x INT+ 2")
        target_antiderivative = parse_prefix_string("pow x INT+ 3")

        for residual_mode in ("none", "symbolic", "numeric", "both"):
            with self.subTest(residual_mode=residual_mode):
                with patch(
                    "src.tree_diffusion.training_examples.generate_current_candidate",
                    return_value=(current, 1, False),
                ):
                    example = generate_training_example(
                        target_integrand,
                        target_antiderivative,
                        tokenizer=TreeDiffusionTokenizer(max_positions=64),
                        rng=random.Random(9),
                        rho=0.0,
                        smax=1,
                        sigma_small=2,
                        residual_mode=residual_mode,
                    )

                residual_tokens = _section(example.input_tokens, "<RES>", "</RES>")
                numeric_tokens = _section(example.input_tokens, "<NUM>", "</NUM>")
                if residual_mode == "none":
                    self.assertEqual(residual_tokens, ["<NO_RES>"])
                    self.assertEqual(numeric_tokens, ["<NO_NUM>"])
                elif residual_mode == "symbolic":
                    self.assertNotEqual(residual_tokens, ["<NO_RES>"])
                    self.assertEqual(numeric_tokens, ["<NO_NUM>"])
                elif residual_mode == "numeric":
                    self.assertEqual(residual_tokens, ["<NO_RES>"])
                    self.assertNotEqual(numeric_tokens, ["<NO_NUM>"])
                else:
                    self.assertNotEqual(residual_tokens, ["<NO_RES>"])
                    self.assertNotEqual(numeric_tokens, ["<NO_NUM>"])

    def test_retry_on_already_target_candidate(self) -> None:
        target_integrand = parse_prefix_string("pow x INT+ 2")
        target_antiderivative = parse_prefix_string("pow x INT+ 3")
        current = parse_prefix_string("pow x INT+ 5")

        with patch(
            "src.tree_diffusion.training_examples.generate_current_candidate",
            side_effect=[
                (target_antiderivative, 0, False),
                (current, 1, False),
            ],
        ):
            example = generate_training_example(
                target_integrand,
                target_antiderivative,
                tokenizer=TreeDiffusionTokenizer(max_positions=64),
                rng=random.Random(10),
                rho=0.0,
                smax=1,
                sigma_small=2,
            )

        self.assertEqual(example.attempts, 2)
        self.assertIsNotNone(example.edit_target)
        self.assertEqual(canonicalize(example.current_antiderivative), canonicalize(current))


def _generate_simple_example(
    *,
    seed: int,
    tokenizer: TreeDiffusionTokenizer | None = None,
    encode: bool = False,
    max_input_length: int | None = None,
    max_target_length: int | None = None,
) -> TreeDiffusionTrainingExample:
    return generate_training_example(
        parse_prefix_string("pow x INT+ 2"),
        parse_prefix_string("div pow x INT+ 3 INT+ 3"),
        tokenizer=tokenizer or TreeDiffusionTokenizer(max_positions=64),
        rng=random.Random(seed),
        rho=0.0,
        smax=2,
        sigma_small=2,
        encode=encode,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )


def _section(tokens: list[str], start_token: str, end_token: str) -> list[str]:
    start = tokens.index(start_token) + 1
    end = tokens.index(end_token, start)
    return tokens[start:end]


if __name__ == "__main__":
    unittest.main()
