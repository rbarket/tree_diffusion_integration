from __future__ import annotations

import random
import unittest
from fractions import Fraction

from src.mathlang.ast import BinaryOp, Const, Var
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.mutation import local_replace_once, replace_subtree_by_node_id, sample_const_replacement, sample_valid_subtree
from src.tree_diffusion.mutation_grammar import can_locally_replace
from tests.mutation_test_utils import num, validate_mutation_result


class ConstantMutationTests(unittest.TestCase):
    def assert_leaf_transition_roundtrips(self, original_expression: str, replacement) -> None:
        original = parse_prefix_string(original_expression)

        self.assertNotEqual(original, replacement)
        self.assertTrue(can_locally_replace(original, replacement))

        mutated = canonicalize(replace_subtree_by_node_id(original, 0, replacement))
        serialized = serialize_prefix_string(mutated)
        self.assertEqual(mutated, canonicalize(parse_prefix_string(serialized)))

    def test_explicit_leaf_transition_matrix(self) -> None:
        self.assert_leaf_transition_roundtrips("INT+ 2", num(3))
        self.assert_leaf_transition_roundtrips("INT+ 2", parse_prefix_string("Pi"))
        self.assert_leaf_transition_roundtrips("INT+ 2", Var(name="x"))
        self.assert_leaf_transition_roundtrips("Pi", num(1))
        self.assert_leaf_transition_roundtrips("x", num(1))
        self.assert_leaf_transition_roundtrips("x", parse_prefix_string("Pi"))

    def test_local_replace_once_hits_explicit_leaf_mutation_cases(self) -> None:
        cases = (
            ("x", 1, "INT+ 1"),
            ("x", 5, "Pi"),
            ("INT+ 2", 5, "x"),
            ("Pi", 1, "INT+ 1"),
        )

        for expression, seed, expected_replacement in cases:
            with self.subTest(expression=expression, seed=seed):
                result = local_replace_once(parse_prefix_string(expression), selected_node_id=0, rng=random.Random(seed))
                validate_mutation_result(self, parse_prefix_string(expression), result)
                assert result is not None
                self.assertEqual(serialize_prefix_string(result.replacement_subtree), expected_replacement)

    def test_sample_const_replacement_handles_named_negative_and_rational_constants(self) -> None:
        cases = (
            (Const(symbol="Pi"), random.Random(0)),
            (Const(value=Fraction(-3, 1)), random.Random(1)),
            (Const(value=Fraction(-3, 4)), random.Random(8)),
        )

        for original, rng in cases:
            with self.subTest(original=serialize_prefix_string(original)):
                replacement = sample_const_replacement(original, rng)
                self.assertIsInstance(replacement, Const)
                self.assertNotEqual(replacement, original)

                if replacement.is_numeric:
                    self.assertNotEqual(replacement.value.denominator, 0)

                serialized = serialize_prefix_string(replacement)
                self.assertEqual(canonicalize(parse_prefix_string(serialized)), canonicalize(replacement))

    def test_sampled_div_subtrees_never_use_zero_numeric_denominator(self) -> None:
        for seed in range(50):
            with self.subTest(seed=seed):
                subtree = sample_valid_subtree("DIV_EXPR", sigma_small=2, rng=random.Random(seed))
                self.assertIsInstance(subtree, BinaryOp)

                if isinstance(subtree.right, Const) and subtree.right.is_numeric:
                    self.assertNotEqual(subtree.right.value, 0)

                serialized = serialize_prefix_string(subtree)
                self.assertEqual(canonicalize(parse_prefix_string(serialized)), canonicalize(subtree))
