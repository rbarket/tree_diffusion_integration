from __future__ import annotations

import random
import unittest
from fractions import Fraction

from src.mathlang.ast import Const
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.mutation import (
    mutate_once,
    replace_subtree_by_node_id,
    sample_const_replacement,
    sample_valid_subtree,
)
from src.tree_diffusion.mutation_grammar import can_replace
from src.tree_diffusion.positions import index_tree_positions


class MutationTests(unittest.TestCase):
    def test_unary_family_swap_is_legal(self) -> None:
        self.assertTrue(
            can_replace(parse_prefix_string("sin x"), parse_prefix_string("cos x"))
        )

    def test_pow_exponent_constant_mutation_is_legal(self) -> None:
        expr = canonicalize(parse_prefix_string("pow x INT+ 5"))
        index = index_tree_positions(expr, sigma_small=0)
        exponent_position = next(
            position
            for position in index.positions
            if position.production_family == "CONST" and position.token_start > 0
        )
        mutated = replace_subtree_by_node_id(
            expr,
            exponent_position.node_id,
            Const(value=Fraction(3, 1)),
        )
        self.assertEqual(serialize_prefix_string(mutated), "pow x INT+ 3")

    def test_const_cannot_be_replaced_with_unary_subtree(self) -> None:
        self.assertFalse(
            can_replace(parse_prefix_string("INT+ 2"), parse_prefix_string("sin x"))
        )

    def test_pow_to_div_direct_replacement_is_illegal(self) -> None:
        self.assertFalse(
            can_replace(parse_prefix_string("pow x INT+ 2"), parse_prefix_string("div x INT+ 2"))
        )

    def test_sampled_subtrees_roundtrip(self) -> None:
        rng = random.Random(0)
        for family in ("CONST", "UNARY_EXPR", "ADD_EXPR", "MUL_EXPR", "POW_EXPR", "DIV_EXPR"):
            with self.subTest(family=family):
                subtree = sample_valid_subtree(family, sigma_small=2, rng=rng)
                reparsed = parse_prefix_string(serialize_prefix_string(subtree))
                self.assertEqual(canonicalize(subtree), canonicalize(reparsed))

    def test_mutation_respects_sigma_small(self) -> None:
        result = mutate_once(parse_prefix_string("pow x INT+ 5"), sigma_small=0, rng=random.Random(0))
        self.assertIsNotNone(result)
        self.assertEqual(result.selected_family, "CONST")
        self.assertEqual(serialize_prefix_string(result.mutated_expr).split()[0], "pow")

    def test_repeated_mutation_does_not_crash(self) -> None:
        rng = random.Random(7)
        expr = parse_prefix_string("add sin x pow x INT+ 2")
        for _ in range(20):
            result = mutate_once(expr, sigma_small=2, rng=rng)
            self.assertIsNotNone(result)
            expr = result.mutated_expr
            reparsed = parse_prefix_string(serialize_prefix_string(expr))
            self.assertEqual(expr, canonicalize(reparsed))

    def test_single_variable_leaf_has_no_mutation(self) -> None:
        self.assertIsNone(mutate_once(parse_prefix_string("x"), sigma_small=0, rng=random.Random(0)))

    def test_const_replacement_is_local(self) -> None:
        replacement = sample_const_replacement(Const(value=Fraction(2, 1)), random.Random(0))
        self.assertIsInstance(replacement, Const)
        self.assertNotEqual(replacement.value, Fraction(2, 1))
