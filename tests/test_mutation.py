from __future__ import annotations

import random
import unittest
from fractions import Fraction

from src.mathlang.ast import BinaryOp, Const, UnaryOp, Var
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.mutation import (
    LOCAL_SAME_ARITY_REPLACEMENT,
    SAMPLED_SMALL_SUBTREE_REPLACEMENT,
    local_replace_once,
    mutate_once,
    replace_subtree_by_node_id,
    sample_const_replacement,
    sample_valid_subtree,
)
from src.tree_diffusion.mutation_grammar import can_locally_replace, can_sampled_subtree_replace
from src.tree_diffusion.positions import index_tree_positions


class MutationTests(unittest.TestCase):
    def test_local_replace_once_on_constant_leaf(self) -> None:
        expr = canonicalize(parse_prefix_string("pow x INT+ 5"))
        index = index_tree_positions(expr)
        exponent_position = next(
            position
            for position in index.positions
            if position.production_family == "CONST" and position.token_start > 0
        )

        result = local_replace_once(expr, exponent_position.node_id, rng=random.Random(0))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.mutation_kind, LOCAL_SAME_ARITY_REPLACEMENT)
        self.assertTrue(can_locally_replace(result.original_subtree, result.replacement_subtree))
        reparsed = parse_prefix_string(serialize_prefix_string(result.mutated_expr))
        self.assertEqual(result.mutated_expr, canonicalize(reparsed))

    def test_local_replace_once_on_variable_leaf(self) -> None:
        result = local_replace_once(parse_prefix_string("x"), selected_node_id=0, rng=random.Random(0))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.selected_family, "VAR")
        self.assertEqual(result.mutation_kind, LOCAL_SAME_ARITY_REPLACEMENT)
        self.assertNotIsInstance(result.replacement_subtree, Var)
        reparsed = parse_prefix_string(serialize_prefix_string(result.mutated_expr))
        self.assertEqual(result.mutated_expr, canonicalize(reparsed))

    def test_local_replace_once_on_unary_node_preserves_child(self) -> None:
        result = local_replace_once(parse_prefix_string("sin x"), selected_node_id=0, rng=random.Random(0))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.mutation_kind, LOCAL_SAME_ARITY_REPLACEMENT)
        self.assertIsInstance(result.replacement_subtree, UnaryOp)
        self.assertEqual(result.replacement_subtree.operand, result.original_subtree.operand)
        self.assertTrue(can_locally_replace(result.original_subtree, result.replacement_subtree))

    def test_local_replace_once_on_binary_node_preserves_children(self) -> None:
        result = local_replace_once(parse_prefix_string("pow x INT+ 2"), selected_node_id=0, rng=random.Random(0))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.mutation_kind, LOCAL_SAME_ARITY_REPLACEMENT)
        self.assertIsInstance(result.replacement_subtree, BinaryOp)
        self.assertEqual(result.replacement_subtree.left, result.original_subtree.left)
        self.assertEqual(result.replacement_subtree.right, result.original_subtree.right)
        self.assertEqual(serialize_prefix_string(result.replacement_subtree), "mul x INT+ 2")

    def test_local_replace_once_on_binary_add_mul_node_preserves_children(self) -> None:
        expr = BinaryOp(
            op="mul",
            left=Var(name="x"),
            right=Const(value=Fraction(2, 1)),
        )
        result = local_replace_once(expr, selected_node_id=0, rng=random.Random(0))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.mutation_kind, LOCAL_SAME_ARITY_REPLACEMENT)
        self.assertIsInstance(result.replacement_subtree, BinaryOp)
        self.assertEqual(result.replacement_subtree.left, result.original_subtree.left)
        self.assertEqual(result.replacement_subtree.right, result.original_subtree.right)
        reparsed = parse_prefix_string(serialize_prefix_string(result.mutated_expr))
        self.assertEqual(result.mutated_expr, canonicalize(reparsed))

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

    def test_sampled_subtrees_roundtrip(self) -> None:
        rng = random.Random(0)
        for family in ("EXPR", "CONST", "UNARY_EXPR", "ADD_EXPR", "MUL_EXPR", "POW_EXPR", "DIV_EXPR"):
            with self.subTest(family=family):
                subtree = sample_valid_subtree(family, sigma_small=2, rng=rng)
                if family in {"ADD_EXPR", "MUL_EXPR"}:
                    self.assertIsInstance(subtree, BinaryOp)
                    self.assertIn(subtree.op, {"add", "mul"})
                reparsed = parse_prefix_string(serialize_prefix_string(subtree))
                self.assertEqual(canonicalize(subtree), canonicalize(reparsed))

    def test_mutation_respects_sigma_small(self) -> None:
        result = mutate_once(parse_prefix_string("pow x INT+ 5"), sigma_small=0, rng=random.Random(0))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn(result.selected_family, {"CONST", "VAR"})
        reparsed = parse_prefix_string(serialize_prefix_string(result.mutated_expr))
        self.assertEqual(result.mutated_expr, canonicalize(reparsed))

    def test_repeated_mutation_does_not_crash(self) -> None:
        rng = random.Random(7)
        expr = parse_prefix_string("add sin x pow x INT+ 2")
        for _ in range(20):
            result = mutate_once(expr, sigma_small=2, rng=rng)
            self.assertIsNotNone(result)
            assert result is not None
            expr = result.mutated_expr
            reparsed = parse_prefix_string(serialize_prefix_string(expr))
            self.assertEqual(expr, canonicalize(reparsed))

    def test_single_variable_leaf_now_has_a_valid_mutation(self) -> None:
        result = mutate_once(parse_prefix_string("x"), sigma_small=0, rng=random.Random(0))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.selected_family, "VAR")
        self.assertIn(result.mutation_kind, {LOCAL_SAME_ARITY_REPLACEMENT, SAMPLED_SMALL_SUBTREE_REPLACEMENT})

    def test_const_replacement_is_local(self) -> None:
        replacement = sample_const_replacement(Const(value=Fraction(2, 1)), random.Random(0))
        self.assertIsInstance(replacement, Const)
        self.assertNotEqual(replacement.value, Fraction(2, 1))

    def test_subtree_replacement_now_supports_root_cross_shape_changes(self) -> None:
        original = parse_prefix_string("pow x INT+ 2")
        replacement = parse_prefix_string("sin x")
        self.assertTrue(can_sampled_subtree_replace(original, replacement))
        self.assertFalse(can_locally_replace(original, replacement))
        mutated = canonicalize(replace_subtree_by_node_id(canonicalize(original), 0, replacement))
        reparsed = parse_prefix_string(serialize_prefix_string(mutated))
        self.assertEqual(mutated, canonicalize(reparsed))
