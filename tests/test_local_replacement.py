from __future__ import annotations

import random
import unittest

from src.mathlang.ast import BinaryOp, UnaryOp, Var
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.mutation import local_replace_once, replace_subtree_by_node_id
from src.tree_diffusion.mutation_grammar import can_locally_replace, node_arity, node_shape
from tests.mutation_test_utils import canonical_expr, num, validate_mutation_result


class LocalReplacementTests(unittest.TestCase):
    def assert_local_leaf_replacement(self, original_expression: str, replacement) -> None:
        original = parse_prefix_string(original_expression)

        self.assertNotEqual(original, replacement)
        self.assertTrue(can_locally_replace(original, replacement))
        self.assertEqual(node_shape(original), node_shape(replacement))
        self.assertEqual(node_arity(original), node_arity(replacement))

        mutated = canonicalize(replace_subtree_by_node_id(original, 0, replacement))
        serialized = serialize_prefix_string(mutated)
        self.assertEqual(canonicalize(parse_prefix_string(serialized)), mutated)

    def test_explicit_leaf_to_leaf_local_replacements_are_legal(self) -> None:
        self.assert_local_leaf_replacement("INT+ 2", num(3))
        self.assert_local_leaf_replacement("INT+ 2", num(1, 2))
        self.assert_local_leaf_replacement("INT+ 2", Var(name="x"))
        self.assert_local_leaf_replacement("INT+ 2", num(-2))
        self.assert_local_leaf_replacement("INT+ 2", parse_prefix_string("Pi"))
        self.assert_local_leaf_replacement("Pi", num(1))
        self.assert_local_leaf_replacement("x", num(1))
        self.assert_local_leaf_replacement("x", parse_prefix_string("Pi"))

    def test_explicit_operator_local_replacements_are_legal(self) -> None:
        self.assertTrue(can_locally_replace(parse_prefix_string("sin x"), parse_prefix_string("cos x")))
        self.assertTrue(can_locally_replace(parse_prefix_string("pow x INT+ 2"), parse_prefix_string("div x INT+ 2")))
        self.assertTrue(
            can_locally_replace(
                parse_prefix_string("add x pow x INT+ 2"),
                parse_prefix_string("mul x pow x INT+ 2"),
            )
        )

    def test_cross_shape_and_cross_arity_local_replacements_are_illegal(self) -> None:
        self.assertFalse(can_locally_replace(parse_prefix_string("sin x"), parse_prefix_string("pow x INT+ 2")))
        self.assertFalse(can_locally_replace(parse_prefix_string("pow x INT+ 2"), parse_prefix_string("INT+ 2")))
        self.assertTrue(can_locally_replace(parse_prefix_string("add x INT+ 1"), parse_prefix_string("pow x INT+ 1")))
        self.assertFalse(
            can_locally_replace(
                parse_prefix_string("add x add INT+ 1 INT+ 2"),
                parse_prefix_string("mul x INT+ 1"),
            )
        )

    def test_local_replace_once_preserves_shape_and_children_for_operator_roots(self) -> None:
        cases = (
            ("unary", canonical_expr("sin x")),
            ("binary", canonical_expr("pow x INT+ 2")),
            ("associative_chain", canonical_expr("add x add sin x add pow x INT+ 2 ln x")),
        )

        for case_name, expr in cases:
            with self.subTest(case=case_name):
                result = local_replace_once(expr, selected_node_id=0, rng=random.Random(0))
                validated = validate_mutation_result(self, expr, result)
                assert result is not None

                self.assertEqual(node_shape(result.original_subtree), node_shape(result.replacement_subtree))
                self.assertEqual(node_arity(result.original_subtree), node_arity(result.replacement_subtree))
                self.assertTrue(validated.has_local_like_kind)

                if isinstance(result.original_subtree, UnaryOp):
                    self.assertEqual(result.original_subtree.operand, result.replacement_subtree.operand)
                elif isinstance(result.original_subtree, BinaryOp):
                    self.assertEqual(result.original_subtree.left, result.replacement_subtree.left)
                    self.assertEqual(result.original_subtree.right, result.replacement_subtree.right)

    def test_local_replace_once_can_mutate_an_associative_binary_root(self) -> None:
        expr = canonical_expr("add x add sin x add pow x INT+ 2 add ln x cos x")
        result = local_replace_once(expr, selected_node_id=0, rng=random.Random(0))
        validate_mutation_result(self, expr, result)
        assert result is not None

        self.assertIsInstance(result.original_subtree, BinaryOp)
        self.assertIsInstance(result.replacement_subtree, BinaryOp)
        self.assertEqual(result.original_subtree.left, result.replacement_subtree.left)
        self.assertEqual(result.original_subtree.right, result.replacement_subtree.right)
        self.assertEqual(result.original_subtree.op, "add")
        self.assertNotEqual(result.replacement_subtree.op, "add")
