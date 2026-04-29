from __future__ import annotations

import unittest
from fractions import Fraction

from src.mathlang.ast import BinaryOp, Const, UnaryOp, Var
from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion.mutation_grammar import (
    ADD_EXPR_FAMILY,
    CONST_FAMILY,
    DIV_EXPR_FAMILY,
    MUL_EXPR_FAMILY,
    POW_EXPR_FAMILY,
    UNARY_EXPR_FAMILY,
    VAR_FAMILY,
    can_locally_replace,
    can_replace,
    can_sampled_subtree_replace,
    local_replacement_candidates,
    node_arity,
    node_shape,
    production_family,
    subtree_size,
)


class MutationGrammarTests(unittest.TestCase):
    def test_family_classification(self) -> None:
        self.assertEqual(production_family(Const(value=Fraction(1, 1))), CONST_FAMILY)
        self.assertEqual(production_family(Var(name="x")), VAR_FAMILY)
        self.assertEqual(production_family(UnaryOp(op="sin", operand=Var(name="x"))), UNARY_EXPR_FAMILY)
        self.assertEqual(
            production_family(BinaryOp(op="add", left=Var(name="x"), right=Const(value=Fraction(1, 1)))),
            ADD_EXPR_FAMILY,
        )
        self.assertEqual(
            production_family(BinaryOp(op="mul", left=Var(name="x"), right=Const(value=Fraction(2, 1)))),
            MUL_EXPR_FAMILY,
        )
        self.assertEqual(
            production_family(BinaryOp(op="pow", left=Var(name="x"), right=Const(value=Fraction(2, 1)))),
            POW_EXPR_FAMILY,
        )
        self.assertEqual(
            production_family(BinaryOp(op="div", left=Var(name="x"), right=Const(value=Fraction(2, 1)))),
            DIV_EXPR_FAMILY,
        )

    def test_node_shape_and_arity_follow_constructor_shape(self) -> None:
        self.assertEqual(node_shape(Const(value=Fraction(1, 1))), "leaf")
        self.assertEqual(node_shape(Var(name="x")), "leaf")
        self.assertEqual(node_shape(UnaryOp(op="sin", operand=Var(name="x"))), "unary")
        self.assertEqual(node_shape(BinaryOp(op="pow", left=Var(name="x"), right=Const(value=Fraction(2, 1)))), "binary")
        self.assertEqual(
            node_shape(BinaryOp(op="add", left=Var(name="x"), right=Const(value=Fraction(1, 1)))),
            "binary",
        )
        self.assertEqual(node_arity(Const(value=Fraction(1, 1))), 0)
        self.assertEqual(node_arity(Var(name="x")), 0)
        self.assertEqual(node_arity(UnaryOp(op="sin", operand=Var(name="x"))), 1)
        self.assertEqual(node_arity(BinaryOp(op="pow", left=Var(name="x"), right=Const(value=Fraction(2, 1)))), 2)
        self.assertEqual(
            node_arity(BinaryOp(op="add", left=Var(name="x"), right=Const(value=Fraction(1, 1)))),
            2,
        )

    def test_leaf_local_replacements_allow_const_named_const_and_x(self) -> None:
        const_expr = parse_prefix_string("INT+ 2")
        self.assertTrue(can_locally_replace(const_expr, Var(name="x")))
        self.assertTrue(can_locally_replace(const_expr, Const(symbol="Pi")))

        var_expr = parse_prefix_string("x")
        self.assertTrue(can_locally_replace(var_expr, Const(value=Fraction(1, 1))))
        self.assertTrue(can_locally_replace(var_expr, Const(symbol="Pi")))
        self.assertFalse(can_locally_replace(var_expr, Var(name="x")))

        const_candidates = {spec.leaf_kind for spec in local_replacement_candidates(const_expr)}
        var_candidates = {spec.leaf_kind for spec in local_replacement_candidates(var_expr)}
        self.assertEqual(const_candidates, {"numeric_const", "named_const", "var"})
        self.assertEqual(var_candidates, {"numeric_const", "named_const"})

    def test_unary_local_replacements_require_unary_shape(self) -> None:
        self.assertTrue(can_locally_replace(parse_prefix_string("sin x"), parse_prefix_string("cos x")))
        self.assertTrue(can_locally_replace(parse_prefix_string("exp x"), parse_prefix_string("ln x")))
        self.assertFalse(can_locally_replace(parse_prefix_string("sin x"), parse_prefix_string("pow x INT+ 2")))
        self.assertFalse(can_locally_replace(parse_prefix_string("sin x"), parse_prefix_string("div x INT+ 2")))
        self.assertTrue(can_replace(parse_prefix_string("sin x"), parse_prefix_string("cos x")))

    def test_binary_local_replacements_require_binary_shape(self) -> None:
        self.assertTrue(can_locally_replace(parse_prefix_string("pow x INT+ 2"), parse_prefix_string("div x INT+ 2")))
        self.assertTrue(can_locally_replace(parse_prefix_string("div x INT+ 2"), parse_prefix_string("pow x INT+ 2")))
        self.assertFalse(can_locally_replace(parse_prefix_string("pow x INT+ 2"), parse_prefix_string("INT+ 2")))

    def test_binary_add_mul_local_replacements_preserve_children(self) -> None:
        original = BinaryOp(
            op="add",
            left=Var(name="x"),
            right=Const(value=Fraction(1, 1)),
        )
        candidate = BinaryOp(
            op="mul",
            left=Var(name="x"),
            right=Const(value=Fraction(1, 1)),
        )
        pow_candidate = BinaryOp(
            op="pow",
            left=Var(name="x"),
            right=Const(value=Fraction(1, 1)),
        )
        self.assertTrue(can_locally_replace(original, candidate))
        self.assertTrue(can_locally_replace(original, pow_candidate))
        self.assertTrue(can_locally_replace(parse_prefix_string("add x INT+ 1"), parse_prefix_string("pow x INT+ 1")))

    def test_identical_or_cross_shape_local_replacements_are_rejected(self) -> None:
        self.assertFalse(can_locally_replace(parse_prefix_string("sin x"), parse_prefix_string("sin x")))
        self.assertFalse(can_locally_replace(parse_prefix_string("pow x INT+ 2"), parse_prefix_string("INT+ 2")))
        self.assertFalse(can_locally_replace(parse_prefix_string("x"), parse_prefix_string("add x INT+ 1")))
        self.assertFalse(can_locally_replace(parse_prefix_string("INT+ 2"), parse_prefix_string("exp x")))
        self.assertFalse(can_locally_replace(parse_prefix_string("add x INT+ 1"), parse_prefix_string("sin x")))

    def test_sampled_subtree_replacement_allows_cross_shape_expr_replacements(self) -> None:
        self.assertTrue(can_sampled_subtree_replace(parse_prefix_string("pow x INT+ 2"), parse_prefix_string("sin x")))
        self.assertTrue(can_sampled_subtree_replace(parse_prefix_string("sin x"), parse_prefix_string("div x INT+ 2")))
        self.assertTrue(can_sampled_subtree_replace(parse_prefix_string("x"), parse_prefix_string("add x INT+ 1")))
        self.assertTrue(can_sampled_subtree_replace(parse_prefix_string("INT+ 2"), parse_prefix_string("exp x")))

    def test_subtree_size_counts_operator_nodes_only(self) -> None:
        expr = parse_prefix_string("add sin x pow x INT+ 2")
        self.assertEqual(subtree_size(expr), 3)
