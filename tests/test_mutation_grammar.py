from __future__ import annotations

import unittest
from fractions import Fraction

from src.mathlang.ast import BinaryOp, Const, NaryOp, UnaryOp, Var
from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion.mutation_grammar import (
    ADD_EXPR_FAMILY,
    CONST_FAMILY,
    DIV_EXPR_FAMILY,
    MUL_EXPR_FAMILY,
    POW_EXPR_FAMILY,
    UNARY_EXPR_FAMILY,
    VAR_FAMILY,
    can_replace,
    compatible_replacement_families,
    production_family,
    subtree_size,
)


class MutationGrammarTests(unittest.TestCase):
    def test_family_classification(self) -> None:
        self.assertEqual(production_family(Const(value=Fraction(1, 1))), CONST_FAMILY)
        self.assertEqual(production_family(Var(name="x")), VAR_FAMILY)
        self.assertEqual(production_family(UnaryOp(op="sin", operand=Var(name="x"))), UNARY_EXPR_FAMILY)
        self.assertEqual(production_family(NaryOp(op="add", operands=(Var(name="x"), Const(value=Fraction(1, 1))))), ADD_EXPR_FAMILY)
        self.assertEqual(production_family(NaryOp(op="mul", operands=(Var(name="x"), Const(value=Fraction(2, 1))))), MUL_EXPR_FAMILY)
        self.assertEqual(production_family(BinaryOp(op="pow", left=Var(name="x"), right=Const(value=Fraction(2, 1)))), POW_EXPR_FAMILY)
        self.assertEqual(production_family(BinaryOp(op="div", left=Var(name="x"), right=Const(value=Fraction(2, 1)))), DIV_EXPR_FAMILY)

    def test_same_family_compatibility_only(self) -> None:
        expr = parse_prefix_string("sin x")
        self.assertEqual(compatible_replacement_families(expr), (UNARY_EXPR_FAMILY,))
        self.assertTrue(can_replace(expr, parse_prefix_string("cos x")))
        self.assertFalse(can_replace(expr, parse_prefix_string("pow x INT+ 2")))

    def test_const_rules_reject_non_const(self) -> None:
        expr = parse_prefix_string("INT+ 2")
        self.assertTrue(can_replace(expr, Const(value=Fraction(3, 1))))
        self.assertFalse(can_replace(expr, parse_prefix_string("sin x")))

    def test_var_is_non_mutable(self) -> None:
        expr = parse_prefix_string("x")
        self.assertEqual(compatible_replacement_families(expr), ())
        self.assertFalse(can_replace(expr, Var(name="x")))

    def test_pow_family_allows_non_constant_exponents(self) -> None:
        expr = parse_prefix_string("pow x sin x")
        candidate = parse_prefix_string("pow x add x INT+ 1")
        self.assertTrue(can_replace(expr, candidate))

    def test_subtree_size_counts_operator_nodes_only(self) -> None:
        expr = parse_prefix_string("add sin x pow x INT+ 2")
        self.assertEqual(subtree_size(expr), 3)
