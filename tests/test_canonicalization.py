from __future__ import annotations

import unittest
from fractions import Fraction

from src.mathlang.ast import Const, NaryOp
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string


class CanonicalizationTests(unittest.TestCase):
    def canonical(self, expression: str):
        return canonicalize(parse_prefix_string(expression))

    def test_commutative_reordering_normalization(self) -> None:
        left = self.canonical("add pow x INT+ 2 x")
        right = self.canonical("add x pow x INT+ 2")
        self.assertEqual(left, right)
        self.assertEqual(serialize_prefix_string(left), "add x pow x INT+ 2")

    def test_associative_normalization(self) -> None:
        expr = self.canonical("add sin x add x pow x INT+ 3")
        self.assertIsInstance(expr, NaryOp)
        self.assertEqual(
            [serialize_prefix_string(operand) for operand in expr.operands],
            ["x", "sin x", "pow x INT+ 3"],
        )

    def test_signed_rational_normalization(self) -> None:
        expr = self.canonical("div INT- 2 INT- 4")
        self.assertIsInstance(expr, Const)
        self.assertEqual(expr.value, Fraction(1, 2))
        self.assertEqual(serialize_prefix_string(expr), "div INT+ 1 INT+ 2")

    def test_idempotence(self) -> None:
        expr = self.canonical("add INT+ 5 add x INT+ 2")
        self.assertEqual(expr, canonicalize(expr))

    def test_top_level_additive_constant_stripping(self) -> None:
        expr = self.canonical("add INT+ 7 add x INT+ 2")
        self.assertEqual(serialize_prefix_string(expr), "x")

    def test_non_top_level_constants_are_not_stripped(self) -> None:
        expr = self.canonical("mul add x INT+ 1 INT+ 2")
        self.assertEqual(serialize_prefix_string(expr), "mul INT+ 2 add INT+ 1 x")
