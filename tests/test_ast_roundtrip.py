from __future__ import annotations

import unittest
from fractions import Fraction

from src.mathlang.ast import BinaryOp, Const, NaryOp, UnaryOp
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string


class AstRoundtripTests(unittest.TestCase):
    def assert_roundtrip(self, expression: str) -> None:
        expr = parse_prefix_string(expression)
        reparsed = parse_prefix_string(serialize_prefix_string(expr))
        self.assertEqual(expr, reparsed)

    def test_simple_unary_example(self) -> None:
        expr = parse_prefix_string("sin x")
        self.assertIsInstance(expr, UnaryOp)
        self.assertEqual(serialize_prefix_string(expr), "sin x")
        self.assert_roundtrip("sin x")

    def test_simple_binary_example(self) -> None:
        expr = parse_prefix_string("pow x INT+ 2")
        self.assertIsInstance(expr, BinaryOp)
        self.assertEqual(serialize_prefix_string(expr), "pow x INT+ 2")
        self.assert_roundtrip("pow x INT+ 2")

    def test_nested_expression_example(self) -> None:
        expr = parse_prefix_string("add mul INT+ 2 x pow x INT+ 3")
        self.assertIsInstance(expr, NaryOp)
        self.assertEqual(serialize_prefix_string(expr), "add mul INT+ 2 x pow x INT+ 3")
        self.assert_roundtrip("add mul INT+ 2 x pow x INT+ 3")

    def test_negative_integer_constant_example(self) -> None:
        expr = parse_prefix_string("INT- 1 0")
        self.assertIsInstance(expr, Const)
        self.assertEqual(expr.value, Fraction(-10, 1))
        self.assertEqual(serialize_prefix_string(expr), "INT- 1 0")
        self.assert_roundtrip("INT- 1 0")

    def test_rational_constant_example(self) -> None:
        expr = parse_prefix_string("div INT- 1 INT+ 2")
        self.assertIsInstance(expr, Const)
        self.assertEqual(expr.value, Fraction(-1, 2))
        self.assertEqual(serialize_prefix_string(expr), "div INT- 1 INT+ 2")
        self.assert_roundtrip("div INT- 1 INT+ 2")
