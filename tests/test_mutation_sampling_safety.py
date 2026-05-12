from __future__ import annotations

import random
import unittest
from fractions import Fraction

from src.mathlang.ast import BinaryOp, Const, Var
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_tokens
from src.tree_diffusion.mutation import (
    is_obviously_zero,
    mutate_once,
    sample_random_expr,
    sample_valid_subtree,
)
from src.tree_diffusion.training_examples import generate_training_example


class MutationSamplingSafetyTests(unittest.TestCase):
    def test_default_random_sampling_does_not_introduce_complex_constant(self) -> None:
        for seed in range(250):
            expr = sample_random_expr(rng=random.Random(seed), max_size=3)
            self.assertNotIn("I", serialize_prefix_tokens(expr))

    def test_default_unary_sampling_does_not_introduce_sign(self) -> None:
        for seed in range(250):
            expr = sample_valid_subtree("UNARY_EXPR", sigma_small=2, rng=random.Random(seed))
            self.assertNotIn("sign", serialize_prefix_tokens(expr))

    def test_complex_constants_can_be_enabled_for_random_sampling(self) -> None:
        sampled = [
            serialize_prefix_tokens(
                sample_valid_subtree(
                    "CONST",
                    sigma_small=0,
                    rng=random.Random(seed),
                    allow_complex_constants=True,
                )
            )
            for seed in range(500)
        ]

        self.assertTrue(any(tokens == ["I"] for tokens in sampled))

    def test_distributional_unary_ops_can_be_enabled_for_random_sampling(self) -> None:
        sampled = [
            serialize_prefix_tokens(
                sample_valid_subtree(
                    "UNARY_EXPR",
                    sigma_small=1,
                    rng=random.Random(seed),
                    allow_distributional_unary_ops=True,
                )
            )
            for seed in range(500)
        ]

        self.assertTrue(any(tokens[0] == "sign" for tokens in sampled))

    def test_is_obviously_zero_is_conservative_but_catches_simple_cases(self) -> None:
        zero = Const(value=Fraction(0, 1))
        one = Const(value=Fraction(1, 1))
        x = Var(name="x")

        self.assertTrue(is_obviously_zero(zero))
        self.assertFalse(is_obviously_zero(one))
        self.assertTrue(is_obviously_zero(BinaryOp(op="mul", left=zero, right=x)))
        self.assertTrue(is_obviously_zero(BinaryOp(op="div", left=zero, right=x)))
        self.assertTrue(
            is_obviously_zero(
                BinaryOp(op="pow", left=zero, right=Const(value=Fraction(3, 1)))
            )
        )
        self.assertTrue(
            is_obviously_zero(
                BinaryOp(
                    op="add",
                    left=Const(value=Fraction(1, 1)),
                    right=Const(value=Fraction(-1, 1)),
                )
            )
        )
        self.assertFalse(is_obviously_zero(BinaryOp(op="add", left=x, right=zero)))

    def test_sampled_div_denominators_are_not_obviously_zero(self) -> None:
        for seed in range(150):
            expr = sample_valid_subtree("DIV_EXPR", sigma_small=3, rng=random.Random(seed))
            for denominator in _div_denominators(expr):
                self.assertFalse(
                    is_obviously_zero(denominator),
                    msg=f"seed={seed} denominator={serialize_prefix_tokens(denominator)}",
                )

    def test_mutate_once_avoids_obvious_division_by_zero(self) -> None:
        expr = parse_prefix_string("div x INT+ 1")
        for seed in range(150):
            mutation = mutate_once(expr, sigma_small=1, rng=random.Random(seed))
            if mutation is None:
                continue
            for denominator in _div_denominators(mutation.mutated_expr):
                self.assertFalse(
                    is_obviously_zero(denominator),
                    msg=f"seed={seed} mutated={serialize_prefix_tokens(mutation.mutated_expr)}",
                )

    def test_default_generated_examples_avoid_sign_and_complex_constant_failures(self) -> None:
        rng = random.Random(123)
        for _ in range(12):
            example = generate_training_example(
                parse_prefix_string("pow x INT+ 2"),
                parse_prefix_string("div pow x INT+ 3 INT+ 3"),
                rng=rng,
                rho=1.0,
                max_random_size=3,
                residual_mode="both",
                max_attempts=64,
            )
            self.assertNotEqual(example.observation.status, "derivative_failed")
            self.assertNotIn("I", example.input_tokens)


def _div_denominators(expr):
    if isinstance(expr, BinaryOp):
        if expr.op == "div":
            yield expr.right
        yield from _div_denominators(expr.left)
        yield from _div_denominators(expr.right)
    else:
        for child in expr.children():
            yield from _div_denominators(child)


if __name__ == "__main__":
    unittest.main()
