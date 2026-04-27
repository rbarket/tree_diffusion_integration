from __future__ import annotations

from fractions import Fraction

from src.mathlang.ast import BinaryOp, Const, Expr, UnaryOp, Var


def serialize_prefix_string(expr: Expr) -> str:
    return " ".join(serialize_prefix_tokens(expr))


def serialize_prefix_tokens(expr: Expr) -> list[str]:
    if isinstance(expr, Const):
        if expr.is_named:
            return [expr.symbol]
        return _serialize_fraction(expr.value)

    if isinstance(expr, Var):
        return [expr.name]

    if isinstance(expr, UnaryOp):
        return [expr.op] + serialize_prefix_tokens(expr.operand)

    if isinstance(expr, BinaryOp):
        return [expr.op] + serialize_prefix_tokens(expr.left) + serialize_prefix_tokens(expr.right)

    raise TypeError(f"Unsupported expression type: {type(expr).__name__}")


def _serialize_fraction(value: Fraction) -> list[str]:
    if value.denominator == 1:
        return _serialize_signed_int(value.numerator)
    return ["div"] + _serialize_signed_int(value.numerator) + _serialize_signed_int(value.denominator)


def _serialize_signed_int(value: int) -> list[str]:
    sign_token = "INT-" if value < 0 else "INT+"
    digits = list(str(abs(value)))
    return [sign_token] + digits
