from __future__ import annotations

from fractions import Fraction

from src.mathlang.ast import BinaryOp, Const, Expr, NaryOp, UnaryOp, Var


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

    if isinstance(expr, NaryOp):
        return _serialize_nary(expr.op, expr.operands)

    raise TypeError(f"Unsupported expression type: {type(expr).__name__}")


def _serialize_nary(op: str, operands: tuple[Expr, ...]) -> list[str]:
    if len(operands) < 2:
        raise ValueError(f"Cannot serialize n-ary operator '{op}' with fewer than two operands.")

    first = operands[0]
    if len(operands) == 2:
        return [op] + serialize_prefix_tokens(first) + serialize_prefix_tokens(operands[1])

    return [op] + serialize_prefix_tokens(first) + _serialize_nary(op, operands[1:])


def _serialize_fraction(value: Fraction) -> list[str]:
    if value.denominator == 1:
        return _serialize_signed_int(value.numerator)
    return ["div"] + _serialize_signed_int(value.numerator) + _serialize_signed_int(value.denominator)


def _serialize_signed_int(value: int) -> list[str]:
    sign_token = "INT-" if value < 0 else "INT+"
    digits = list(str(abs(value)))
    return [sign_token] + digits
