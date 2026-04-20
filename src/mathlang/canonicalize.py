from __future__ import annotations

from fractions import Fraction

from src.mathlang.ast import BinaryOp, Const, Expr, NaryOp, UnaryOp, Var
from src.mathlang.grammar import is_commutative, normalize_token


def canonicalize(expr: Expr, *, variable: str = "x") -> Expr:
    normalized = _canonicalize_node(expr)
    return _strip_top_level_constants(normalized, variable=variable)


def _canonicalize_node(expr: Expr) -> Expr:
    if isinstance(expr, Const):
        if expr.is_named:
            return Const(
                symbol=normalize_token(expr.symbol),
                token_start=expr.token_start,
                token_end=expr.token_end,
            )
        return Const(
            value=Fraction(expr.value),
            token_start=expr.token_start,
            token_end=expr.token_end,
        )

    if isinstance(expr, Var):
        return expr

    if isinstance(expr, UnaryOp):
        return UnaryOp(
            op=normalize_token(expr.op),
            operand=_canonicalize_node(expr.operand),
            token_start=expr.token_start,
            token_end=expr.token_end,
        )

    if isinstance(expr, BinaryOp):
        left = _canonicalize_node(expr.left)
        right = _canonicalize_node(expr.right)
        if expr.op == "div" and isinstance(left, Const) and isinstance(right, Const) and left.is_numeric and right.is_numeric:
            if right.value != 0:
                return Const(
                    value=left.value / right.value,
                    token_start=expr.token_start,
                    token_end=expr.token_end,
                )
        return BinaryOp(
            op=normalize_token(expr.op),
            left=left,
            right=right,
            token_start=expr.token_start,
            token_end=expr.token_end,
        )

    if isinstance(expr, NaryOp):
        operands: list[Expr] = []
        for operand in expr.operands:
            child = _canonicalize_node(operand)
            if isinstance(child, NaryOp) and child.op == expr.op:
                operands.extend(child.operands)
            else:
                operands.append(child)

        if is_commutative(expr.op):
            operands = sorted(operands, key=_structural_key)

        if len(operands) == 1:
            return operands[0]

        return NaryOp(
            op=normalize_token(expr.op),
            operands=tuple(operands),
            token_start=expr.token_start,
            token_end=expr.token_end,
        )

    raise TypeError(f"Unsupported expression type: {type(expr).__name__}")


def _structural_key(expr: Expr) -> tuple:
    if isinstance(expr, Const):
        if expr.is_named:
            return (1, expr.symbol)
        return (0, expr.value.numerator, expr.value.denominator)
    if isinstance(expr, Var):
        return (2, expr.name)
    if isinstance(expr, UnaryOp):
        return (3, expr.op, _structural_key(expr.operand))
    if isinstance(expr, BinaryOp):
        return (4, expr.op, _structural_key(expr.left), _structural_key(expr.right))
    if isinstance(expr, NaryOp):
        return (5, expr.op, tuple(_structural_key(operand) for operand in expr.operands))
    raise TypeError(f"Unsupported expression type: {type(expr).__name__}")


def _strip_top_level_constants(expr: Expr, *, variable: str) -> Expr:
    if not isinstance(expr, NaryOp) or expr.op != "add":
        return expr

    variable_terms = [operand for operand in expr.operands if operand.contains_var(variable)]
    if not variable_terms or len(variable_terms) == len(expr.operands):
        return expr

    if len(variable_terms) == 1:
        return variable_terms[0]

    return NaryOp(
        op="add",
        operands=tuple(variable_terms),
        token_start=expr.token_start,
        token_end=expr.token_end,
    )
