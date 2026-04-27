from __future__ import annotations

from fractions import Fraction

from src.mathlang.ast import BinaryOp, Const, Expr, UnaryOp, Var
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
        op = normalize_token(expr.op)
        left = _canonicalize_node(expr.left)
        right = _canonicalize_node(expr.right)

        if op in {"add", "mul"}:
            terms = _flatten_binary_associative(op, left) + _flatten_binary_associative(op, right)
            if is_commutative(op):
                terms = sorted(terms, key=_structural_key)
            return _build_right_nested(
                op,
                terms,
                token_start=expr.token_start,
                token_end=expr.token_end,
            )

        if op == "div" and isinstance(left, Const) and isinstance(right, Const) and left.is_numeric and right.is_numeric:
            if right.value != 0:
                return Const(
                    value=left.value / right.value,
                    token_start=expr.token_start,
                    token_end=expr.token_end,
                )
        return BinaryOp(
            op=op,
            left=left,
            right=right,
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
    raise TypeError(f"Unsupported expression type: {type(expr).__name__}")


def _strip_top_level_constants(expr: Expr, *, variable: str) -> Expr:
    if not isinstance(expr, BinaryOp) or expr.op != "add":
        return expr

    terms = _flatten_binary_associative("add", expr)
    variable_terms = [term for term in terms if term.contains_var(variable)]
    if not variable_terms or len(variable_terms) == len(terms):
        return expr

    if len(variable_terms) == 1:
        return variable_terms[0]

    return _build_right_nested(
        "add",
        variable_terms,
        token_start=expr.token_start,
        token_end=expr.token_end,
    )


def _flatten_binary_associative(op: str, expr: Expr) -> list[Expr]:
    if isinstance(expr, BinaryOp) and expr.op == op:
        return _flatten_binary_associative(op, expr.left) + _flatten_binary_associative(op, expr.right)
    return [expr]


def _build_right_nested(
    op: str,
    terms: list[Expr],
    *,
    token_start: int | None = None,
    token_end: int | None = None,
) -> Expr:
    if not terms:
        raise ValueError("Cannot build a binary expression without terms.")
    if len(terms) == 1:
        return terms[0]

    right: Expr = terms[-1]
    for index in range(len(terms) - 2, -1, -1):
        right = BinaryOp(
            op=op,
            left=terms[index],
            right=right,
            token_start=token_start if index == 0 else None,
            token_end=token_end if index == 0 else None,
        )
    return right
