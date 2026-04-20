from __future__ import annotations

from fractions import Fraction

from src.mathlang.ast import BinaryOp, Const, Expr, NaryOp, UnaryOp, Var
from src.mathlang.grammar import NAMED_CONSTANT_TOKENS, OPERATOR_SPECS


EXPR_FAMILY = "EXPR"
CONST_FAMILY = "CONST"
VAR_FAMILY = "VAR"
UNARY_EXPR_FAMILY = "UNARY_EXPR"
ADD_EXPR_FAMILY = "ADD_EXPR"
MUL_EXPR_FAMILY = "MUL_EXPR"
POW_EXPR_FAMILY = "POW_EXPR"
DIV_EXPR_FAMILY = "DIV_EXPR"

UNARY_OPERATORS = tuple(
    token for token, spec in OPERATOR_SPECS.items() if spec.arity_kind == "unary"
)
COMMUTATIVE_FAMILIES = frozenset({ADD_EXPR_FAMILY, MUL_EXPR_FAMILY})
NUMERIC_CONSTANT_BANK = (
    Fraction(-1, 1),
    Fraction(0, 1),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(1, 1),
    Fraction(2, 1),
    Fraction(3, 1),
)
NAMED_CONSTANT_BANK = tuple(sorted(NAMED_CONSTANT_TOKENS))


def production_family(node: Expr) -> str:
    if isinstance(node, Const):
        return CONST_FAMILY
    if isinstance(node, Var):
        return VAR_FAMILY
    if isinstance(node, UnaryOp):
        return UNARY_EXPR_FAMILY
    if isinstance(node, NaryOp):
        if node.op == "add":
            return ADD_EXPR_FAMILY
        if node.op == "mul":
            return MUL_EXPR_FAMILY
    if isinstance(node, BinaryOp):
        if node.op == "pow":
            return POW_EXPR_FAMILY
        if node.op == "div":
            return DIV_EXPR_FAMILY
    raise TypeError(f"Unsupported expression type: {type(node).__name__}")


def compatible_replacement_families(node: Expr) -> tuple[str, ...]:
    family = production_family(node)
    if family == VAR_FAMILY:
        return ()
    return (family,)


def can_replace(node: Expr, candidate_subtree: Expr) -> bool:
    if production_family(candidate_subtree) not in compatible_replacement_families(node):
        return False

    if isinstance(node, Const):
        if not isinstance(candidate_subtree, Const):
            return False
        if node.is_numeric:
            return candidate_subtree.is_numeric
        return candidate_subtree.is_named

    if isinstance(node, Var):
        return False

    if isinstance(node, UnaryOp) and isinstance(candidate_subtree, UnaryOp):
        return candidate_subtree.op in UNARY_OPERATORS

    if isinstance(node, BinaryOp) and isinstance(candidate_subtree, BinaryOp):
        return node.op == candidate_subtree.op

    if isinstance(node, NaryOp) and isinstance(candidate_subtree, NaryOp):
        return node.op == candidate_subtree.op

    return True


def subtree_size(node: Expr) -> int:
    if isinstance(node, (Const, Var)):
        return 0
    return 1 + sum(subtree_size(child) for child in node.children())
