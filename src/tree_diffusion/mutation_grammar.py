from __future__ import annotations

from dataclasses import dataclass
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
BINARY_OPERATORS = tuple(
    token for token, spec in OPERATOR_SPECS.items() if spec.arity_kind == "binary"
)
NARY_OPERATORS = tuple(
    token for token, spec in OPERATOR_SPECS.items() if spec.arity_kind == "nary"
)
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

LEAF_SHAPE = "leaf"
UNARY_SHAPE = "unary"
BINARY_SHAPE = "binary"
NARY_SHAPE = "nary"
NUMERIC_CONST_LEAF = "numeric_const"
NAMED_CONST_LEAF = "named_const"
VAR_LEAF = "var"


@dataclass(frozen=True)
class LocalReplacementSpec:
    shape: str
    op: str | None = None
    leaf_kind: str | None = None
    child_count: int | None = None


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


def node_shape(node: Expr) -> str:
    if isinstance(node, (Const, Var)):
        return LEAF_SHAPE
    if isinstance(node, UnaryOp):
        return UNARY_SHAPE
    if isinstance(node, BinaryOp):
        return BINARY_SHAPE
    if isinstance(node, NaryOp):
        return NARY_SHAPE
    raise TypeError(f"Unsupported expression type: {type(node).__name__}")


def node_arity(node: Expr) -> int:
    if isinstance(node, (Const, Var)):
        return 0
    if isinstance(node, UnaryOp):
        return 1
    if isinstance(node, BinaryOp):
        return 2
    if isinstance(node, NaryOp):
        return len(node.operands)
    raise TypeError(f"Unsupported expression type: {type(node).__name__}")


def local_replacement_candidates(node: Expr) -> tuple[LocalReplacementSpec, ...]:
    if isinstance(node, (Const, Var)):
        candidates = (
            LocalReplacementSpec(shape=LEAF_SHAPE, leaf_kind=NUMERIC_CONST_LEAF, child_count=0),
            LocalReplacementSpec(shape=LEAF_SHAPE, leaf_kind=NAMED_CONST_LEAF, child_count=0),
            LocalReplacementSpec(shape=LEAF_SHAPE, leaf_kind=VAR_LEAF, child_count=0),
        )
        return tuple(spec for spec in candidates if can_locally_replace(node, spec))

    if isinstance(node, UnaryOp):
        return _operator_replacement_candidates(node, shape=UNARY_SHAPE, operators=UNARY_OPERATORS)

    if isinstance(node, BinaryOp):
        return _operator_replacement_candidates(node, shape=BINARY_SHAPE, operators=BINARY_OPERATORS)

    if isinstance(node, NaryOp):
        return _operator_replacement_candidates(node, shape=NARY_SHAPE, operators=NARY_OPERATORS)

    raise TypeError(f"Unsupported expression type: {type(node).__name__}")


def has_local_replacement(node: Expr) -> bool:
    if isinstance(node, Const):
        return True
    if isinstance(node, Var):
        return True
    if isinstance(node, UnaryOp):
        return len(UNARY_OPERATORS) > 1
    if isinstance(node, BinaryOp):
        return len(BINARY_OPERATORS) > 1
    if isinstance(node, NaryOp):
        return len(NARY_OPERATORS) > 1
    raise TypeError(f"Unsupported expression type: {type(node).__name__}")


def can_locally_replace(node: Expr, candidate: Expr | LocalReplacementSpec) -> bool:
    if isinstance(candidate, LocalReplacementSpec):
        return _can_locally_replace_spec(node, candidate)

    if node_shape(candidate) != node_shape(node):
        return False
    if node_arity(candidate) != node_arity(node):
        return False

    if isinstance(node, (Const, Var)) and isinstance(candidate, (Const, Var)):
        if isinstance(candidate, Var) and candidate.name != "x":
            return False
        return candidate != node

    if isinstance(node, UnaryOp) and isinstance(candidate, UnaryOp):
        return candidate.op in UNARY_OPERATORS and candidate.operand == node.operand and candidate != node

    if isinstance(node, BinaryOp) and isinstance(candidate, BinaryOp):
        return (
            candidate.op in BINARY_OPERATORS
            and candidate.left == node.left
            and candidate.right == node.right
            and candidate != node
        )

    if isinstance(node, NaryOp) and isinstance(candidate, NaryOp):
        return (
            candidate.op in NARY_OPERATORS
            and len(candidate.operands) == len(node.operands)
            and candidate.operands == node.operands
            and candidate != node
        )

    return False


def can_sampled_subtree_replace(node: Expr, candidate_subtree: Expr) -> bool:
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


def can_replace(node: Expr, candidate_subtree: Expr) -> bool:
    return can_locally_replace(node, candidate_subtree)


def subtree_size(node: Expr) -> int:
    if isinstance(node, (Const, Var)):
        return 0
    return 1 + sum(subtree_size(child) for child in node.children())


def _can_locally_replace_spec(node: Expr, spec: LocalReplacementSpec) -> bool:
    if spec.shape != node_shape(node):
        return False
    if spec.child_count is not None and spec.child_count != node_arity(node):
        return False

    if isinstance(node, (Const, Var)):
        if spec.leaf_kind == NUMERIC_CONST_LEAF:
            return True
        if spec.leaf_kind == NAMED_CONST_LEAF:
            return bool(NAMED_CONSTANT_BANK)
        if spec.leaf_kind == VAR_LEAF:
            return not isinstance(node, Var)
        return False

    if isinstance(node, UnaryOp):
        return spec.op in UNARY_OPERATORS and spec.op != node.op

    if isinstance(node, BinaryOp):
        return spec.op in BINARY_OPERATORS and spec.op != node.op

    if isinstance(node, NaryOp):
        return spec.op in NARY_OPERATORS and spec.op != node.op

    return False


def _operator_replacement_candidates(
    node: UnaryOp | BinaryOp | NaryOp,
    *,
    shape: str,
    operators: tuple[str, ...],
) -> tuple[LocalReplacementSpec, ...]:
    child_count = node_arity(node)
    return tuple(
        spec
        for spec in (LocalReplacementSpec(shape=shape, op=op, child_count=child_count) for op in operators)
        if can_locally_replace(node, spec)
    )
