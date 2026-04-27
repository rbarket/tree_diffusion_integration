from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction

from src.mathlang.ast import BinaryOp, Const, Expr, UnaryOp, Var
from src.mathlang.canonicalize import canonicalize
from src.tree_diffusion.mutation_grammar import (
    ADD_EXPR_FAMILY,
    CONST_FAMILY,
    DIV_EXPR_FAMILY,
    EXPR_FAMILY,
    NAMED_CONST_LEAF,
    LocalReplacementSpec,
    MUL_EXPR_FAMILY,
    NAMED_CONSTANT_BANK,
    NUMERIC_CONST_LEAF,
    NUMERIC_CONSTANT_BANK,
    POW_EXPR_FAMILY,
    UNARY_EXPR_FAMILY,
    UNARY_OPERATORS,
    VAR_LEAF,
    can_locally_replace,
    can_sampled_subtree_replace,
    compatible_replacement_families,
    has_local_replacement,
    local_replacement_candidates,
)
from src.tree_diffusion.positions import NodePosition, PositionIndex, index_tree_positions

LOCAL_CONST_EDIT = "local_const_edit"
LOCAL_SAME_ARITY_REPLACEMENT = "local_same_arity_replacement"
SAMPLED_SMALL_SUBTREE_REPLACEMENT = "sampled_small_subtree_replacement"


@dataclass(frozen=True)
class MutationResult:
    mutated_expr: Expr
    selected_node_id: int
    selected_family: str
    original_subtree: Expr
    replacement_subtree: Expr
    selected_token_start: int
    selected_token_end: int


def collect_candidate_nodes(expr: Expr, sigma_small: int) -> dict[str, list[NodePosition]]:
    canonical_expr = canonicalize(expr)
    index = index_tree_positions(canonical_expr, sigma_small=sigma_small)
    return _collect_candidate_nodes_from_index(index)


def sample_valid_subtree(family: str, sigma_small: int, rng: random.Random) -> Expr:
    if sigma_small < 0:
        raise ValueError("sigma_small must be non-negative.")

    if family == EXPR_FAMILY:
        return _sample_expr(sigma_small, rng)
    if family == CONST_FAMILY:
        return _sample_any_const(rng)
    if family == "VAR":
        return Var(name="x")
    if family == UNARY_EXPR_FAMILY:
        if sigma_small < 1:
            raise ValueError("UNARY_EXPR requires sigma_small >= 1.")
        return UnaryOp(op=rng.choice(UNARY_OPERATORS), operand=_sample_expr(sigma_small - 1, rng))
    if family == ADD_EXPR_FAMILY:
        return _sample_binary_expr("add", sigma_small, rng)
    if family == MUL_EXPR_FAMILY:
        return _sample_binary_expr("mul", sigma_small, rng)
    if family == POW_EXPR_FAMILY:
        if sigma_small < 1:
            raise ValueError("POW_EXPR requires sigma_small >= 1.")
        remaining = sigma_small - 1
        left_budget, right_budget = _split_budget(remaining, 2, rng)
        right = _sample_pow_exponent(right_budget, rng)
        return BinaryOp(
            op="pow",
            left=_sample_expr(left_budget, rng),
            right=right,
        )
    if family == DIV_EXPR_FAMILY:
        return _sample_div_expr(sigma_small, rng)

    raise ValueError(f"Unsupported family: {family}")


def mutate_once(
    expr: Expr,
    sigma_small: int,
    rng: random.Random,
    max_attempts: int = 64,
) -> MutationResult | None:
    canonical_expr = canonicalize(expr)
    index = index_tree_positions(canonical_expr, sigma_small=sigma_small)
    candidates_by_family = _collect_candidate_nodes_from_index(index)

    if not candidates_by_family:
        return None

    families = sorted(candidates_by_family)
    for _ in range(max_attempts):
        family = rng.choice(families)
        selected_position = rng.choice(candidates_by_family[family])
        original_subtree = index.node_id_to_node[selected_position.node_id]
        mutation_kind = _sample_mutation_kind(original_subtree, rng)
        if mutation_kind is None:
            continue

        if mutation_kind == LOCAL_CONST_EDIT:
            replacement = sample_const_replacement(original_subtree, rng)
            if replacement == original_subtree:
                continue
            result = _apply_replacement(
                canonical_expr,
                selected_position,
                original_subtree,
                replacement,
            )
            if result is not None:
                return result
            continue

        if mutation_kind == LOCAL_SAME_ARITY_REPLACEMENT:
            result = _local_replace_selected_node(
                canonical_expr,
                selected_position,
                original_subtree,
                rng,
            )
            if result is not None:
                return result
            continue

        replacement = sample_valid_subtree(family, sigma_small, rng)
        if not can_sampled_subtree_replace(original_subtree, replacement):
            continue
        if replacement == original_subtree:
            continue
        result = _apply_replacement(
            canonical_expr,
            selected_position,
            original_subtree,
            replacement,
        )
        if result is not None:
            return result

    return None


def local_replace_once(
    expr: Expr,
    selected_node_id: int,
    rng: random.Random,
    max_attempts: int = 32,
) -> MutationResult | None:
    canonical_expr = canonicalize(expr)
    index = index_tree_positions(canonical_expr)
    if selected_node_id < 0 or selected_node_id >= len(index.positions):
        raise KeyError(f"Unknown node_id: {selected_node_id}")
    selected_position = index.positions[selected_node_id]
    original_subtree = index.node_id_to_node[selected_position.node_id]
    return _local_replace_selected_node(
        canonical_expr,
        selected_position,
        original_subtree,
        rng,
        max_attempts=max_attempts,
    )


def sample_const_replacement(node: Expr, rng: random.Random) -> Const:
    if not isinstance(node, Const):
        raise TypeError("sample_const_replacement expects a Const node.")

    if node.is_named:
        choices = [symbol for symbol in NAMED_CONSTANT_BANK if symbol != node.symbol]
        if not choices:
            return node
        return Const(symbol=rng.choice(choices))

    assert node.value is not None
    value = node.value
    candidates: set[Fraction] = set(NUMERIC_CONSTANT_BANK)
    candidates.add(-value)
    candidates.add(value + 1)
    candidates.add(value - 1)
    if value.denominator > 1:
        candidates.add(Fraction(value.numerator + 1, value.denominator))
        candidates.add(Fraction(value.numerator - 1, value.denominator))
        if value.denominator > 2:
            candidates.add(Fraction(value.numerator, value.denominator - 1))
        candidates.add(Fraction(value.numerator, value.denominator + 1))

    ordered = [candidate for candidate in sorted(candidates) if candidate != value]
    if not ordered:
        return node
    return Const(value=rng.choice(ordered))


def replace_subtree_by_node_id(expr: Expr, node_id: int, replacement: Expr) -> Expr:
    replaced, _, found = _replace_subtree(expr, target_id=node_id, replacement=replacement, next_id=0)
    if not found:
        raise KeyError(f"Unknown node_id: {node_id}")
    return replaced


def _collect_candidate_nodes_from_index(index: PositionIndex) -> dict[str, list[NodePosition]]:
    candidates_by_family: dict[str, list[NodePosition]] = {}
    for position in index.positions:
        if not position.is_mutable:
            continue
        candidates_by_family.setdefault(position.production_family, []).append(position)
    return candidates_by_family


def _sample_expr(sigma_small: int, rng: random.Random) -> Expr:
    if sigma_small <= 0:
        return _sample_leaf(rng)

    families = [
        CONST_FAMILY,
        "VAR",
        UNARY_EXPR_FAMILY,
        ADD_EXPR_FAMILY,
        MUL_EXPR_FAMILY,
        POW_EXPR_FAMILY,
        DIV_EXPR_FAMILY,
    ]
    weights = [4, 1, 2, 1, 1, 1, 1]
    family = rng.choices(families, weights=weights, k=1)[0]
    if family == CONST_FAMILY:
        return _sample_any_const(rng)
    if family == "VAR":
        return Var(name="x")
    return sample_valid_subtree(family, sigma_small, rng)


def _sample_leaf(rng: random.Random) -> Expr:
    if rng.random() < 0.75:
        return _sample_any_const(rng)
    return Var(name="x")


def _sample_any_const(rng: random.Random) -> Const:
    if rng.random() < 0.9:
        return Const(value=rng.choice(NUMERIC_CONSTANT_BANK))
    return Const(symbol=rng.choice(NAMED_CONSTANT_BANK))


def _sample_binary_expr(op: str, sigma_small: int, rng: random.Random) -> BinaryOp:
    if sigma_small < 1:
        raise ValueError(f"{op} requires sigma_small >= 1.")
    remaining = sigma_small - 1
    left_budget, right_budget = _split_budget(remaining, 2, rng)
    return BinaryOp(
        op=op,
        left=_sample_expr(left_budget, rng),
        right=_sample_expr(right_budget, rng),
    )


def _sample_pow_exponent(sigma_small: int, rng: random.Random) -> Expr:
    if sigma_small <= 0 or rng.random() < 0.7:
        return _sample_any_const(rng)
    return _sample_expr(sigma_small, rng)


def _sample_div_expr(sigma_small: int, rng: random.Random) -> BinaryOp:
    if sigma_small < 1:
        raise ValueError("DIV_EXPR requires sigma_small >= 1.")

    remaining = sigma_small - 1
    for _ in range(16):
        left_budget, right_budget = _split_budget(remaining, 2, rng)
        left = _sample_expr(left_budget, rng)
        right = _coerce_nonzero_denominator(_sample_expr(right_budget, rng), rng)
        if not (_is_numeric_const(left) and _is_numeric_const(right)):
            return BinaryOp(op="div", left=left, right=right)

    return BinaryOp(op="div", left=Var(name="x"), right=Const(value=Fraction(2, 1)))


def _coerce_nonzero_denominator(expr: Expr, rng: random.Random) -> Expr:
    if _is_numeric_const(expr) and isinstance(expr, Const) and expr.value == 0:
        return Const(value=rng.choice([Fraction(1, 1), Fraction(2, 1), Fraction(3, 1)]))
    return expr


def _is_numeric_const(expr: Expr) -> bool:
    return isinstance(expr, Const) and expr.is_numeric


def _split_budget(total: int, parts: int, rng: random.Random) -> tuple[int, ...]:
    budgets = [0] * parts
    for _ in range(total):
        budgets[rng.randrange(parts)] += 1
    return tuple(budgets)


def _node_count(node: Expr) -> int:
    return 1 + sum(_node_count(child) for child in node.children())


def _sample_mutation_kind(node: Expr, rng: random.Random) -> str | None:
    kinds: list[str] = []
    if isinstance(node, Const):
        kinds.append(LOCAL_CONST_EDIT)
        if has_local_replacement(node):
            kinds.append(LOCAL_SAME_ARITY_REPLACEMENT)
    elif isinstance(node, Var):
        if has_local_replacement(node):
            kinds.append(LOCAL_SAME_ARITY_REPLACEMENT)
    elif isinstance(node, (UnaryOp, BinaryOp)):
        if has_local_replacement(node):
            kinds.append(LOCAL_SAME_ARITY_REPLACEMENT)
        if compatible_replacement_families(node):
            kinds.append(SAMPLED_SMALL_SUBTREE_REPLACEMENT)

    if not kinds:
        return None
    return rng.choice(kinds)


def _local_replace_selected_node(
    canonical_expr: Expr,
    selected_position: NodePosition,
    original_subtree: Expr,
    rng: random.Random,
    max_attempts: int = 32,
) -> MutationResult | None:
    candidates = local_replacement_candidates(original_subtree)
    if not candidates:
        return None

    for _ in range(max_attempts):
        spec = rng.choice(candidates)
        replacement = _materialize_local_replacement(original_subtree, spec, rng)
        if not can_locally_replace(original_subtree, replacement):
            continue
        result = _apply_replacement(
            canonical_expr,
            selected_position,
            original_subtree,
            replacement,
        )
        if result is not None:
            return result

    return None


def _materialize_local_replacement(
    node: Expr,
    spec: LocalReplacementSpec,
    rng: random.Random,
) -> Expr:
    if isinstance(node, (Const, Var)):
        if spec.leaf_kind == NUMERIC_CONST_LEAF:
            if isinstance(node, Const) and node.is_numeric:
                return sample_const_replacement(node, rng)
            return Const(value=rng.choice(NUMERIC_CONSTANT_BANK))
        if spec.leaf_kind == NAMED_CONST_LEAF:
            if isinstance(node, Const) and node.is_named:
                return sample_const_replacement(node, rng)
            return Const(symbol=rng.choice(NAMED_CONSTANT_BANK))
        if spec.leaf_kind == VAR_LEAF:
            return Var(name="x")
        raise ValueError(f"Unsupported leaf replacement kind: {spec.leaf_kind}")

    if isinstance(node, UnaryOp):
        if spec.op is None:
            raise ValueError("Unary replacement spec must include an operator.")
        return UnaryOp(op=spec.op, operand=node.operand)

    if isinstance(node, BinaryOp):
        if spec.op is None:
            raise ValueError("Binary replacement spec must include an operator.")
        return BinaryOp(op=spec.op, left=node.left, right=node.right)

    raise TypeError(f"Unsupported expression type: {type(node).__name__}")


def _apply_replacement(
    canonical_expr: Expr,
    selected_position: NodePosition,
    original_subtree: Expr,
    replacement: Expr,
) -> MutationResult | None:
    mutated_expr = replace_subtree_by_node_id(canonical_expr, selected_position.node_id, replacement)
    canonical_mutated = canonicalize(mutated_expr)
    if canonical_mutated == canonical_expr:
        return None

    return MutationResult(
        mutated_expr=canonical_mutated,
        selected_node_id=selected_position.node_id,
        selected_family=selected_position.production_family,
        original_subtree=original_subtree,
        replacement_subtree=replacement,
        selected_token_start=selected_position.token_start,
        selected_token_end=selected_position.token_end,
    )


def _replace_subtree(
    node: Expr,
    *,
    target_id: int,
    replacement: Expr,
    next_id: int,
) -> tuple[Expr, int, bool]:
    current_id = next_id
    next_id += 1

    if current_id == target_id:
        return replacement, current_id + _node_count(node), True

    if isinstance(node, (Const, Var)):
        return node, next_id, False

    if isinstance(node, UnaryOp):
        operand, next_id, found = _replace_subtree(
            node.operand,
            target_id=target_id,
            replacement=replacement,
            next_id=next_id,
        )
        return UnaryOp(op=node.op, operand=operand), next_id, found

    if isinstance(node, BinaryOp):
        left, next_id, found_left = _replace_subtree(
            node.left,
            target_id=target_id,
            replacement=replacement,
            next_id=next_id,
        )
        right, next_id, found_right = _replace_subtree(
            node.right,
            target_id=target_id,
            replacement=replacement,
            next_id=next_id,
        )
        return BinaryOp(op=node.op, left=left, right=right), next_id, found_left or found_right

    raise TypeError(f"Unsupported expression type: {type(node).__name__}")
