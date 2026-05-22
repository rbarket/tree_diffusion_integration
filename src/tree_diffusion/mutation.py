from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Collection

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
    VAR_FAMILY,
    VAR_LEAF,
    can_locally_replace,
    can_sampled_subtree_replace,
    has_local_replacement,
    local_replacement_candidates,
)
from src.tree_diffusion.positions import NodePosition, PositionIndex, index_tree_positions

LOCAL_CONST_EDIT = "local_const_edit"
LOCAL_SAME_ARITY_REPLACEMENT = "local_same_arity_replacement"
SAMPLED_SMALL_SUBTREE_REPLACEMENT = "sampled_small_subtree_replacement"
_COMPLEX_CONSTANT_TOKEN = "I"
_DISTRIBUTIONAL_UNARY_TOKENS = frozenset({"sign"})


@dataclass(frozen=True)
class MutationResult:
    mutated_expr: Expr
    selected_node_id: int
    selected_family: str
    mutation_kind: str
    original_subtree: Expr
    replacement_subtree: Expr
    selected_token_start: int
    selected_token_end: int


@dataclass(frozen=True)
class MutationSamplingOptions:
    allow_complex_constants: bool = False
    allow_distributional_unary_ops: bool = False
    excluded_random_tokens: tuple[str, ...] = ()


def collect_candidate_nodes(expr: Expr, sigma_small: int) -> dict[str, list[NodePosition]]:
    canonical_expr = canonicalize(expr)
    index = index_tree_positions(canonical_expr, sigma_small=sigma_small)
    return _collect_candidate_nodes_from_index(index)


def sample_valid_subtree(
    family: str,
    sigma_small: int,
    rng: random.Random,
    *,
    allow_complex_constants: bool = False,
    allow_distributional_unary_ops: bool = False,
    excluded_random_tokens: Collection[str] | None = None,
) -> Expr:
    if sigma_small < 0:
        raise ValueError("sigma_small must be non-negative.")
    options = _mutation_sampling_options(
        allow_complex_constants=allow_complex_constants,
        allow_distributional_unary_ops=allow_distributional_unary_ops,
        excluded_random_tokens=excluded_random_tokens,
    )

    if family == EXPR_FAMILY:
        return _sample_expr(sigma_small, rng, options=options)
    if family == CONST_FAMILY:
        return _sample_any_const(rng, options=options)
    if family == VAR_FAMILY:
        return Var(name="x")
    if family == UNARY_EXPR_FAMILY:
        if sigma_small < 1:
            raise ValueError("UNARY_EXPR requires sigma_small >= 1.")
        return UnaryOp(
            op=rng.choice(_allowed_unary_operators(options)),
            operand=_sample_expr(sigma_small - 1, rng, options=options),
        )
    if family == ADD_EXPR_FAMILY:
        return _sample_binary_expr("add", sigma_small, rng, options=options)
    if family == MUL_EXPR_FAMILY:
        return _sample_binary_expr("mul", sigma_small, rng, options=options)
    if family == POW_EXPR_FAMILY:
        if sigma_small < 1:
            raise ValueError("POW_EXPR requires sigma_small >= 1.")
        remaining = sigma_small - 1
        left_budget, right_budget = _split_budget(remaining, 2, rng)
        right = _sample_pow_exponent(right_budget, rng, options=options)
        return BinaryOp(
            op="pow",
            left=_sample_expr(left_budget, rng, options=options),
            right=right,
        )
    if family == DIV_EXPR_FAMILY:
        return _sample_div_expr(sigma_small, rng, options=options)

    raise ValueError(f"Unsupported family: {family}")


def sample_random_expr(
    *,
    rng: random.Random | None = None,
    max_size: int = 4,
    allow_complex_constants: bool = False,
    allow_distributional_unary_ops: bool = False,
    excluded_random_tokens: Collection[str] | None = None,
) -> Expr:
    if max_size < 0:
        raise ValueError("max_size must be non-negative.")

    rng = rng or random.Random()
    options = _mutation_sampling_options(
        allow_complex_constants=allow_complex_constants,
        allow_distributional_unary_ops=allow_distributional_unary_ops,
        excluded_random_tokens=excluded_random_tokens,
    )
    return canonicalize(
        sample_valid_subtree(
            EXPR_FAMILY,
            max_size,
            rng,
            allow_complex_constants=options.allow_complex_constants,
            allow_distributional_unary_ops=options.allow_distributional_unary_ops,
            excluded_random_tokens=options.excluded_random_tokens,
        )
    )


def mutate_once(
    expr: Expr,
    sigma_small: int,
    rng: random.Random,
    max_attempts: int = 64,
    *,
    allow_complex_constants: bool = False,
    allow_distributional_unary_ops: bool = False,
    excluded_random_tokens: Collection[str] | None = None,
) -> MutationResult | None:
    options = _mutation_sampling_options(
        allow_complex_constants=allow_complex_constants,
        allow_distributional_unary_ops=allow_distributional_unary_ops,
        excluded_random_tokens=excluded_random_tokens,
    )
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
            replacement = sample_const_replacement(
                original_subtree,
                rng,
                allow_complex_constants=options.allow_complex_constants,
                excluded_random_tokens=options.excluded_random_tokens,
            )
            if replacement == original_subtree:
                continue
            result = _apply_replacement(
                canonical_expr,
                selected_position,
                original_subtree,
                replacement,
                mutation_kind=LOCAL_CONST_EDIT,
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
                options=options,
            )
            if result is not None:
                return result
            continue

        replacement = sample_valid_subtree(
            EXPR_FAMILY,
            sigma_small,
            rng,
            allow_complex_constants=options.allow_complex_constants,
            allow_distributional_unary_ops=options.allow_distributional_unary_ops,
            excluded_random_tokens=options.excluded_random_tokens,
        )
        if not can_sampled_subtree_replace(original_subtree, replacement):
            continue
        if replacement == original_subtree:
            continue
        result = _apply_replacement(
            canonical_expr,
            selected_position,
            original_subtree,
            replacement,
            mutation_kind=SAMPLED_SMALL_SUBTREE_REPLACEMENT,
        )
        if result is not None:
            return result

    return None


def local_replace_once(
    expr: Expr,
    selected_node_id: int,
    rng: random.Random,
    max_attempts: int = 32,
    *,
    allow_complex_constants: bool = False,
    allow_distributional_unary_ops: bool = False,
    excluded_random_tokens: Collection[str] | None = None,
) -> MutationResult | None:
    options = _mutation_sampling_options(
        allow_complex_constants=allow_complex_constants,
        allow_distributional_unary_ops=allow_distributional_unary_ops,
        excluded_random_tokens=excluded_random_tokens,
    )
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
        options=options,
    )


def sample_const_replacement(
    node: Expr,
    rng: random.Random,
    *,
    allow_complex_constants: bool = False,
    excluded_random_tokens: Collection[str] | None = None,
) -> Const:
    if not isinstance(node, Const):
        raise TypeError("sample_const_replacement expects a Const node.")
    options = _mutation_sampling_options(
        allow_complex_constants=allow_complex_constants,
        excluded_random_tokens=excluded_random_tokens,
    )

    if node.is_named:
        choices = [
            symbol
            for symbol in _allowed_named_constants(options)
            if symbol != node.symbol
        ]
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


def is_obviously_zero(expr: Expr) -> bool:
    constant_value = _constant_fraction_value(expr)
    if constant_value is not None:
        return constant_value == 0

    if isinstance(expr, UnaryOp) and expr.op in {"neg", "minus"}:
        return is_obviously_zero(expr.operand)

    if isinstance(expr, BinaryOp):
        if expr.op == "mul":
            return is_obviously_zero(expr.left) or is_obviously_zero(expr.right)
        if expr.op == "div":
            return is_obviously_zero(expr.left) and not is_obviously_zero(expr.right)
        if expr.op == "pow":
            return is_obviously_zero(expr.left) and _is_positive_integer_const(expr.right)

    return False


def _collect_candidate_nodes_from_index(index: PositionIndex) -> dict[str, list[NodePosition]]:
    candidates_by_family: dict[str, list[NodePosition]] = {}
    for position in index.positions:
        if not position.is_mutable:
            continue
        candidates_by_family.setdefault(position.production_family, []).append(position)
    return candidates_by_family


def _sample_expr(
    sigma_small: int,
    rng: random.Random,
    *,
    options: MutationSamplingOptions,
) -> Expr:
    if sigma_small <= 0:
        return _sample_leaf(rng, options=options)

    families = [
        CONST_FAMILY,
        VAR_FAMILY,
        UNARY_EXPR_FAMILY,
        ADD_EXPR_FAMILY,
        MUL_EXPR_FAMILY,
        POW_EXPR_FAMILY,
        DIV_EXPR_FAMILY,
    ]
    weights = [4, 1, 2, 1, 1, 1, 1]
    family = rng.choices(families, weights=weights, k=1)[0]
    if family == CONST_FAMILY:
        return _sample_any_const(rng, options=options)
    if family == VAR_FAMILY:
        return Var(name="x")
    return sample_valid_subtree(
        family,
        sigma_small,
        rng,
        allow_complex_constants=options.allow_complex_constants,
        allow_distributional_unary_ops=options.allow_distributional_unary_ops,
        excluded_random_tokens=options.excluded_random_tokens,
    )


def _sample_leaf(
    rng: random.Random,
    *,
    options: MutationSamplingOptions,
) -> Expr:
    if rng.random() < 0.75:
        return _sample_any_const(rng, options=options)
    return Var(name="x")


def _sample_any_const(
    rng: random.Random,
    *,
    options: MutationSamplingOptions,
) -> Const:
    named_constants = _allowed_named_constants(options)
    if rng.random() < 0.9 or not named_constants:
        return Const(value=rng.choice(NUMERIC_CONSTANT_BANK))
    return Const(symbol=rng.choice(named_constants))


def _sample_binary_expr(
    op: str,
    sigma_small: int,
    rng: random.Random,
    *,
    options: MutationSamplingOptions,
) -> BinaryOp:
    if sigma_small < 1:
        raise ValueError(f"{op} requires sigma_small >= 1.")
    remaining = sigma_small - 1
    left_budget, right_budget = _split_budget(remaining, 2, rng)
    return BinaryOp(
        op=op,
        left=_sample_expr(left_budget, rng, options=options),
        right=_sample_expr(right_budget, rng, options=options),
    )


def _sample_pow_exponent(
    sigma_small: int,
    rng: random.Random,
    *,
    options: MutationSamplingOptions,
) -> Expr:
    if sigma_small <= 0 or rng.random() < 0.7:
        return _sample_any_const(rng, options=options)
    return _sample_expr(sigma_small, rng, options=options)


def _sample_div_expr(
    sigma_small: int,
    rng: random.Random,
    *,
    options: MutationSamplingOptions,
) -> BinaryOp:
    if sigma_small < 1:
        raise ValueError("DIV_EXPR requires sigma_small >= 1.")

    remaining = sigma_small - 1
    for _ in range(16):
        left_budget, right_budget = _split_budget(remaining, 2, rng)
        left = _sample_expr(left_budget, rng, options=options)
        right = _coerce_nonzero_denominator(
            _sample_expr(right_budget, rng, options=options),
            rng,
        )
        if is_obviously_zero(right):
            continue
        if not (_is_numeric_const(left) and _is_numeric_const(right)):
            return BinaryOp(op="div", left=left, right=right)

    return BinaryOp(op="div", left=Var(name="x"), right=Const(value=Fraction(2, 1)))


def _coerce_nonzero_denominator(expr: Expr, rng: random.Random) -> Expr:
    if is_obviously_zero(expr):
        return Const(value=rng.choice([Fraction(1, 1), Fraction(2, 1), Fraction(3, 1)]))
    return expr


def _contains_obviously_zero_denominator(expr: Expr) -> bool:
    if isinstance(expr, BinaryOp):
        if expr.op == "div" and is_obviously_zero(expr.right):
            return True
        return _contains_obviously_zero_denominator(expr.left) or _contains_obviously_zero_denominator(expr.right)
    if isinstance(expr, UnaryOp):
        return _contains_obviously_zero_denominator(expr.operand)
    return False


def _constant_fraction_value(expr: Expr) -> Fraction | None:
    if isinstance(expr, Const):
        return expr.value if expr.is_numeric else None

    if isinstance(expr, UnaryOp) and expr.op in {"neg", "minus"}:
        operand_value = _constant_fraction_value(expr.operand)
        return None if operand_value is None else -operand_value

    if not isinstance(expr, BinaryOp):
        return None

    left = _constant_fraction_value(expr.left)
    right = _constant_fraction_value(expr.right)
    if left is None or right is None:
        return None

    if expr.op == "add":
        return left + right
    if expr.op == "mul":
        return left * right
    if expr.op == "div":
        if right == 0:
            return None
        return left / right
    if expr.op == "pow" and right.denominator == 1:
        exponent = right.numerator
        if exponent < 0 and left == 0:
            return None
        try:
            return left**exponent
        except ZeroDivisionError:
            return None

    return None


def _is_positive_integer_const(expr: Expr) -> bool:
    return isinstance(expr, Const) and expr.is_numeric and expr.value.denominator == 1 and expr.value > 0


def _excluded_random_token_set(excluded_random_tokens: Collection[str] | None) -> frozenset[str]:
    return frozenset(str(token) for token in (excluded_random_tokens or ()))


def _mutation_sampling_options(
    *,
    allow_complex_constants: bool,
    allow_distributional_unary_ops: bool = False,
    excluded_random_tokens: Collection[str] | None,
) -> MutationSamplingOptions:
    return MutationSamplingOptions(
        allow_complex_constants=bool(allow_complex_constants),
        allow_distributional_unary_ops=bool(allow_distributional_unary_ops),
        excluded_random_tokens=tuple(str(token) for token in (excluded_random_tokens or ())),
    )


def _excluded_tokens(options: MutationSamplingOptions) -> frozenset[str]:
    return _excluded_random_token_set(options.excluded_random_tokens)


def _allowed_named_constants(options: MutationSamplingOptions) -> tuple[str, ...]:
    excluded = _excluded_tokens(options)
    return tuple(
        symbol
        for symbol in NAMED_CONSTANT_BANK
        if symbol not in excluded and (options.allow_complex_constants or symbol != _COMPLEX_CONSTANT_TOKEN)
    )


def _allowed_unary_operators(options: MutationSamplingOptions) -> tuple[str, ...]:
    excluded = _excluded_tokens(options)
    operators = tuple(
        op
        for op in UNARY_OPERATORS
        if op not in excluded
        and (options.allow_distributional_unary_ops or op not in _DISTRIBUTIONAL_UNARY_TOKENS)
    )
    if not operators:
        raise ValueError("No unary operators are available after applying random-token exclusions.")
    return operators


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
    if isinstance(node, Var):
        if has_local_replacement(node):
            kinds.append(LOCAL_SAME_ARITY_REPLACEMENT)
    if isinstance(node, (UnaryOp, BinaryOp)):
        if has_local_replacement(node):
            kinds.append(LOCAL_SAME_ARITY_REPLACEMENT)
    if isinstance(node, (Const, Var, UnaryOp, BinaryOp)):
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
    *,
    options: MutationSamplingOptions,
) -> MutationResult | None:
    candidates = _filter_local_replacement_candidates(
        local_replacement_candidates(original_subtree),
        options=options,
    )
    if not candidates:
        return None

    for _ in range(max_attempts):
        spec = rng.choice(candidates)
        replacement = _materialize_local_replacement(
            original_subtree,
            spec,
            rng,
            options=options,
        )
        if not can_locally_replace(original_subtree, replacement):
            continue
        result = _apply_replacement(
            canonical_expr,
            selected_position,
            original_subtree,
            replacement,
            mutation_kind=LOCAL_SAME_ARITY_REPLACEMENT,
        )
        if result is not None:
            return result

    return None


def _materialize_local_replacement(
    node: Expr,
    spec: LocalReplacementSpec,
    rng: random.Random,
    *,
    options: MutationSamplingOptions,
) -> Expr:
    if isinstance(node, (Const, Var)):
        if spec.leaf_kind == NUMERIC_CONST_LEAF:
            if isinstance(node, Const) and node.is_numeric:
                return sample_const_replacement(
                    node,
                    rng,
                    allow_complex_constants=options.allow_complex_constants,
                    excluded_random_tokens=options.excluded_random_tokens,
                )
            return Const(value=rng.choice(NUMERIC_CONSTANT_BANK))
        if spec.leaf_kind == NAMED_CONST_LEAF:
            if isinstance(node, Const) and node.is_named:
                return sample_const_replacement(
                    node,
                    rng,
                    allow_complex_constants=options.allow_complex_constants,
                    excluded_random_tokens=options.excluded_random_tokens,
                )
            choices = _allowed_named_constants(options)
            if not choices:
                return node
            return Const(symbol=rng.choice(choices))
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


def _filter_local_replacement_candidates(
    candidates: tuple[LocalReplacementSpec, ...],
    *,
    options: MutationSamplingOptions,
) -> tuple[LocalReplacementSpec, ...]:
    excluded = _excluded_tokens(options)
    return tuple(
        spec
        for spec in candidates
        if spec.op is None
        or (
            spec.op not in excluded
            and (options.allow_distributional_unary_ops or spec.op not in _DISTRIBUTIONAL_UNARY_TOKENS)
        )
    )


def _apply_replacement(
    canonical_expr: Expr,
    selected_position: NodePosition,
    original_subtree: Expr,
    replacement: Expr,
    *,
    mutation_kind: str,
) -> MutationResult | None:
    mutated_expr = replace_subtree_by_node_id(canonical_expr, selected_position.node_id, replacement)
    canonical_mutated = canonicalize(mutated_expr)
    if canonical_mutated == canonical_expr:
        return None
    if _contains_obviously_zero_denominator(canonical_mutated):
        return None

    return MutationResult(
        mutated_expr=canonical_mutated,
        selected_node_id=selected_position.node_id,
        selected_family=selected_position.production_family,
        mutation_kind=mutation_kind,
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
