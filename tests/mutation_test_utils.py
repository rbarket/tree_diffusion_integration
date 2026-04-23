from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

from src.mathlang.ast import BinaryOp, Const, Expr, NaryOp, UnaryOp, Var
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string, serialize_prefix_tokens
from src.tree_diffusion.mutation import (
    LOCAL_CONST_EDIT,
    LOCAL_SAME_ARITY_REPLACEMENT,
    MutationResult,
    SAMPLED_SMALL_SUBTREE_REPLACEMENT,
)
from src.tree_diffusion.mutation_grammar import can_locally_replace, can_sampled_subtree_replace
from src.tree_diffusion.positions import NodePosition, PositionIndex, index_tree_positions


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_prefix_filtered.parquet"


@dataclass(frozen=True)
class ValidatedMutation:
    source_expr: Expr
    pre_index: PositionIndex
    post_index: PositionIndex
    pre_position: NodePosition
    serialized_mutated: str
    reparsed_mutated: Expr
    possible_kinds: frozenset[str]

    @property
    def has_local_like_kind(self) -> bool:
        return bool({LOCAL_CONST_EDIT, LOCAL_SAME_ARITY_REPLACEMENT} & self.possible_kinds)

    @property
    def has_subtree_kind(self) -> bool:
        return SAMPLED_SMALL_SUBTREE_REPLACEMENT in self.possible_kinds


def canonical_expr(expression: str | Expr) -> Expr:
    if isinstance(expression, str):
        expression = parse_prefix_string(expression)
    return canonicalize(expression)


def num(numerator: int, denominator: int = 1) -> Const:
    return Const(value=Fraction(numerator, denominator))


def hand_built_mutation_cases() -> tuple[tuple[str, Expr, int], ...]:
    return (
        ("leaf_only", canonical_expr("x"), 0),
        ("unary", canonical_expr("sin x"), 1),
        ("binary", canonical_expr("pow x INT+ 2"), 1),
        ("nary_add", canonical_expr("add x add sin x pow x INT+ 2"), 2),
        ("nary_mul", canonical_expr("mul x mul sin x pow x INT+ 2"), 2),
        ("mixed", canonical_expr("add div sin x INT+ 2 add mul pow x INT+ 3 cos x ln x"), 3),
    )


def assert_index_spans_match(testcase, index: PositionIndex) -> None:
    for position in index.positions:
        subtree_tokens = serialize_prefix_tokens(index.node_id_to_node[position.node_id])
        span_tokens = list(index.serialized_tokens[position.token_start:position.token_end])
        testcase.assertEqual(
            subtree_tokens,
            span_tokens,
            msg=(
                f"node_id={position.node_id} span={position.token_start}:{position.token_end} "
                f"subtree={subtree_tokens} span_tokens={span_tokens}"
            ),
        )


def deepest_node_id(
    expr: Expr,
    *,
    predicate: Callable[[Expr], bool] | None = None,
) -> int:
    index = index_tree_positions(canonical_expr(expr))
    positions = [
        position
        for position in index.positions
        if predicate is None or predicate(index.node_id_to_node[position.node_id])
    ]
    if not positions:
        raise LookupError("No matching node found.")
    return max(positions, key=lambda position: (position.depth, position.node_id)).node_id


def first_node_id(
    expr: Expr,
    *,
    predicate: Callable[[Expr], bool],
) -> int:
    index = index_tree_positions(canonical_expr(expr))
    for position in index.positions:
        node = index.node_id_to_node[position.node_id]
        if predicate(node):
            return position.node_id
    raise LookupError("No matching node found.")


def infer_possible_mutation_kinds(result: MutationResult) -> frozenset[str]:
    original = result.original_subtree
    replacement = result.replacement_subtree
    possible: set[str] = set()

    if replacement == original:
        return frozenset()

    if can_locally_replace(original, replacement):
        if isinstance(original, Const):
            if isinstance(replacement, Const) and _same_const_leaf_kind(original, replacement):
                possible.add(LOCAL_CONST_EDIT)
            possible.add(LOCAL_SAME_ARITY_REPLACEMENT)
        else:
            possible.add(LOCAL_SAME_ARITY_REPLACEMENT)

    if can_sampled_subtree_replace(original, replacement):
        if isinstance(original, Const) and isinstance(replacement, Const):
            if _same_const_leaf_kind(original, replacement):
                possible.add(SAMPLED_SMALL_SUBTREE_REPLACEMENT)
        elif isinstance(original, UnaryOp) and isinstance(replacement, UnaryOp):
            possible.add(SAMPLED_SMALL_SUBTREE_REPLACEMENT)
        elif isinstance(original, BinaryOp) and isinstance(replacement, BinaryOp):
            if replacement.op == original.op:
                possible.add(SAMPLED_SMALL_SUBTREE_REPLACEMENT)
        elif isinstance(original, NaryOp) and isinstance(replacement, NaryOp):
            if replacement.op == original.op:
                possible.add(SAMPLED_SMALL_SUBTREE_REPLACEMENT)

    return frozenset(possible)


def validate_mutation_result(
    testcase,
    source_expr: Expr | str,
    result: MutationResult | None,
    *,
    sigma_small: int | None = None,
) -> ValidatedMutation:
    testcase.assertIsNotNone(result)
    assert result is not None

    canonical_source = canonical_expr(source_expr)
    pre_index = index_tree_positions(canonical_source)

    testcase.assertIn(result.selected_node_id, pre_index.node_id_to_node)
    pre_position = pre_index.positions[result.selected_node_id]
    original_from_index = pre_index.node_id_to_node[result.selected_node_id]

    testcase.assertEqual(original_from_index, result.original_subtree)
    testcase.assertEqual(pre_position.production_family, result.selected_family)
    testcase.assertEqual(pre_position.token_start, result.selected_token_start)
    testcase.assertEqual(pre_position.token_end, result.selected_token_end)
    testcase.assertNotEqual(result.original_subtree, result.replacement_subtree)
    testcase.assertNotEqual(canonical_source, result.mutated_expr)

    if sigma_small is not None:
        testcase.assertLessEqual(pre_position.subtree_size, sigma_small)

    selected_span_tokens = list(
        pre_index.serialized_tokens[result.selected_token_start:result.selected_token_end]
    )
    testcase.assertEqual(selected_span_tokens, serialize_prefix_tokens(result.original_subtree))

    serialized_mutated = serialize_prefix_string(result.mutated_expr)
    reparsed_mutated = parse_prefix_string(serialized_mutated)

    testcase.assertEqual(result.mutated_expr, canonicalize(result.mutated_expr))
    testcase.assertEqual(result.mutated_expr, canonicalize(reparsed_mutated))

    post_index = index_tree_positions(result.mutated_expr)
    assert_index_spans_match(testcase, post_index)

    possible_kinds = infer_possible_mutation_kinds(result)
    testcase.assertTrue(
        possible_kinds,
        msg=(
            "Mutation metadata did not admit any consistent local/subtree classification: "
            f"original={serialize_prefix_string(result.original_subtree)!r} "
            f"replacement={serialize_prefix_string(result.replacement_subtree)!r}"
        ),
    )

    return ValidatedMutation(
        source_expr=canonical_source,
        pre_index=pre_index,
        post_index=post_index,
        pre_position=pre_position,
        serialized_mutated=serialized_mutated,
        reparsed_mutated=reparsed_mutated,
        possible_kinds=possible_kinds,
    )


def load_dataset_expressions(*, limit: int, column: str = "integrand_prefix") -> list[str]:
    import pandas as pd

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    sample = pd.read_parquet(DATASET_PATH, columns=[column]).head(limit)
    return sample[column].astype(str).tolist()


def _same_const_leaf_kind(left: Const, right: Const) -> bool:
    return (left.is_numeric and right.is_numeric) or (left.is_named and right.is_named)
