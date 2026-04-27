from __future__ import annotations

from src.mathlang.ast import Const
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string, serialize_prefix_tokens
from src.tree_diffusion.edit_path import EditTarget, structural_distance
from src.tree_diffusion.mutation import (
    LOCAL_CONST_EDIT,
    LOCAL_SAME_ARITY_REPLACEMENT,
    SAMPLED_SMALL_SUBTREE_REPLACEMENT,
    replace_subtree_by_node_id,
)
from src.tree_diffusion.mutation_grammar import can_locally_replace, can_sampled_subtree_replace, subtree_size
from src.tree_diffusion.positions import index_tree_positions


def assert_edit_is_legal(testcase, source_expr, edit: EditTarget, *, sigma_small: int) -> None:
    canonical_source = canonicalize(source_expr)
    index = index_tree_positions(canonical_source)

    testcase.assertIn(edit.selected_node_id, index.node_id_to_node)
    testcase.assertEqual(index.node_id_to_node[edit.selected_node_id], edit.original_subtree)
    testcase.assertEqual(index.node_id_to_span[edit.selected_node_id], edit.selected_node_span)
    testcase.assertNotEqual(edit.original_subtree, edit.replacement_subtree)

    span_tokens = list(index.serialized_tokens[edit.selected_node_span[0]:edit.selected_node_span[1]])
    testcase.assertEqual(span_tokens, serialize_prefix_tokens(edit.original_subtree))

    expected_result = canonicalize(
        replace_subtree_by_node_id(canonical_source, edit.selected_node_id, edit.replacement_subtree)
    )
    testcase.assertEqual(edit.resulting_tree, expected_result)
    testcase.assertEqual(edit.resulting_tree, canonicalize(edit.resulting_tree))

    serialized_result = serialize_prefix_string(edit.resulting_tree)
    testcase.assertEqual(edit.resulting_tree, canonicalize(parse_prefix_string(serialized_result)))

    if edit.mutation_kind == LOCAL_CONST_EDIT:
        testcase.assertIsInstance(edit.original_subtree, Const)
        testcase.assertIsInstance(edit.replacement_subtree, Const)
        testcase.assertTrue(_same_const_leaf_kind(edit.original_subtree, edit.replacement_subtree))
        testcase.assertTrue(can_locally_replace(edit.original_subtree, edit.replacement_subtree))
        return

    if edit.mutation_kind == LOCAL_SAME_ARITY_REPLACEMENT:
        testcase.assertTrue(can_locally_replace(edit.original_subtree, edit.replacement_subtree))
        return

    if edit.mutation_kind == SAMPLED_SMALL_SUBTREE_REPLACEMENT:
        testcase.assertLessEqual(subtree_size(edit.replacement_subtree), sigma_small)
        testcase.assertTrue(can_sampled_subtree_replace(edit.original_subtree, edit.replacement_subtree))
        return

    testcase.fail(f"Unknown edit mutation kind: {edit.mutation_kind!r}")


def assert_edit_reduces_distance(testcase, source_expr, target_expr, edit: EditTarget) -> None:
    before = structural_distance(source_expr, target_expr)
    after = structural_distance(edit.resulting_tree, target_expr)
    testcase.assertLess(after, before, msg=f"distance did not reduce: before={before} after={after}")


def _same_const_leaf_kind(left: Const, right: Const) -> bool:
    return (left.is_numeric and right.is_numeric) or (left.is_named and right.is_named)
