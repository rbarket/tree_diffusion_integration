from __future__ import annotations

import random
import unittest

from src.mathlang.ast import Const, Var
from src.tree_diffusion.mutation import local_replace_once, mutate_once
from src.tree_diffusion.positions import index_tree_positions
from tests.mutation_test_utils import assert_index_spans_match, canonical_expr, deepest_node_id, validate_mutation_result


class MutationPositionTests(unittest.TestCase):
    def test_pre_mutation_metadata_spans_only_apply_to_the_pre_mutation_tree(self) -> None:
        expr = canonical_expr("sin x")
        result = mutate_once(expr, sigma_small=1, rng=random.Random(1))
        validated = validate_mutation_result(self, expr, result, sigma_small=1)
        assert result is not None

        self.assertEqual((result.selected_token_start, result.selected_token_end), (0, 2))
        self.assertNotEqual(
            validated.post_index.node_id_to_span[0],
            (result.selected_token_start, result.selected_token_end),
        )
        self.assertEqual(
            validated.post_index.node_id_to_span[0],
            (0, len(validated.post_index.serialized_tokens)),
        )

    def test_reindex_after_deepest_leaf_mutation_matches_current_serialization(self) -> None:
        expr = canonical_expr("add div sin x INT+ 2 add mul pow x INT+ 3 cos x ln x")
        deepest_leaf_id = deepest_node_id(expr, predicate=lambda node: isinstance(node, (Const, Var)))

        result = local_replace_once(expr, selected_node_id=deepest_leaf_id, rng=random.Random(0))
        validated = validate_mutation_result(self, expr, result)

        assert result is not None
        assert_index_spans_match(self, validated.post_index)
        self.assertEqual(
            [position.node_id for position in validated.post_index.positions],
            list(range(len(validated.post_index.positions))),
        )

    def test_reindex_after_root_local_mutation_of_associative_binary_tree(self) -> None:
        expr = canonical_expr("add x add sin x add pow x INT+ 2 add ln x cos x")
        result = local_replace_once(expr, selected_node_id=0, rng=random.Random(0))
        validated = validate_mutation_result(self, expr, result)

        assert result is not None
        assert_index_spans_match(self, validated.post_index)
        self.assertEqual(validated.post_index.positions[0].depth, 0)

    def test_reindex_after_subtree_mutation_uses_current_spans_not_stale_ones(self) -> None:
        expr = canonical_expr("mul x mul sin x pow x INT+ 2")
        result = mutate_once(expr, sigma_small=3, rng=random.Random(1))
        validated = validate_mutation_result(self, expr, result, sigma_small=3)
        assert result is not None

        fresh_index = index_tree_positions(result.mutated_expr)
        assert_index_spans_match(self, fresh_index)
        self.assertNotEqual(
            fresh_index.node_id_to_span[result.selected_node_id],
            validated.pre_index.node_id_to_span[result.selected_node_id],
        )
