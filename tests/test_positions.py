from __future__ import annotations

import unittest

from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_tokens
from src.tree_diffusion.positions import index_tree_positions


class PositionIndexTests(unittest.TestCase):
    def test_preorder_ids_are_deterministic(self) -> None:
        expr = canonicalize(parse_prefix_string("add pow x INT+ 2 sin x"))
        first = index_tree_positions(expr)
        second = index_tree_positions(expr)

        self.assertEqual(first.positions, second.positions)
        self.assertEqual(first.node_id_to_span, second.node_id_to_span)
        self.assertEqual(first.serialized_tokens, second.serialized_tokens)

    def test_spans_match_current_subtree_serialization(self) -> None:
        expr = canonicalize(parse_prefix_string("add sin x add x pow x INT+ 3"))
        index = index_tree_positions(expr)

        for position in index.positions:
            subtree_tokens = serialize_prefix_tokens(index.node_id_to_node[position.node_id])
            span_tokens = list(index.serialized_tokens[position.token_start:position.token_end])
            self.assertEqual(subtree_tokens, span_tokens)

    def test_canonicalization_recomputes_spans_from_current_tree(self) -> None:
        original = parse_prefix_string("add pow x INT+ 2 x")
        canonical_expr = canonicalize(original)
        index = index_tree_positions(canonical_expr)

        self.assertEqual(list(index.serialized_tokens), ["add", "x", "pow", "x", "INT+", "2"])
        root = index.positions[0]
        self.assertEqual((root.token_start, root.token_end), (0, 6))

    def test_parent_ids_and_depths_follow_preorder_tree(self) -> None:
        expr = canonicalize(parse_prefix_string("mul exp x pow x INT+ 2"))
        index = index_tree_positions(expr)

        self.assertEqual(index.positions[0].parent_id, None)
        self.assertEqual(index.positions[0].depth, 0)
        self.assertTrue(all(position.node_id == position.preorder_index for position in index.positions))
        self.assertEqual(index.positions[1].parent_id, 0)
        self.assertEqual(index.positions[1].depth, 1)
