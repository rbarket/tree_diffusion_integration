from __future__ import annotations

import random
import unittest

from src.mathlang.ast import BinaryOp, UnaryOp, Var
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.mutation import (
    SAMPLED_SMALL_SUBTREE_REPLACEMENT,
    mutate_once,
    replace_subtree_by_node_id,
    sample_valid_subtree,
)
from src.tree_diffusion.mutation_grammar import can_locally_replace, can_sampled_subtree_replace, subtree_size
from src.tree_diffusion.positions import index_tree_positions
from tests.mutation_test_utils import canonical_expr, first_node_id, validate_mutation_result


class SubtreeReplacementTests(unittest.TestCase):
    def test_mutate_once_can_do_unary_subtree_replacement_at_the_root(self) -> None:
        expr = canonical_expr("sin x")
        sigma_small = 1

        result = mutate_once(expr, sigma_small=sigma_small, rng=random.Random(1))
        validated = validate_mutation_result(self, expr, result, sigma_small=sigma_small)
        assert result is not None

        self.assertEqual(result.selected_node_id, 0)
        self.assertEqual(validated.possible_kinds, frozenset({SAMPLED_SMALL_SUBTREE_REPLACEMENT}))
        self.assertFalse(can_locally_replace(result.original_subtree, result.replacement_subtree))
        self.assertTrue(can_sampled_subtree_replace(result.original_subtree, result.replacement_subtree))
        self.assertIsInstance(result.replacement_subtree, UnaryOp)

    def test_mutate_once_can_do_binary_subtree_replacement_at_the_root(self) -> None:
        expr = canonical_expr("pow x INT+ 2")
        sigma_small = 1

        result = mutate_once(expr, sigma_small=sigma_small, rng=random.Random(7))
        validated = validate_mutation_result(self, expr, result, sigma_small=sigma_small)
        assert result is not None

        self.assertEqual(result.selected_node_id, 0)
        self.assertEqual(validated.possible_kinds, frozenset({SAMPLED_SMALL_SUBTREE_REPLACEMENT}))
        self.assertFalse(can_locally_replace(result.original_subtree, result.replacement_subtree))
        self.assertTrue(can_sampled_subtree_replace(result.original_subtree, result.replacement_subtree))
        self.assertIsInstance(result.replacement_subtree, BinaryOp)

    def test_mutate_once_can_do_add_mul_binary_subtree_replacement(self) -> None:
        expr = canonical_expr("mul x mul sin x pow x INT+ 2")
        sigma_small = 3

        result = mutate_once(expr, sigma_small=sigma_small, rng=random.Random(1))
        validated = validate_mutation_result(self, expr, result, sigma_small=sigma_small)
        assert result is not None

        self.assertEqual(result.selected_family, "MUL_EXPR")
        self.assertEqual(validated.possible_kinds, frozenset({SAMPLED_SMALL_SUBTREE_REPLACEMENT}))
        self.assertFalse(can_locally_replace(result.original_subtree, result.replacement_subtree))
        self.assertTrue(can_sampled_subtree_replace(result.original_subtree, result.replacement_subtree))
        self.assertIsInstance(result.replacement_subtree, BinaryOp)
        self.assertEqual(result.original_subtree.op, result.replacement_subtree.op)

    def test_manual_sampled_subtree_replacement_roundtrips_for_a_non_root_subtree(self) -> None:
        expr = canonical_expr("add div sin x INT+ 2 add mul pow x INT+ 3 cos x ln x")
        sigma_small = 3
        selected_node_id = first_node_id(expr, predicate=lambda node: isinstance(node, BinaryOp) and node.op == "div")
        index = index_tree_positions(expr, sigma_small=sigma_small)
        selected_position = index.positions[selected_node_id]

        replacement = sample_valid_subtree("DIV_EXPR", sigma_small=2, rng=random.Random(2))
        self.assertLessEqual(selected_position.subtree_size, sigma_small)
        self.assertLessEqual(subtree_size(replacement), 2)
        self.assertTrue(can_sampled_subtree_replace(index.node_id_to_node[selected_node_id], replacement))
        self.assertFalse(can_locally_replace(index.node_id_to_node[selected_node_id], replacement))
        self.assertNotEqual(index.node_id_to_node[selected_node_id], replacement)

        mutated = canonicalize(replace_subtree_by_node_id(expr, selected_node_id, replacement))
        serialized = serialize_prefix_string(mutated)
        self.assertEqual(mutated, canonicalize(parse_prefix_string(serialized)))

    def test_var_leaf_to_non_leaf_subtree_replacement_is_rejected_in_current_design(self) -> None:
        self.assertFalse(
            can_sampled_subtree_replace(
                Var(name="x"),
                sample_valid_subtree("UNARY_EXPR", sigma_small=1, rng=random.Random(2)),
            )
        )
