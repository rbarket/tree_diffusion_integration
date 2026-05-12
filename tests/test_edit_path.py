from __future__ import annotations

import random
import unittest
from fractions import Fraction

from src.mathlang.ast import Const, Var
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.edit_path import (
    compute_edit_path,
    find_first_mismatch,
    first_edit_toward_target,
    is_small_enough,
    structural_distance,
    trees_equal,
)
from src.tree_diffusion.mutation import (
    LOCAL_CONST_EDIT,
    LOCAL_SAME_ARITY_REPLACEMENT,
    SAMPLED_SMALL_SUBTREE_REPLACEMENT,
)
from src.tree_diffusion.positions import index_tree_positions
from tests.edit_path_test_utils import assert_edit_is_legal, assert_edit_reduces_distance
from tests.mutation_test_utils import canonical_expr, num


class EditPathTests(unittest.TestCase):
    def test_pow_exponent_directly_changes_to_target_constant(self) -> None:
        current = canonical_expr("pow x INT+ 5")
        target = canonical_expr("pow x INT+ 3")

        edit = first_edit_toward_target(current, target, sigma_small=0, rng=random.Random(0))

        self.assertIsNotNone(edit)
        assert edit is not None
        assert_edit_is_legal(self, current, edit, sigma_small=0)
        assert_edit_reduces_distance(self, current, target, edit)
        self.assertEqual(edit.mutation_kind, LOCAL_CONST_EDIT)
        self.assertEqual(edit.replacement_subtree, num(3))
        self.assertEqual(edit.resulting_tree, target)

        index = index_tree_positions(current)
        self.assertEqual(index.node_id_to_node[edit.selected_node_id], Const(value=Fraction(5, 1)))
        self.assertEqual(edit.selected_node_span, index.node_id_to_span[edit.selected_node_id])

    def test_unary_operator_directly_swaps_to_target_operator(self) -> None:
        current = canonical_expr("sin x")
        target = canonical_expr("cos x")

        edit = first_edit_toward_target(current, target, sigma_small=1)

        self.assertIsNotNone(edit)
        assert edit is not None
        assert_edit_is_legal(self, current, edit, sigma_small=1)
        self.assertEqual(edit.selected_node_id, 0)
        self.assertEqual(edit.mutation_kind, LOCAL_SAME_ARITY_REPLACEMENT)
        self.assertEqual(serialize_prefix_string(edit.replacement_subtree), "cos x")
        self.assertEqual(edit.resulting_tree, target)

    def test_top_level_additive_constant_difference_is_canonicalized_away(self) -> None:
        current = parse_prefix_string("add x INT+ 5")
        target = parse_prefix_string("x")

        self.assertTrue(trees_equal(current, target))
        self.assertIsNone(first_edit_toward_target(current, target, sigma_small=1))

    def test_first_mismatch_is_canonical_structural_mismatch(self) -> None:
        current = canonical_expr("pow x INT+ 5")
        target = canonical_expr("pow x INT+ 3")

        mismatch = find_first_mismatch(current, target)

        self.assertIsNotNone(mismatch)
        assert mismatch is not None
        self.assertEqual(mismatch.path, (1,))
        self.assertEqual(mismatch.current_node_id, 2)
        self.assertEqual(mismatch.current_subtree, num(5))
        self.assertEqual(mismatch.target_subtree, num(3))

    def test_constant_correction_targets_goal_not_arbitrary_prior_value(self) -> None:
        current = Const(value=Fraction(5, 1))
        target = Const(value=Fraction(1, 1))

        edit = first_edit_toward_target(current, target, sigma_small=0, rng=random.Random(123))

        self.assertIsNotNone(edit)
        assert edit is not None
        assert_edit_is_legal(self, current, edit, sigma_small=0)
        self.assertEqual(edit.mutation_kind, LOCAL_CONST_EDIT)
        self.assertEqual(edit.replacement_subtree, target)
        self.assertNotEqual(edit.replacement_subtree, num(3))
        self.assertEqual(edit.resulting_tree, target)

    def test_operator_correction_targets_goal_not_sampled_operator_chain(self) -> None:
        current = canonical_expr("sin x")
        target = canonical_expr("tan x")

        edit = first_edit_toward_target(current, target, sigma_small=1, rng=random.Random(99))

        self.assertIsNotNone(edit)
        assert edit is not None
        assert_edit_is_legal(self, current, edit, sigma_small=1)
        self.assertEqual(edit.replacement_subtree, target)
        self.assertEqual(edit.resulting_tree, target)

    def test_cross_shape_small_target_uses_direct_sampled_subtree_replacement(self) -> None:
        current = canonical_expr("sin x")
        target = canonical_expr("div x INT+ 2")

        edit = first_edit_toward_target(current, target, sigma_small=1, rng=random.Random(0))

        self.assertIsNotNone(edit)
        assert edit is not None
        assert_edit_is_legal(self, current, edit, sigma_small=1)
        assert_edit_reduces_distance(self, current, target, edit)
        self.assertEqual(edit.selected_node_id, 0)
        self.assertEqual(edit.mutation_kind, SAMPLED_SMALL_SUBTREE_REPLACEMENT)
        self.assertEqual(edit.replacement_subtree, target)
        self.assertEqual(edit.resulting_tree, target)

    def test_oversized_target_uses_direct_small_subtree_when_target_subtree_fits(self) -> None:
        current = canonical_expr("pow x INT+ 5")
        target = canonical_expr("pow x add x pow x INT+ 2")

        self.assertGreater(structural_distance(current, target), 0)
        self.assertFalse(is_small_enough(target, sigma_small=2))

        edit = first_edit_toward_target(current, target, sigma_small=2, rng=random.Random(0))

        self.assertIsNotNone(edit)
        assert edit is not None
        assert_edit_is_legal(self, current, edit, sigma_small=2)
        assert_edit_reduces_distance(self, current, target, edit)
        self.assertEqual(edit.selected_node_id, 2)
        self.assertEqual(edit.mutation_kind, SAMPLED_SMALL_SUBTREE_REPLACEMENT)
        self.assertEqual(serialize_prefix_string(edit.replacement_subtree), "add x pow x INT+ 2")
        self.assertEqual(edit.resulting_tree, target)

    def test_compute_edit_path_reaches_target_on_simple_oversized_case(self) -> None:
        current = canonical_expr("pow x INT+ 5")
        target = canonical_expr("pow x add x pow x INT+ 2")

        path = compute_edit_path(current, target, sigma_small=2, rng=random.Random(0), max_steps=8)

        self.assertEqual(len(path), 1)
        self.assertEqual(path[-1].resulting_tree, target)

        source = current
        previous_distance = structural_distance(source, target)
        for edit in path:
            with self.subTest(edit=edit):
                assert_edit_is_legal(self, source, edit, sigma_small=2)
                next_distance = structural_distance(edit.resulting_tree, target)
                self.assertLess(next_distance, previous_distance)
                source = edit.resulting_tree
                previous_distance = next_distance

    def test_direct_child_repair_handles_leaf_to_nonleaf_child(self) -> None:
        current = canonical_expr("add x x")
        target = canonical_expr("add x pow x INT+ 2")

        edit = first_edit_toward_target(current, target, sigma_small=2)

        self.assertIsNotNone(edit)
        assert edit is not None
        assert_edit_is_legal(self, current, edit, sigma_small=2)
        assert_edit_reduces_distance(self, current, target, edit)
        self.assertEqual(edit.resulting_tree, target)
        self.assertEqual(edit.selected_node_id, 2)
        self.assertEqual(edit.mutation_kind, SAMPLED_SMALL_SUBTREE_REPLACEMENT)
        self.assertEqual(serialize_prefix_string(edit.replacement_subtree), "pow x INT+ 2")

    def test_public_small_helpers_follow_operator_node_budget(self) -> None:
        self.assertTrue(is_small_enough(parse_prefix_string("x"), 0))
        self.assertTrue(is_small_enough(parse_prefix_string("sin x"), 1))
        self.assertFalse(is_small_enough(parse_prefix_string("sin pow x INT+ 2"), 1))
        self.assertEqual(structural_distance(Var(name="x"), Var(name="x")), 0)


if __name__ == "__main__":
    unittest.main()
