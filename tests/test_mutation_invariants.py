from __future__ import annotations

import random
import unittest

from src.tree_diffusion.mutation import mutate_once
from tests.mutation_test_utils import canonical_expr, hand_built_mutation_cases, validate_mutation_result


CASE_SEEDS = {
    "leaf_only": 1,
    "unary": 1,
    "binary": 0,
    "nary_add": 1,
    "nary_mul": 4,
    "mixed": 4,
}


class MutationInvariantTests(unittest.TestCase):
    def test_mutate_once_preserves_core_invariants_on_hand_built_expressions(self) -> None:
        saw_local_like = False
        saw_subtree = False

        for case_name, expr, sigma_small in hand_built_mutation_cases():
            with self.subTest(case=case_name):
                result = mutate_once(expr, sigma_small=sigma_small, rng=random.Random(CASE_SEEDS[case_name]))
                validated = validate_mutation_result(self, expr, result, sigma_small=sigma_small)
                saw_local_like = saw_local_like or validated.has_local_like_kind
                saw_subtree = saw_subtree or validated.has_subtree_kind

        self.assertTrue(saw_local_like)
        self.assertTrue(saw_subtree)

    def test_sigma_small_zero_limits_mutation_to_leaves(self) -> None:
        expr = canonical_expr("add div sin x INT+ 2 add mul pow x INT+ 3 cos x ln x")
        result = mutate_once(expr, sigma_small=0, rng=random.Random(0))
        validated = validate_mutation_result(self, expr, result, sigma_small=0)

        self.assertIn(result.selected_family, {"CONST", "VAR"})
        self.assertEqual(validated.pre_position.subtree_size, 0)

    def test_sigma_small_one_can_mutate_the_root_of_a_small_tree(self) -> None:
        expr = canonical_expr("sin x")
        result = mutate_once(expr, sigma_small=1, rng=random.Random(1))
        validated = validate_mutation_result(self, expr, result, sigma_small=1)

        self.assertEqual(result.selected_node_id, 0)
        self.assertEqual(validated.pre_position.subtree_size, 1)
        self.assertTrue(validated.has_subtree_kind)

    def test_repeated_mutation_walk_stays_parseable_and_canonical(self) -> None:
        expr = canonical_expr("add div sin x INT+ 2 add mul pow x INT+ 3 cos x ln x")
        rng = random.Random(17)

        for step in range(12):
            with self.subTest(step=step):
                result = mutate_once(expr, sigma_small=3, rng=rng)
                validate_mutation_result(self, expr, result, sigma_small=3)
                assert result is not None
                expr = result.mutated_expr
