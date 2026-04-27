from __future__ import annotations

import random
import unittest

from src.tree_diffusion.edit_path import first_edit_toward_target, structural_distance
from src.tree_diffusion.mutation import local_replace_once
from src.tree_diffusion.positions import index_tree_positions
from tests.edit_path_test_utils import assert_edit_is_legal
from tests.mutation_test_utils import DATASET_PATH, canonical_expr, load_dataset_expressions, validate_mutation_result


SAMPLE_SIZE = 12
SIGMA_SMALL = 3


class EditPathDatasetSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_PATH.exists():
            raise unittest.SkipTest(f"Dataset artifact not found: {DATASET_PATH}")
        try:
            cls.expressions = load_dataset_expressions(limit=SAMPLE_SIZE, column="integrand_prefix")
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"Dataset dependency not available: {exc.name}") from exc

    def test_dataset_sample_local_corruptions_get_legal_reverse_edits(self) -> None:
        checked = 0

        for row_index, expression in enumerate(self.expressions):
            target = canonical_expr(expression)
            index = index_tree_positions(target)
            candidate_positions = [
                position
                for position in index.positions
                if position.is_mutable and position.subtree_size <= SIGMA_SMALL
            ]
            if not candidate_positions:
                continue

            selected = candidate_positions[row_index % len(candidate_positions)]
            mutation = local_replace_once(target, selected.node_id, rng=random.Random(20_000 + row_index))
            if mutation is None:
                continue

            validate_mutation_result(self, target, mutation, sigma_small=SIGMA_SMALL)
            corrupted = mutation.mutated_expr

            edit = first_edit_toward_target(corrupted, target, SIGMA_SMALL, rng=random.Random(30_000 + row_index))

            self.assertIsNotNone(edit)
            assert edit is not None
            assert_edit_is_legal(self, corrupted, edit, sigma_small=SIGMA_SMALL)
            self.assertLess(structural_distance(edit.resulting_tree, target), structural_distance(corrupted, target))
            checked += 1

        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
