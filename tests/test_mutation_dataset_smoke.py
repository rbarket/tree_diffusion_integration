from __future__ import annotations

import random
import unittest

from src.tree_diffusion.mutation import mutate_once
from tests.mutation_test_utils import DATASET_PATH, canonical_expr, load_dataset_expressions, validate_mutation_result


SAMPLE_SIZE = 25
MUTATIONS_PER_EXAMPLE = 4
SIGMA_SMALL = 3


class MutationDatasetSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_PATH.exists():
            raise unittest.SkipTest(f"Dataset artifact not found: {DATASET_PATH}")
        cls.expressions = load_dataset_expressions(limit=SAMPLE_SIZE, column="integrand_prefix")

    def test_processed_dataset_sample_mutates_without_crashing(self) -> None:
        local_like_hits = 0
        subtree_hits = 0
        successful_mutations = 0
        total_mutations = len(self.expressions) * MUTATIONS_PER_EXAMPLE

        for row_index, expression in enumerate(self.expressions):
            current = canonical_expr(expression)
            rng = random.Random(10_000 + row_index)

            for step in range(MUTATIONS_PER_EXAMPLE):
                with self.subTest(row_index=row_index, step=step):
                    result = mutate_once(current, sigma_small=SIGMA_SMALL, rng=rng)
                    validated = validate_mutation_result(self, current, result, sigma_small=SIGMA_SMALL)
                    successful_mutations += 1
                    local_like_hits += int(validated.has_local_like_kind)
                    subtree_hits += int(validated.has_subtree_kind)
                    assert result is not None
                    current = result.mutated_expr

        self.assertEqual(successful_mutations, total_mutations)
        self.assertGreater(local_like_hits, 0)
        self.assertGreater(subtree_hits, 0)
