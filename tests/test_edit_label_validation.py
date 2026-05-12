from __future__ import annotations

import random
import unittest
from dataclasses import replace

from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion.edit_path import EditTarget, first_edit_toward_target
from src.tree_diffusion.label_validation import validate_edit_label_progress
from src.tree_diffusion.mutation import replace_subtree_by_node_id
from src.tree_diffusion.training_examples import generate_training_example


class EditLabelValidationTests(unittest.TestCase):
    def test_simple_useful_edit_validates(self) -> None:
        current = parse_prefix_string("pow x INT+ 5")
        target = parse_prefix_string("pow x INT+ 3")
        edit = first_edit_toward_target(current, target, sigma_small=1, rng=random.Random(0))
        self.assertIsNotNone(edit)
        assert edit is not None

        result = validate_edit_label_progress(current, target, edit)

        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.nonincreasing_distance)
        self.assertTrue(result.strict_improvement)

    def test_bad_position_fails_validation(self) -> None:
        current = parse_prefix_string("pow x INT+ 5")
        target = parse_prefix_string("pow x INT+ 3")
        edit = first_edit_toward_target(current, target, sigma_small=1, rng=random.Random(0))
        self.assertIsNotNone(edit)
        assert edit is not None

        bad_edit = replace(edit, selected_node_id=999)
        result = validate_edit_label_progress(current, target, bad_edit)

        self.assertFalse(result.ok)
        self.assertFalse(result.valid_position)

    def test_distance_increase_fails_validation(self) -> None:
        current = parse_prefix_string("x")
        target = parse_prefix_string("x")
        replacement = parse_prefix_string("sin x")
        resulting_tree = replace_subtree_by_node_id(current, 0, replacement)
        edit = EditTarget(
            selected_node_id=0,
            selected_node_span=(0, 1),
            original_subtree=current,
            replacement_subtree=replacement,
            mutation_kind="sampled_small_subtree_replacement",
            reason="unit_distance_increase",
            resulting_tree=resulting_tree,
        )

        result = validate_edit_label_progress(current, target, edit)

        self.assertFalse(result.ok)
        self.assertFalse(result.nonincreasing_distance)
        self.assertEqual(result.distance_before, 0)
        self.assertGreater(result.distance_after or 0, 0)

    def test_generated_examples_validate(self) -> None:
        pairs = (
            ("pow x INT+ 2", "div pow x INT+ 3 INT+ 3"),
            ("cos x", "sin x"),
            ("exp x", "exp x"),
        )

        for seed, (integrand, integral) in enumerate(pairs):
            with self.subTest(seed=seed, integrand=integrand):
                example = generate_training_example(
                    parse_prefix_string(integrand),
                    parse_prefix_string(integral),
                    rng=random.Random(seed),
                    rho=0.0,
                    smax=2,
                    sigma_small=2,
                )
                result = validate_edit_label_progress(
                    example.current_antiderivative,
                    example.target_antiderivative,
                    example.edit_target,
                )

                self.assertTrue(result.ok, result.error)


if __name__ == "__main__":
    unittest.main()
