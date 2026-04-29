from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import pandas as pd

from src.mathlang.ast import Expr
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.audit_observation import (
    build_gold_timing_cases,
    run_timing_cases,
    summarize_timing_records,
    write_timing_summary,
)
from src.tree_diffusion.observation import Observation, build_observation


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_prefix_filtered.parquet"
SAMPLE_SIZE = 25
TIMING_PROFILE = os.getenv("TREE_DIFFUSION_TIMING_PROFILE", "light").lower()
DEFAULT_LIGHT_TIMING_SAMPLE_SIZE = 10
DEFAULT_FULL_TIMING_SAMPLE_SIZE = 100
DEFAULT_LIGHT_TIMING_REPEATS = 1
DEFAULT_FULL_TIMING_REPEATS = 3
TIMING_SAMPLE_SIZE = int(
    os.getenv(
        "TREE_DIFFUSION_TIMING_DATASET_SAMPLE",
        str(
            DEFAULT_FULL_TIMING_SAMPLE_SIZE
            if TIMING_PROFILE == "full"
            else DEFAULT_LIGHT_TIMING_SAMPLE_SIZE
        ),
    )
)
TIMING_REPEATS = int(
    os.getenv(
        "TREE_DIFFUSION_TIMING_REPEATS",
        str(
            DEFAULT_FULL_TIMING_REPEATS
            if TIMING_PROFILE == "full"
            else DEFAULT_LIGHT_TIMING_REPEATS
        ),
    )
)
TIMING_SUMMARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "test_summaries"
    / "observation_dataset_timing_summary.json"
)
ZERO_TOLERANCE = 1e-8


class ObservationDatasetSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_PATH.exists():
            raise unittest.SkipTest(f"Dataset artifact not found: {DATASET_PATH}")

        cls.sample = pd.read_parquet(
            DATASET_PATH,
            columns=["integrand_prefix", "integral_prefix"],
        ).head(SAMPLE_SIZE)
        cls.timing_sample = pd.read_parquet(
            DATASET_PATH,
            columns=["integrand_prefix", "integral_prefix"],
        ).head(TIMING_SAMPLE_SIZE)

    def test_build_observations_on_dataset_sample(self) -> None:
        observations: list[Observation] = []
        derivative_present = 0
        numeric_present = 0
        symbolic_present = 0
        finite_positive = 0
        symbolic_zero_rows = 0

        for row_index, row in enumerate(self.sample.itertuples(index=False)):
            target = parse_prefix_string(str(row.integrand_prefix))
            current = parse_prefix_string(str(row.integral_prefix))

            with self.subTest(row_index=row_index):
                observation = build_observation(target, current, residual_mode="both")
                observations.append(observation)

                self.assertIsInstance(observation, Observation)
                self.assertIsInstance(observation.target_integrand, Expr)
                self.assertIsInstance(observation.current_antiderivative, Expr)
                self.assertNotEqual(observation.status, "derivative_failed")

                if observation.current_derivative is None:
                    self.assertTrue(
                        any(
                            warning.startswith(prefix)
                            for prefix in (
                                "derivative_failed:",
                                "derivative_ast_conversion_failed:",
                                "current_derivative_token_cap_exceeded:",
                            )
                            for warning in observation.warnings
                        )
                    )
                else:
                    derivative_present += 1

                if observation.numeric_probes is None:
                    self.assertTrue(
                        any(
                            warning.startswith(prefix)
                            for prefix in (
                                "numeric_probe_failed:",
                                "target_integrand_sympy_failed:",
                            )
                            for warning in observation.warnings
                        )
                    )
                else:
                    numeric_present += 1
                    if observation.numeric_probes.fraction_finite > 0.0:
                        finite_positive += 1

                if observation.symbolic_residual is None:
                    self.assertTrue(
                        any(
                            warning.startswith(prefix)
                            for prefix in (
                                "symbolic_residual_failed:",
                                "target_integrand_sympy_failed:",
                                "symbolic_residual_token_cap_exceeded:",
                            )
                            for warning in observation.warnings
                        )
                    )
                else:
                    symbolic_present += 1
                    if serialize_prefix_string(observation.symbolic_residual) == "INT+ 0":
                        symbolic_zero_rows += 1
                        if observation.numeric_probes is not None:
                            if observation.numeric_probes.mean_abs_residual is not None:
                                self.assertLessEqual(
                                    observation.numeric_probes.mean_abs_residual,
                                    ZERO_TOLERANCE,
                                )
                            if observation.numeric_probes.mean_squared_abs_residual is not None:
                                self.assertLessEqual(
                                    observation.numeric_probes.mean_squared_abs_residual,
                                    ZERO_TOLERANCE * ZERO_TOLERANCE,
                                )
                            if observation.numeric_probes.max_abs_residual is not None:
                                self.assertLessEqual(
                                    observation.numeric_probes.max_abs_residual,
                                    ZERO_TOLERANCE,
                                )

        self.assertEqual(len(observations), SAMPLE_SIZE)
        self.assertEqual(derivative_present, SAMPLE_SIZE)
        self.assertEqual(numeric_present, SAMPLE_SIZE)
        self.assertGreaterEqual(symbolic_present, SAMPLE_SIZE - 1)
        self.assertGreaterEqual(finite_positive, 20)
        self.assertGreater(symbolic_zero_rows, 0)

    def test_write_dataset_timing_summary(self) -> None:
        cases = build_gold_timing_cases(self.timing_sample)
        records = run_timing_cases(
            cases,
            residual_mode="both",
            repeats=TIMING_REPEATS,
        )
        summary = summarize_timing_records(
            records,
            summary_name="gold_observation_timing_from_tests",
            residual_mode="both",
            repeats=TIMING_REPEATS,
        )
        output_path = write_timing_summary(summary, TIMING_SUMMARY_PATH)

        self.assertTrue(output_path.exists())
        loaded_summary = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded_summary["total_cases"], len(cases))
        self.assertEqual(loaded_summary["total_records"], len(cases) * TIMING_REPEATS)
        self.assertEqual(len(loaded_summary["per_case"]), len(cases))
        self.assertEqual(loaded_summary["repeats"], TIMING_REPEATS)
        self.assertIn("average_seconds", loaded_summary["overall_timing"])
        self.assertIn("mean_seconds", loaded_summary["overall_timing"])
        self.assertIn("median_seconds", loaded_summary["overall_timing"])


if __name__ == "__main__":
    unittest.main()
