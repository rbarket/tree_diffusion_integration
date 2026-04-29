from __future__ import annotations

import json
import os
import random
import unittest
from pathlib import Path

import pandas as pd

from src.mathlang.ast import Expr
from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion.audit_observation import (
    build_corrupted_timing_cases,
    run_timing_cases,
    summarize_timing_records,
    write_timing_summary,
)
from src.tree_diffusion.mutation import MutationResult, mutate_once
from src.tree_diffusion.observation import Observation, build_observation


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_prefix_filtered.parquet"
FULL_SCENARIOS = (
    {"row_index": 0, "mutation_count": 1, "seed": 0},
    {"row_index": 0, "mutation_count": 2, "seed": 0},
    {"row_index": 1, "mutation_count": 1, "seed": 0},
    {"row_index": 1, "mutation_count": 2, "seed": 0},
    {"row_index": 2, "mutation_count": 1, "seed": 0},
    {"row_index": 2, "mutation_count": 2, "seed": 0},
    {"row_index": 4, "mutation_count": 1, "seed": 0},
    {"row_index": 4, "mutation_count": 2, "seed": 0},
    {"row_index": 5, "mutation_count": 1, "seed": 0},
    {"row_index": 5, "mutation_count": 2, "seed": 0},
    {"row_index": 6, "mutation_count": 1, "seed": 0},
    {"row_index": 6, "mutation_count": 2, "seed": 0},
)
LIGHT_SCENARIOS = (
    {"row_index": 0, "mutation_count": 1, "seed": 0},
    {"row_index": 1, "mutation_count": 1, "seed": 0},
    {"row_index": 4, "mutation_count": 2, "seed": 0},
    {"row_index": 5, "mutation_count": 2, "seed": 0},
)
# Manual reproduction for the slow symbolic-residual simplify path seen in the
# observation audit. This uses the audit-style per-row RNG seed:
#   row_index=3, mutation_count=1, rng_seed=seed + row_index = 3
SLOW_SYMBOLIC_RESIDUAL_SCENARIO = {"row_index": 3, "mutation_count": 1, "seed": 3}
SIGMA_SMALL = 0
TIMING_PROFILE = os.getenv("TREE_DIFFUSION_TIMING_PROFILE", "light").lower()
SCENARIOS = FULL_SCENARIOS if TIMING_PROFILE == "full" else LIGHT_SCENARIOS
TIMING_REPEATS = int(
    os.getenv(
        "TREE_DIFFUSION_TIMING_REPEATS",
        "3" if TIMING_PROFILE == "full" else "1",
    )
)
TIMING_SUMMARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "test_summaries"
    / "observation_corrupted_timing_summary.json"
)


class ObservationCorruptedSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_PATH.exists():
            raise unittest.SkipTest(f"Dataset artifact not found: {DATASET_PATH}")

        max_row = max(scenario["row_index"] for scenario in SCENARIOS) + 1
        cls.sample = pd.read_parquet(
            DATASET_PATH,
            columns=["integrand_prefix", "integral_prefix"],
        ).head(max(max_row, SLOW_SYMBOLIC_RESIDUAL_SCENARIO["row_index"] + 1))

    def test_build_observations_on_corrupted_candidates(self) -> None:
        observations: list[Observation] = []
        derivative_present = 0
        numeric_present = 0
        symbolic_present = 0
        finite_positive = 0
        warningful_observations = 0

        for scenario in SCENARIOS:
            row = self.sample.iloc[scenario["row_index"]]
            target = parse_prefix_string(str(row.integrand_prefix))
            current = parse_prefix_string(str(row.integral_prefix))
            rng = random.Random(scenario["seed"])
            mutations: list[MutationResult] = []

            with self.subTest(
                row_index=scenario["row_index"],
                mutation_count=scenario["mutation_count"],
                seed=scenario["seed"],
            ):
                for _ in range(scenario["mutation_count"]):
                    mutation = mutate_once(current, sigma_small=SIGMA_SMALL, rng=rng)
                    self.assertIsNotNone(
                        mutation,
                        msg=(
                            "Expected deterministic mutation sequence to be fully applicable: "
                            f"{scenario}"
                        ),
                    )
                    assert mutation is not None
                    mutations.append(mutation)
                    current = mutation.mutated_expr

                self.assertEqual(len(mutations), scenario["mutation_count"])

                observation = build_observation(
                    target_integrand=target,
                    current_antiderivative=current,
                    residual_mode="both",
                )
                observations.append(observation)

                self.assertIsInstance(observation, Observation)
                self.assertIsInstance(observation.target_integrand, Expr)
                self.assertIsInstance(observation.current_antiderivative, Expr)

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

                if observation.warnings:
                    warningful_observations += 1

                self.assertIn(observation.status, {"ok", "partial", "derivative_failed"})

        self.assertEqual(len(observations), len(SCENARIOS))
        self.assertGreaterEqual(derivative_present, len(SCENARIOS) - 1)
        self.assertGreaterEqual(numeric_present, len(SCENARIOS) - 1)
        self.assertGreaterEqual(symbolic_present, len(SCENARIOS) - 1)
        self.assertGreaterEqual(finite_positive, len(SCENARIOS) - 2)
        self.assertGreaterEqual(warningful_observations, 1)

    def test_write_corrupted_timing_summary(self) -> None:
        cases = build_corrupted_timing_cases(
            self.sample,
            list(SCENARIOS),
            sigma_small=SIGMA_SMALL,
        )
        records = run_timing_cases(
            cases,
            residual_mode="both",
            repeats=TIMING_REPEATS,
        )
        summary = summarize_timing_records(
            records,
            summary_name="corrupted_observation_timing_from_tests",
            residual_mode="both",
            repeats=TIMING_REPEATS,
        )
        output_path = write_timing_summary(summary, TIMING_SUMMARY_PATH)

        self.assertTrue(output_path.exists())
        loaded_summary = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded_summary["total_cases"], len(SCENARIOS))
        self.assertEqual(loaded_summary["total_records"], len(SCENARIOS) * TIMING_REPEATS)
        self.assertIn("overall_timing", loaded_summary)
        self.assertEqual(
            len(loaded_summary["per_case"]),
            len(SCENARIOS),
        )
        self.assertEqual(loaded_summary["repeats"], TIMING_REPEATS)
        self.assertIn("average_seconds", loaded_summary["overall_timing"])
        self.assertIn("mean_seconds", loaded_summary["overall_timing"])
        self.assertIn("median_seconds", loaded_summary["overall_timing"])

    @unittest.skip("Manual reproduction for the historically slow symbolic-residual simplify case.")
    def test_manual_slow_symbolic_residual_case(self) -> None:
        scenario = SLOW_SYMBOLIC_RESIDUAL_SCENARIO
        row = self.sample.iloc[scenario["row_index"]]
        target = parse_prefix_string(str(row.integrand_prefix))
        current = parse_prefix_string(str(row.integral_prefix))
        rng = random.Random(scenario["seed"])

        for _ in range(scenario["mutation_count"]):
            mutation = mutate_once(current, sigma_small=SIGMA_SMALL, rng=rng)
            self.assertIsNotNone(mutation, msg=f"Expected deterministic mutation for {scenario}")
            assert mutation is not None
            current = mutation.mutated_expr

        observation = build_observation(
            target_integrand=target,
            current_antiderivative=current,
            residual_mode="both",
        )

        self.assertIsInstance(observation, Observation)
        self.assertIsNotNone(observation.current_derivative)
        self.assertIsNotNone(observation.symbolic_residual)
        self.assertIsNotNone(observation.numeric_probes)


if __name__ == "__main__":
    unittest.main()
