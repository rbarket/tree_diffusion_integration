from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.observation import build_observation


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_prefix_filtered.parquet"
SAMPLE_SIZE = 25


class ObservationDatasetSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_PATH.exists():
            raise unittest.SkipTest(f"Dataset artifact not found: {DATASET_PATH}")

        cls.sample = pd.read_parquet(
            DATASET_PATH,
            columns=["integrand_prefix", "integral_prefix"],
        ).head(SAMPLE_SIZE)

    def test_build_observations_on_dataset_sample(self) -> None:
        observations = []
        finite_positive = 0
        symbolic_zero_rows = 0

        for row_index, row in enumerate(self.sample.itertuples(index=False)):
            target = parse_prefix_string(str(row.integrand_prefix))
            current = parse_prefix_string(str(row.integral_prefix))

            with self.subTest(row_index=row_index):
                observation = build_observation(target, current, residual_mode="both")
                observations.append(observation)

                self.assertNotEqual(observation.status, "derivative_failed")
                self.assertIsNotNone(observation.current_derivative)
                self.assertIsNotNone(observation.numeric_probes)

                assert observation.numeric_probes is not None
                if observation.numeric_probes.fraction_finite > 0.0:
                    finite_positive += 1

                if observation.symbolic_residual is None:
                    self.assertTrue(
                        any(warning.startswith("symbolic_residual_failed:") for warning in observation.warnings)
                        or any(warning.startswith("target_integrand_sympy_failed:") for warning in observation.warnings)
                    )
                    continue

                if serialize_prefix_string(observation.symbolic_residual) == "INT+ 0":
                    symbolic_zero_rows += 1
                    if observation.numeric_probes.mean_abs_residual is not None:
                        self.assertLessEqual(observation.numeric_probes.mean_abs_residual, 1e-8)
                    if observation.numeric_probes.max_abs_residual is not None:
                        self.assertLessEqual(observation.numeric_probes.max_abs_residual, 1e-8)

        self.assertEqual(len(observations), SAMPLE_SIZE)
        self.assertTrue(all(observation.current_derivative is not None for observation in observations))
        self.assertTrue(all(observation.numeric_probes is not None for observation in observations))
        self.assertGreaterEqual(finite_positive, 20)
        self.assertGreater(symbolic_zero_rows, 0)


if __name__ == "__main__":
    unittest.main()
