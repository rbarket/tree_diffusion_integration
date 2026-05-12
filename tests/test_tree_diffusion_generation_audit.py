from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.tree_diffusion.audit_generation import main, run_generation_audit
from tests.tree_diffusion_test_utils import write_toy_parquet


class TreeDiffusionGenerationAuditTests(unittest.TestCase):
    def test_generation_audit_writes_summary_and_validates_labels(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = write_toy_parquet(work_dir / "toy.parquet")
            output = work_dir / "audit.json"

            summary = run_generation_audit(
                data=parquet,
                num_examples=6,
                output=output,
                seed=123,
                sigma_small=2,
                smax=2,
                rho=0.0,
                residual_mode="both",
                validate_labels=True,
            )

            self.assertTrue(output.exists())
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["total_attempted"], 6)
            self.assertEqual(loaded["total_attempted"], 6)
            self.assertGreater(loaded["total_success"], 0)
            for field in (
                "failure_by_exception_type",
                "observation_status_counts",
                "warning_counts",
                "input_length",
                "target_length",
                "selected_node_id",
                "distance_before",
                "distance_after",
                "label_validation_failure_count",
                "nonincreasing_distance_rate",
                "strict_improvement_rate",
                "derivative_missing_rate",
                "residual_missing_rate",
                "numeric_missing_rate",
            ):
                self.assertIn(field, loaded)
            self.assertTrue(loaded["validate_labels"])
            self.assertGreaterEqual(loaded["nonincreasing_distance_rate"], 0.0)
            self.assertLessEqual(loaded["nonincreasing_distance_rate"], 1.0)

    def test_generation_audit_cli_runs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = write_toy_parquet(work_dir / "toy.parquet")
            output = work_dir / "audit_cli.json"

            result = main(
                [
                    "--data",
                    str(parquet),
                    "--num-examples",
                    "4",
                    "--output",
                    str(output),
                    "--seed",
                    "321",
                    "--smax",
                    "2",
                    "--rho",
                    "0.0",
                    "--validate-labels",
                    "true",
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
