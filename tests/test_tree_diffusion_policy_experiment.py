from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.tree_diffusion.dataset import load_integration_pairs_from_parquet
from src.tree_diffusion.experiments.policy_validation_experiment import (
    compare_policy_experiment_summaries,
    load_policy_experiment_config,
    run_policy_experiment,
)
from src.training.workflows.tree_diffusion import split_pairs_for_training
from tests.tree_diffusion_test_utils import tiny_training_config_values, write_toy_parquet


class TreeDiffusionPolicyExperimentTests(unittest.TestCase):
    def test_experiment_config_loading_and_validation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            config_path = _write_experiment_config(work_dir / "config.json", parquet)

            config = load_policy_experiment_config(config_path)

            self.assertEqual(config.experiment_name, "unit_policy_experiment")
            self.assertEqual(config.training.train_data, str(parquet))
            self.assertEqual(config.final_eval.val_batches, 50)
            self.assertEqual(config.final_eval.diagnostic_batches, 20)
            self.assertEqual(config.final_eval.checkpoint, "best")

            root_unknown = _experiment_config_dict(parquet)
            root_unknown["surprise"] = True
            (work_dir / "root_unknown.json").write_text(json.dumps(root_unknown), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown experiment"):
                load_policy_experiment_config(work_dir / "root_unknown.json")

            training_unknown = _experiment_config_dict(parquet)
            training_unknown["training"]["surprise"] = True
            (work_dir / "training_unknown.json").write_text(
                json.dumps(training_unknown),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unknown training"):
                load_policy_experiment_config(work_dir / "training_unknown.json")

            bad_steps = _experiment_config_dict(parquet)
            bad_steps["training"]["num_epochs"] = 0
            (work_dir / "bad_steps.json").write_text(json.dumps(bad_steps), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "num_epochs"):
                load_policy_experiment_config(work_dir / "bad_steps.json")

            bad_batch = _experiment_config_dict(parquet)
            bad_batch["training"]["batch_size"] = 0
            (work_dir / "bad_batch.json").write_text(json.dumps(bad_batch), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch_size"):
                load_policy_experiment_config(work_dir / "bad_batch.json")

            bad_timeout = _experiment_config_dict(parquet)
            bad_timeout["training"]["observation_timeout_seconds"] = 0.0
            (work_dir / "bad_timeout.json").write_text(json.dumps(bad_timeout), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "observation_timeout_seconds"):
                load_policy_experiment_config(work_dir / "bad_timeout.json")

            bad_final_eval = _experiment_config_dict(parquet)
            bad_final_eval["final_eval"] = {"val_batches": 0}
            (work_dir / "bad_final_eval.json").write_text(
                json.dumps(bad_final_eval),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "final_eval.val_batches"):
                load_policy_experiment_config(work_dir / "bad_final_eval.json")

    def test_tiny_experiment_runs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            output_dir = work_dir / "run"
            config_path = _write_experiment_config(
                work_dir / "config.json",
                parquet,
                output_dir=output_dir,
                final_eval={"val_batches": 1, "diagnostic_batches": 1, "checkpoint": "best"},
            )

            summary = run_policy_experiment(config_path)

            self.assertIsInstance(summary, dict)
            self.assertTrue((output_dir / "experiment_summary.json").exists())
            self.assertTrue((output_dir / "metrics.jsonl").exists())
            self.assertTrue((output_dir / "checkpoint_step_latest.pt").exists())
            for key in (
                "final_val_loss",
                "final_val_position_accuracy",
                "final_val_token_accuracy",
                "final_diagnostic_valid_position_rate",
                "final_diagnostic_parseable_replacement_rate",
                "final_diagnostic_applicable_edit_rate",
                "final_diagnostic_structural_improvement_rate",
                "final_diagnostic_exact_target_rate",
            ):
                self.assertIn(key, summary)
            for key in (
                "final_diagnostic_valid_position_rate",
                "final_diagnostic_parseable_replacement_rate",
                "final_diagnostic_applicable_edit_rate",
                "final_diagnostic_structural_improvement_rate",
                "final_diagnostic_exact_target_rate",
            ):
                self.assertGreaterEqual(float(summary[key]), 0.0)
                self.assertLessEqual(float(summary[key]), 1.0)
            self.assertTrue(math.isfinite(float(summary["final_train_loss"])))
            self.assertTrue(math.isfinite(float(summary["final_val_loss"])))

    def test_held_out_split_helper_is_deterministic_and_non_overlapping(self) -> None:
        with TemporaryDirectory() as temp_dir:
            parquet = _write_parquet(Path(temp_dir) / "toy.parquet")
            pairs = load_integration_pairs_from_parquet(parquet, limit=5)

            train_a, val_a = split_pairs_for_training(
                pairs,
                val_fraction=0.4,
                seed=99,
                train_limit=5,
                val_limit=2,
            )
            train_b, val_b = split_pairs_for_training(
                pairs,
                val_fraction=0.4,
                seed=99,
                train_limit=5,
                val_limit=2,
            )

            train_indices = {pair.index for pair in train_a}
            val_indices = {pair.index for pair in val_a}
            self.assertEqual(train_indices & val_indices, set())
            self.assertEqual([pair.index for pair in train_a], [pair.index for pair in train_b])
            self.assertEqual([pair.index for pair in val_a], [pair.index for pair in val_b])

    def test_final_evaluation_records_checkpoint_used(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            output_dir = work_dir / "run"
            config_path = _write_experiment_config(
                work_dir / "config.json",
                parquet,
                output_dir=output_dir,
                final_eval={"val_batches": 1, "diagnostic_batches": 1, "checkpoint": "best"},
            )

            summary = run_policy_experiment(config_path)

            self.assertTrue(summary["best_checkpoint"] or summary["last_checkpoint"])
            self.assertIn(summary["final_eval_checkpoint_kind"], {"best", "last_fallback"})
            self.assertTrue(Path(summary["final_eval_checkpoint"]).exists())

    def test_resume_from_override_is_supported(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            resume_path = work_dir / "checkpoint_step_latest.pt"
            resume_path.write_bytes(b"placeholder")
            config_path = _write_experiment_config(work_dir / "config.json", parquet)

            config = load_policy_experiment_config(
                config_path,
                overrides={
                    "resume_from": str(resume_path),
                    "num_workers": 3,
                    "observation_timeout_seconds": 7.5,
                },
            )

            self.assertEqual(config.training.resume_from, str(resume_path))
            self.assertEqual(config.training.num_workers, 3)
            self.assertEqual(config.training.observation_timeout_seconds, 7.5)

    def test_compare_summaries_utility(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            first = work_dir / "first.json"
            second = work_dir / "second.json"
            _write_summary(first, experiment_name="first", residual_mode="both")
            _write_summary(second, experiment_name="second", residual_mode="none")

            rows = compare_policy_experiment_summaries([first, second])

            self.assertEqual([row["experiment_name"] for row in rows], ["first", "second"])
            self.assertEqual([row["residual_mode"] for row in rows], ["both", "none"])
            for field in (
                "best_val_loss",
                "final_val_position_accuracy",
                "final_val_token_accuracy",
                "valid_position_rate",
                "applicable_edit_rate",
                "structural_improvement_rate",
                "exact_target_rate",
            ):
                self.assertIn(field, rows[0])


def _write_parquet(path: Path) -> Path:
    return write_toy_parquet(path, include_zero_row=True)


def _experiment_config_dict(
    parquet: Path,
    *,
    output_dir: Path | None = None,
    final_eval: dict | None = None,
) -> dict:
    return {
        "experiment_name": "unit_policy_experiment",
        "training": tiny_training_config_values(parquet, output_dir=output_dir),
        **({"final_eval": final_eval} if final_eval is not None else {}),
    }


def _write_experiment_config(
    path: Path,
    parquet: Path,
    *,
    output_dir: Path | None = None,
    final_eval: dict | None = None,
) -> Path:
    path.write_text(
        json.dumps(_experiment_config_dict(parquet, output_dir=output_dir, final_eval=final_eval)),
        encoding="utf-8",
    )
    return path


def _write_summary(path: Path, *, experiment_name: str, residual_mode: str) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "residual_mode": residual_mode,
                "best_val_loss": 1.0,
                "final_val_position_accuracy": 0.25,
                "final_val_token_accuracy": 0.5,
                "final_diagnostic_valid_position_rate": 0.3,
                "final_diagnostic_applicable_edit_rate": 0.2,
                "final_diagnostic_structural_improvement_rate": 0.1,
                "final_diagnostic_exact_target_rate": 0.05,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
