from __future__ import annotations

import json
import math
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.tree_diffusion.dataset import load_integration_pairs_from_parquet
from src.tree_diffusion.experiments.policy_validation_experiment import (
    compare_policy_experiment_summaries,
    load_policy_experiment_config,
    run_policy_experiment,
)
from src.training.workflows.tree_diffusion import split_pairs_for_training


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
            bad_steps["training"]["max_steps"] = 0
            (work_dir / "bad_steps.json").write_text(json.dumps(bad_steps), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "max_steps"):
                load_policy_experiment_config(work_dir / "bad_steps.json")

            bad_batch = _experiment_config_dict(parquet)
            bad_batch["training"]["batch_size"] = 0
            (work_dir / "bad_batch.json").write_text(json.dumps(bad_batch), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch_size"):
                load_policy_experiment_config(work_dir / "bad_batch.json")

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
            self.assertTrue((output_dir / "checkpoint_last.pt").exists())
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

    def test_residual_ablation_configs_are_valid_and_matched(self) -> None:
        both = load_policy_experiment_config(
            "config/experiments/tree_diffusion_policy_residual_ablation_both.json"
        )
        none = load_policy_experiment_config(
            "config/experiments/tree_diffusion_policy_residual_ablation_none.json"
        )

        self.assertEqual(both.training.residual_mode, "both")
        self.assertEqual(none.training.residual_mode, "none")

        both_training = asdict(both.training)
        none_training = asdict(none.training)
        for key in ("output_dir", "residual_mode"):
            both_training.pop(key)
            none_training.pop(key)
        self.assertEqual(both_training, none_training)
        self.assertEqual(both.final_eval, none.final_eval)

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
    pd.DataFrame(
        [
            {"integrand_prefix": "pow x INT+ 2", "integral_prefix": "div pow x INT+ 3 INT+ 3"},
            {"integrand_prefix": "cos x", "integral_prefix": "sin x"},
            {"integrand_prefix": "exp x", "integral_prefix": "exp x"},
            {"integrand_prefix": "INT+ 1", "integral_prefix": "x"},
            {"integrand_prefix": "INT+ 0", "integral_prefix": "INT+ 0"},
        ]
    ).to_parquet(path)
    return path


def _experiment_config_dict(
    parquet: Path,
    *,
    output_dir: Path | None = None,
    final_eval: dict | None = None,
) -> dict:
    return {
        "experiment_name": "unit_policy_experiment",
        "training": {
            "train_data": str(parquet),
            "val_data": None,
            "output_dir": str(output_dir or parquet.parent / "run"),
            "train_limit": 4,
            "val_limit": 2,
            "val_fraction": 0.25,
            "seed": 123,
            "device": "cpu",
            "max_steps": 2,
            "batch_size": 2,
            "num_workers": 0,
            "sigma_small": 2,
            "smax": 3,
            "rho": 0.2,
            "residual_mode": "both",
            "max_input_length": 128,
            "max_target_length": 32,
            "max_positions": 128,
            "max_random_size": None,
            "max_attempts": 32,
            "d_model": 32,
            "n_heads": 4,
            "d_ff": 64,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "dropout": 0.0,
            "norm_first": True,
            "tie_embeddings": True,
            "lr": 0.003,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "grad_clip_norm": 1.0,
            "log_every": 1,
            "val_every": 1,
            "checkpoint_every": 2,
            "val_batches": 1,
            "diagnostic_batches": 1,
            "resume_from": None,
            "save_best": True,
            "save_last": True,
        },
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
