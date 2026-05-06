from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch

from src.tree_diffusion.dataset import load_integration_pairs_from_parquet, make_tree_diffusion_dataloader
from src.tree_diffusion.model import TreeDiffusionModelConfig, TreeDiffusionPolicyModel
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.training.workflows.tree_diffusion import (
    TreeDiffusionTrainingConfig,
    evaluate_tree_diffusion_policy,
    load_checkpoint,
    load_training_config,
    main,
    save_checkpoint,
    train_tree_diffusion_policy,
)


class TreeDiffusionTrainingWorkflowTests(unittest.TestCase):
    def test_config_loading_and_validation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            config_path = _write_config(work_dir / "config.json", parquet, output_dir=work_dir / "run")

            config = load_training_config(config_path)

            self.assertEqual(config.train_data, str(parquet))
            self.assertEqual(config.batch_size, 2)
            self.assertEqual(config.betas, (0.9, 0.999))

            bad_path = _write_config(work_dir / "missing.json", work_dir / "missing.parquet")
            with self.assertRaisesRegex(ValueError, "train_data"):
                load_training_config(bad_path)

            invalid_fraction = _config_dict(parquet)
            invalid_fraction["val_fraction"] = 1.0
            (work_dir / "invalid_fraction.json").write_text(json.dumps(invalid_fraction), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "val_fraction"):
                load_training_config(work_dir / "invalid_fraction.json")

            invalid_steps = _config_dict(parquet)
            invalid_steps["max_steps"] = 0
            (work_dir / "invalid_steps.json").write_text(json.dumps(invalid_steps), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "max_steps"):
                load_training_config(work_dir / "invalid_steps.json")

            unknown = _config_dict(parquet)
            unknown["surprise"] = True
            (work_dir / "unknown.json").write_text(json.dumps(unknown), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_training_config(work_dir / "unknown.json")

    def test_train_tree_diffusion_policy_smoke(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            config = TreeDiffusionTrainingConfig(**_config_dict(parquet, output_dir=work_dir / "run"))

            summary = train_tree_diffusion_policy(config)

            self.assertEqual(summary["final_step"], 2)
            self.assertTrue(Path(summary["output_dir"]).exists())
            self.assertTrue((Path(summary["output_dir"]) / "metrics.jsonl").exists())
            self.assertTrue((Path(summary["output_dir"]) / "checkpoint_last.pt").exists())
            rows = _read_metrics(Path(summary["output_dir"]) / "metrics.jsonl")
            self.assertTrue(any(row["split"] == "train" for row in rows))
            self.assertTrue(any(row["split"] == "val" for row in rows))
            self.assertTrue(any(row["split"] == "diagnostic" for row in rows))
            for row in rows:
                if "loss" in row:
                    self.assertTrue(math.isfinite(float(row["loss"])))

    def test_checkpoint_save_load(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            config = TreeDiffusionTrainingConfig(**_config_dict(parquet, output_dir=work_dir / "run"))
            tokenizer = TreeDiffusionTokenizer(max_positions=config.max_positions)
            model = _small_model(tokenizer)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
            path = work_dir / "checkpoint.pt"

            saved = [parameter.detach().clone() for parameter in model.parameters()]
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=7,
                best_val_loss=1.25,
                tokenizer=tokenizer,
                extra={"note": "unit"},
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1.0)

            checkpoint = load_checkpoint(path, model=model, optimizer=None)

            for parameter, expected in zip(model.parameters(), saved):
                self.assertTrue(torch.allclose(parameter.detach(), expected))
            self.assertEqual(checkpoint["step"], 7)
            self.assertIn("config", checkpoint)
            self.assertEqual(checkpoint["tokenizer"]["vocab_size"], tokenizer.vocab_size)

    def test_resume_training(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            output_dir = work_dir / "run"
            first = TreeDiffusionTrainingConfig(**_config_dict(parquet, output_dir=output_dir, max_steps=2))
            train_tree_diffusion_policy(first)
            checkpoint = output_dir / "checkpoint_last.pt"

            resumed_values = _config_dict(
                parquet,
                output_dir=output_dir,
                max_steps=3,
            )
            resumed_values["resume_from"] = str(checkpoint)
            summary = train_tree_diffusion_policy(TreeDiffusionTrainingConfig(**resumed_values))

            self.assertEqual(summary["final_step"], 3)
            rows = _read_metrics(output_dir / "metrics.jsonl")
            self.assertTrue(any(row["split"] == "train" and row["step"] == 3 for row in rows))

    def test_evaluate_tree_diffusion_policy_returns_averages(self) -> None:
        with TemporaryDirectory() as temp_dir:
            parquet = _write_parquet(Path(temp_dir) / "toy.parquet")
            tokenizer = TreeDiffusionTokenizer(max_positions=128)
            pairs = load_integration_pairs_from_parquet(parquet, limit=4)
            loader = make_tree_diffusion_dataloader(
                pairs,
                tokenizer=tokenizer,
                batch_size=2,
                num_workers=0,
                max_input_length=128,
                max_target_length=32,
                base_seed=123,
                include_metadata=True,
            )
            model = _small_model(tokenizer)

            metrics = evaluate_tree_diffusion_policy(
                model,
                loader,
                tokenizer=tokenizer,
                device="cpu",
                num_batches=2,
            )

            for key in (
                "loss",
                "position_accuracy",
                "token_accuracy",
                "input_length_mean",
                "target_length_mean",
                "random_init_fraction",
                "num_mutations_mean",
            ):
                self.assertIn(key, metrics)
                self.assertTrue(math.isfinite(metrics[key]))

    def test_cli_main_runs_with_overrides(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            config_path = _write_config(work_dir / "config.json", parquet, output_dir=work_dir / "unused")
            output_dir = work_dir / "cli_run"

            result = main(
                [
                    "--config",
                    str(config_path),
                    "--max-steps",
                    "1",
                    "--batch-size",
                    "2",
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cpu",
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "metrics.jsonl").exists())
            self.assertTrue((output_dir / "checkpoint_last.pt").exists())

    def test_compatibility_wrapper_imports(self) -> None:
        import training.workflows.tree_diffusion as compat

        self.assertTrue(callable(compat.main))

    def test_deterministic_short_run(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            run_a = work_dir / "run_a"
            run_b = work_dir / "run_b"

            train_tree_diffusion_policy(
                TreeDiffusionTrainingConfig(**_config_dict(parquet, output_dir=run_a, max_steps=2))
            )
            train_tree_diffusion_policy(
                TreeDiffusionTrainingConfig(**_config_dict(parquet, output_dir=run_b, max_steps=2))
            )

            first_train_a = _first_metric(run_a / "metrics.jsonl", "train")
            first_train_b = _first_metric(run_b / "metrics.jsonl", "train")
            first_val_a = _first_metric(run_a / "metrics.jsonl", "val")
            first_val_b = _first_metric(run_b / "metrics.jsonl", "val")
            self.assertAlmostEqual(first_train_a["loss"], first_train_b["loss"], places=6)
            self.assertAlmostEqual(first_val_a["loss"], first_val_b["loss"], places=6)


def _write_parquet(path: Path) -> Path:
    pd.DataFrame(
        [
            {"integrand_prefix": "pow x INT+ 2", "integral_prefix": "div pow x INT+ 3 INT+ 3"},
            {"integrand_prefix": "cos x", "integral_prefix": "sin x"},
            {"integrand_prefix": "exp x", "integral_prefix": "exp x"},
            {"integrand_prefix": "INT+ 1", "integral_prefix": "x"},
        ]
    ).to_parquet(path)
    return path


def _config_dict(
    parquet: Path,
    *,
    output_dir: Path | None = None,
    max_steps: int = 2,
) -> dict:
    return {
        "train_data": str(parquet),
        "val_data": None,
        "output_dir": str(output_dir or parquet.parent / "run"),
        "train_limit": 4,
        "val_limit": 2,
        "val_fraction": 0.25,
        "seed": 123,
        "device": "cpu",
        "max_steps": max_steps,
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
    }


def _write_config(path: Path, parquet: Path, *, output_dir: Path | None = None) -> Path:
    path.write_text(json.dumps(_config_dict(parquet, output_dir=output_dir)), encoding="utf-8")
    return path


def _small_model(tokenizer: TreeDiffusionTokenizer) -> TreeDiffusionPolicyModel:
    return TreeDiffusionPolicyModel(
        TreeDiffusionModelConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_id,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            max_input_length=128,
            max_target_length=32,
            d_model=32,
            n_heads=4,
            d_ff=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            dropout=0.0,
        )
    )


def _read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _first_metric(path: Path, split: str) -> dict:
    for row in _read_metrics(path):
        if row["split"] == split:
            return row
    raise AssertionError(f"No {split!r} metric row found.")


if __name__ == "__main__":
    unittest.main()
