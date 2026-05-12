from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch

from src.tree_diffusion.dataset import load_integration_pairs_from_parquet, make_tree_diffusion_dataloader
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.training.lightning.tree_diffusion_wandb import (
    GLOBAL_STEP_METRIC,
    build_tree_diffusion_wandb_tracker,
)
from src.training.workflows.tree_diffusion import (
    TreeDiffusionTrainingConfig,
    evaluate_tree_diffusion_policy,
    load_checkpoint,
    load_training_config,
    main,
    save_checkpoint,
    train_tree_diffusion_policy,
)
from tests.tree_diffusion_test_utils import (
    small_policy_model,
    tiny_training_config_values,
    write_toy_parquet,
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
            invalid_steps["num_epochs"] = 0
            (work_dir / "invalid_steps.json").write_text(json.dumps(invalid_steps), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "num_epochs"):
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
            self.assertTrue((Path(summary["output_dir"]) / "lightning" / "last.ckpt").exists())
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
            first = TreeDiffusionTrainingConfig(**_config_dict(parquet, output_dir=output_dir, num_epochs=1))
            train_tree_diffusion_policy(first)
            checkpoint = output_dir / "checkpoint_last.pt"
            legacy_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertIn("lightning_resume_ckpt", legacy_checkpoint)
            self.assertTrue(Path(legacy_checkpoint["lightning_resume_ckpt"]).exists())

            resumed_values = _config_dict(
                parquet,
                output_dir=output_dir,
                num_epochs=2,
            )
            resumed_values["resume_from"] = str(checkpoint)
            summary = train_tree_diffusion_policy(TreeDiffusionTrainingConfig(**resumed_values))

            self.assertEqual(summary["final_step"], 4)
            rows = _read_metrics(output_dir / "metrics.jsonl")
            self.assertTrue(any(row["split"] == "train" and row["step"] == 4 for row in rows))

    def test_resume_training_from_lightning_checkpoint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            output_dir = work_dir / "run"
            first = TreeDiffusionTrainingConfig(**_config_dict(parquet, output_dir=output_dir, num_epochs=1))
            train_tree_diffusion_policy(first)
            lightning_checkpoint = output_dir / "lightning" / "last.ckpt"

            resumed_values = _config_dict(
                parquet,
                output_dir=output_dir,
                num_epochs=2,
            )
            resumed_values["resume_from"] = str(lightning_checkpoint)
            summary = train_tree_diffusion_policy(TreeDiffusionTrainingConfig(**resumed_values))

            self.assertEqual(summary["final_step"], 4)
            self.assertTrue((output_dir / "checkpoint_last.pt").exists())

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
                    "--num-epochs",
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
                TreeDiffusionTrainingConfig(**_config_dict(parquet, output_dir=run_a, num_epochs=1))
            )
            train_tree_diffusion_policy(
                TreeDiffusionTrainingConfig(**_config_dict(parquet, output_dir=run_b, num_epochs=1))
            )

            first_train_a = _first_metric(run_a / "metrics.jsonl", "train")
            first_train_b = _first_metric(run_b / "metrics.jsonl", "train")
            first_val_a = _first_metric(run_a / "metrics.jsonl", "val")
            first_val_b = _first_metric(run_b / "metrics.jsonl", "val")
            self.assertAlmostEqual(first_train_a["loss"], first_train_b["loss"], places=6)
            self.assertAlmostEqual(first_val_a["loss"], first_val_b["loss"], places=6)

    def test_wandb_tracker_disabled_is_noop(self) -> None:
        cfg = _wandb_cfg(enable_wandb=False)
        tracker = build_tree_diffusion_wandb_tracker(cfg, _model_cfg())

        tracker.track_many({"train/loss": 1.0}, step=1)

        self.assertIsNone(tracker.run)

    def test_wandb_tracker_logs_prefixed_metrics_with_resume(self) -> None:
        captured_kwargs = {}

        class _FakeWandbSdk:
            @staticmethod
            def init(**kwargs):
                captured_kwargs.update(kwargs)
                return _FakeRun(run_id=str(kwargs.get("id")), run_name=str(kwargs.get("name")))

        cfg = _wandb_cfg(enable_wandb=True)
        with mock.patch(
            "src.training.lightning.tree_diffusion_wandb._import_wandb_sdk",
            return_value=_FakeWandbSdk(),
        ):
            tracker = build_tree_diffusion_wandb_tracker(
                cfg,
                _model_cfg(),
                run_id="resume-123",
                resume="allow",
            )

        tracker.track_many({"train/loss": 1.25, "train/lr": 0.003}, step=7)
        tracker.track_prefixed_metrics({"loss": 0.9, "position_accuracy": 0.4}, prefix="val", step=8)
        tracker.track_prefixed_metrics({"valid_position_rate": 0.5}, prefix="diagnostic", step=8)

        self.assertEqual(captured_kwargs["id"], "resume-123")
        self.assertEqual(captured_kwargs["resume"], "allow")
        self.assertEqual(captured_kwargs["project"], "tree-tests")
        self.assertIsNotNone(tracker.run)
        assert tracker.run is not None
        self.assertEqual(
            tracker.run.logged[0],
            {GLOBAL_STEP_METRIC: 7, "train/loss": 1.25, "train/lr": 0.003},
        )
        self.assertEqual(
            tracker.run.logged[1],
            {GLOBAL_STEP_METRIC: 8, "val/loss": 0.9, "val/position_accuracy": 0.4},
        )
        self.assertEqual(
            tracker.run.logged[2],
            {GLOBAL_STEP_METRIC: 8, "diagnostic/valid_position_rate": 0.5},
        )


def _write_parquet(path: Path) -> Path:
    return write_toy_parquet(path)


def _config_dict(
    parquet: Path,
    *,
    output_dir: Path | None = None,
    num_epochs: int = 1,
) -> dict:
    return tiny_training_config_values(parquet, output_dir=output_dir, num_epochs=num_epochs)


def _write_config(path: Path, parquet: Path, *, output_dir: Path | None = None) -> Path:
    path.write_text(json.dumps(_config_dict(parquet, output_dir=output_dir)), encoding="utf-8")
    return path


def _small_model(tokenizer: TreeDiffusionTokenizer):
    return small_policy_model(tokenizer)


def _read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _first_metric(path: Path, split: str) -> dict:
    for row in _read_metrics(path):
        if row["split"] == split:
            return row
    raise AssertionError(f"No {split!r} metric row found.")


class _FakeConfig(dict):
    def update(self, values, allow_val_change=False):  # type: ignore[override]
        self["_allow_val_change"] = bool(allow_val_change)
        super().update(values)


class _FakeRun:
    def __init__(self, *, run_id: str, run_name: str) -> None:
        self.id = run_id
        self.name = run_name
        self.logged: list[dict[str, float]] = []
        self.defined_metrics: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.config = _FakeConfig()

    def log(self, data):
        self.logged.append(dict(data))

    def define_metric(self, *args, **kwargs):
        self.defined_metrics.append((args, dict(kwargs)))

    def finish(self) -> None:
        pass


def _wandb_cfg(*, enable_wandb: bool) -> SimpleNamespace:
    return SimpleNamespace(
        enable_wandb=enable_wandb,
        wandb_project="tree-tests",
        wandb_run_name="tree-run",
        wandb_run_id=None,
        wandb_resume=None,
        wandb_entity=None,
        wandb_dir=None,
        wandb_mode="offline",
        batch_size=2,
        num_epochs=2,
        lr=0.003,
    )


def _model_cfg() -> SimpleNamespace:
    return SimpleNamespace(d_model=32, n_heads=4, d_ff=64)


if __name__ == "__main__":
    unittest.main()
