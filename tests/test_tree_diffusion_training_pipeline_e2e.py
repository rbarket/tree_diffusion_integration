from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch

from src.tree_diffusion.audit_training_pipeline import main as audit_main
from src.tree_diffusion.dataset import (
    IntegrationPair,
    load_integration_pairs_from_parquet,
    make_tree_diffusion_dataloader,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.train_step import (
    inspect_batch_predictions,
    tree_diffusion_eval_step,
    tree_diffusion_train_step,
    validate_tree_diffusion_batch,
)
from tests.tree_diffusion_test_utils import sample_integration_pairs, small_policy_model


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_prefix_filtered.parquet"


class TreeDiffusionTrainingPipelineE2ETests(unittest.TestCase):
    def test_full_preflight_on_hand_pairs(self) -> None:
        torch.manual_seed(123)
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        batch = _batch(_pairs(), tokenizer=tokenizer, batch_size=4)

        validate_tree_diffusion_batch(batch, pad_token_id=tokenizer.pad_id, require_metadata=True)
        eval_output = tree_diffusion_eval_step(model, batch, tokenizer=tokenizer)
        train_output = tree_diffusion_train_step(model, batch, optimizer, tokenizer=tokenizer, grad_clip_norm=1.0)
        predictions = inspect_batch_predictions(model, batch, tokenizer, num_examples=2)

        self.assertTrue(torch.isfinite(torch.tensor(eval_output.loss)).item())
        self.assertTrue(torch.isfinite(torch.tensor(train_output.loss)).item())
        self.assertIsNotNone(train_output.grad_norm)
        assert train_output.grad_norm is not None
        self.assertGreater(train_output.grad_norm, 0.0)
        self.assertEqual(batch["input_ids"].shape, (4, 128))
        self.assertEqual(batch["target_ids"].shape, (4, 32))
        self.assertGreater(len(predictions), 0)

    def test_real_dataset_smoke_if_available(self) -> None:
        if not DATASET_PATH.exists():
            raise unittest.SkipTest(f"Dataset artifact not found: {DATASET_PATH}")

        torch.manual_seed(123)
        pairs = load_integration_pairs_from_parquet(DATASET_PATH, limit=16)
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        batch = _batch(pairs, tokenizer=tokenizer, batch_size=4)

        eval_output = tree_diffusion_eval_step(model, batch, tokenizer=tokenizer)
        train_output = tree_diffusion_train_step(model, batch, optimizer, tokenizer=tokenizer, grad_clip_norm=1.0)

        self.assertTrue(torch.isfinite(torch.tensor(eval_output.loss)).item())
        self.assertTrue(torch.isfinite(torch.tensor(train_output.loss)).item())
        self.assertIsNotNone(train_output.grad_norm)
        assert train_output.grad_norm is not None
        self.assertGreater(train_output.grad_norm, 0.0)

    def test_audit_main_runs_on_temp_parquet_config(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet_path = work_dir / "toy.parquet"
            config_path = work_dir / "preflight.json"
            pd.DataFrame(
                [
                    {"integrand_prefix": "pow x INT+ 2", "integral_prefix": "div pow x INT+ 3 INT+ 3"},
                    {"integrand_prefix": "cos x", "integral_prefix": "sin x"},
                    {"integrand_prefix": "exp x", "integral_prefix": "exp x"},
                ]
            ).to_parquet(parquet_path)
            config_path.write_text(
                json.dumps(
                    {
                        "data": {"parquet": str(parquet_path), "num_pairs": 3},
                        "dataset": {
                            "batch_size": 2,
                            "max_input_length": 128,
                            "max_target_length": 32,
                            "sigma_small": 2,
                            "smax": 2,
                            "rho": 0.0,
                            "residual_mode": "both",
                            "num_workers": 0,
                            "shuffle_pairs": False,
                        },
                        "model": {
                            "d_model": 32,
                            "n_heads": 4,
                            "d_ff": 64,
                            "n_encoder_layers": 1,
                            "n_decoder_layers": 1,
                            "dropout": 0.0,
                        },
                        "training": {"lr": 0.003, "grad_clip_norm": 1.0},
                        "runtime": {"device": "cpu", "seed": 123},
                        "audit": {
                            "steps": 2,
                            "num_prediction_examples": 2,
                            "require_loss_decrease_after_steps": 5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(audit_main(["--config", str(config_path)]), 0)

    def test_compatibility_wrapper_imports(self) -> None:
        import tree_diffusion.audit_training_pipeline as compat

        self.assertTrue(callable(compat.main))


def _pairs() -> list[IntegrationPair]:
    return sample_integration_pairs()


def _batch(
    pairs: list[IntegrationPair],
    *,
    tokenizer: TreeDiffusionTokenizer,
    batch_size: int,
) -> dict:
    loader = make_tree_diffusion_dataloader(
        pairs,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=0,
        sigma_small=2,
        smax=3,
        rho=0.2,
        max_input_length=128,
        max_target_length=32,
        simplify_symbolic_residual=False,
        base_seed=123,
        shuffle_pairs=False,
        include_metadata=True,
    )
    return next(iter(loader))


def _small_model(tokenizer: TreeDiffusionTokenizer):
    return small_policy_model(tokenizer)


if __name__ == "__main__":
    unittest.main()
