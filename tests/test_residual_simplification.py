from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.dataset import IntegrationPair, TreeDiffusionIterableDataset
from src.tree_diffusion.observation import compute_symbolic_residual
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.training_examples import generate_training_example
from src.training.workflows.tree_diffusion import load_training_config


class ResidualSimplificationTests(unittest.TestCase):
    def test_simplified_residual_preserves_additive_constant(self) -> None:
        residual = compute_symbolic_residual(
            current_derivative=parse_prefix_string("add x INT+ 1"),
            target_integrand=parse_prefix_string("x"),
            simplify_residual=True,
        )

        self.assertEqual(serialize_prefix_string(residual), "INT+ 1")

    def test_simplification_can_visibly_reduce_symbolic_residual(self) -> None:
        current_derivative = parse_prefix_string("add pow sin x INT+ 2 pow cos x INT+ 2")
        target_integrand = parse_prefix_string("INT+ 0")

        simplified = compute_symbolic_residual(
            current_derivative=current_derivative,
            target_integrand=target_integrand,
            simplify_residual=True,
        )
        unsimplified = compute_symbolic_residual(
            current_derivative=current_derivative,
            target_integrand=target_integrand,
            simplify_residual=False,
        )

        self.assertEqual(serialize_prefix_string(simplified), "INT+ 1")
        self.assertNotEqual(serialize_prefix_string(unsimplified), "INT+ 1")

    def test_training_example_accepts_simplification_flag(self) -> None:
        target_integrand = parse_prefix_string("INT+ 0")
        target_antiderivative = parse_prefix_string("INT+ 0")
        current = parse_prefix_string("add div pow sin x INT+ 3 INT+ 3 x")

        for simplify in (True, False):
            with self.subTest(simplify=simplify):
                with patch(
                    "src.tree_diffusion.training_examples.generate_current_candidate",
                    return_value=(current, 1, False),
                ):
                    example = generate_training_example(
                        target_integrand,
                        target_antiderivative,
                        tokenizer=TreeDiffusionTokenizer(max_positions=64),
                        rng=random.Random(1),
                        residual_mode="symbolic",
                        sigma_small=3,
                        simplify_symbolic_residual=simplify,
                    )

                self.assertIsNotNone(example.observation.symbolic_residual)

    def test_dataset_accepts_simplification_flag(self) -> None:
        dataset = TreeDiffusionIterableDataset(
            [
                IntegrationPair(
                    target_integrand=parse_prefix_string("pow x INT+ 2"),
                    target_antiderivative=parse_prefix_string("div pow x INT+ 3 INT+ 3"),
                )
            ],
            tokenizer=TreeDiffusionTokenizer(max_positions=64),
            residual_mode="both",
            simplify_symbolic_residual=False,
            max_input_length=128,
            max_target_length=32,
            rho=0.0,
            smax=1,
        )

        item = next(iter(dataset))
        self.assertIn("input_tokens", item)

    def test_training_config_preserves_simplification_choice(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = work_dir / "toy.parquet"
            pd.DataFrame(
                [{"integrand_prefix": "x", "integral_prefix": "div pow x INT+ 2 INT+ 2"}]
            ).to_parquet(parquet)
            config_path = work_dir / "config.json"
            config = _minimal_training_config(parquet)
            config["simplify_symbolic_residual"] = False
            config_path.write_text(json.dumps(config), encoding="utf-8")

            loaded = load_training_config(config_path)

        self.assertFalse(loaded.simplify_symbolic_residual)


def _minimal_training_config(parquet: Path) -> dict:
    return {
        "train_data": str(parquet),
        "device": "cpu",
        "num_epochs": 1,
        "batch_size": 1,
        "num_workers": 0,
        "train_limit": 1,
        "val_limit": 1,
        "val_fraction": 0.0,
        "max_input_length": 128,
        "max_target_length": 32,
        "max_positions": 64,
        "d_model": 32,
        "n_heads": 4,
        "d_ff": 64,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "log_every": 1,
        "val_every": 1,
        "checkpoint_every": 1,
        "val_batches": 1,
        "diagnostic_batches": 1,
    }


if __name__ == "__main__":
    unittest.main()
