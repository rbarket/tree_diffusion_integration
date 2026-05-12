from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion.dataset import IntegrationPair
from src.tree_diffusion.model import TreeDiffusionModelConfig, TreeDiffusionPolicyModel
from src.tree_diffusion.precompute_dataset import (
    TreeDiffusionPrecomputeConfig,
    precompute_tree_diffusion_dataset,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


TOY_PREFIX_ROWS = [
    {"integrand_prefix": "pow x INT+ 2", "integral_prefix": "div pow x INT+ 3 INT+ 3"},
    {"integrand_prefix": "cos x", "integral_prefix": "sin x"},
    {"integrand_prefix": "exp x", "integral_prefix": "exp x"},
    {"integrand_prefix": "INT+ 1", "integral_prefix": "x"},
]
EXTENDED_TOY_PREFIX_ROWS = [
    *TOY_PREFIX_ROWS,
    {
        "integrand_prefix": "add x INT+ 1",
        "integral_prefix": "add div pow x INT+ 2 INT+ 2 x",
    },
]


def write_toy_parquet(path: Path, *, include_zero_row: bool = False) -> Path:
    rows = list(TOY_PREFIX_ROWS)
    if include_zero_row:
        rows.append({"integrand_prefix": "INT+ 0", "integral_prefix": "INT+ 0"})
    pd.DataFrame(rows).to_parquet(path)
    return path


def write_extended_toy_parquet(path: Path) -> Path:
    pd.DataFrame(EXTENDED_TOY_PREFIX_ROWS).to_parquet(path)
    return path


def run_tiny_precompute(work_dir: Path) -> Path:
    parquet = write_extended_toy_parquet(work_dir / "toy.parquet")
    output_dir = work_dir / "precomputed"
    config = TreeDiffusionPrecomputeConfig(
        input_data=str(parquet),
        output_dir=str(output_dir),
        train_limit=4,
        val_limit=1,
        val_fraction=0.2,
        examples_per_pair_train=2,
        examples_per_pair_val=2,
        shard_size=2,
        overwrite=True,
        sigma_small=2,
        smax=3,
        rho=0.2,
        max_input_length=256,
        max_target_length=64,
        max_positions=128,
        max_attempts=32,
    )
    precompute_tree_diffusion_dataset(config)
    return output_dir


def sample_integration_pairs() -> list[IntegrationPair]:
    return [
        IntegrationPair(
            target_integrand=parse_prefix_string("pow x INT+ 2"),
            target_antiderivative=parse_prefix_string("div pow x INT+ 3 INT+ 3"),
            source="unit",
            index=0,
        ),
        IntegrationPair(
            target_integrand=parse_prefix_string("cos x"),
            target_antiderivative=parse_prefix_string("sin x"),
            source="unit",
            index=1,
        ),
        IntegrationPair(
            target_integrand=parse_prefix_string("exp x"),
            target_antiderivative=parse_prefix_string("exp x"),
            source="unit",
            index=2,
        ),
    ]


def small_policy_model(
    tokenizer: TreeDiffusionTokenizer,
    *,
    max_input_length: int = 128,
    max_target_length: int = 32,
    d_model: int = 32,
    n_heads: int = 4,
    d_ff: int = 64,
    n_encoder_layers: int = 1,
    n_decoder_layers: int = 1,
    dropout: float = 0.0,
    **overrides: Any,
) -> TreeDiffusionPolicyModel:
    values = {
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_id,
        "bos_token_id": tokenizer.bos_id,
        "eos_token_id": tokenizer.eos_id,
        "max_input_length": max_input_length,
        "max_target_length": max_target_length,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "n_encoder_layers": n_encoder_layers,
        "n_decoder_layers": n_decoder_layers,
        "dropout": dropout,
    }
    values.update(overrides)
    return TreeDiffusionPolicyModel(TreeDiffusionModelConfig(**values))


def tiny_training_config_values(
    parquet: Path,
    *,
    output_dir: Path | None = None,
    num_epochs: int = 1,
) -> dict[str, Any]:
    return {
        "train_data": str(parquet),
        "val_data": None,
        "output_dir": str(output_dir or parquet.parent / "run"),
        "train_limit": 4,
        "val_limit": 2,
        "val_fraction": 0.25,
        "seed": 123,
        "device": "cpu",
        "num_epochs": num_epochs,
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
