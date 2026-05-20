from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Sequence

from src.tree_diffusion.model import TreeDiffusionModelConfig


@dataclass
class TreeDiffusionTrainingConfig:
    train_data: str | None = None
    val_data: str | None = None
    precomputed_data_dir: str | None = None
    use_precomputed: bool = False
    output_dir: str = "runs/tree_diffusion"

    integrand_column: str = "integrand_prefix"
    integral_column: str = "integral_prefix"

    train_limit: int | None = None
    val_limit: int | None = 512
    val_fraction: float = 0.05

    seed: int = 123
    device: str = "auto"

    num_epochs: int = 1
    batch_size: int = 32
    num_workers: int = 0

    sigma_small: int = 2
    smax: int = 5
    rho: float = 0.2
    residual_mode: str = "both"
    simplify_symbolic_residual: bool = True
    max_input_length: int = 1024
    max_target_length: int = 128
    max_positions: int = 512
    max_random_size: int | None = None
    max_attempts: int = 32
    observation_timeout_seconds: float | None = 5.0
    allow_complex_constants: bool = False
    allow_distributional_unary_ops: bool = False
    excluded_random_tokens: tuple[str, ...] = ()
    validate_generated_labels: bool = False
    max_derivative_tokens: int | None = None
    max_residual_tokens: int | None = None

    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    dropout: float = 0.1
    norm_first: bool = True
    tie_embeddings: bool = True

    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip_norm: float | None = 1.0

    log_every: int = 50
    val_every: int = 500
    checkpoint_every: int = 1000
    val_batches: int = 20
    diagnostic_batches: int = 5
    diagnostic_timeout_seconds: float | None = 120.0
    diagnostic_example_timeout_seconds: float | None = 5.0
    diagnostic_numeric_timeout_seconds: float | None = 2.0

    resume_from: str | None = None
    save_best: bool = True
    save_last: bool = True

    accelerator: str | None = None
    devices: Any = None
    precision: str = "32-true"
    log_every_n_steps: int | None = None
    num_sanity_val_steps: int = 0
    enable_progress_bar: bool = True
    deterministic: bool = True

    enable_wandb: bool = False
    wandb_project: str = "tree_diffusion_train"
    wandb_run_name: str | None = None
    wandb_run_id: str | None = None
    wandb_resume: str | None = None
    wandb_entity: str | None = None
    wandb_dir: str | None = ".wandb"
    wandb_mode: str | None = None

    def __post_init__(self) -> None:
        self.betas = _normalize_betas(self.betas)
        self.excluded_random_tokens = tuple(str(token) for token in self.excluded_random_tokens)
        if self.wandb_resume is not None:
            normalized_wandb_resume = str(self.wandb_resume).strip().lower()
            if normalized_wandb_resume not in {"allow", "must", "never"}:
                raise ValueError("wandb_resume must be one of: allow, must, never.")
            self.wandb_resume = normalized_wandb_resume
        _validate_training_config(self)


def load_training_config(path: str | Path) -> TreeDiffusionTrainingConfig:
    return TreeDiffusionTrainingConfig(**_load_training_config_values(path))


def _load_training_config_values(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Training config root must be a JSON object.")

    known_fields = {field.name for field in fields(TreeDiffusionTrainingConfig)}
    unknown = sorted(set(raw) - known_fields)
    if unknown:
        raise ValueError(f"Unknown training config field(s): {', '.join(unknown)}.")
    return raw


def _validate_training_config(config: TreeDiffusionTrainingConfig) -> None:
    if _use_precomputed(config):
        if not config.precomputed_data_dir:
            raise ValueError("precomputed_data_dir is required when use_precomputed=True.")
        if not Path(config.precomputed_data_dir).exists():
            raise ValueError(f"precomputed_data_dir does not exist: {config.precomputed_data_dir}")
    else:
        if not config.train_data:
            raise ValueError("train_data is required unless precomputed_data_dir is set.")
        if not Path(config.train_data).exists():
            raise ValueError(f"train_data does not exist: {config.train_data}")
    if config.val_data is not None and not Path(config.val_data).exists():
        raise ValueError(f"val_data does not exist: {config.val_data}")
    if config.resume_from is not None and not Path(config.resume_from).exists():
        raise ValueError(f"resume_from does not exist: {config.resume_from}")

    if config.train_limit is not None and config.train_limit < 1:
        raise ValueError("train_limit must be >= 1 when provided.")
    if config.val_limit is not None and config.val_limit < 1:
        raise ValueError("val_limit must be >= 1 when provided.")
    if not 0.0 <= config.val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1.")
    if config.num_epochs < 1:
        raise ValueError("num_epochs must be >= 1.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if config.num_workers < 0:
        raise ValueError("num_workers must be >= 0.")
    if config.sigma_small < 1:
        raise ValueError("sigma_small must be >= 1.")
    if config.smax < 1:
        raise ValueError("smax must be >= 1.")
    if not 0.0 <= config.rho <= 1.0:
        raise ValueError("rho must satisfy 0 <= rho <= 1.")
    if config.max_positions < 1:
        raise ValueError("max_positions must be >= 1.")
    if config.max_random_size is not None and config.max_random_size < 0:
        raise ValueError("max_random_size must be >= 0 when provided.")
    if config.max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")
    if config.observation_timeout_seconds is not None and config.observation_timeout_seconds <= 0.0:
        raise ValueError("observation_timeout_seconds must be > 0 when provided.")
    if config.max_derivative_tokens is not None and config.max_derivative_tokens < 1:
        raise ValueError("max_derivative_tokens must be >= 1 when provided.")
    if config.max_residual_tokens is not None and config.max_residual_tokens < 1:
        raise ValueError("max_residual_tokens must be >= 1 when provided.")
    if config.lr <= 0.0:
        raise ValueError("lr must be > 0.")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be >= 0.")
    if config.grad_clip_norm is not None and config.grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be > 0 when provided.")
    for name in ("log_every", "val_every", "checkpoint_every", "val_batches", "diagnostic_batches"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be >= 1.")
    for name in (
        "diagnostic_timeout_seconds",
        "diagnostic_example_timeout_seconds",
        "diagnostic_numeric_timeout_seconds",
    ):
        value = getattr(config, name)
        if value is not None and value <= 0.0:
            raise ValueError(f"{name} must be > 0 when provided.")
    if config.log_every_n_steps is not None and config.log_every_n_steps < 1:
        raise ValueError("log_every_n_steps must be >= 1 when provided.")
    if config.num_sanity_val_steps < 0:
        raise ValueError("num_sanity_val_steps must be >= 0.")
    if config.enable_wandb and not str(config.wandb_project).strip():
        raise ValueError("wandb_project must be non-empty when enable_wandb=True.")

    TreeDiffusionModelConfig(
        vocab_size=3,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_encoder_layers=config.n_encoder_layers,
        n_decoder_layers=config.n_decoder_layers,
        dropout=config.dropout,
        norm_first=config.norm_first,
        tie_embeddings=config.tie_embeddings,
    )


def _normalize_betas(value: Sequence[float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("betas must contain exactly two values.")
    beta1, beta2 = float(value[0]), float(value[1])
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("betas must satisfy 0 <= beta < 1.")
    return beta1, beta2


def _use_precomputed(config: TreeDiffusionTrainingConfig) -> bool:
    return bool(config.use_precomputed or config.precomputed_data_dir)


__all__ = [
    "TreeDiffusionTrainingConfig",
    "_load_training_config_values",
    "_normalize_betas",
    "_use_precomputed",
    "_validate_training_config",
    "load_training_config",
]
