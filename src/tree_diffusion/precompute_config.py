from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Sequence

from src.tree_diffusion._common import json_safe


VALID_PRECOMPUTE_SPLITS = frozenset({"train", "val"})
VALID_VAL_TRAJECTORY_MODES = frozenset({"none", "forward_and_repair"})

_RESUME_REQUIRED_CONFIG_FIELDS = (
    "input_data",
    "integrand_column",
    "integral_column",
    "train_limit",
    "val_limit",
    "val_fraction",
    "seed",
    "shuffle_before_limit",
    "examples_per_pair_train",
    "examples_per_pair_val",
    "sigma_small",
    "smax",
    "rho",
    "residual_mode",
    "simplify_symbolic_residual",
    "max_input_length",
    "max_target_length",
    "max_positions",
    "max_random_size",
    "max_attempts",
    "observation_timeout_seconds",
    "observation_timeout_retries",
    "allow_complex_constants",
    "allow_distributional_unary_ops",
    "excluded_random_tokens",
    "validate_labels",
    "require_strict_label_improvement",
)


@dataclass
class TreeDiffusionPrecomputeConfig:
    input_data: str
    output_dir: str = "data/precomputed/tree_diffusion"

    integrand_column: str = "integrand_prefix"
    integral_column: str = "integral_prefix"

    train_limit: int | None = None
    val_limit: int | None = 10000
    val_fraction: float = 0.05
    seed: int = 123
    shuffle_before_limit: bool = True

    examples_per_pair_train: int = 2
    examples_per_pair_val: int = 2

    shard_size: int = 50000
    overwrite: bool = False
    resume: bool = False

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
    observation_timeout_retries: int = 3

    allow_complex_constants: bool = False
    allow_distributional_unary_ops: bool = False
    excluded_random_tokens: tuple[str, ...] = ()

    validate_labels: bool = True
    require_strict_label_improvement: bool = False

    max_failures: int | None = None
    write_failed_examples: bool = True
    failed_examples_limit: int = 100
    num_workers: int = 1
    splits: tuple[str, ...] = ("train", "val")
    val_trajectory_mode: str = "none"

    def __post_init__(self) -> None:
        self.excluded_random_tokens = tuple(str(token) for token in self.excluded_random_tokens)
        self.splits = _normalize_precompute_splits(self.splits)
        validate_precompute_config(self)


def load_precompute_config(path: str | Path) -> TreeDiffusionPrecomputeConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Precompute config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Precompute config root must be a JSON object.")

    known_fields = {field.name for field in fields(TreeDiffusionPrecomputeConfig)}
    unknown = sorted(set(raw) - known_fields)
    if unknown:
        raise ValueError(f"Unknown precompute config field(s): {', '.join(unknown)}.")
    if "excluded_random_tokens" in raw:
        raw["excluded_random_tokens"] = tuple(raw["excluded_random_tokens"])
    if "splits" in raw and not isinstance(raw["splits"], str):
        raw["splits"] = tuple(raw["splits"])
    return TreeDiffusionPrecomputeConfig(**raw)


def _normalize_precompute_splits(value: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_splits = value.split(",")
    else:
        raw_splits = list(value)

    splits: list[str] = []
    for raw_split in raw_splits:
        split = str(raw_split).strip()
        if not split:
            continue
        if split not in VALID_PRECOMPUTE_SPLITS:
            raise ValueError("splits must contain only 'train' and/or 'val'.")
        if split in splits:
            raise ValueError("splits must not contain duplicates.")
        splits.append(split)

    if not splits:
        raise ValueError("splits must include at least one of: train, val.")
    return tuple(splits)


def validate_precompute_config(config: TreeDiffusionPrecomputeConfig) -> None:
    if not config.input_data:
        raise ValueError("input_data is required.")
    if not Path(config.input_data).exists():
        raise ValueError(f"input_data does not exist: {config.input_data}")
    if config.overwrite and config.resume:
        raise ValueError("overwrite and resume are mutually exclusive.")
    output_dir = Path(config.output_dir)
    if config.resume and not output_dir.exists():
        raise ValueError(f"Cannot resume because output_dir does not exist: {output_dir}")
    if output_dir.exists() and not config.overwrite and not config.resume:
        raise ValueError(f"output_dir already exists and overwrite=False: {output_dir}")
    if config.train_limit is not None and config.train_limit < 1:
        raise ValueError("train_limit must be >= 1 when provided.")
    if config.val_limit is not None and config.val_limit < 0:
        raise ValueError("val_limit must be >= 0 when provided.")
    if not 0.0 <= config.val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1.")
    if config.examples_per_pair_train < 1:
        raise ValueError("examples_per_pair_train must be >= 1.")
    if config.examples_per_pair_val < 1:
        raise ValueError("examples_per_pair_val must be >= 1.")
    if config.shard_size < 1:
        raise ValueError("shard_size must be >= 1.")
    if config.sigma_small < 1:
        raise ValueError("sigma_small must be >= 1.")
    if config.smax < 1:
        raise ValueError("smax must be >= 1.")
    if not 0.0 <= config.rho <= 1.0:
        raise ValueError("rho must satisfy 0 <= rho <= 1.")
    if config.residual_mode not in {"none", "symbolic", "numeric", "both"}:
        raise ValueError("residual_mode must be one of: none, symbolic, numeric, both.")
    if config.max_input_length < 1:
        raise ValueError("max_input_length must be >= 1.")
    if config.max_target_length < 1:
        raise ValueError("max_target_length must be >= 1.")
    if config.max_positions < 1:
        raise ValueError("max_positions must be >= 1.")
    if config.max_random_size is not None and config.max_random_size < 0:
        raise ValueError("max_random_size must be >= 0 when provided.")
    if config.max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")
    if config.observation_timeout_seconds is not None and config.observation_timeout_seconds <= 0.0:
        raise ValueError("observation_timeout_seconds must be > 0 when provided.")
    if config.observation_timeout_retries < 0:
        raise ValueError("observation_timeout_retries must be >= 0.")
    if config.max_failures is not None and config.max_failures < 0:
        raise ValueError("max_failures must be >= 0 when provided.")
    if config.failed_examples_limit < 0:
        raise ValueError("failed_examples_limit must be >= 0.")
    if config.num_workers < 1:
        raise ValueError("num_workers must be >= 1.")
    if config.val_trajectory_mode not in VALID_VAL_TRAJECTORY_MODES:
        raise ValueError("val_trajectory_mode must be one of: none, forward_and_repair.")


def _config_dict(config: TreeDiffusionPrecomputeConfig) -> dict[str, Any]:
    return json_safe(asdict(config))


__all__ = [
    "TreeDiffusionPrecomputeConfig",
    "VALID_PRECOMPUTE_SPLITS",
    "VALID_VAL_TRAJECTORY_MODES",
    "_RESUME_REQUIRED_CONFIG_FIELDS",
    "_config_dict",
    "_normalize_precompute_splits",
    "load_precompute_config",
    "validate_precompute_config",
]
