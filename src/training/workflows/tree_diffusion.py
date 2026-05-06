from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from src.tree_diffusion.dataset import (
    IntegrationPair,
    load_integration_pairs_from_parquet,
    make_tree_diffusion_dataloader,
)
from src.tree_diffusion.model import TreeDiffusionModelConfig, TreeDiffusionPolicyModel
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.train_step import (
    TrainStepOutput,
    tree_diffusion_eval_step,
    tree_diffusion_train_step,
)
from src.tree_diffusion.validation import (
    OneStepEditDiagnosticSummary,
    run_one_step_edit_diagnostics,
)
from src.utils.seeding import set_global_seed


DEFAULT_CONFIG_PATH = "config/train/tree_diffusion.json"


@dataclass
class TreeDiffusionTrainingConfig:
    train_data: str
    val_data: str | None = None
    output_dir: str = "runs/tree_diffusion"

    integrand_column: str = "integrand_prefix"
    integral_column: str = "integral_prefix"

    train_limit: int | None = None
    val_limit: int | None = 512
    val_fraction: float = 0.05

    seed: int = 123
    device: str = "auto"

    max_steps: int = 10000
    batch_size: int = 32
    num_workers: int = 0

    sigma_small: int = 2
    smax: int = 5
    rho: float = 0.2
    residual_mode: str = "both"
    max_input_length: int = 512
    max_target_length: int = 128
    max_positions: int = 512
    max_random_size: int | None = None
    max_attempts: int = 32

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

    resume_from: str | None = None
    save_best: bool = True
    save_last: bool = True

    def __post_init__(self) -> None:
        self.betas = _normalize_betas(self.betas)
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


def train_tree_diffusion_policy(config: TreeDiffusionTrainingConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    start_time = time.time()

    train_pairs, val_pairs, validation_held_out = _load_train_val_pairs(config)
    tokenizer = TreeDiffusionTokenizer(max_positions=config.max_positions)
    train_loader = _make_loader(
        train_pairs,
        tokenizer=tokenizer,
        config=config,
        base_seed=config.seed,
        shuffle_pairs=True,
    )
    val_loader = _make_loader(
        val_pairs,
        tokenizer=tokenizer,
        config=config,
        base_seed=config.seed + 10_000,
        shuffle_pairs=False,
    )

    model = _build_model(config, tokenizer).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    start_step = 0
    best_val_loss: float | None = None
    last_checkpoint: str | None = None
    best_checkpoint: str | None = None
    if config.resume_from is not None:
        checkpoint = load_checkpoint(
            config.resume_from,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        start_step = int(checkpoint.get("step", 0))
        raw_best = checkpoint.get("best_val_loss")
        best_val_loss = None if raw_best is None else float(raw_best)
        best_checkpoint = str(output_dir / "checkpoint_best.pt") if best_val_loss is not None else None
        print(f"Resumed from {config.resume_from}: step={start_step} best_val_loss={best_val_loss}")

    train_iter = iter(train_loader)
    final_train: TrainStepOutput | None = None

    for step in range(start_step + 1, config.max_steps + 1):
        batch = next(train_iter)
        train_output = tree_diffusion_train_step(
            model,
            batch,
            optimizer,
            tokenizer=tokenizer,
            grad_clip_norm=config.grad_clip_norm,
            device=device,
        )
        final_train = train_output

        if step % config.log_every == 0 or step == start_step + 1 or step == config.max_steps:
            row = _metrics_row(
                split="train",
                step=step,
                elapsed_seconds=time.time() - start_time,
                lr=_current_lr(optimizer),
                metrics=_train_step_metrics(train_output),
            )
            _append_jsonl(metrics_path, row)
            _print_train_progress(step, train_output, _current_lr(optimizer))

        if step % config.val_every == 0 or step == config.max_steps:
            val_metrics = evaluate_tree_diffusion_policy(
                model,
                val_loader,
                tokenizer=tokenizer,
                device=device,
                num_batches=config.val_batches,
            )
            val_metrics["validation_held_out"] = float(validation_held_out)
            _append_jsonl(
                metrics_path,
                _metrics_row(
                    split="val",
                    step=step,
                    elapsed_seconds=time.time() - start_time,
                    lr=_current_lr(optimizer),
                    metrics=val_metrics,
                ),
            )
            print(
                f"[step {step}] val_loss={val_metrics['loss']:.4f} "
                f"pos_acc={val_metrics['position_accuracy']:.4f} "
                f"tok_acc={val_metrics['token_accuracy']:.4f}"
            )

            diagnostics = run_one_step_edit_diagnostics(
                model,
                val_loader,
                tokenizer=tokenizer,
                device=device,
                num_batches=config.diagnostic_batches,
            )
            _append_jsonl(
                metrics_path,
                _metrics_row(
                    split="diagnostic",
                    step=step,
                    elapsed_seconds=time.time() - start_time,
                    lr=_current_lr(optimizer),
                    metrics=_diagnostic_metrics(diagnostics),
                ),
            )

            val_loss = float(val_metrics["loss"])
            if config.save_best and (best_val_loss is None or val_loss < best_val_loss):
                best_val_loss = val_loss
                best_path = output_dir / "checkpoint_best.pt"
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    best_val_loss=best_val_loss,
                    tokenizer=tokenizer,
                    extra={"validation_held_out": validation_held_out, "val_metrics": val_metrics},
                )
                best_checkpoint = str(best_path)

        if step % config.checkpoint_every == 0:
            step_path = output_dir / f"checkpoint_step_{step}.pt"
            save_checkpoint(
                step_path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                best_val_loss=best_val_loss,
                tokenizer=tokenizer,
                extra={"validation_held_out": validation_held_out},
            )
            last_checkpoint = str(step_path)

    if final_train is None:
        raise RuntimeError("Training loop did not run any steps.")

    if config.save_last:
        last_path = output_dir / "checkpoint_last.pt"
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            config=config,
            step=config.max_steps,
            best_val_loss=best_val_loss,
            tokenizer=tokenizer,
            extra={"validation_held_out": validation_held_out},
        )
        last_checkpoint = str(last_path)

    return {
        "final_step": config.max_steps,
        "final_train_loss": float(final_train.loss),
        "best_val_loss": best_val_loss,
        "output_dir": str(output_dir),
        "last_checkpoint": last_checkpoint,
        "best_checkpoint": best_checkpoint,
    }


@torch.no_grad()
def evaluate_tree_diffusion_policy(
    model: TreeDiffusionPolicyModel,
    dataloader: DataLoader,
    *,
    tokenizer: TreeDiffusionTokenizer,
    device: torch.device | str,
    num_batches: int,
) -> dict[str, float]:
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1.")

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    iterator = iter(dataloader)
    for _ in range(num_batches):
        output = tree_diffusion_eval_step(
            model,
            next(iterator),
            tokenizer=tokenizer,
            device=device,
        )
        for name, value in _train_step_metrics(output).items():
            if value is None:
                continue
            totals[name] = totals.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1

    expected = (
        "loss",
        "position_accuracy",
        "token_accuracy",
        "input_length_mean",
        "target_length_mean",
        "random_init_fraction",
        "num_mutations_mean",
    )
    return {
        name: (totals[name] / counts[name]) if counts.get(name, 0) > 0 else 0.0
        for name in expected
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: TreeDiffusionPolicyModel,
    optimizer: torch.optim.Optimizer,
    config: TreeDiffusionTrainingConfig,
    step: int,
    best_val_loss: float | None,
    tokenizer: TreeDiffusionTokenizer,
    extra: Mapping[str, Any] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "step": int(step),
            "best_val_loss": best_val_loss,
            "tokenizer": {
                "vocab_size": tokenizer.vocab_size,
                "max_positions": tokenizer.max_positions,
                "pad_id": tokenizer.pad_id,
                "bos_id": tokenizer.bos_id,
                "eos_id": tokenizer.eos_id,
            },
            "extra": dict(extra or {}),
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: TreeDiffusionPolicyModel,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}.")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the symbolic-integration tree-diffusion edit policy.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="JSON training config path.")
    parser.add_argument("--train-data", default=None, help="Override train_data.")
    parser.add_argument("--val-data", default=None, help="Override val_data.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size.")
    parser.add_argument("--device", default=None, help="Override device.")
    parser.add_argument("--resume-from", default=None, help="Override resume_from.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    args = parser.parse_args(argv)

    values = _load_training_config_values(args.config)
    overrides = {
        "train_data": args.train_data,
        "val_data": args.val_data,
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "device": args.device,
        "resume_from": args.resume_from,
        "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    config = TreeDiffusionTrainingConfig(**values)
    summary = train_tree_diffusion_policy(config)
    print("training_summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _validate_training_config(config: TreeDiffusionTrainingConfig) -> None:
    if not config.train_data:
        raise ValueError("train_data is required.")
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
    if config.max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
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
    if config.lr <= 0.0:
        raise ValueError("lr must be > 0.")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be >= 0.")
    if config.grad_clip_norm is not None and config.grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be > 0 when provided.")
    for name in ("log_every", "val_every", "checkpoint_every", "val_batches", "diagnostic_batches"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be >= 1.")

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


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is not available.")
    return resolved


def _load_train_val_pairs(
    config: TreeDiffusionTrainingConfig,
) -> tuple[list[IntegrationPair], list[IntegrationPair], bool]:
    train_pairs = load_integration_pairs_from_parquet(
        config.train_data,
        integrand_column=config.integrand_column,
        integral_column=config.integral_column,
        limit=config.train_limit,
    )
    if not train_pairs:
        raise ValueError("No training pairs were loaded.")

    if config.val_data is not None:
        val_pairs = load_integration_pairs_from_parquet(
            config.val_data,
            integrand_column=config.integrand_column,
            integral_column=config.integral_column,
            limit=config.val_limit,
        )
        if not val_pairs:
            raise ValueError("No validation pairs were loaded.")
        return train_pairs, val_pairs, True

    if config.val_fraction > 0.0 and len(train_pairs) >= 2:
        train_split, val_pairs = split_pairs_for_training(
            train_pairs,
            val_fraction=config.val_fraction,
            seed=config.seed,
            train_limit=None,
            val_limit=config.val_limit,
        )
        return train_split, val_pairs, True

    val_count = len(train_pairs) if config.val_limit is None else min(len(train_pairs), config.val_limit)
    return train_pairs, train_pairs[:val_count], False


def split_pairs_for_training(
    pairs: Sequence[IntegrationPair],
    *,
    val_fraction: float,
    seed: int,
    train_limit: int | None,
    val_limit: int | None,
) -> tuple[list[IntegrationPair], list[IntegrationPair]]:
    if train_limit is not None and train_limit < 1:
        raise ValueError("train_limit must be >= 1 when provided.")
    if val_limit is not None and val_limit < 1:
        raise ValueError("val_limit must be >= 1 when provided.")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1.")

    candidate_pairs = list(pairs)
    if train_limit is not None:
        candidate_pairs = candidate_pairs[:train_limit]
    if not candidate_pairs:
        raise ValueError("No training pairs were available after applying train_limit.")

    if val_fraction > 0.0 and len(candidate_pairs) >= 2:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(candidate_pairs), generator=generator).tolist()
        val_count = max(1, int(round(len(candidate_pairs) * val_fraction)))
        val_count = min(val_count, len(candidate_pairs) - 1)
        if val_limit is not None:
            val_count = min(val_count, val_limit)
        val_indices = set(order[:val_count])
        val_pairs = [pair for index, pair in enumerate(candidate_pairs) if index in val_indices]
        train_split = [
            pair for index, pair in enumerate(candidate_pairs) if index not in val_indices
        ]
        return train_split, val_pairs

    val_count = len(candidate_pairs) if val_limit is None else min(len(candidate_pairs), val_limit)
    return candidate_pairs, candidate_pairs[:val_count]


def _make_loader(
    pairs: Sequence[IntegrationPair],
    *,
    tokenizer: TreeDiffusionTokenizer,
    config: TreeDiffusionTrainingConfig,
    base_seed: int,
    shuffle_pairs: bool,
) -> DataLoader:
    return make_tree_diffusion_dataloader(
        pairs,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sigma_small=config.sigma_small,
        smax=config.smax,
        rho=config.rho,
        residual_mode=config.residual_mode,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
        base_seed=base_seed,
        shuffle_pairs=shuffle_pairs,
        max_attempts=config.max_attempts,
        max_random_size=config.max_random_size,
        include_metadata=True,
        pin_memory=_resolve_device(config.device).type == "cuda" if config.device != "auto" else torch.cuda.is_available(),
    )


def _build_model(
    config: TreeDiffusionTrainingConfig,
    tokenizer: TreeDiffusionTokenizer,
) -> TreeDiffusionPolicyModel:
    return TreeDiffusionPolicyModel(
        TreeDiffusionModelConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_id,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
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
    )


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _metrics_row(
    *,
    split: str,
    step: int,
    elapsed_seconds: float,
    lr: float,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "split": split,
        "step": int(step),
        "elapsed_seconds": float(elapsed_seconds),
        "lr": float(lr),
        **dict(metrics),
    }


def _train_step_metrics(output: TrainStepOutput) -> dict[str, float | None]:
    return {
        "loss": output.loss,
        "position_accuracy": output.position_accuracy,
        "token_accuracy": output.token_accuracy,
        "grad_norm": output.grad_norm,
        "input_length_mean": output.input_length_mean,
        "target_length_mean": output.target_length_mean,
        "random_init_fraction": output.random_init_fraction,
        "num_mutations_mean": output.num_mutations_mean,
    }


def _diagnostic_metrics(summary: OneStepEditDiagnosticSummary) -> dict[str, float | int | None]:
    return {
        "examples": summary.examples,
        "valid_position_rate": summary.valid_position_rate,
        "parseable_replacement_rate": summary.parseable_replacement_rate,
        "applicable_edit_rate": summary.applicable_edit_rate,
        "structural_improvement_rate": summary.structural_improvement_rate,
        "numeric_residual_improvement_rate": summary.numeric_residual_improvement_rate,
        "exact_target_rate": summary.exact_target_rate,
        "mean_structural_distance_before": summary.mean_structural_distance_before,
        "mean_structural_distance_after": summary.mean_structural_distance_after,
    }


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _print_train_progress(step: int, output: TrainStepOutput, lr: float) -> None:
    print(
        f"[step {step}] train_loss={output.loss:.4f} "
        f"pos_acc={_format_optional(output.position_accuracy)} "
        f"tok_acc={_format_optional(output.token_accuracy)} "
        f"grad_norm={_format_optional(output.grad_norm)} lr={lr:.6g}"
    )


def _format_optional(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:.4f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    return value


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "TreeDiffusionTrainingConfig",
    "evaluate_tree_diffusion_policy",
    "load_checkpoint",
    "load_training_config",
    "main",
    "save_checkpoint",
    "split_pairs_for_training",
    "train_tree_diffusion_policy",
]


if __name__ == "__main__":
    raise SystemExit(main())
