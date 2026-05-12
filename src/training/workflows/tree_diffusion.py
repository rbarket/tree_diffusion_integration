from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from src.tree_diffusion._common import (
    parse_bool as _parse_bool,
    resolve_device as _resolve_device,
)
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
)
from src.training.lightning.tree_diffusion_callbacks import build_tree_diffusion_callbacks
from src.training.lightning.tree_diffusion_data import TreeDiffusionDataModule
from src.training.lightning.tree_diffusion_module import TreeDiffusionLightningModule
from src.utils.seeding import set_global_seed


DEFAULT_CONFIG_PATH = "config/train/tree_diffusion.json"


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


def train_tree_diffusion_policy(config: TreeDiffusionTrainingConfig) -> dict[str, Any]:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import CSVLogger

    set_global_seed(config.seed)
    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    _log_training(
        "tree_diffusion_training_start "
        f"output_dir={output_dir} device={device} seed={config.seed} "
        f"num_epochs={config.num_epochs} batch_size={config.batch_size}"
    )
    _log_training(
        "training_schedule "
        f"log_every={config.log_every} val_every={config.val_every} "
        f"checkpoint_every={config.checkpoint_every} "
        f"val_batches={config.val_batches} diagnostic_batches={config.diagnostic_batches}"
    )
    _log_training(
        "data_config "
        f"train_data={config.train_data} val_data={config.val_data} "
        f"precomputed_data_dir={config.precomputed_data_dir} "
        f"use_precomputed={_use_precomputed(config)} "
        f"train_limit={config.train_limit} val_limit={config.val_limit} "
        f"val_fraction={config.val_fraction}"
    )
    _log_training(
        "edit_noise_config "
        f"sigma_small={config.sigma_small} smax={config.smax} rho={config.rho} "
        f"residual_mode={config.residual_mode} "
        f"simplify_symbolic_residual={config.simplify_symbolic_residual} "
        f"allow_complex_constants={config.allow_complex_constants} "
        f"allow_distributional_unary_ops={config.allow_distributional_unary_ops} "
        f"excluded_random_tokens={config.excluded_random_tokens} "
        f"validate_generated_labels={config.validate_generated_labels} "
        f"observation_timeout_seconds={config.observation_timeout_seconds}"
    )
    _log_training(
        "lightning_config "
        f"accelerator={config.accelerator} devices={config.devices} "
        f"precision={config.precision} log_every_n_steps={config.log_every_n_steps} "
        f"num_sanity_val_steps={config.num_sanity_val_steps} "
        f"enable_progress_bar={config.enable_progress_bar} deterministic={config.deterministic}"
    )
    _log_training(
        "wandb_config "
        f"enable_wandb={config.enable_wandb} project={config.wandb_project} "
        f"run_name={config.wandb_run_name} mode={config.wandb_mode}"
    )

    if _use_precomputed(config):
        train_pairs = None
        val_pairs = None
        validation_held_out = True
        _log_training(f"loaded_precomputed_data data_dir={config.precomputed_data_dir}")
    else:
        train_pairs, val_pairs, validation_held_out = _load_train_val_pairs(config)
        _log_training(
            "loaded_pairs "
            f"train_pairs={len(train_pairs)} val_pairs={len(val_pairs)} "
            f"validation_held_out={validation_held_out}"
        )
    tokenizer = _build_tokenizer(config)
    _log_training(
        "tokenizer_ready "
        f"vocab_size={tokenizer.vocab_size} max_positions={tokenizer.max_positions}"
    )
    train_loader = make_loader_for_training_config(
        train_pairs,
        tokenizer=tokenizer,
        config=config,
        base_seed=config.seed,
        shuffle_pairs=True,
        precomputed_split="train",
    )
    val_loader = make_loader_for_training_config(
        val_pairs,
        tokenizer=tokenizer,
        config=config,
        base_seed=config.seed + 10_000,
        shuffle_pairs=False,
        precomputed_split="val",
    )
    _log_training(
        "dataloaders_ready "
        f"train_seed={config.seed} val_seed={config.seed + 10_000} "
        f"num_workers={config.num_workers}"
    )
    target_step = _resolve_epoch_training_steps(config, train_loader)
    _log_training(
        "epoch_schedule_resolved "
        f"num_epochs={config.num_epochs} train_batches_per_epoch={len(train_loader)} "
        f"total_training_steps={target_step}"
    )

    model = build_policy_model_for_config(config, tokenizer).to(device)
    _log_training(
        "model_ready "
        f"d_model={config.d_model} n_heads={config.n_heads} d_ff={config.d_ff} "
        f"encoder_layers={config.n_encoder_layers} decoder_layers={config.n_decoder_layers} "
        f"parameters={_count_parameters(model):,}"
    )
    _log_training(
        "optimizer_ready "
        f"lr={config.lr} weight_decay={config.weight_decay} betas={config.betas} "
        f"grad_clip_norm={config.grad_clip_norm}"
    )

    start_step = 0
    best_val_loss: float | None = None
    legacy_optimizer_state: Mapping[str, Any] | None = None
    resume_ckpt_path = _resolve_lightning_resume_ckpt_path(config.resume_from)
    resume_wandb_run_id, resume_wandb_run_name = _load_resume_wandb_state(config.resume_from)
    wandb_run_id = config.wandb_run_id or resume_wandb_run_id
    wandb_resume = config.wandb_resume
    if wandb_run_id is not None and wandb_resume is None:
        wandb_resume = "allow"
    if config.resume_from is not None:
        if resume_ckpt_path is None:
            checkpoint = load_checkpoint(
                config.resume_from,
                model=model,
                optimizer=None,
                map_location=device,
            )
            start_step = int(checkpoint.get("step", 0))
            raw_best = checkpoint.get("best_val_loss")
            best_val_loss = None if raw_best is None else float(raw_best)
            raw_optimizer = checkpoint.get("optimizer_state_dict")
            legacy_optimizer_state = raw_optimizer if isinstance(raw_optimizer, dict) else None
        _log_training(
            f"resumed_from={config.resume_from} step={start_step} "
            f"lightning_resume_ckpt={resume_ckpt_path} best_val_loss={best_val_loss}"
        )

    trainer_step_budget = target_step if resume_ckpt_path is not None else target_step - start_step
    if trainer_step_budget < 1:
        raise RuntimeError("Training loop did not run any steps.")

    datamodule = TreeDiffusionDataModule(
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        validation_held_out=validation_held_out,
    )
    module = TreeDiffusionLightningModule(
        model=model,
        model_cfg=model.config,
        cfg=config,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
        validation_held_out=validation_held_out,
        target_step=target_step,
        legacy_step=start_step,
        best_val_loss=best_val_loss,
        legacy_optimizer_state=legacy_optimizer_state,
        wandb_run_id=wandb_run_id,
        wandb_resume=wandb_resume,
        wandb_run_name=config.wandb_run_name or resume_wandb_run_name,
    )

    accelerator, devices = _resolve_lightning_accelerator_and_devices(config, device)
    callbacks = build_tree_diffusion_callbacks(
        output_dir=str(output_dir),
        evaluate_policy=evaluate_tree_diffusion_policy,
        save_legacy_checkpoint=save_checkpoint,
        enable_progress_bar=bool(config.enable_progress_bar),
    )
    csv_logger = CSVLogger(save_dir=str(output_dir / "lightning"), name="csv")
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_steps=int(trainer_step_budget),
        max_epochs=-1,
        accelerator=accelerator,
        devices=devices,
        precision=str(config.precision),
        log_every_n_steps=int(config.log_every_n_steps or config.log_every),
        num_sanity_val_steps=int(config.num_sanity_val_steps),
        limit_val_batches=0,
        deterministic=bool(config.deterministic),
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=bool(config.enable_progress_bar),
        logger=csv_logger,
        callbacks=callbacks,
    )

    _log_training(f"training_loop_start start_step={start_step} target_step={target_step}")
    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt_path)
    final_train = module.final_train_output
    if final_train is None:
        raise RuntimeError("Training loop did not run any steps.")

    _log_training(
        "tree_diffusion_training_complete "
        f"final_step={module.legacy_step} final_train_loss={final_train.loss:.4f} "
        f"best_val_loss={_format_optional(module.best_val_loss)} "
        f"elapsed={time.time() - start_time:.1f}s"
    )

    return {
        "final_step": int(module.legacy_step),
        "final_train_loss": float(final_train.loss),
        "best_val_loss": module.best_val_loss,
        "output_dir": str(output_dir),
        "last_checkpoint": module.last_checkpoint,
        "best_checkpoint": module.best_checkpoint,
        "num_epochs": int(config.num_epochs),
        "total_training_steps": int(target_step),
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
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        output = tree_diffusion_eval_step(
            model,
            batch,
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
    lightning_resume_ckpt: str | None = None,
    wandb_run_id: str | None = None,
    wandb_run_name: str | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
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
    }
    if lightning_resume_ckpt is not None:
        payload["lightning_resume_ckpt"] = str(lightning_resume_ckpt)
    if wandb_run_id is not None:
        payload["wandb_run_id"] = str(wandb_run_id)
    if wandb_run_name is not None:
        payload["wandb_run_name"] = str(wandb_run_name)
    torch.save(payload, checkpoint_path)


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
    if "state_dict" in checkpoint and "model_state_dict" not in checkpoint:
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError(f"Lightning checkpoint state_dict must be a mapping: {checkpoint_path}")
        model_state = {
            str(key).removeprefix("model."): value
            for key, value in state_dict.items()
            if str(key).startswith("model.")
        }
        if not model_state:
            raise KeyError(f"Lightning checkpoint missing model.* state_dict keys: {checkpoint_path}")
        model.load_state_dict(model_state)
        if optimizer is not None:
            optimizer_states = checkpoint.get("optimizer_states")
            if isinstance(optimizer_states, (list, tuple)) and optimizer_states:
                optimizer.load_state_dict(optimizer_states[0])
        return checkpoint
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def _resolve_lightning_accelerator_and_devices(
    config: TreeDiffusionTrainingConfig,
    resolved_device: torch.device,
) -> tuple[str, Any]:
    if config.accelerator is not None:
        accelerator = str(config.accelerator)
        devices = 1 if config.devices is None else config.devices
        return accelerator, devices
    if config.devices is not None:
        return "auto", config.devices
    if resolved_device.type == "cuda":
        if resolved_device.index is None:
            return "gpu", 1
        return "gpu", [int(resolved_device.index)]
    if resolved_device.type == "mps":
        return "mps", 1
    if resolved_device.type == "cpu":
        return "cpu", 1
    return "auto", 1


def _resolve_lightning_resume_ckpt_path(resume_from: str | None) -> str | None:
    if resume_from is None:
        return None
    checkpoint_path = Path(resume_from)
    if checkpoint_path.suffix == ".ckpt":
        return str(checkpoint_path)
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(raw, dict):
        return None
    paired = raw.get("lightning_resume_ckpt")
    if paired is None:
        return None
    paired_path = Path(str(paired))
    if not paired_path.exists():
        raise FileNotFoundError(f"Legacy checkpoint references missing lightning_resume_ckpt: {paired_path}")
    return str(paired_path)


def _load_resume_wandb_state(resume_from: str | None) -> tuple[str | None, str | None]:
    if resume_from is None:
        return None, None
    candidates = [Path(resume_from)]
    if candidates[0].suffix != ".ckpt":
        try:
            raw = torch.load(candidates[0], map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(candidates[0], map_location="cpu")
        if isinstance(raw, dict):
            paired = raw.get("lightning_resume_ckpt")
            if paired is not None and Path(str(paired)).exists():
                candidates.append(Path(str(paired)))

    for candidate in candidates:
        try:
            raw = torch.load(candidate, map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(candidate, map_location="cpu")
        if not isinstance(raw, dict):
            continue
        run_id = raw.get("wandb_run_id")
        run_name = raw.get("wandb_run_name")
        if run_id is not None or run_name is not None:
            return (
                None if run_id is None else str(run_id),
                None if run_name is None else str(run_name),
            )
    return None, None


def _parse_devices_override(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _resolve_epoch_training_steps(
    config: TreeDiffusionTrainingConfig,
    train_loader: DataLoader,
) -> int:
    try:
        train_batches_per_epoch = len(train_loader)
    except TypeError as exc:
        raise ValueError(
            "num_epochs requires a finite-size training dataloader. "
            "Use precomputed data, or a finite dataset with __len__."
        ) from exc
    if train_batches_per_epoch < 1:
        raise ValueError("num_epochs requires at least one training batch per epoch.")
    return int(train_batches_per_epoch) * int(config.num_epochs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the symbolic-integration tree-diffusion edit policy.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="JSON training config path.")
    parser.add_argument("--train-data", default=None, help="Override train_data.")
    parser.add_argument("--val-data", default=None, help="Override val_data.")
    parser.add_argument("--precomputed-data-dir", default=None, help="Use precomputed tree-diffusion data.")
    parser.add_argument("--use-precomputed", action="store_true", help="Use precomputed tree-diffusion data.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override num_epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size.")
    parser.add_argument("--device", default=None, help="Override device.")
    parser.add_argument("--resume-from", default=None, help="Override resume_from.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers.")
    parser.add_argument(
        "--simplify-symbolic-residual",
        type=_parse_bool,
        default=None,
        help="Override simplify_symbolic_residual.",
    )
    parser.add_argument(
        "--allow-complex-constants",
        type=_parse_bool,
        default=None,
        help="Override allow_complex_constants.",
    )
    parser.add_argument(
        "--allow-distributional-unary-ops",
        type=_parse_bool,
        default=None,
        help="Override allow_distributional_unary_ops.",
    )
    parser.add_argument(
        "--excluded-random-tokens",
        nargs="*",
        default=None,
        help="Override excluded_random_tokens.",
    )
    parser.add_argument(
        "--validate-generated-labels",
        type=_parse_bool,
        default=None,
        help="Override validate_generated_labels.",
    )
    parser.add_argument("--max-derivative-tokens", type=int, default=None)
    parser.add_argument("--max-residual-tokens", type=int, default=None)
    parser.add_argument(
        "--observation-timeout-seconds",
        type=float,
        default=None,
        help="Override observation_timeout_seconds.",
    )
    parser.add_argument("--accelerator", default=None, help="Override Lightning accelerator.")
    parser.add_argument("--devices", default=None, help="Override Lightning devices.")
    parser.add_argument("--precision", default=None, help="Override Lightning precision.")
    parser.add_argument("--log-every-n-steps", type=int, default=None)
    parser.add_argument("--num-sanity-val-steps", type=int, default=None)
    parser.add_argument("--enable-progress-bar", type=_parse_bool, default=None)
    parser.add_argument("--deterministic", type=_parse_bool, default=None)
    parser.add_argument("--enable-wandb", type=_parse_bool, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-resume", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument("--wandb-mode", default=None)
    args = parser.parse_args(argv)

    values = _load_training_config_values(args.config)
    overrides = {
        "train_data": args.train_data,
        "val_data": args.val_data,
        "precomputed_data_dir": args.precomputed_data_dir,
        "output_dir": args.output_dir,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "resume_from": args.resume_from,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "simplify_symbolic_residual": args.simplify_symbolic_residual,
        "allow_complex_constants": args.allow_complex_constants,
        "allow_distributional_unary_ops": args.allow_distributional_unary_ops,
        "excluded_random_tokens": tuple(args.excluded_random_tokens) if args.excluded_random_tokens is not None else None,
        "validate_generated_labels": args.validate_generated_labels,
        "max_derivative_tokens": args.max_derivative_tokens,
        "max_residual_tokens": args.max_residual_tokens,
        "observation_timeout_seconds": args.observation_timeout_seconds,
        "accelerator": args.accelerator,
        "devices": _parse_devices_override(args.devices),
        "precision": args.precision,
        "log_every_n_steps": args.log_every_n_steps,
        "num_sanity_val_steps": args.num_sanity_val_steps,
        "enable_progress_bar": args.enable_progress_bar,
        "deterministic": args.deterministic,
        "enable_wandb": args.enable_wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "wandb_run_id": args.wandb_run_id,
        "wandb_resume": args.wandb_resume,
        "wandb_entity": args.wandb_entity,
        "wandb_dir": args.wandb_dir,
        "wandb_mode": args.wandb_mode,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    if args.use_precomputed:
        values["use_precomputed"] = True
    config = TreeDiffusionTrainingConfig(**values)
    summary = train_tree_diffusion_policy(config)
    print("training_summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


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


def _load_train_val_pairs(
    config: TreeDiffusionTrainingConfig,
) -> tuple[list[IntegrationPair], list[IntegrationPair], bool]:
    assert config.train_data is not None
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


def make_loader_for_training_config(
    pairs: Sequence[IntegrationPair] | None,
    *,
    tokenizer: TreeDiffusionTokenizer,
    config: TreeDiffusionTrainingConfig,
    base_seed: int,
    shuffle_pairs: bool,
    precomputed_split: str,
) -> DataLoader:
    return make_tree_diffusion_dataloader(
        pairs,
        tokenizer=tokenizer,
        precomputed_data_dir=config.precomputed_data_dir if _use_precomputed(config) else None,
        precomputed_split=precomputed_split,
        precomputed_limit=config.train_limit if precomputed_split == "train" else config.val_limit,
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
        observation_timeout_seconds=config.observation_timeout_seconds,
        simplify_symbolic_residual=config.simplify_symbolic_residual,
        allow_complex_constants=config.allow_complex_constants,
        allow_distributional_unary_ops=config.allow_distributional_unary_ops,
        excluded_random_tokens=config.excluded_random_tokens,
        validate_generated_labels=config.validate_generated_labels,
        max_derivative_tokens=config.max_derivative_tokens,
        max_residual_tokens=config.max_residual_tokens,
        include_metadata=True,
        pin_memory=_resolve_device(config.device).type == "cuda" if config.device != "auto" else torch.cuda.is_available(),
    )


def _use_precomputed(config: TreeDiffusionTrainingConfig) -> bool:
    return bool(config.use_precomputed or config.precomputed_data_dir)


def _build_tokenizer(config: TreeDiffusionTrainingConfig) -> TreeDiffusionTokenizer:
    if not _use_precomputed(config):
        return TreeDiffusionTokenizer(max_positions=config.max_positions)

    from src.tree_diffusion.precomputed_dataset import load_precomputed_tokenizer_metadata

    assert config.precomputed_data_dir is not None
    metadata = load_precomputed_tokenizer_metadata(config.precomputed_data_dir)
    tokenizer = TreeDiffusionTokenizer(
        max_positions=int(metadata.get("max_positions", config.max_positions)),
        numeric_log_min=int(metadata.get("numeric_log_min", -12)),
        numeric_log_max=int(metadata.get("numeric_log_max", 12)),
    )
    if tokenizer.max_positions != config.max_positions:
        raise ValueError(
            "Precomputed tokenizer max_positions does not match training config: "
            f"metadata={tokenizer.max_positions}, config={config.max_positions}."
        )
    _validate_precomputed_lengths(config, metadata)
    return tokenizer


def _validate_precomputed_lengths(
    config: TreeDiffusionTrainingConfig,
    tokenizer_metadata: Mapping[str, Any],
) -> None:
    metadata_path = Path(config.precomputed_data_dir or "") / "metadata.json"
    if not metadata_path.exists():
        return
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    precompute_config = metadata.get("config", {})
    if not isinstance(precompute_config, Mapping):
        return
    for name in ("max_input_length", "max_target_length"):
        if name in precompute_config and int(precompute_config[name]) != int(getattr(config, name)):
            raise ValueError(
                f"Precomputed {name}={precompute_config[name]} does not match "
                f"training config {name}={getattr(config, name)}."
            )


def build_policy_model_for_config(
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


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _log_training(message: str) -> None:
    print(message, flush=True)


def _format_optional(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:.4f}"


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "TreeDiffusionTrainingConfig",
    "build_policy_model_for_config",
    "evaluate_tree_diffusion_policy",
    "load_checkpoint",
    "load_training_config",
    "main",
    "make_loader_for_training_config",
    "save_checkpoint",
    "split_pairs_for_training",
    "train_tree_diffusion_policy",
]


if __name__ == "__main__":
    raise SystemExit(main())
