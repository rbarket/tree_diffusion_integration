from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.tree_diffusion._common import (
    diagnostic_metrics as _diagnostic_metrics,
    json_safe as _json_safe,
    parse_bool as _parse_bool,
    resolve_device as _resolve_device,
    write_json as _write_json,
)
from src.tree_diffusion.dataset import (
    IntegrationPair,
    load_integration_pairs_from_parquet,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.validation import (
    run_one_step_edit_diagnostics,
)
from src.training.workflows.tree_diffusion import (
    TreeDiffusionTrainingConfig,
    build_policy_model_for_config,
    evaluate_tree_diffusion_policy,
    load_checkpoint,
    make_loader_for_training_config,
    split_pairs_for_training,
    train_tree_diffusion_policy,
)


DEFAULT_EXPERIMENT_CONFIG_PATH: str | None = None
_ROOT_FIELDS = {"experiment_name", "training", "final_eval"}
_FINAL_EVAL_FIELDS = {"val_batches", "diagnostic_batches", "checkpoint"}
_COMPARISON_FIELDS = (
    ("experiment_name", "experiment_name"),
    ("residual_mode", "residual_mode"),
    ("best_val_loss", "best_val_loss"),
    ("final_val_position_accuracy", "final_val_position_accuracy"),
    ("final_val_token_accuracy", "final_val_token_accuracy"),
    ("valid_position_rate", "final_diagnostic_valid_position_rate"),
    ("applicable_edit_rate", "final_diagnostic_applicable_edit_rate"),
    ("structural_improvement_rate", "final_diagnostic_structural_improvement_rate"),
    ("exact_target_rate", "final_diagnostic_exact_target_rate"),
)


@dataclass(frozen=True)
class FinalEvalConfig:
    val_batches: int = 50
    diagnostic_batches: int = 20
    checkpoint: str = "best"

    def __post_init__(self) -> None:
        if self.val_batches < 1:
            raise ValueError("final_eval.val_batches must be >= 1.")
        if self.diagnostic_batches < 1:
            raise ValueError("final_eval.diagnostic_batches must be >= 1.")
        if self.checkpoint not in {"best", "last"}:
            raise ValueError("final_eval.checkpoint must be 'best' or 'last'.")


@dataclass(frozen=True)
class PolicyExperimentConfig:
    experiment_name: str
    training: TreeDiffusionTrainingConfig
    final_eval: FinalEvalConfig = FinalEvalConfig()

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty.")


def load_policy_experiment_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> PolicyExperimentConfig:
    raw = _load_experiment_config_values(path)
    override_values = {key: value for key, value in dict(overrides or {}).items() if value is not None}

    training_values = _training_values_from_raw(raw)
    for key, value in override_values.items():
        if key not in _training_field_names():
            raise ValueError(f"Unknown training override field: {key}.")
        training_values[key] = value

    return PolicyExperimentConfig(
        experiment_name=_experiment_name_from_raw(raw),
        training=TreeDiffusionTrainingConfig(**training_values),
        final_eval=_final_eval_from_raw(raw),
    )


def run_policy_experiment(
    config_path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    start_time = time.time()
    experiment = load_policy_experiment_config(config_path, overrides=overrides)
    config = experiment.training
    output_dir = Path(config.output_dir)
    _log_experiment(
        "policy_experiment_start "
        f"name={experiment.experiment_name} config={config_path} output_dir={output_dir}"
    )
    _log_experiment(
        "policy_experiment_training_config "
        f"train_data={config.train_data} val_data={config.val_data} "
        f"train_limit={config.train_limit} val_limit={config.val_limit} "
        f"num_epochs={config.num_epochs} batch_size={config.batch_size} "
        f"num_workers={config.num_workers} "
        f"simplify_symbolic_residual={config.simplify_symbolic_residual} "
        f"allow_complex_constants={config.allow_complex_constants} "
        f"allow_distributional_unary_ops={config.allow_distributional_unary_ops} "
        f"validate_generated_labels={config.validate_generated_labels} "
        f"observation_timeout_seconds={config.observation_timeout_seconds} "
        f"device={config.device}"
    )
    _log_experiment(
        "policy_experiment_final_eval_config "
        f"val_batches={experiment.final_eval.val_batches} "
        f"diagnostic_batches={experiment.final_eval.diagnostic_batches} "
        f"checkpoint={experiment.final_eval.checkpoint}"
    )

    train_pairs, val_pairs, split_description = _load_experiment_pairs(config)
    _log_experiment(
        "policy_experiment_split "
        f"train_pairs={len(train_pairs)} val_pairs={len(val_pairs)} "
        f"held_out={split_description.get('held_out_by_pair')} "
        f"mode={split_description.get('mode')}"
    )

    training_summary = train_tree_diffusion_policy(config)
    _log_experiment(
        "policy_experiment_training_finished "
        f"final_train_loss={training_summary.get('final_train_loss')} "
        f"best_val_loss={training_summary.get('best_val_loss')}"
    )
    checkpoint_path, checkpoint_kind = _select_final_eval_checkpoint(
        final_eval=experiment.final_eval,
        best_checkpoint=training_summary.get("best_checkpoint"),
        last_checkpoint=training_summary.get("last_checkpoint"),
    )
    _log_experiment(
        "policy_experiment_final_checkpoint "
        f"kind={checkpoint_kind} path={checkpoint_path}"
    )

    device = _resolve_device(config.device)
    tokenizer = TreeDiffusionTokenizer(max_positions=config.max_positions)
    model = build_policy_model_for_config(config, tokenizer).to(device)
    load_checkpoint(checkpoint_path, model=model, optimizer=None, map_location=device)

    val_loader = make_loader_for_training_config(
        val_pairs,
        tokenizer=tokenizer,
        config=config,
        base_seed=config.seed + 10_000,
        shuffle_pairs=False,
        precomputed_split="val",
    )
    _log_experiment(
        "policy_experiment_final_ce_eval_start "
        f"num_batches={experiment.final_eval.val_batches}"
    )
    final_val_metrics = evaluate_tree_diffusion_policy(
        model,
        val_loader,
        tokenizer=tokenizer,
        device=device,
        num_batches=experiment.final_eval.val_batches,
    )
    _log_experiment(
        "policy_experiment_final_ce_eval "
        f"loss={final_val_metrics['loss']:.4f} "
        f"position_accuracy={final_val_metrics['position_accuracy']:.4f} "
        f"token_accuracy={final_val_metrics['token_accuracy']:.4f}"
    )
    _log_experiment(
        "policy_experiment_final_diagnostics_start "
        f"num_batches={experiment.final_eval.diagnostic_batches}"
    )
    final_diagnostics = run_one_step_edit_diagnostics(
        model,
        val_loader,
        tokenizer=tokenizer,
        device=device,
        num_batches=experiment.final_eval.diagnostic_batches,
    )
    _log_experiment(
        "policy_experiment_final_diagnostics "
        f"valid_pos={final_diagnostics.valid_position_rate:.4f} "
        f"parseable={final_diagnostics.parseable_replacement_rate:.4f} "
        f"applicable={final_diagnostics.applicable_edit_rate:.4f} "
        f"struct_improve={final_diagnostics.structural_improvement_rate:.4f} "
        f"exact={final_diagnostics.exact_target_rate:.4f}"
    )

    summary = _build_summary(
        experiment=experiment,
        training_summary=training_summary,
        split_description=split_description,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        checkpoint_path=checkpoint_path,
        checkpoint_kind=checkpoint_kind,
        final_val_metrics=final_val_metrics,
        final_diagnostics=final_diagnostics,
        wall_clock_seconds=time.time() - start_time,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "experiment_summary.json", summary)
    _log_experiment(f"policy_experiment_summary_written path={output_dir / 'experiment_summary.json'}")
    print("experiment_summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def compare_policy_experiment_summaries(
    summary_paths: Sequence[str | Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in summary_paths:
        summary_path = Path(path)
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if not isinstance(summary, dict):
            raise ValueError(f"Summary must be a JSON object: {summary_path}")
        rows.append({name: summary.get(source) for name, source in _COMPARISON_FIELDS})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run tree-diffusion policy validation experiments.")
    parser.add_argument("--config", default=DEFAULT_EXPERIMENT_CONFIG_PATH, help="Experiment config path.")
    parser.add_argument("--compare", nargs="+", default=None, help="Compare experiment_summary.json files.")
    parser.add_argument("--train-data", default=None, help="Override training.train_data.")
    parser.add_argument("--val-data", default=None, help="Override training.val_data.")
    parser.add_argument("--output-dir", default=None, help="Override training.output_dir.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override training.num_epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument("--device", default=None, help="Override training.device.")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed.")
    parser.add_argument("--train-limit", type=int, default=None, help="Override training.train_limit.")
    parser.add_argument("--val-limit", type=int, default=None, help="Override training.val_limit.")
    parser.add_argument("--residual-mode", default=None, help="Override training.residual_mode.")
    parser.add_argument("--resume-from", default=None, help="Override training.resume_from.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers.")
    parser.add_argument("--simplify-symbolic-residual", type=_parse_bool, default=None)
    parser.add_argument("--allow-complex-constants", type=_parse_bool, default=None)
    parser.add_argument("--allow-distributional-unary-ops", type=_parse_bool, default=None)
    parser.add_argument("--excluded-random-tokens", nargs="*", default=None)
    parser.add_argument("--validate-generated-labels", type=_parse_bool, default=None)
    parser.add_argument("--max-derivative-tokens", type=int, default=None)
    parser.add_argument("--max-residual-tokens", type=int, default=None)
    parser.add_argument(
        "--observation-timeout-seconds",
        type=float,
        default=None,
        help="Override training.observation_timeout_seconds.",
    )
    parser.add_argument("--accelerator", default=None, help="Override training.accelerator.")
    parser.add_argument("--devices", default=None, help="Override training.devices.")
    parser.add_argument("--precision", default=None, help="Override training.precision.")
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

    if args.compare is not None:
        rows = compare_policy_experiment_summaries(args.compare)
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if args.config is None:
        parser.error("--config is required unless --compare is used.")

    overrides = {
        "train_data": args.train_data,
        "val_data": args.val_data,
        "output_dir": args.output_dir,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "residual_mode": args.residual_mode,
        "resume_from": args.resume_from,
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
    run_policy_experiment(args.config, overrides=overrides)
    return 0


def _load_experiment_config_values(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Experiment config root must be a JSON object.")

    unknown = sorted(set(raw) - _ROOT_FIELDS)
    if unknown:
        raise ValueError(f"Unknown experiment config field(s): {', '.join(unknown)}.")
    return raw


def _experiment_name_from_raw(raw: Mapping[str, Any]) -> str:
    value = raw.get("experiment_name")
    if not isinstance(value, str):
        raise ValueError("experiment_name must be a string.")
    return value


def _training_values_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    training = raw.get("training")
    if not isinstance(training, dict):
        raise ValueError("Experiment config must contain a training object.")
    unknown = sorted(set(training) - _training_field_names())
    if unknown:
        raise ValueError(f"Unknown training config field(s): {', '.join(unknown)}.")
    return dict(training)


def _final_eval_from_raw(raw: Mapping[str, Any]) -> FinalEvalConfig:
    final_eval = raw.get("final_eval", {})
    if not isinstance(final_eval, dict):
        raise ValueError("final_eval must be a JSON object when provided.")
    unknown = sorted(set(final_eval) - _FINAL_EVAL_FIELDS)
    if unknown:
        raise ValueError(f"Unknown final_eval field(s): {', '.join(unknown)}.")
    return FinalEvalConfig(**final_eval)


def _training_field_names() -> set[str]:
    return {field.name for field in fields(TreeDiffusionTrainingConfig)}


def _parse_devices_override(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _load_experiment_pairs(
    config: TreeDiffusionTrainingConfig,
) -> tuple[list[IntegrationPair], list[IntegrationPair], dict[str, Any]]:
    train_candidates = load_integration_pairs_from_parquet(
        config.train_data,
        integrand_column=config.integrand_column,
        integral_column=config.integral_column,
        limit=config.train_limit,
    )
    if not train_candidates:
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
        return train_candidates, val_pairs, {
            "mode": "explicit_val_data",
            "held_out_by_pair": True,
            "train_limit_applied_before_split": config.train_limit,
            "val_limit": config.val_limit,
            "val_fraction": config.val_fraction,
            "split_seed": config.seed,
        }

    train_pairs, val_pairs = split_pairs_for_training(
        train_candidates,
        val_fraction=config.val_fraction,
        seed=config.seed,
        train_limit=None,
        val_limit=config.val_limit,
    )
    held_out = config.val_fraction > 0.0 and len(train_candidates) >= 2
    return train_pairs, val_pairs, {
        "mode": "deterministic_train_split" if held_out else "train_prefix_reuse",
        "held_out_by_pair": held_out,
        "train_limit_applied_before_split": config.train_limit,
        "val_limit": config.val_limit,
        "val_fraction": config.val_fraction,
        "split_seed": config.seed,
    }


def _select_final_eval_checkpoint(
    *,
    final_eval: FinalEvalConfig,
    best_checkpoint: Any,
    last_checkpoint: Any,
) -> tuple[str, str]:
    best_path = Path(str(best_checkpoint)) if best_checkpoint else None
    last_path = Path(str(last_checkpoint)) if last_checkpoint else None

    if final_eval.checkpoint == "best" and best_path is not None and best_path.exists():
        return str(best_path), "best"
    if last_path is not None and last_path.exists():
        fallback = "last" if final_eval.checkpoint == "last" else "last_fallback"
        return str(last_path), fallback
    if best_path is not None and best_path.exists():
        return str(best_path), "best_fallback"
    raise FileNotFoundError("No usable checkpoint was produced for final evaluation.")


def _build_summary(
    *,
    experiment: PolicyExperimentConfig,
    training_summary: Mapping[str, Any],
    split_description: Mapping[str, Any],
    train_pairs: Sequence[IntegrationPair],
    val_pairs: Sequence[IntegrationPair],
    checkpoint_path: str,
    checkpoint_kind: str,
    final_val_metrics: Mapping[str, float],
    final_diagnostics: Any,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    config = experiment.training
    diagnostic_metrics = _diagnostic_metrics(final_diagnostics)
    summary = {
        "experiment_name": experiment.experiment_name,
        "output_dir": config.output_dir,
        "train_data": config.train_data,
        "val_data": config.val_data,
        "split_description": dict(split_description),
        "train_pairs_count": len(train_pairs),
        "val_pairs_count": len(val_pairs),
        "seed": config.seed,
        "num_epochs": config.num_epochs,
        "total_training_steps": training_summary.get("total_training_steps"),
        "residual_mode": config.residual_mode,
        "simplify_symbolic_residual": config.simplify_symbolic_residual,
        "sigma_small": config.sigma_small,
        "smax": config.smax,
        "rho": config.rho,
        "allow_complex_constants": config.allow_complex_constants,
        "allow_distributional_unary_ops": config.allow_distributional_unary_ops,
        "excluded_random_tokens": list(config.excluded_random_tokens),
        "validate_generated_labels": config.validate_generated_labels,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "d_ff": config.d_ff,
        "n_encoder_layers": config.n_encoder_layers,
        "n_decoder_layers": config.n_decoder_layers,
        "dropout": config.dropout,
        "max_input_length": config.max_input_length,
        "max_target_length": config.max_target_length,
        "max_positions": config.max_positions,
        "num_workers": config.num_workers,
        "observation_timeout_seconds": config.observation_timeout_seconds,
        "max_derivative_tokens": config.max_derivative_tokens,
        "max_residual_tokens": config.max_residual_tokens,
        "best_checkpoint": training_summary.get("best_checkpoint"),
        "last_checkpoint": training_summary.get("last_checkpoint"),
        "final_eval_checkpoint": checkpoint_path,
        "final_eval_checkpoint_kind": checkpoint_kind,
        "best_val_loss": training_summary.get("best_val_loss"),
        "final_train_loss": training_summary.get("final_train_loss"),
        "final_val_loss": final_val_metrics.get("loss"),
        "final_val_position_accuracy": final_val_metrics.get("position_accuracy"),
        "final_val_token_accuracy": final_val_metrics.get("token_accuracy"),
        "final_diagnostic_valid_position_rate": diagnostic_metrics["valid_position_rate"],
        "final_diagnostic_parseable_replacement_rate": diagnostic_metrics[
            "parseable_replacement_rate"
        ],
        "final_diagnostic_applicable_edit_rate": diagnostic_metrics["applicable_edit_rate"],
        "final_diagnostic_structural_improvement_rate": diagnostic_metrics[
            "structural_improvement_rate"
        ],
        "final_diagnostic_exact_target_rate": diagnostic_metrics["exact_target_rate"],
        "final_diagnostic_mean_structural_distance_before": diagnostic_metrics[
            "mean_structural_distance_before"
        ],
        "final_diagnostic_mean_structural_distance_after": diagnostic_metrics[
            "mean_structural_distance_after"
        ],
        "final_eval_val_batches": experiment.final_eval.val_batches,
        "final_eval_diagnostic_batches": experiment.final_eval.diagnostic_batches,
        "wall_clock_seconds": wall_clock_seconds,
    }
    return _json_safe(summary)


def _log_experiment(message: str) -> None:
    print(message, flush=True)


__all__ = [
    "DEFAULT_EXPERIMENT_CONFIG_PATH",
    "FinalEvalConfig",
    "PolicyExperimentConfig",
    "compare_policy_experiment_summaries",
    "load_policy_experiment_config",
    "main",
    "run_policy_experiment",
]


if __name__ == "__main__":
    raise SystemExit(main())
