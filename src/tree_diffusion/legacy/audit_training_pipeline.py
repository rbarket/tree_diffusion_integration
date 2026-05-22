from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.tree_diffusion.dataset import load_integration_pairs_from_parquet, make_tree_diffusion_dataloader
from src.tree_diffusion.model import build_tree_diffusion_policy_model
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.train_step import (
    inspect_batch_predictions,
    overfit_fixed_batch,
    tree_diffusion_eval_step,
    tree_diffusion_train_step,
    validate_tree_diffusion_batch,
)
from src.utils.seeding import set_global_seed


DEFAULT_CONFIG_PATH = "config/audit/preflight.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight-audit the tree-diffusion edit policy pipeline.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="JSON preflight config path.")
    parser.add_argument("--data", default=None, help="Optional parquet path override.")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    if args.data is not None:
        cfg = _with_data_override(cfg, args.data)
    _validate_config(cfg)

    runtime_cfg = cfg["runtime"]
    data_cfg = cfg["data"]
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    audit_cfg = cfg["audit"]

    set_global_seed(int(runtime_cfg["seed"]))
    device = torch.device(str(runtime_cfg["device"]))

    tokenizer = TreeDiffusionTokenizer(max_positions=int(dataset_cfg["max_input_length"]))
    pairs = load_integration_pairs_from_parquet(
        str(data_cfg["parquet"]),
        limit=int(data_cfg["num_pairs"]),
    )
    if not pairs:
        raise RuntimeError("No integration pairs were loaded for the preflight audit.")

    dataloader = make_tree_diffusion_dataloader(
        pairs,
        tokenizer=tokenizer,
        batch_size=int(dataset_cfg["batch_size"]),
        num_workers=int(dataset_cfg["num_workers"]),
        sigma_small=int(dataset_cfg["sigma_small"]),
        smax=int(dataset_cfg["smax"]),
        rho=float(dataset_cfg["rho"]),
        residual_mode=str(dataset_cfg["residual_mode"]),
        simplify_symbolic_residual=bool(dataset_cfg.get("simplify_symbolic_residual", True)),
        allow_complex_constants=bool(dataset_cfg.get("allow_complex_constants", False)),
        allow_distributional_unary_ops=bool(dataset_cfg.get("allow_distributional_unary_ops", False)),
        excluded_random_tokens=tuple(dataset_cfg.get("excluded_random_tokens", ())),
        validate_generated_labels=bool(dataset_cfg.get("validate_generated_labels", False)),
        max_derivative_tokens=dataset_cfg.get("max_derivative_tokens"),
        max_residual_tokens=dataset_cfg.get("max_residual_tokens"),
        max_input_length=int(dataset_cfg["max_input_length"]),
        max_target_length=int(dataset_cfg["max_target_length"]),
        base_seed=int(runtime_cfg["seed"]),
        shuffle_pairs=bool(dataset_cfg["shuffle_pairs"]),
        include_metadata=True,
    )
    batch = next(iter(dataloader))
    validate_tree_diffusion_batch(batch, pad_token_id=tokenizer.pad_id, require_metadata=True)

    model = build_tree_diffusion_policy_model(
        tokenizer,
        max_input_length=int(dataset_cfg["max_input_length"]),
        max_target_length=int(dataset_cfg["max_target_length"]),
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        d_ff=int(model_cfg["d_ff"]),
        n_encoder_layers=int(model_cfg["n_encoder_layers"]),
        n_decoder_layers=int(model_cfg["n_decoder_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["lr"]))

    before_predictions = inspect_batch_predictions(
        model,
        batch,
        tokenizer,
        num_examples=int(audit_cfg["num_prediction_examples"]),
        device=device,
    )
    initial = tree_diffusion_eval_step(model, batch, tokenizer=tokenizer, device=device)
    first_train = tree_diffusion_train_step(
        model,
        batch,
        optimizer,
        tokenizer=tokenizer,
        grad_clip_norm=float(training_cfg["grad_clip_norm"]) if training_cfg["grad_clip_norm"] is not None else None,
        device=device,
    )
    history = overfit_fixed_batch(
        model,
        batch,
        optimizer,
        tokenizer=tokenizer,
        steps=int(audit_cfg["steps"]),
        grad_clip_norm=float(training_cfg["grad_clip_norm"]) if training_cfg["grad_clip_norm"] is not None else None,
        device=device,
    )
    final = history[-1]
    after_predictions = inspect_batch_predictions(
        model,
        batch,
        tokenizer,
        num_examples=int(audit_cfg["num_prediction_examples"]),
        device=device,
    )

    if not _is_finite(initial.loss) or not _is_finite(final.loss):
        raise RuntimeError("Preflight produced a non-finite loss.")
    if first_train.grad_norm is None or first_train.grad_norm <= 0.0 or not _is_finite(first_train.grad_norm):
        raise RuntimeError("Preflight produced no finite gradients.")

    required_decrease_steps = int(audit_cfg["require_loss_decrease_after_steps"])
    if int(audit_cfg["steps"]) >= required_decrease_steps and final.loss >= initial.loss:
        raise RuntimeError(
            f"Fixed-batch overfit did not reduce loss: initial={initial.loss:.6f}, final={final.loss:.6f}."
        )

    _print_report(
        cfg=cfg,
        batch=batch,
        tokenizer=tokenizer,
        initial=initial,
        first_train=first_train,
        final=final,
        before_predictions=before_predictions,
        after_predictions=after_predictions,
    )
    return 0


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a JSON object.")
    return cfg


def _with_data_override(cfg: Mapping[str, Any], data_path: str) -> dict[str, Any]:
    copied = {str(key): dict(value) if isinstance(value, dict) else value for key, value in cfg.items()}
    data = dict(copied.get("data", {}))
    data["parquet"] = data_path
    copied["data"] = data
    return copied


def _validate_config(cfg: Mapping[str, Any]) -> None:
    required_sections = ("data", "dataset", "model", "training", "runtime", "audit")
    for section in required_sections:
        if section not in cfg or not isinstance(cfg[section], Mapping):
            raise ValueError(f"Config section {section!r} must be present and be an object.")

    if int(cfg["data"]["num_pairs"]) < 1:
        raise ValueError("data.num_pairs must be >= 1.")
    if int(cfg["dataset"]["batch_size"]) < 1:
        raise ValueError("dataset.batch_size must be >= 1.")
    if int(cfg["audit"]["steps"]) < 1:
        raise ValueError("audit.steps must be >= 1.")


def _is_finite(value: float) -> bool:
    return torch.isfinite(torch.tensor(float(value))).item()


def _print_report(
    *,
    cfg: Mapping[str, Any],
    batch: Mapping[str, Any],
    tokenizer: TreeDiffusionTokenizer,
    initial: Any,
    first_train: Any,
    final: Any,
    before_predictions: list[dict[str, Any]],
    after_predictions: list[dict[str, Any]],
) -> None:
    print("tree_diffusion_training_preflight")
    print(f"data: {cfg['data']['parquet']}")
    print(f"device: {cfg['runtime']['device']}")
    print("batch shapes:")
    for field in ("input_ids", "input_attention_mask", "target_ids", "target_attention_mask", "labels"):
        print(f"  {field}: {tuple(batch[field].shape)}")
    print("metrics:")
    print(f"  initial_loss: {initial.loss:.6f}")
    print(f"  first_train_loss: {first_train.loss:.6f}")
    print(f"  final_loss: {final.loss:.6f}")
    print(f"  initial_position_accuracy: {initial.position_accuracy}")
    print(f"  final_position_accuracy: {final.position_accuracy}")
    print(f"  initial_token_accuracy: {initial.token_accuracy}")
    print(f"  final_token_accuracy: {final.token_accuracy}")
    print(f"  grad_norm: {first_train.grad_norm}")
    print(f"  input_length_mean: {final.input_length_mean}")
    print(f"  target_length_mean: {final.target_length_mean}")
    print(f"  random_init_fraction: {final.random_init_fraction}")
    print(f"  num_mutations_mean: {final.num_mutations_mean}")

    target_tokens = batch["target_tokens"][0]
    decoder_input_tokens = [tokenizer.bos_token, *target_tokens[:-1]]
    print("target alignment example:")
    print(f"  decoder input: {decoder_input_tokens}")
    print(f"  labels:        {target_tokens}")

    print("predictions before:")
    _print_prediction_records(before_predictions)
    print("predictions after:")
    _print_prediction_records(after_predictions)


def _print_prediction_records(records: list[dict[str, Any]]) -> None:
    for record in records:
        print(f"  example {record['index']}:")
        if "target_tokens" in record:
            print(f"    target:    {_compact_tokens(record['target_tokens'])}")
        print(f"    predicted: {_compact_tokens(record['predicted_tokens'])}")


def _compact_tokens(tokens: list[str], *, limit: int = 24) -> list[str]:
    if len(tokens) <= limit:
        return tokens
    head_count = max(1, limit - 3)
    return [*tokens[:head_count], "...", f"<{len(tokens) - head_count} more>"]


if __name__ == "__main__":
    raise SystemExit(main())
