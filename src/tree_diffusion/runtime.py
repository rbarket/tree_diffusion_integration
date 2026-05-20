from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import torch

from src.tree_diffusion.dataset import (
    load_integration_pairs_from_parquet,
    make_tree_diffusion_dataloader,
)
from src.tree_diffusion.model import TreeDiffusionModelConfig, TreeDiffusionPolicyModel
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


def load_model_and_tokenizer_for_inference(
    *,
    checkpoint: str | Path | None,
    precomputed_data_dir: str | Path | None,
    allow_random_init_model: bool,
) -> tuple[TreeDiffusionTokenizer, TreeDiffusionPolicyModel]:
    if checkpoint is None:
        if not allow_random_init_model:
            raise ValueError("Provide a checkpoint or explicitly allow a random model.")
        tokenizer = tokenizer_from_precomputed(precomputed_data_dir) or TreeDiffusionTokenizer()
        from src.training.tree_diffusion_builders import build_policy_model_for_config
        from src.training.tree_diffusion_config import TreeDiffusionTrainingConfig

        model = build_policy_model_for_config(TreeDiffusionTrainingConfig(), tokenizer)
        return tokenizer, model

    checkpoint_path = Path(checkpoint)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(payload).__name__}.")

    tokenizer = tokenizer_from_checkpoint(payload) or tokenizer_from_precomputed(precomputed_data_dir)
    if tokenizer is None:
        tokenizer = TreeDiffusionTokenizer()

    model = resolve_checkpoint_model(payload, tokenizer=tokenizer, checkpoint_path=checkpoint_path)
    return tokenizer, model


def build_evaluation_dataloader(
    *,
    data: str | Path | None,
    precomputed_data_dir: str | Path | None,
    precomputed_split: str = "val",
    tokenizer: TreeDiffusionTokenizer,
    model: TreeDiffusionPolicyModel,
    num_pairs: int,
    batch_size: int,
    seed: int,
):
    if precomputed_data_dir is not None:
        return make_tree_diffusion_dataloader(
            tokenizer=tokenizer,
            precomputed_data_dir=str(precomputed_data_dir),
            precomputed_split=precomputed_split,
            precomputed_limit=num_pairs,
            batch_size=batch_size,
            shuffle_pairs=False,
            include_metadata=True,
        )

    if data is None:
        raise ValueError("Provide data or precomputed_data_dir.")
    pairs = load_integration_pairs_from_parquet(str(data), limit=num_pairs)
    return make_tree_diffusion_dataloader(
        pairs,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=0,
        max_input_length=model.config.max_input_length,
        max_target_length=model.config.max_target_length,
        base_seed=seed,
        shuffle_pairs=False,
        include_metadata=True,
    )


def tokenizer_from_checkpoint(payload: Mapping[str, Any]) -> TreeDiffusionTokenizer | None:
    metadata = payload.get("tokenizer")
    if not isinstance(metadata, Mapping):
        return None
    return TreeDiffusionTokenizer(
        max_positions=int(metadata.get("max_positions", 512)),
        numeric_log_min=int(metadata.get("numeric_log_min", -12)),
        numeric_log_max=int(metadata.get("numeric_log_max", 12)),
    )


def tokenizer_from_precomputed(data_dir: str | Path | None) -> TreeDiffusionTokenizer | None:
    if data_dir is None:
        return None
    from src.tree_diffusion.precomputed_dataset import load_precomputed_tokenizer_metadata

    metadata = load_precomputed_tokenizer_metadata(str(data_dir))
    return TreeDiffusionTokenizer(
        max_positions=int(metadata.get("max_positions", 512)),
        numeric_log_min=int(metadata.get("numeric_log_min", -12)),
        numeric_log_max=int(metadata.get("numeric_log_max", 12)),
    )


def model_config_from_checkpoint(
    payload: Mapping[str, Any],
    *,
    tokenizer: TreeDiffusionTokenizer,
) -> TreeDiffusionModelConfig:
    raw_model_cfg = payload.get("model_cfg")
    if isinstance(raw_model_cfg, Mapping):
        allowed = {field.name for field in fields(TreeDiffusionModelConfig)}
        values = {str(key): value for key, value in raw_model_cfg.items() if str(key) in allowed}
        values["vocab_size"] = tokenizer.vocab_size
        values["pad_token_id"] = tokenizer.pad_id
        values["bos_token_id"] = tokenizer.bos_id
        values["eos_token_id"] = tokenizer.eos_id
        return TreeDiffusionModelConfig(**values)

    raw_training_cfg = payload.get("config")
    if isinstance(raw_training_cfg, Mapping):
        from src.training.tree_diffusion_builders import build_policy_model_for_config
        from src.training.tree_diffusion_config import TreeDiffusionTrainingConfig

        model = build_policy_model_for_config(
            TreeDiffusionTrainingConfig(**dict(raw_training_cfg)),
            tokenizer,
        )
        return model.config

    raise ValueError("Checkpoint does not contain model_cfg or training config metadata.")


def load_model_state(
    model: TreeDiffusionPolicyModel,
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> None:
    if "state_dict" in payload and "model_state_dict" not in payload:
        state_dict = payload["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Lightning checkpoint state_dict must be a mapping: {checkpoint_path}")
        model_state = {
            str(key).removeprefix("model."): value
            for key, value in state_dict.items()
            if str(key).startswith("model.")
        }
        if not model_state:
            raise KeyError(f"Lightning checkpoint missing model.* state_dict keys: {checkpoint_path}")
        model.load_state_dict(model_state)
        return

    if "model_state_dict" not in payload:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")
    model.load_state_dict(payload["model_state_dict"])


def resolve_checkpoint_model(
    payload: Mapping[str, Any],
    *,
    tokenizer: TreeDiffusionTokenizer,
    checkpoint_path: str | Path,
) -> TreeDiffusionPolicyModel:
    path = Path(checkpoint_path)
    model_config = model_config_from_checkpoint(payload, tokenizer=tokenizer)
    model = TreeDiffusionPolicyModel(model_config)
    load_model_state(model, payload, checkpoint_path=path)
    return model


def batch_size(batch: Mapping[str, Any] | torch.Tensor) -> int:
    if isinstance(batch, torch.Tensor):
        return _tensor_batch_size(batch)

    for key in ("current_prefix", "target_integrand_prefix", "target_antiderivative_prefix"):
        value = batch.get(key)
        if isinstance(value, (list, tuple)):
            return len(value)
    input_ids = batch.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        return _tensor_batch_size(input_ids)
    return 1


def tensor_row(tensor: torch.Tensor, row_index: int) -> torch.Tensor:
    if tensor.ndim == 1:
        if row_index != 0:
            raise IndexError("Cannot index more than one row from a 1-D tensor.")
        return tensor
    return tensor[row_index]


def required_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Batch is missing required tensor field {key!r}.")
    return value


def required_metadata(batch: Mapping[str, Any], key: str, row_index: int) -> Any:
    if key not in batch:
        raise ValueError(f"Batch is missing required metadata field {key!r}.")
    raw_value = batch[key]
    if isinstance(raw_value, (list, tuple)):
        try:
            value = raw_value[row_index]
        except IndexError as exc:
            raise ValueError(f"Metadata field {key!r} is shorter than the batch.") from exc
    else:
        value = metadata_item(batch, key, row_index, default=None)
    if value is None:
        raise ValueError(f"Metadata field {key!r} contains None.")
    return value


def metadata_item(
    batch: Mapping[str, Any],
    key: str,
    row_index: int,
    default: Any = None,
) -> Any:
    if key not in batch:
        return default
    value = batch[key]
    if isinstance(value, (list, tuple)):
        try:
            return value[row_index]
        except IndexError:
            return default
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value[row_index].detach().cpu().tolist()
    return value


def _tensor_batch_size(tensor: torch.Tensor) -> int:
    if tensor.ndim == 1:
        return 1
    if tensor.ndim == 2:
        return int(tensor.size(0))
    raise ValueError("input_ids must have shape (L,) or (B, L).")


__all__ = [
    "batch_size",
    "build_evaluation_dataloader",
    "load_model_and_tokenizer_for_inference",
    "load_model_state",
    "metadata_item",
    "model_config_from_checkpoint",
    "required_metadata",
    "required_tensor",
    "resolve_checkpoint_model",
    "tensor_row",
    "tokenizer_from_checkpoint",
    "tokenizer_from_precomputed",
]
