from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from src.tree_diffusion.model import TreeDiffusionPolicyModel
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


@dataclass(frozen=True)
class TrainStepOutput:
    loss: float
    position_accuracy: float | None
    token_accuracy: float | None
    grad_norm: float | None
    input_length_mean: float | None = None
    target_length_mean: float | None = None
    random_init_fraction: float | None = None
    num_mutations_mean: float | None = None


def validate_tree_diffusion_batch(
    batch: Mapping[str, Any],
    *,
    pad_token_id: int,
    require_metadata: bool = False,
) -> None:
    required_fields = (
        "input_ids",
        "input_attention_mask",
        "target_ids",
        "target_attention_mask",
        "labels",
    )
    for field in required_fields:
        if field not in batch:
            raise ValueError(f"Missing required batch field: {field}.")
        if not isinstance(batch[field], torch.Tensor):
            raise ValueError(f"Batch field {field} must be a torch.Tensor.")

    input_ids = batch["input_ids"]
    input_attention_mask = batch["input_attention_mask"]
    target_ids = batch["target_ids"]
    target_attention_mask = batch["target_attention_mask"]
    labels = batch["labels"]

    _require_long_rank2("input_ids", input_ids)
    _require_long_rank2("target_ids", target_ids)
    _require_long_rank2("labels", labels)
    _require_rank2("input_attention_mask", input_attention_mask)
    _require_rank2("target_attention_mask", target_attention_mask)

    if input_attention_mask.shape != input_ids.shape:
        raise ValueError("input_attention_mask shape must match input_ids shape.")
    if target_attention_mask.shape != target_ids.shape:
        raise ValueError("target_attention_mask shape must match target_ids shape.")
    if labels.shape != target_ids.shape:
        raise ValueError("labels shape must match target_ids shape.")
    if input_ids.size(0) != target_ids.size(0):
        raise ValueError("input_ids and target_ids batch sizes must match.")

    _validate_binary_attention_mask("input_attention_mask", input_attention_mask)
    _validate_binary_attention_mask("target_attention_mask", target_attention_mask)

    expected_input_mask = input_ids.ne(pad_token_id)
    expected_target_mask = target_ids.ne(pad_token_id)
    if not torch.equal(input_attention_mask.bool(), expected_input_mask):
        raise ValueError("input_attention_mask must equal input_ids != pad_token_id.")
    if not torch.equal(target_attention_mask.bool(), expected_target_mask):
        raise ValueError("target_attention_mask must equal target_ids != pad_token_id.")

    target_pad = target_ids.eq(pad_token_id)
    if target_pad.any() and not labels[target_pad].eq(-100).all():
        raise ValueError("labels must be -100 where target_ids are pad tokens.")
    target_nonpad = ~target_pad
    if target_nonpad.any() and not torch.equal(labels[target_nonpad], target_ids[target_nonpad]):
        raise ValueError("labels must equal target_ids where target_ids are not pad tokens.")

    if require_metadata:
        _validate_metadata(batch=batch, batch_size=input_ids.size(0))


def compute_gradient_norm(
    parameters: Iterable[torch.nn.Parameter],
    *,
    norm_type: float = 2.0,
) -> float:
    gradients: list[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            raise RuntimeError("Gradient contains NaN or Inf.")
        gradients.append(grad)

    if not gradients:
        return 0.0

    if norm_type == float("inf"):
        return float(max(grad.abs().max().item() for grad in gradients))

    total = torch.zeros((), device=gradients[0].device, dtype=torch.float32)
    for grad in gradients:
        total = total + grad.float().norm(norm_type).pow(norm_type)
    return float(total.pow(1.0 / norm_type).item())


def tree_diffusion_train_step(
    model: TreeDiffusionPolicyModel,
    batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    tokenizer: TreeDiffusionTokenizer | None = None,
    grad_clip_norm: float | None = None,
    device: torch.device | str | None = None,
    validate_batch: bool = True,
) -> TrainStepOutput:
    model.train()
    working_batch = _move_tensor_batch(batch, device=device)
    if validate_batch:
        validate_tree_diffusion_batch(
            working_batch,
            pad_token_id=_pad_token_id(model=model, tokenizer=tokenizer),
        )

    optimizer.zero_grad(set_to_none=True)
    output = model(
        input_ids=working_batch["input_ids"],
        input_attention_mask=working_batch["input_attention_mask"],
        target_ids=working_batch["target_ids"],
        target_attention_mask=working_batch["target_attention_mask"],
        labels=working_batch["labels"],
    )
    if output.loss is None:
        raise RuntimeError("Model output loss is None.")
    if not torch.isfinite(output.loss):
        raise RuntimeError("Model loss is not finite.")

    output.loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    grad_norm = compute_gradient_norm(model.parameters())
    if grad_norm <= 0.0:
        raise RuntimeError("No finite nonzero gradients were produced.")
    optimizer.step()

    return _train_step_output(output, working_batch, grad_norm=grad_norm)


@torch.no_grad()
def tree_diffusion_eval_step(
    model: TreeDiffusionPolicyModel,
    batch: Mapping[str, Any],
    *,
    tokenizer: TreeDiffusionTokenizer | None = None,
    device: torch.device | str | None = None,
    validate_batch: bool = True,
) -> TrainStepOutput:
    model.eval()
    working_batch = _move_tensor_batch(batch, device=device)
    if validate_batch:
        validate_tree_diffusion_batch(
            working_batch,
            pad_token_id=_pad_token_id(model=model, tokenizer=tokenizer),
        )

    output = model(
        input_ids=working_batch["input_ids"],
        input_attention_mask=working_batch["input_attention_mask"],
        target_ids=working_batch["target_ids"],
        target_attention_mask=working_batch["target_attention_mask"],
        labels=working_batch["labels"],
    )
    if output.loss is None:
        raise RuntimeError("Model output loss is None.")
    if not torch.isfinite(output.loss):
        raise RuntimeError("Model loss is not finite.")
    return _train_step_output(output, working_batch, grad_norm=None)


@torch.no_grad()
def inspect_batch_predictions(
    model: TreeDiffusionPolicyModel,
    batch: Mapping[str, Any],
    tokenizer: TreeDiffusionTokenizer,
    *,
    num_examples: int = 4,
    device: torch.device | str | None = None,
    use_greedy_decode: bool = True,
) -> list[dict[str, Any]]:
    if num_examples < 1:
        raise ValueError("num_examples must be >= 1.")

    model.eval()
    working_batch = _move_tensor_batch(batch, device=device)
    input_ids = working_batch["input_ids"]
    input_attention_mask = working_batch["input_attention_mask"]
    limit = min(num_examples, input_ids.size(0))

    if use_greedy_decode and hasattr(model, "greedy_decode"):
        predicted_ids = model.greedy_decode(
            input_ids[:limit],
            input_attention_mask=input_attention_mask[:limit],
            max_length=working_batch["target_ids"].size(1),
        )
    else:
        output = model(
            input_ids=input_ids[:limit],
            input_attention_mask=input_attention_mask[:limit],
            target_ids=working_batch["target_ids"][:limit],
            target_attention_mask=working_batch["target_attention_mask"][:limit],
            labels=working_batch["labels"][:limit],
        )
        predicted_ids = output.logits.argmax(dim=-1)

    if not predicted_ids.ge(0).all() or not predicted_ids.lt(tokenizer.vocab_size).all():
        raise RuntimeError("Predicted token ids contain values outside the tokenizer vocabulary.")

    records: list[dict[str, Any]] = []
    for index, ids in enumerate(predicted_ids.detach().cpu().tolist()):
        predicted_tokens = tokenizer.decode_ids(ids, strip_pad=True)
        record: dict[str, Any] = {
            "index": index,
            "predicted_tokens": predicted_tokens,
        }
        if "input_tokens" in batch:
            record["input_tokens"] = batch["input_tokens"][index]
        if "target_tokens" in batch:
            record["target_tokens"] = batch["target_tokens"][index]
        records.append(record)
    return records


def overfit_fixed_batch(
    model: TreeDiffusionPolicyModel,
    batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    tokenizer: TreeDiffusionTokenizer | None = None,
    steps: int = 50,
    grad_clip_norm: float | None = None,
    device: torch.device | str | None = None,
) -> list[TrainStepOutput]:
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    return [
        tree_diffusion_train_step(
            model,
            batch,
            optimizer,
            tokenizer=tokenizer,
            grad_clip_norm=grad_clip_norm,
            device=device,
        )
        for _ in range(steps)
    ]


def _move_tensor_batch(
    batch: Mapping[str, Any],
    *,
    device: torch.device | str | None,
) -> dict[str, Any]:
    if device is None:
        return dict(batch)
    target_device = torch.device(device)
    return {
        key: value.to(target_device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _train_step_output(
    model_output: Any,
    batch: Mapping[str, Any],
    *,
    grad_norm: float | None,
) -> TrainStepOutput:
    assert model_output.loss is not None
    return TrainStepOutput(
        loss=float(model_output.loss.detach().cpu()),
        position_accuracy=_optional_tensor_float(model_output.position_accuracy),
        token_accuracy=_optional_tensor_float(model_output.token_accuracy),
        grad_norm=None if grad_norm is None else float(grad_norm),
        input_length_mean=_optional_batch_mean(batch, "input_length"),
        target_length_mean=_optional_batch_mean(batch, "target_length"),
        random_init_fraction=_optional_batch_mean(batch, "used_random_init"),
        num_mutations_mean=_optional_batch_mean(batch, "num_mutations"),
    )


def _optional_tensor_float(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().cpu())


def _optional_batch_mean(batch: Mapping[str, Any], field: str) -> float | None:
    value = batch.get(field)
    if not isinstance(value, torch.Tensor):
        return None
    return float(value.detach().float().mean().cpu())


def _pad_token_id(
    *,
    model: TreeDiffusionPolicyModel,
    tokenizer: TreeDiffusionTokenizer | None,
) -> int:
    return tokenizer.pad_id if tokenizer is not None else model.config.pad_token_id


def _require_rank2(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [B, L].")


def _require_long_rank2(name: str, value: torch.Tensor) -> None:
    if value.dtype != torch.long:
        raise ValueError(f"{name} must be torch.long.")
    _require_rank2(name, value)


def _validate_binary_attention_mask(name: str, value: torch.Tensor) -> None:
    if value.dtype == torch.bool:
        return
    if not torch.logical_or(value.eq(0), value.eq(1)).all():
        raise ValueError(f"{name} must contain only 0/1 values or be bool.")


def _validate_metadata(batch: Mapping[str, Any], *, batch_size: int) -> None:
    for field in ("input_tokens", "target_tokens"):
        if field not in batch:
            raise ValueError(f"Missing required metadata field: {field}.")
        if not isinstance(batch[field], list):
            raise ValueError(f"Metadata field {field} must be a list.")
        if len(batch[field]) != batch_size:
            raise ValueError(f"Metadata field {field} length must match batch size.")

    for index, input_tokens in enumerate(batch["input_tokens"]):
        if not input_tokens or input_tokens[-1] != "<EDIT>":
            raise ValueError(f"input_tokens[{index}] must end with <EDIT>.")
    for index, target_tokens in enumerate(batch["target_tokens"]):
        if not target_tokens or not str(target_tokens[0]).startswith("<POS_"):
            raise ValueError(f"target_tokens[{index}] must start with a <POS_...> token.")
        if target_tokens[-1] != "<eos>":
            raise ValueError(f"target_tokens[{index}] must end with <eos>.")


__all__ = [
    "TrainStepOutput",
    "compute_gradient_norm",
    "inspect_batch_predictions",
    "overfit_fixed_batch",
    "tree_diffusion_eval_step",
    "tree_diffusion_train_step",
    "validate_tree_diffusion_batch",
]
