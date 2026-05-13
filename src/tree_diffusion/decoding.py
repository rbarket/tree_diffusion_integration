from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
import torch.nn.functional as F

from src.mathlang.ast import Expr
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_tokens
from src.tree_diffusion.label_validation import apply_subtree_replacement_by_position
from src.tree_diffusion.model import TreeDiffusionPolicyModel
from src.tree_diffusion.positions import index_tree_positions
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


@dataclass(frozen=True)
class DecodedEdit:
    selected_node_id: int | None
    replacement_subtree: Expr | None
    generated_tokens: list[str]
    normalized_tokens: list[str]
    replacement_tokens: list[str]
    status: str
    logprob: float | None = None
    error: str | None = None


def decode_edit_tokens(
    tokens: Sequence[str],
    *,
    tokenizer: TreeDiffusionTokenizer,
    current_tree: Expr,
    stop_at_eos: bool = True,
) -> DecodedEdit:
    generated_tokens = list(tokens)
    normalized_tokens = _normalize_generated_tokens(
        generated_tokens,
        tokenizer=tokenizer,
        stop_at_eos=stop_at_eos,
    )

    if not normalized_tokens:
        return _decoded_failure(
            generated_tokens=generated_tokens,
            normalized_tokens=normalized_tokens,
            status="empty",
            error="No tokens remained after cleanup.",
        )

    first_token = normalized_tokens[0]
    try:
        selected_node_id = tokenizer.token_to_position(first_token)
    except Exception as exc:
        return _decoded_failure(
            generated_tokens=generated_tokens,
            normalized_tokens=normalized_tokens,
            status="invalid_position_token",
            error=str(exc),
        )

    index = index_tree_positions(current_tree)
    if selected_node_id not in index.node_id_to_node:
        return DecodedEdit(
            selected_node_id=selected_node_id,
            replacement_subtree=None,
            generated_tokens=generated_tokens,
            normalized_tokens=normalized_tokens,
            replacement_tokens=_replacement_tokens(normalized_tokens, tokenizer=tokenizer),
            status="position_out_of_range",
            error=f"Position {selected_node_id} does not exist in current tree.",
        )

    replacement_tokens = _replacement_tokens(normalized_tokens, tokenizer=tokenizer)
    if not replacement_tokens:
        return DecodedEdit(
            selected_node_id=selected_node_id,
            replacement_subtree=None,
            generated_tokens=generated_tokens,
            normalized_tokens=normalized_tokens,
            replacement_tokens=replacement_tokens,
            status="missing_replacement",
            error="No replacement subtree tokens were decoded.",
        )

    try:
        replacement_subtree = parse_prefix_tokens(list(replacement_tokens))
    except Exception as exc:
        return DecodedEdit(
            selected_node_id=selected_node_id,
            replacement_subtree=None,
            generated_tokens=generated_tokens,
            normalized_tokens=normalized_tokens,
            replacement_tokens=replacement_tokens,
            status="replacement_parse_failed",
            error=str(exc),
        )

    return DecodedEdit(
        selected_node_id=selected_node_id,
        replacement_subtree=replacement_subtree,
        generated_tokens=generated_tokens,
        normalized_tokens=normalized_tokens,
        replacement_tokens=replacement_tokens,
        status="ok",
    )


def apply_decoded_edit(
    current_tree: Expr,
    decoded_edit: DecodedEdit,
    *,
    canonicalize_result: bool = True,
) -> Expr:
    if decoded_edit.status != "ok":
        raise ValueError(f"Cannot apply decoded edit with status={decoded_edit.status!r}.")
    if decoded_edit.selected_node_id is None:
        raise ValueError("Cannot apply decoded edit without a selected node id.")
    if decoded_edit.replacement_subtree is None:
        raise ValueError("Cannot apply decoded edit without a replacement subtree.")

    edited_tree = apply_subtree_replacement_by_position(
        current_tree,
        decoded_edit.selected_node_id,
        decoded_edit.replacement_subtree,
    )
    if canonicalize_result:
        return canonicalize(edited_tree)
    return edited_tree


def valid_position_token_ids(
    current_tree: Expr,
    tokenizer: TreeDiffusionTokenizer,
) -> list[int]:
    index = index_tree_positions(current_tree)
    token_ids: list[int] = []
    for position in sorted(index.positions, key=lambda item: item.node_id):
        if position.node_id >= tokenizer.max_positions:
            raise ValueError(
                f"Current tree node_id={position.node_id} cannot be represented by "
                f"tokenizer.max_positions={tokenizer.max_positions}."
            )
        token_ids.append(tokenizer.position_id(position.node_id))
    return token_ids


@torch.no_grad()
def greedy_decode_edit_tokens(
    model: TreeDiffusionPolicyModel,
    input_ids: torch.Tensor,
    *,
    tokenizer: TreeDiffusionTokenizer,
    current_tree: Expr,
    input_attention_mask: torch.Tensor | None = None,
    max_length: int | None = None,
    constrain_position: bool = True,
    device: torch.device | str | None = None,
) -> tuple[list[str], float | None]:
    input_ids = _single_example_ids("input_ids", input_ids)
    if input_attention_mask is not None:
        input_attention_mask = _single_example_ids("input_attention_mask", input_attention_mask)
        if input_attention_mask.shape != input_ids.shape:
            raise ValueError("input_attention_mask shape must match input_ids shape.")

    if device is not None:
        target_device = torch.device(device)
        input_ids = input_ids.to(target_device)
        if input_attention_mask is not None:
            input_attention_mask = input_attention_mask.to(target_device)

    decode_length = _resolve_decode_length(model, max_length)

    memory, memory_padding_mask = model.encode(
        input_ids,
        input_attention_mask=input_attention_mask,
    )
    decoder_input_ids = input_ids.new_full((1, 1), tokenizer.bos_id)
    generated_ids: list[int] = []
    logprob = 0.0

    for step in range(decode_length):
        target_attention_mask = torch.ones_like(decoder_input_ids)
        hidden = model.decode(
            decoder_input_ids,
            memory=memory,
            memory_padding_mask=memory_padding_mask,
            target_attention_mask=target_attention_mask,
        )
        logits = model.lm_head(hidden)
        next_logits = logits[:, -1, :]
        if step == 0 and constrain_position:
            next_logits = _mask_to_token_ids(
                next_logits,
                valid_position_token_ids(current_tree, tokenizer),
            )

        log_probs = F.log_softmax(next_logits, dim=-1)
        next_token = next_logits.argmax(dim=-1)
        next_id = int(next_token.item())
        generated_ids.append(next_id)
        logprob += float(log_probs[0, next_id].detach().cpu())

        if next_id == tokenizer.eos_id:
            break
        decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(1)], dim=1)

    return tokenizer.decode_ids(generated_ids), logprob


@torch.no_grad()
def predict_greedy_edit(
    model: TreeDiffusionPolicyModel,
    input_ids: torch.Tensor,
    *,
    tokenizer: TreeDiffusionTokenizer,
    current_tree: Expr,
    input_attention_mask: torch.Tensor | None = None,
    max_length: int | None = None,
    constrain_position: bool = True,
    device: torch.device | str | None = None,
) -> DecodedEdit:
    tokens, logprob = greedy_decode_edit_tokens(
        model,
        input_ids,
        tokenizer=tokenizer,
        current_tree=current_tree,
        input_attention_mask=input_attention_mask,
        max_length=max_length,
        constrain_position=constrain_position,
        device=device,
    )
    decoded = decode_edit_tokens(
        tokens,
        tokenizer=tokenizer,
        current_tree=current_tree,
    )
    return replace(decoded, logprob=logprob)


def _normalize_generated_tokens(
    tokens: Sequence[str],
    *,
    tokenizer: TreeDiffusionTokenizer,
    stop_at_eos: bool,
) -> list[str]:
    normalized = list(tokens)
    while normalized and normalized[-1] == tokenizer.pad_token:
        normalized.pop()
    if normalized and normalized[0] == tokenizer.bos_token:
        normalized = normalized[1:]
    if stop_at_eos:
        try:
            eos_index = normalized.index(tokenizer.eos_token)
        except ValueError:
            pass
        else:
            normalized = normalized[:eos_index]
    return normalized


def _replacement_tokens(
    normalized_tokens: Sequence[str],
    *,
    tokenizer: TreeDiffusionTokenizer,
) -> list[str]:
    tokens = list(normalized_tokens[1:])
    try:
        eos_index = tokens.index(tokenizer.eos_token)
    except ValueError:
        return tokens
    return tokens[:eos_index]


def _decoded_failure(
    *,
    generated_tokens: list[str],
    normalized_tokens: list[str],
    status: str,
    error: str,
) -> DecodedEdit:
    return DecodedEdit(
        selected_node_id=None,
        replacement_subtree=None,
        generated_tokens=generated_tokens,
        normalized_tokens=normalized_tokens,
        replacement_tokens=[],
        status=status,
        error=error,
    )


def _single_example_ids(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.long:
        raise TypeError(f"{name} must be a torch.long tensor.")
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (L,) or (1, L).")
    if value.size(0) != 1:
        raise ValueError(f"{name} must contain exactly one example.")
    if value.size(1) < 1:
        raise ValueError(f"{name} sequence length must be >= 1.")
    return value


def _resolve_decode_length(
    model: TreeDiffusionPolicyModel,
    max_length: int | None,
) -> int:
    model_config = getattr(model, "config", None)
    configured_length = getattr(model_config, "max_target_length", None)
    if max_length is None:
        if configured_length is None:
            raise ValueError("max_length is required when model.config.max_target_length is absent.")
        max_length = int(configured_length)
    if max_length < 1:
        raise ValueError("max_length must be >= 1.")
    if configured_length is not None and max_length > int(configured_length):
        raise ValueError(
            f"max_length={max_length} exceeds max_target_length={int(configured_length)}."
        )
    return int(max_length)


def _mask_to_token_ids(logits: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    if not token_ids:
        raise ValueError("At least one valid position token id is required.")
    masked = logits.new_full(logits.shape, float("-inf"))
    allowed = torch.tensor(list(token_ids), device=logits.device, dtype=torch.long)
    masked[:, allowed] = logits[:, allowed]
    return masked


__all__ = [
    "DecodedEdit",
    "apply_decoded_edit",
    "decode_edit_tokens",
    "greedy_decode_edit_tokens",
    "predict_greedy_edit",
    "valid_position_token_ids",
]
