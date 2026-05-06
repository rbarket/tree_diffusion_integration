from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion.edit_path import structural_distance
from src.tree_diffusion.model import TreeDiffusionPolicyModel
from src.tree_diffusion.mutation import replace_subtree_by_node_id
from src.tree_diffusion.positions import index_tree_positions
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


@dataclass(frozen=True)
class OneStepEditDiagnosticSummary:
    examples: int
    valid_position_rate: float
    parseable_replacement_rate: float
    applicable_edit_rate: float
    structural_improvement_rate: float
    numeric_residual_improvement_rate: float | None
    exact_target_rate: float
    mean_structural_distance_before: float | None
    mean_structural_distance_after: float | None


@torch.no_grad()
def run_one_step_edit_diagnostics(
    model: TreeDiffusionPolicyModel,
    dataloader: DataLoader,
    *,
    tokenizer: TreeDiffusionTokenizer,
    device: torch.device | str,
    num_batches: int,
) -> OneStepEditDiagnosticSummary:
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1.")

    model.eval()
    iterator = iter(dataloader)
    target_device = torch.device(device)

    examples = 0
    valid_positions = 0
    parseable_replacements = 0
    applicable_edits = 0
    structural_improvements = 0
    exact_targets = 0
    before_distances: list[float] = []
    after_distances: list[float] = []

    for _ in range(num_batches):
        batch = next(iterator)
        working_batch = _move_tensor_batch(batch, device=target_device)
        predicted_ids = _predict_ids(
            model,
            working_batch,
            target_length=working_batch["target_ids"].size(1),
        )
        predicted_ids = predicted_ids.detach().cpu()

        for row_index, row_ids in enumerate(predicted_ids.tolist()):
            examples += 1
            predicted_tokens = tokenizer.decode_ids(row_ids, strip_pad=True)
            first_token, replacement_tokens = _extract_edit_tokens(predicted_tokens, tokenizer=tokenizer)

            try:
                current_tree = canonicalize(parse_prefix_string(str(batch["current_prefix"][row_index])))
                target_tree = canonicalize(parse_prefix_string(str(batch["target_antiderivative_prefix"][row_index])))
            except Exception:
                continue

            try:
                selected_node_id = tokenizer.token_to_position(first_token)
                index = index_tree_positions(current_tree)
                if selected_node_id not in index.node_id_to_node:
                    raise ValueError("Predicted position does not exist in current tree.")
            except Exception:
                continue
            valid_positions += 1

            try:
                replacement_subtree = canonicalize(parse_prefix_string(" ".join(replacement_tokens)))
            except Exception:
                continue
            parseable_replacements += 1

            try:
                edited_tree = canonicalize(
                    replace_subtree_by_node_id(current_tree, selected_node_id, replacement_subtree)
                )
            except Exception:
                continue
            applicable_edits += 1

            before = float(structural_distance(current_tree, target_tree))
            after = float(structural_distance(edited_tree, target_tree))
            before_distances.append(before)
            after_distances.append(after)
            if after < before:
                structural_improvements += 1
            if edited_tree == target_tree:
                exact_targets += 1

    if examples == 0:
        return OneStepEditDiagnosticSummary(
            examples=0,
            valid_position_rate=0.0,
            parseable_replacement_rate=0.0,
            applicable_edit_rate=0.0,
            structural_improvement_rate=0.0,
            numeric_residual_improvement_rate=None,
            exact_target_rate=0.0,
            mean_structural_distance_before=None,
            mean_structural_distance_after=None,
        )

    return OneStepEditDiagnosticSummary(
        examples=examples,
        valid_position_rate=valid_positions / examples,
        parseable_replacement_rate=parseable_replacements / examples,
        applicable_edit_rate=applicable_edits / examples,
        structural_improvement_rate=structural_improvements / examples,
        numeric_residual_improvement_rate=None,
        exact_target_rate=exact_targets / examples,
        mean_structural_distance_before=_mean_or_none(before_distances),
        mean_structural_distance_after=_mean_or_none(after_distances),
    )


def _predict_ids(
    model: TreeDiffusionPolicyModel,
    batch: Mapping[str, Any],
    *,
    target_length: int,
) -> torch.Tensor:
    if hasattr(model, "greedy_decode"):
        return model.greedy_decode(
            batch["input_ids"],
            input_attention_mask=batch["input_attention_mask"],
            max_length=target_length,
        )

    output = model(
        input_ids=batch["input_ids"],
        input_attention_mask=batch["input_attention_mask"],
        target_ids=batch["target_ids"],
        target_attention_mask=batch["target_attention_mask"],
        labels=batch["labels"],
    )
    return output.logits.argmax(dim=-1)


def _extract_edit_tokens(
    tokens: list[str],
    *,
    tokenizer: TreeDiffusionTokenizer,
) -> tuple[str, list[str]]:
    trimmed: list[str] = []
    for token in tokens:
        if token == tokenizer.pad_token:
            break
        if token == tokenizer.eos_token:
            break
        trimmed.append(token)

    if not trimmed:
        return "", []
    return trimmed[0], trimmed[1:]


def _move_tensor_batch(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


__all__ = [
    "OneStepEditDiagnosticSummary",
    "run_one_step_edit_diagnostics",
]
