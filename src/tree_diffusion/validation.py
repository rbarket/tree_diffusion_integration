from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from src.tree_diffusion.eval_one_step import evaluate_one_step_edits
from src.tree_diffusion.model import TreeDiffusionPolicyModel
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
    decoded_ok_rate: float | None = None
    nonincreasing_structural_rate: float | None = None
    mean_numeric_residual_before: float | None = None
    mean_numeric_residual_after: float | None = None
    status_counts: dict[str, int] | None = None


@torch.no_grad()
def run_one_step_edit_diagnostics(
    model: TreeDiffusionPolicyModel,
    dataloader: DataLoader,
    *,
    tokenizer: TreeDiffusionTokenizer,
    device: torch.device | str,
    num_batches: int,
) -> OneStepEditDiagnosticSummary:
    evaluation = evaluate_one_step_edits(
        model,
        dataloader,
        tokenizer=tokenizer,
        device=device,
        num_batches=num_batches,
        constrain_position=True,
        compute_numeric_residual=True,
    )
    return OneStepEditDiagnosticSummary(
        examples=evaluation.examples,
        valid_position_rate=evaluation.valid_position_rate,
        parseable_replacement_rate=evaluation.parseable_replacement_rate,
        applicable_edit_rate=evaluation.applicable_edit_rate,
        structural_improvement_rate=evaluation.structural_improvement_rate,
        numeric_residual_improvement_rate=evaluation.numeric_residual_improvement_rate,
        exact_target_rate=evaluation.exact_target_rate,
        mean_structural_distance_before=evaluation.mean_structural_distance_before,
        mean_structural_distance_after=evaluation.mean_structural_distance_after,
        decoded_ok_rate=evaluation.decoded_ok_rate,
        nonincreasing_structural_rate=evaluation.nonincreasing_structural_rate,
        mean_numeric_residual_before=evaluation.mean_numeric_residual_before,
        mean_numeric_residual_after=evaluation.mean_numeric_residual_after,
        status_counts=dict(evaluation.status_counts),
    )


__all__ = [
    "OneStepEditDiagnosticSummary",
    "run_one_step_edit_diagnostics",
]
