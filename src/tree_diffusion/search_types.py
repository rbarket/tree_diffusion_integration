from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepairStep:
    step_index: int
    current_prefix: str
    chosen_prefix: str | None
    decoded_status: str | None
    selected_node_id: int | None
    replacement_tokens: list[str]
    replacement_subtree_prefix: str | None
    candidate_rank: int | None
    policy_logprob: float | None
    numeric_residual_before: float | None
    numeric_residual_after: float | None
    best_numeric_residual_so_far: float | None
    score: float | None
    structural_distance_before: int | None = None
    structural_distance_after: int | None = None
    exact_symbolic_match: bool = False
    stop_reason: str | None = None


__all__ = [
    "RepairStep",
]
