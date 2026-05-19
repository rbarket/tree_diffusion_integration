from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import sympy as sp
import torch

from src.mathlang.ast import Expr
from src.mathlang.canonicalize import canonicalize
from src.mathlang.conversions import ast_to_sympy
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.decoding import DecodedEdit, apply_decoded_edit, decode_edit_candidates
from src.tree_diffusion.edit_path import structural_distance
from src.tree_diffusion.eval_one_step import numeric_residual_score
from src.tree_diffusion.model import TreeDiffusionPolicyModel
from src.tree_diffusion.observation import (
    _observation_timeout,
    build_observation,
    compute_current_derivative,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


_MISSING_NUMERIC_RESIDUAL_PENALTY = 1e12


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
    score: float | None
    structural_distance_before: int | None = None
    structural_distance_after: int | None = None
    exact_symbolic_match: bool = False
    stop_reason: str | None = None


@dataclass(frozen=True)
class RepairResult:
    target_integrand_prefix: str
    initial_prefix: str
    final_prefix: str
    success: bool
    stop_reason: str
    steps_taken: int
    initial_numeric_residual: float | None
    final_numeric_residual: float | None
    exact_symbolic_match: bool
    repeated_state: bool
    no_candidate: bool
    steps: list[RepairStep]


@dataclass(frozen=True)
class RepairScoringConfig:
    lambda_size: float = 1e-3
    lambda_policy: float = 1e-2
    require_numeric_improvement: bool = False


@dataclass(frozen=True)
class _ScoredRepairCandidate:
    rank: int
    decoded: DecodedEdit
    edited_tree: Expr
    edited_prefix: str
    numeric_residual_after: float | None
    score: float
    structural_distance_after: int | None
    exact_symbolic_match: bool


def tree_size(expr: Expr) -> int:
    return 1 + sum(tree_size(child) for child in expr.children())


def score_repair_candidate(
    *,
    numeric_residual: float | None,
    tree_size_value: int,
    policy_logprob: float | None,
    config: RepairScoringConfig,
) -> float:
    residual_term = (
        _MISSING_NUMERIC_RESIDUAL_PENALTY
        if numeric_residual is None
        else float(numeric_residual)
    )
    policy_term = 0.0 if policy_logprob is None else float(policy_logprob)
    return residual_term + (float(config.lambda_size) * int(tree_size_value)) - (
        float(config.lambda_policy) * policy_term
    )


def derivative_matches_target(
    antiderivative: Expr,
    target_integrand: Expr,
    *,
    simplify: bool = True,
    timeout_seconds: float | None = None,
) -> bool:
    try:
        with _observation_timeout(timeout_seconds):
            derivative = compute_current_derivative(
                antiderivative,
                simplify_derivative=simplify,
            )
            if simplify:
                residual = ast_to_sympy(derivative) - ast_to_sympy(
                    canonicalize(target_integrand, strip_additive_constants=False)
                )
                return bool(sp.simplify(residual) == 0)
            return canonicalize(derivative, strip_additive_constants=False) == canonicalize(
                target_integrand,
                strip_additive_constants=False,
            )
    except Exception:
        return False


@torch.no_grad()
def greedy_repair(
    model: TreeDiffusionPolicyModel,
    target_integrand: Expr,
    initial_antiderivative: Expr,
    *,
    tokenizer: TreeDiffusionTokenizer,
    device: torch.device | str,
    max_steps: int = 10,
    candidate_k: int = 8,
    numeric_tol: float = 1e-10,
    patience: int = 2,
    constrain_position: bool = True,
    max_decode_length: int | None = None,
    scoring: RepairScoringConfig | None = None,
    residual_mode: str = "both",
    simplify_symbolic_residual: bool = True,
    numeric_residual_timeout_seconds: float | None = 2.0,
    symbolic_check_timeout_seconds: float | None = 2.0,
    target_antiderivative: Expr | None = None,
) -> RepairResult:
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0.")
    if candidate_k < 1:
        raise ValueError("candidate_k must be >= 1.")
    if numeric_tol < 0.0:
        raise ValueError("numeric_tol must be >= 0.")
    if patience < 0:
        raise ValueError("patience must be >= 0.")

    config = scoring or RepairScoringConfig()
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    canonical_target_integrand = canonicalize(
        target_integrand,
        strip_additive_constants=False,
    )
    canonical_target_antiderivative = (
        None if target_antiderivative is None else canonicalize(target_antiderivative)
    )
    current = canonicalize(initial_antiderivative)
    initial_prefix = serialize_prefix_string(current)
    target_integrand_prefix = serialize_prefix_string(canonical_target_integrand)
    current_numeric = numeric_residual_score(
        current,
        canonical_target_integrand,
        timeout_seconds=numeric_residual_timeout_seconds,
    )
    initial_numeric = current_numeric
    exact_match = derivative_matches_target(
        current,
        canonical_target_integrand,
        timeout_seconds=symbolic_check_timeout_seconds,
    )
    steps: list[RepairStep] = []

    if exact_match:
        return _repair_result(
            target_integrand_prefix=target_integrand_prefix,
            initial_prefix=initial_prefix,
            final_tree=current,
            stop_reason="exact_symbolic_match",
            steps=steps,
            initial_numeric_residual=initial_numeric,
            final_numeric_residual=current_numeric,
            exact_symbolic_match=True,
        )
    if _meets_numeric_tol(current_numeric, numeric_tol):
        return _repair_result(
            target_integrand_prefix=target_integrand_prefix,
            initial_prefix=initial_prefix,
            final_tree=current,
            stop_reason="numeric_tol",
            steps=steps,
            initial_numeric_residual=initial_numeric,
            final_numeric_residual=current_numeric,
            exact_symbolic_match=False,
        )
    if max_steps == 0:
        return _repair_result(
            target_integrand_prefix=target_integrand_prefix,
            initial_prefix=initial_prefix,
            final_tree=current,
            stop_reason="max_steps",
            steps=steps,
            initial_numeric_residual=initial_numeric,
            final_numeric_residual=current_numeric,
            exact_symbolic_match=False,
        )

    visited_prefixes = {initial_prefix}
    non_improving_steps = 0

    for step_index in range(max_steps):
        current_prefix = serialize_prefix_string(current)
        structural_before = _structural_distance_or_none(
            current,
            canonical_target_antiderivative,
        )

        try:
            input_ids, input_attention_mask = _encode_repair_observation(
                model=model,
                tokenizer=tokenizer,
                target_integrand=canonical_target_integrand,
                current_antiderivative=current,
                residual_mode=residual_mode,
                simplify_symbolic_residual=simplify_symbolic_residual,
                device=target_device,
            )
        except Exception:
            steps.append(
                RepairStep(
                    step_index=step_index,
                    current_prefix=current_prefix,
                    chosen_prefix=None,
                    decoded_status=None,
                    selected_node_id=None,
                    replacement_tokens=[],
                    replacement_subtree_prefix=None,
                    candidate_rank=None,
                    policy_logprob=None,
                    numeric_residual_before=current_numeric,
                    numeric_residual_after=None,
                    score=None,
                    structural_distance_before=structural_before,
                    structural_distance_after=None,
                    stop_reason="observation_failed",
                )
            )
            return _repair_result(
                target_integrand_prefix=target_integrand_prefix,
                initial_prefix=initial_prefix,
                final_tree=current,
                stop_reason="observation_failed",
                steps=steps,
                initial_numeric_residual=initial_numeric,
                final_numeric_residual=current_numeric,
                exact_symbolic_match=False,
            )

        try:
            candidates = decode_edit_candidates(
                model,
                input_ids,
                tokenizer=tokenizer,
                current_tree=current,
                input_attention_mask=input_attention_mask,
                k=int(candidate_k),
                max_length=max_decode_length,
                constrain_position=constrain_position,
                device=target_device,
            )
        except Exception:
            steps.append(
                RepairStep(
                    step_index=step_index,
                    current_prefix=current_prefix,
                    chosen_prefix=None,
                    decoded_status=None,
                    selected_node_id=None,
                    replacement_tokens=[],
                    replacement_subtree_prefix=None,
                    candidate_rank=None,
                    policy_logprob=None,
                    numeric_residual_before=current_numeric,
                    numeric_residual_after=None,
                    score=None,
                    structural_distance_before=structural_before,
                    structural_distance_after=None,
                    stop_reason="decode_failed",
                )
            )
            return _repair_result(
                target_integrand_prefix=target_integrand_prefix,
                initial_prefix=initial_prefix,
                final_tree=current,
                stop_reason="decode_failed",
                steps=steps,
                initial_numeric_residual=initial_numeric,
                final_numeric_residual=current_numeric,
                exact_symbolic_match=False,
            )

        scored_candidates: list[_ScoredRepairCandidate] = []
        applicable_candidates = 0
        non_repeated_candidates = 0
        for rank, candidate in enumerate(candidates, start=1):
            if candidate.status != "ok":
                continue
            try:
                edited_tree = canonicalize(apply_decoded_edit(current, candidate))
            except Exception:
                continue
            applicable_candidates += 1

            edited_prefix = serialize_prefix_string(edited_tree)
            if edited_prefix in visited_prefixes:
                continue
            non_repeated_candidates += 1

            numeric_after = numeric_residual_score(
                edited_tree,
                canonical_target_integrand,
                timeout_seconds=numeric_residual_timeout_seconds,
            )
            if config.require_numeric_improvement and not _numeric_improved(
                before=current_numeric,
                after=numeric_after,
            ):
                continue

            scored_candidates.append(
                _ScoredRepairCandidate(
                    rank=rank,
                    decoded=candidate,
                    edited_tree=edited_tree,
                    edited_prefix=edited_prefix,
                    numeric_residual_after=numeric_after,
                    score=score_repair_candidate(
                        numeric_residual=numeric_after,
                        tree_size_value=tree_size(edited_tree),
                        policy_logprob=candidate.logprob,
                        config=config,
                    ),
                    structural_distance_after=_structural_distance_or_none(
                        edited_tree,
                        canonical_target_antiderivative,
                    ),
                    exact_symbolic_match=derivative_matches_target(
                        edited_tree,
                        canonical_target_integrand,
                        timeout_seconds=symbolic_check_timeout_seconds,
                    ),
                )
            )

        if not scored_candidates:
            if applicable_candidates == 0:
                stop_reason = "no_applicable_candidate"
            elif non_repeated_candidates == 0:
                stop_reason = "repeated_state"
            else:
                stop_reason = "no_numeric_improvement"
            top_candidate = candidates[0] if candidates else None
            steps.append(
                _failure_step(
                    step_index=step_index,
                    current_prefix=current_prefix,
                    decoded=top_candidate,
                    numeric_residual_before=current_numeric,
                    structural_distance_before=structural_before,
                    stop_reason=stop_reason,
                )
            )
            return _repair_result(
                target_integrand_prefix=target_integrand_prefix,
                initial_prefix=initial_prefix,
                final_tree=current,
                stop_reason=stop_reason,
                steps=steps,
                initial_numeric_residual=initial_numeric,
                final_numeric_residual=current_numeric,
                exact_symbolic_match=False,
            )

        chosen = min(scored_candidates, key=lambda item: (item.score, item.rank))
        improved = _numeric_improved(
            before=current_numeric,
            after=chosen.numeric_residual_after,
        )
        if improved:
            non_improving_steps = 0
        else:
            non_improving_steps += 1

        next_stop_reason: str | None = None
        if chosen.exact_symbolic_match:
            next_stop_reason = "exact_symbolic_match"
        elif _meets_numeric_tol(chosen.numeric_residual_after, numeric_tol):
            next_stop_reason = "numeric_tol"
        elif non_improving_steps >= patience:
            next_stop_reason = "no_numeric_improvement"

        steps.append(
            RepairStep(
                step_index=step_index,
                current_prefix=current_prefix,
                chosen_prefix=chosen.edited_prefix,
                decoded_status=chosen.decoded.status,
                selected_node_id=chosen.decoded.selected_node_id,
                replacement_tokens=list(chosen.decoded.replacement_tokens),
                replacement_subtree_prefix=_replacement_subtree_prefix(chosen.decoded),
                candidate_rank=chosen.rank,
                policy_logprob=chosen.decoded.logprob,
                numeric_residual_before=current_numeric,
                numeric_residual_after=chosen.numeric_residual_after,
                score=chosen.score,
                structural_distance_before=structural_before,
                structural_distance_after=chosen.structural_distance_after,
                exact_symbolic_match=chosen.exact_symbolic_match,
                stop_reason=next_stop_reason,
            )
        )

        current = chosen.edited_tree
        current_numeric = chosen.numeric_residual_after
        visited_prefixes.add(chosen.edited_prefix)

        if next_stop_reason is not None:
            return _repair_result(
                target_integrand_prefix=target_integrand_prefix,
                initial_prefix=initial_prefix,
                final_tree=current,
                stop_reason=next_stop_reason,
                steps=steps,
                initial_numeric_residual=initial_numeric,
                final_numeric_residual=current_numeric,
                exact_symbolic_match=chosen.exact_symbolic_match,
            )

    return _repair_result(
        target_integrand_prefix=target_integrand_prefix,
        initial_prefix=initial_prefix,
        final_tree=current,
        stop_reason="max_steps",
        steps=steps,
        initial_numeric_residual=initial_numeric,
        final_numeric_residual=current_numeric,
        exact_symbolic_match=derivative_matches_target(
            current,
            canonical_target_integrand,
            timeout_seconds=symbolic_check_timeout_seconds,
        ),
    )


@torch.no_grad()
def greedy_repair_from_seeds(
    model: TreeDiffusionPolicyModel,
    target_integrand: Expr,
    seeds: Sequence[Expr],
    *,
    tokenizer: TreeDiffusionTokenizer,
    device: torch.device | str,
    max_steps: int = 10,
    candidate_k: int = 8,
    numeric_tol: float = 1e-10,
    max_decode_length: int | None = None,
    scoring: RepairScoringConfig | None = None,
    residual_mode: str = "both",
    simplify_symbolic_residual: bool = True,
    numeric_residual_timeout_seconds: float | None = 2.0,
    symbolic_check_timeout_seconds: float | None = 2.0,
    target_antiderivative: Expr | None = None,
) -> RepairResult:
    if not seeds:
        raise ValueError("seeds must be non-empty.")

    results: list[RepairResult] = []
    for seed in seeds:
        result = greedy_repair(
            model,
            target_integrand,
            seed,
            tokenizer=tokenizer,
            device=device,
            max_steps=max_steps,
            candidate_k=candidate_k,
            numeric_tol=numeric_tol,
            max_decode_length=max_decode_length,
            scoring=scoring,
            residual_mode=residual_mode,
            simplify_symbolic_residual=simplify_symbolic_residual,
            numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
            symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
            target_antiderivative=target_antiderivative,
        )
        if result.success:
            return result
        results.append(result)

    finite_results = [
        result
        for result in results
        if result.final_numeric_residual is not None
        and math.isfinite(float(result.final_numeric_residual))
    ]
    if finite_results:
        return min(finite_results, key=lambda result: float(result.final_numeric_residual))
    return results[0]


def _encode_repair_observation(
    *,
    model: TreeDiffusionPolicyModel,
    tokenizer: TreeDiffusionTokenizer,
    target_integrand: Expr,
    current_antiderivative: Expr,
    residual_mode: str,
    simplify_symbolic_residual: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    observation = build_observation(
        target_integrand,
        current_antiderivative,
        residual_mode=residual_mode,
        simplify_symbolic_residual=simplify_symbolic_residual,
    )
    input_tokens = tokenizer.serialize_observation(observation) + ["<EDIT>"]
    max_input_length = getattr(getattr(model, "config", None), "max_input_length", None)
    input_ids = torch.tensor(
        tokenizer.encode_tokens(input_tokens, pad_to_length=max_input_length),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    input_attention_mask = input_ids.ne(tokenizer.pad_id).to(dtype=torch.long)
    return input_ids, input_attention_mask


def _repair_result(
    *,
    target_integrand_prefix: str,
    initial_prefix: str,
    final_tree: Expr,
    stop_reason: str,
    steps: list[RepairStep],
    initial_numeric_residual: float | None,
    final_numeric_residual: float | None,
    exact_symbolic_match: bool,
) -> RepairResult:
    success = stop_reason in {"exact_symbolic_match", "numeric_tol"}
    return RepairResult(
        target_integrand_prefix=target_integrand_prefix,
        initial_prefix=initial_prefix,
        final_prefix=serialize_prefix_string(final_tree),
        success=success,
        stop_reason=stop_reason,
        steps_taken=sum(1 for step in steps if step.chosen_prefix is not None),
        initial_numeric_residual=initial_numeric_residual,
        final_numeric_residual=final_numeric_residual,
        exact_symbolic_match=bool(exact_symbolic_match),
        repeated_state=stop_reason == "repeated_state",
        no_candidate=stop_reason == "no_applicable_candidate",
        steps=list(steps),
    )


def _failure_step(
    *,
    step_index: int,
    current_prefix: str,
    decoded: DecodedEdit | None,
    numeric_residual_before: float | None,
    structural_distance_before: int | None,
    stop_reason: str,
) -> RepairStep:
    return RepairStep(
        step_index=step_index,
        current_prefix=current_prefix,
        chosen_prefix=None,
        decoded_status=None if decoded is None else decoded.status,
        selected_node_id=None if decoded is None else decoded.selected_node_id,
        replacement_tokens=[] if decoded is None else list(decoded.replacement_tokens),
        replacement_subtree_prefix=None if decoded is None else _replacement_subtree_prefix(decoded),
        candidate_rank=None,
        policy_logprob=None if decoded is None else decoded.logprob,
        numeric_residual_before=numeric_residual_before,
        numeric_residual_after=None,
        score=None,
        structural_distance_before=structural_distance_before,
        structural_distance_after=None,
        exact_symbolic_match=False,
        stop_reason=stop_reason,
    )


def _replacement_subtree_prefix(decoded: DecodedEdit) -> str | None:
    if decoded.replacement_subtree is None:
        return None
    return serialize_prefix_string(decoded.replacement_subtree)


def _structural_distance_or_none(
    expr: Expr,
    target_antiderivative: Expr | None,
) -> int | None:
    if target_antiderivative is None:
        return None
    return int(structural_distance(expr, target_antiderivative))


def _numeric_improved(*, before: float | None, after: float | None) -> bool:
    if before is None or after is None:
        return False
    return float(after) < float(before)


def _meets_numeric_tol(value: float | None, numeric_tol: float) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) <= numeric_tol


__all__ = [
    "RepairResult",
    "RepairScoringConfig",
    "RepairStep",
    "derivative_matches_target",
    "greedy_repair",
    "greedy_repair_from_seeds",
    "score_repair_candidate",
    "tree_size",
]
