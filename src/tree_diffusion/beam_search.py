from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass
import math
import time
from typing import Any, Sequence

import torch

from src.mathlang.ast import Expr
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.decoding import apply_decoded_edit, decode_edit_candidates
from src.tree_diffusion.model import TreeDiffusionPolicyModel
from src.tree_diffusion.search_common import (
    best_numeric_residual as _best_numeric_residual,
    deadline_expired as _deadline_expired,
    derivative_matches_target,
    encode_repair_observation,
    is_finite_numeric as _finite_numeric,
    meets_numeric_tol as _meets_numeric_tol,
    numeric_better as _numeric_better,
    numeric_key as _numeric_key,
    numeric_residual_score,
    remaining_timeout as _remaining_timeout,
    structural_better as _structural_better,
    structural_distance_or_none as _structural_distance_or_none,
    tree_size,
)
from src.tree_diffusion.search_types import RepairStep
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


_MISSING_NUMERIC_RESIDUAL_PENALTY = 1e12


@dataclass(frozen=True)
class BeamSearchScoringConfig:
    lambda_residual: float = 1.0
    lambda_size: float = 1e-3
    lambda_steps: float = 1e-3
    lambda_policy: float = 1e-2
    use_log_residual: bool = True


@dataclass(frozen=True)
class BeamSearchStopConfig:
    max_steps: int = 10
    numeric_tol: float = 1e-10
    numeric_patience: int | None = 5
    structural_patience: int | None = None
    max_expanded_states: int | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class BeamSearchState:
    antiderivative: Expr
    prefix: str
    score: float
    numeric_residual: float | None
    symbolic_residual_prefix: str | None
    structural_distance: int | None
    steps: int
    cumulative_policy_logprob: float
    path: list[RepairStep]
    exact_symbolic_match: bool = False


@dataclass(frozen=True)
class BeamSearchResult:
    target_integrand_prefix: str
    initial_prefix: str
    best_prefix: str
    final_beam_prefixes: list[str]
    success: bool
    stop_reason: str
    steps_taken: int
    expanded_states: int
    generated_candidates: int
    applicable_candidates: int
    repeated_candidates: int
    pruned_candidates: int
    initial_numeric_residual: float | None
    best_numeric_residual: float | None
    final_best_numeric_residual: float | None
    best_structural_distance: int | None
    exact_symbolic_match: bool
    best_step_index: int | None
    path: list[RepairStep]
    per_depth_best_numeric_residual: list[float | None]
    per_depth_best_structural_distance: list[int | None]
    stop_diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _BeamCandidateForScoring:
    parent: BeamSearchState
    rank: int
    decoded_generated_tokens: list[str]
    decoded_replacement_tokens: list[str]
    decoded_status: str
    decoded_selected_node_id: int | None
    decoded_replacement_subtree_prefix: str | None
    decoded_logprob: float | None
    edited_tree: Expr
    edited_prefix: str


@dataclass(frozen=True)
class _BeamCandidateScoreRequest:
    edited_prefix: str
    target_integrand_prefix: str
    target_antiderivative_prefix: str | None
    cumulative_policy_logprob: float
    steps: int
    scoring: BeamSearchScoringConfig
    numeric_residual_timeout_seconds: float | None
    symbolic_check_timeout_seconds: float | None


@dataclass(frozen=True)
class _BeamCandidateScoreResult:
    numeric_residual: float | None
    structural_distance: int | None
    exact_symbolic_match: bool
    score: float


def score_beam_state(
    *,
    numeric_residual: float | None,
    tree_size_value: int,
    steps: int,
    cumulative_policy_logprob: float | None,
    config: BeamSearchScoringConfig,
) -> float:
    residual_term = _residual_term(
        numeric_residual,
        use_log_residual=bool(config.use_log_residual),
    )
    policy = 0.0 if cumulative_policy_logprob is None else float(cumulative_policy_logprob)
    return (
        float(config.lambda_residual) * residual_term
        + float(config.lambda_size) * int(tree_size_value)
        + float(config.lambda_steps) * int(steps)
        - float(config.lambda_policy) * policy
    )


@torch.no_grad()
def beam_search_repair(
    model: TreeDiffusionPolicyModel,
    target_integrand: Expr,
    initial_antiderivative: Expr,
    *,
    tokenizer: TreeDiffusionTokenizer,
    device: torch.device | str,
    beam_size: int = 8,
    candidate_k: int = 8,
    scoring: BeamSearchScoringConfig | None = None,
    stopping: BeamSearchStopConfig | None = None,
    constrain_position: bool = True,
    max_decode_length: int | None = None,
    residual_mode: str = "both",
    simplify_symbolic_residual: bool = True,
    observation_timeout_seconds: float | None = 2.0,
    numeric_residual_timeout_seconds: float | None = 2.0,
    symbolic_check_timeout_seconds: float | None = 2.0,
    target_antiderivative: Expr | None = None,
    residual_executor: Executor | None = None,
) -> BeamSearchResult:
    if beam_size <= 0:
        raise ValueError("beam_size must be > 0.")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be > 0.")

    scoring_config = scoring or BeamSearchScoringConfig()
    stop_config = stopping or BeamSearchStopConfig()
    _validate_stop_config(stop_config)
    _validate_scoring_config(scoring_config)

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    deadline = (
        None
        if stop_config.timeout_seconds is None
        else time.monotonic() + float(stop_config.timeout_seconds)
    )
    target = canonicalize(target_integrand, strip_additive_constants=False)
    target_prefix = serialize_prefix_string(target)
    current = canonicalize(initial_antiderivative)
    initial_prefix = serialize_prefix_string(current)
    target_antiderivative_canonical = (
        None if target_antiderivative is None else canonicalize(target_antiderivative)
    )

    initial_structural = _structural_distance_or_none(
        current,
        target_antiderivative_canonical,
    )
    try:
        initial_numeric = numeric_residual_score(
            current,
            target,
            timeout_seconds=_remaining_timeout(
                numeric_residual_timeout_seconds,
                deadline=deadline,
            ),
        )
        initial_exact = derivative_matches_target(
            current,
            target,
            timeout_seconds=_remaining_timeout(
                symbolic_check_timeout_seconds,
                deadline=deadline,
            ),
        )
    except TimeoutError:
        initial_numeric = None
        initial_exact = False
    initial_score = score_beam_state(
        numeric_residual=initial_numeric,
        tree_size_value=tree_size(current),
        steps=0,
        cumulative_policy_logprob=0.0,
        config=scoring_config,
    )
    initial_state = BeamSearchState(
        antiderivative=current,
        prefix=initial_prefix,
        score=initial_score,
        numeric_residual=initial_numeric,
        symbolic_residual_prefix=None,
        structural_distance=initial_structural,
        steps=0,
        cumulative_policy_logprob=0.0,
        path=[],
        exact_symbolic_match=initial_exact,
    )

    per_depth_numeric: list[float | None] = [initial_state.numeric_residual]
    per_depth_structural: list[int | None] = [initial_state.structural_distance]
    counters = _BeamCounters()

    if initial_exact:
        return _beam_result(
            target_integrand_prefix=target_prefix,
            initial_prefix=initial_prefix,
            best_state=initial_state,
            final_beam=[initial_state],
            stop_reason="exact_symbolic_match",
            counters=counters,
            initial_numeric_residual=initial_numeric,
            per_depth_best_numeric_residual=per_depth_numeric,
            per_depth_best_structural_distance=per_depth_structural,
            stop_diagnostics={"depth": 0},
        )
    if _meets_numeric_tol(initial_numeric, stop_config.numeric_tol):
        return _beam_result(
            target_integrand_prefix=target_prefix,
            initial_prefix=initial_prefix,
            best_state=initial_state,
            final_beam=[initial_state],
            stop_reason="numeric_tol",
            counters=counters,
            initial_numeric_residual=initial_numeric,
            per_depth_best_numeric_residual=per_depth_numeric,
            per_depth_best_structural_distance=per_depth_structural,
            stop_diagnostics={"depth": 0},
        )

    beam = [initial_state]
    best_state = initial_state
    visited_prefixes = {initial_prefix}
    numeric_stale_depths = 0
    structural_stale_depths = 0

    for depth in range(1, int(stop_config.max_steps) + 1):
        if _deadline_expired(deadline):
            return _beam_result(
                target_integrand_prefix=target_prefix,
                initial_prefix=initial_prefix,
                best_state=best_state,
                final_beam=beam,
                stop_reason="timeout",
                counters=counters,
                initial_numeric_residual=initial_numeric,
                per_depth_best_numeric_residual=per_depth_numeric,
                per_depth_best_structural_distance=per_depth_structural,
                stop_diagnostics={"depth": depth - 1},
            )

        previous_best_numeric = best_state.numeric_residual
        previous_best_structural = best_state.structural_distance
        score_inputs: list[_BeamCandidateForScoring] = []

        for state in beam:
            if (
                stop_config.max_expanded_states is not None
                and counters.expanded_states >= int(stop_config.max_expanded_states)
            ):
                break
            counters.expanded_states += 1
            try:
                _, input_ids, input_attention_mask = encode_repair_observation(
                    target,
                    state.antiderivative,
                    tokenizer=tokenizer,
                    residual_mode=residual_mode,
                    simplify_symbolic_residual=simplify_symbolic_residual,
                    observation_timeout_seconds=_remaining_timeout(
                        observation_timeout_seconds,
                        deadline=deadline,
                    ),
                    max_input_length=getattr(getattr(model, "config", None), "max_input_length", None),
                )
                input_ids = input_ids.to(target_device)
                input_attention_mask = input_attention_mask.to(target_device)
                decoded_candidates = decode_edit_candidates(
                    model,
                    input_ids,
                    tokenizer=tokenizer,
                    current_tree=state.antiderivative,
                    input_attention_mask=input_attention_mask,
                    k=int(candidate_k),
                    max_length=max_decode_length,
                    constrain_position=constrain_position,
                    device=target_device,
                )
            except TimeoutError:
                return _beam_result(
                    target_integrand_prefix=target_prefix,
                    initial_prefix=initial_prefix,
                    best_state=best_state,
                    final_beam=beam,
                    stop_reason="timeout",
                    counters=counters,
                    initial_numeric_residual=initial_numeric,
                    per_depth_best_numeric_residual=per_depth_numeric,
                    per_depth_best_structural_distance=per_depth_structural,
                    stop_diagnostics={"depth": depth},
                )
            except Exception:
                continue

            counters.generated_candidates += len(decoded_candidates)
            for rank, decoded in enumerate(decoded_candidates, start=1):
                if decoded.status != "ok":
                    continue
                try:
                    edited_tree = canonicalize(
                        apply_decoded_edit(state.antiderivative, decoded)
                    )
                except Exception:
                    continue
                counters.applicable_candidates += 1
                edited_prefix = serialize_prefix_string(edited_tree)
                if edited_prefix in visited_prefixes:
                    counters.repeated_candidates += 1
                    continue

                score_inputs.append(
                    _BeamCandidateForScoring(
                        parent=state,
                        rank=rank,
                        decoded_generated_tokens=list(decoded.generated_tokens),
                        decoded_replacement_tokens=list(decoded.replacement_tokens),
                        decoded_status=decoded.status,
                        decoded_selected_node_id=decoded.selected_node_id,
                        decoded_replacement_subtree_prefix=(
                            None
                            if decoded.replacement_subtree is None
                            else serialize_prefix_string(decoded.replacement_subtree)
                        ),
                        decoded_logprob=decoded.logprob,
                        edited_tree=edited_tree,
                        edited_prefix=edited_prefix,
                    )
                )

        if (
            stop_config.max_expanded_states is not None
            and counters.expanded_states >= int(stop_config.max_expanded_states)
        ):
            stop_reason = "max_expanded_states"
        else:
            stop_reason = None

        try:
            scored = _score_beam_candidates(
                score_inputs,
                target_integrand=target,
                target_integrand_prefix=target_prefix,
                target_antiderivative=target_antiderivative_canonical,
                scoring=scoring_config,
                numeric_residual_timeout_seconds=_remaining_timeout(
                    numeric_residual_timeout_seconds,
                    deadline=deadline,
                ),
                symbolic_check_timeout_seconds=_remaining_timeout(
                    symbolic_check_timeout_seconds,
                    deadline=deadline,
                ),
                residual_executor=residual_executor,
            )
        except TimeoutError:
            return _beam_result(
                target_integrand_prefix=target_prefix,
                initial_prefix=initial_prefix,
                best_state=best_state,
                final_beam=beam,
                stop_reason="timeout",
                counters=counters,
                initial_numeric_residual=initial_numeric,
                per_depth_best_numeric_residual=per_depth_numeric,
                per_depth_best_structural_distance=per_depth_structural,
                stop_diagnostics={"depth": depth},
            )

        candidates_by_prefix: dict[str, BeamSearchState] = {}
        for candidate, score_result in zip(score_inputs, scored, strict=True):
            cumulative_logprob = candidate.parent.cumulative_policy_logprob + (
                0.0
                if candidate.decoded_logprob is None
                else float(candidate.decoded_logprob)
            )
            path_best_numeric = _best_numeric_residual(
                _path_best_numeric(candidate.parent),
                score_result.numeric_residual,
            )
            step = RepairStep(
                step_index=candidate.parent.steps,
                current_prefix=candidate.parent.prefix,
                chosen_prefix=candidate.edited_prefix,
                decoded_status=candidate.decoded_status,
                selected_node_id=candidate.decoded_selected_node_id,
                replacement_tokens=list(candidate.decoded_replacement_tokens),
                replacement_subtree_prefix=candidate.decoded_replacement_subtree_prefix,
                candidate_rank=candidate.rank,
                policy_logprob=candidate.decoded_logprob,
                numeric_residual_before=candidate.parent.numeric_residual,
                numeric_residual_after=score_result.numeric_residual,
                best_numeric_residual_so_far=path_best_numeric,
                score=score_result.score,
                structural_distance_before=candidate.parent.structural_distance,
                structural_distance_after=score_result.structural_distance,
                exact_symbolic_match=score_result.exact_symbolic_match,
            )
            candidate_state = BeamSearchState(
                antiderivative=candidate.edited_tree,
                prefix=candidate.edited_prefix,
                score=score_result.score,
                numeric_residual=score_result.numeric_residual,
                symbolic_residual_prefix=None,
                structural_distance=score_result.structural_distance,
                steps=candidate.parent.steps + 1,
                cumulative_policy_logprob=cumulative_logprob,
                path=[*candidate.parent.path, step],
                exact_symbolic_match=score_result.exact_symbolic_match,
            )
            existing = candidates_by_prefix.get(candidate.edited_prefix)
            if existing is None:
                candidates_by_prefix[candidate.edited_prefix] = candidate_state
            elif _beam_state_dedup_better(candidate_state, existing):
                counters.repeated_candidates += 1
                candidates_by_prefix[candidate.edited_prefix] = candidate_state
            else:
                counters.repeated_candidates += 1

            if _beam_best_better(candidate_state, best_state):
                best_state = candidate_state

        if not candidates_by_prefix:
            per_depth_numeric.append(best_state.numeric_residual)
            per_depth_structural.append(best_state.structural_distance)
            return _beam_result(
                target_integrand_prefix=target_prefix,
                initial_prefix=initial_prefix,
                best_state=best_state,
                final_beam=[],
                stop_reason=stop_reason or "beam_empty",
                counters=counters,
                initial_numeric_residual=initial_numeric,
                per_depth_best_numeric_residual=per_depth_numeric,
                per_depth_best_structural_distance=per_depth_structural,
                stop_diagnostics={"depth": depth},
            )

        ordered_candidates = sorted(
            candidates_by_prefix.values(),
            key=lambda item: (item.score, _numeric_key(item.numeric_residual), item.prefix),
        )
        next_beam = ordered_candidates[: int(beam_size)]
        counters.pruned_candidates += max(0, len(ordered_candidates) - len(next_beam))
        beam = next_beam
        visited_prefixes.update(state.prefix for state in beam)

        per_depth_numeric.append(best_state.numeric_residual)
        per_depth_structural.append(best_state.structural_distance)

        if best_state.exact_symbolic_match:
            stop_reason = "exact_symbolic_match"
        elif _meets_numeric_tol(best_state.numeric_residual, stop_config.numeric_tol):
            stop_reason = "numeric_tol"
        elif stop_reason is None:
            if _numeric_better(best_state.numeric_residual, previous_best_numeric):
                numeric_stale_depths = 0
            else:
                numeric_stale_depths += 1

            if (
                stop_config.structural_patience is not None
                and target_antiderivative_canonical is not None
            ):
                if _structural_better(best_state.structural_distance, previous_best_structural):
                    structural_stale_depths = 0
                else:
                    structural_stale_depths += 1
            if (
                stop_config.numeric_patience is not None
                and numeric_stale_depths >= int(stop_config.numeric_patience)
            ):
                stop_reason = "numeric_patience"
            elif (
                stop_config.structural_patience is not None
                and target_antiderivative_canonical is not None
                and structural_stale_depths >= int(stop_config.structural_patience)
            ):
                stop_reason = "structural_patience"

        if stop_reason is not None:
            return _beam_result(
                target_integrand_prefix=target_prefix,
                initial_prefix=initial_prefix,
                best_state=best_state,
                final_beam=beam,
                stop_reason=stop_reason,
                counters=counters,
                initial_numeric_residual=initial_numeric,
                per_depth_best_numeric_residual=per_depth_numeric,
                per_depth_best_structural_distance=per_depth_structural,
                stop_diagnostics={
                    "depth": depth,
                    "numeric_stale_depths": numeric_stale_depths,
                    "structural_stale_depths": structural_stale_depths,
                },
            )

    return _beam_result(
        target_integrand_prefix=target_prefix,
        initial_prefix=initial_prefix,
        best_state=best_state,
        final_beam=beam,
        stop_reason="max_steps",
        counters=counters,
        initial_numeric_residual=initial_numeric,
        per_depth_best_numeric_residual=per_depth_numeric,
        per_depth_best_structural_distance=per_depth_structural,
        stop_diagnostics={"depth": int(stop_config.max_steps)},
    )


@torch.no_grad()
def beam_search_repair_from_seeds(
    model: TreeDiffusionPolicyModel,
    target_integrand: Expr,
    seeds: Sequence[Expr],
    *,
    tokenizer: TreeDiffusionTokenizer,
    device: torch.device | str,
    beam_size: int = 8,
    candidate_k: int = 8,
    scoring: BeamSearchScoringConfig | None = None,
    stopping: BeamSearchStopConfig | None = None,
    constrain_position: bool = True,
    max_decode_length: int | None = None,
    residual_mode: str = "both",
    simplify_symbolic_residual: bool = True,
    observation_timeout_seconds: float | None = 2.0,
    numeric_residual_timeout_seconds: float | None = 2.0,
    symbolic_check_timeout_seconds: float | None = 2.0,
    target_antiderivative: Expr | None = None,
    residual_executor: Executor | None = None,
) -> BeamSearchResult:
    if not seeds:
        raise ValueError("seeds must be non-empty.")

    results: list[BeamSearchResult] = []
    for seed in seeds:
        result = beam_search_repair(
            model,
            target_integrand,
            seed,
            tokenizer=tokenizer,
            device=device,
            beam_size=beam_size,
            candidate_k=candidate_k,
            scoring=scoring,
            stopping=stopping,
            constrain_position=constrain_position,
            max_decode_length=max_decode_length,
            residual_mode=residual_mode,
            simplify_symbolic_residual=simplify_symbolic_residual,
            observation_timeout_seconds=observation_timeout_seconds,
            numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
            symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
            target_antiderivative=target_antiderivative,
            residual_executor=residual_executor,
        )
        if result.success and result.exact_symbolic_match:
            return result
        results.append(result)

    finite_results = [
        result
        for result in results
        if _finite_numeric(result.best_numeric_residual)
    ]
    if finite_results:
        return min(
            finite_results,
            key=lambda result: (
                float(result.best_numeric_residual),  # type: ignore[arg-type]
                result.steps_taken,
            ),
        )
    return min(results, key=lambda result: result.stop_diagnostics.get("best_score", math.inf))


@dataclass
class _BeamCounters:
    expanded_states: int = 0
    generated_candidates: int = 0
    applicable_candidates: int = 0
    repeated_candidates: int = 0
    pruned_candidates: int = 0


def _beam_result(
    *,
    target_integrand_prefix: str,
    initial_prefix: str,
    best_state: BeamSearchState,
    final_beam: Sequence[BeamSearchState],
    stop_reason: str,
    counters: _BeamCounters,
    initial_numeric_residual: float | None,
    per_depth_best_numeric_residual: list[float | None],
    per_depth_best_structural_distance: list[int | None],
    stop_diagnostics: dict[str, Any],
) -> BeamSearchResult:
    final_best_numeric = _best_numeric_from_states(final_beam)
    diagnostics = {
        **stop_diagnostics,
        "best_score": best_state.score,
        "final_beam_size": len(final_beam),
    }
    return BeamSearchResult(
        target_integrand_prefix=target_integrand_prefix,
        initial_prefix=initial_prefix,
        best_prefix=best_state.prefix,
        final_beam_prefixes=[state.prefix for state in final_beam],
        success=stop_reason in {"exact_symbolic_match", "numeric_tol"},
        stop_reason=stop_reason,
        steps_taken=best_state.steps,
        expanded_states=counters.expanded_states,
        generated_candidates=counters.generated_candidates,
        applicable_candidates=counters.applicable_candidates,
        repeated_candidates=counters.repeated_candidates,
        pruned_candidates=counters.pruned_candidates,
        initial_numeric_residual=initial_numeric_residual,
        best_numeric_residual=best_state.numeric_residual,
        final_best_numeric_residual=final_best_numeric,
        best_structural_distance=best_state.structural_distance,
        exact_symbolic_match=bool(best_state.exact_symbolic_match),
        best_step_index=best_state.steps,
        path=list(best_state.path),
        per_depth_best_numeric_residual=list(per_depth_best_numeric_residual),
        per_depth_best_structural_distance=list(per_depth_best_structural_distance),
        stop_diagnostics=diagnostics,
    )


def _residual_term(numeric_residual: float | None, *, use_log_residual: bool) -> float:
    if numeric_residual is None:
        return _MISSING_NUMERIC_RESIDUAL_PENALTY
    value = float(numeric_residual)
    if not math.isfinite(value):
        return _MISSING_NUMERIC_RESIDUAL_PENALTY
    if not use_log_residual:
        return value
    return math.log1p(max(0.0, value))


def _score_beam_candidates(
    candidates: Sequence[_BeamCandidateForScoring],
    *,
    target_integrand: Expr,
    target_integrand_prefix: str,
    target_antiderivative: Expr | None,
    scoring: BeamSearchScoringConfig,
    numeric_residual_timeout_seconds: float | None,
    symbolic_check_timeout_seconds: float | None,
    residual_executor: Executor | None,
) -> list[_BeamCandidateScoreResult]:
    if not candidates:
        return []
    if residual_executor is None:
        return [
            _score_beam_candidate_local(
                candidate,
                target_integrand=target_integrand,
                target_antiderivative=target_antiderivative,
                scoring=scoring,
                numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
                symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
            )
            for candidate in candidates
        ]

    target_antiderivative_prefix = (
        None
        if target_antiderivative is None
        else serialize_prefix_string(target_antiderivative)
    )
    requests = [
        _BeamCandidateScoreRequest(
            edited_prefix=candidate.edited_prefix,
            target_integrand_prefix=target_integrand_prefix,
            target_antiderivative_prefix=target_antiderivative_prefix,
            cumulative_policy_logprob=candidate.parent.cumulative_policy_logprob
            + (0.0 if candidate.decoded_logprob is None else float(candidate.decoded_logprob)),
            steps=candidate.parent.steps + 1,
            scoring=scoring,
            numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
            symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
        )
        for candidate in candidates
    ]
    return list(residual_executor.map(_score_beam_candidate_worker, requests))


def _score_beam_candidate_local(
    candidate: _BeamCandidateForScoring,
    *,
    target_integrand: Expr,
    target_antiderivative: Expr | None,
    scoring: BeamSearchScoringConfig,
    numeric_residual_timeout_seconds: float | None,
    symbolic_check_timeout_seconds: float | None,
) -> _BeamCandidateScoreResult:
    numeric = numeric_residual_score(
        candidate.edited_tree,
        target_integrand,
        timeout_seconds=numeric_residual_timeout_seconds,
    )
    cumulative_logprob = candidate.parent.cumulative_policy_logprob + (
        0.0 if candidate.decoded_logprob is None else float(candidate.decoded_logprob)
    )
    return _BeamCandidateScoreResult(
        numeric_residual=numeric,
        structural_distance=_structural_distance_or_none(candidate.edited_tree, target_antiderivative),
        exact_symbolic_match=derivative_matches_target(
            candidate.edited_tree,
            target_integrand,
            timeout_seconds=symbolic_check_timeout_seconds,
        ),
        score=score_beam_state(
            numeric_residual=numeric,
            tree_size_value=tree_size(candidate.edited_tree),
            steps=candidate.parent.steps + 1,
            cumulative_policy_logprob=cumulative_logprob,
            config=scoring,
        ),
    )


def _score_beam_candidate_worker(
    request: _BeamCandidateScoreRequest,
) -> _BeamCandidateScoreResult:
    try:
        edited_tree = canonicalize(parse_prefix_string(request.edited_prefix))
        target_integrand = canonicalize(
            parse_prefix_string(request.target_integrand_prefix),
            strip_additive_constants=False,
        )
        target_antiderivative = (
            None
            if request.target_antiderivative_prefix is None
            else canonicalize(parse_prefix_string(request.target_antiderivative_prefix))
        )
        numeric = numeric_residual_score(
            edited_tree,
            target_integrand,
            timeout_seconds=request.numeric_residual_timeout_seconds,
        )
        return _BeamCandidateScoreResult(
            numeric_residual=numeric,
            structural_distance=_structural_distance_or_none(edited_tree, target_antiderivative),
            exact_symbolic_match=derivative_matches_target(
                edited_tree,
                target_integrand,
                timeout_seconds=request.symbolic_check_timeout_seconds,
            ),
            score=score_beam_state(
                numeric_residual=numeric,
                tree_size_value=tree_size(edited_tree),
                steps=request.steps,
                cumulative_policy_logprob=request.cumulative_policy_logprob,
                config=request.scoring,
            ),
        )
    except Exception:
        return _BeamCandidateScoreResult(
            numeric_residual=None,
            structural_distance=None,
            exact_symbolic_match=False,
            score=score_beam_state(
                numeric_residual=None,
                tree_size_value=0,
                steps=request.steps,
                cumulative_policy_logprob=request.cumulative_policy_logprob,
                config=request.scoring,
            ),
        )


def _validate_scoring_config(config: BeamSearchScoringConfig) -> None:
    for name in ("lambda_residual", "lambda_size", "lambda_steps", "lambda_policy"):
        value = getattr(config, name)
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")


def _validate_stop_config(config: BeamSearchStopConfig) -> None:
    if config.max_steps < 0:
        raise ValueError("max_steps must be >= 0.")
    if config.numeric_tol < 0.0:
        raise ValueError("numeric_tol must be >= 0.")
    if config.numeric_patience is not None and config.numeric_patience < 1:
        raise ValueError("numeric_patience must be >= 1 or None.")
    if config.structural_patience is not None and config.structural_patience < 1:
        raise ValueError("structural_patience must be >= 1 or None.")
    if config.max_expanded_states is not None and config.max_expanded_states < 1:
        raise ValueError("max_expanded_states must be >= 1 or None.")
    if config.timeout_seconds is not None and config.timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be > 0 or None.")


def _path_best_numeric(state: BeamSearchState) -> float | None:
    if not state.path:
        return state.numeric_residual
    return state.path[-1].best_numeric_residual_so_far


def _beam_best_better(candidate: BeamSearchState, current: BeamSearchState) -> bool:
    candidate_numeric = _numeric_key(candidate.numeric_residual)
    current_numeric = _numeric_key(current.numeric_residual)
    if candidate_numeric != current_numeric:
        return candidate_numeric < current_numeric
    if candidate.exact_symbolic_match != current.exact_symbolic_match:
        return candidate.exact_symbolic_match
    if candidate.score != current.score:
        return candidate.score < current.score
    return tree_size(candidate.antiderivative) < tree_size(current.antiderivative)


def _beam_state_dedup_better(candidate: BeamSearchState, current: BeamSearchState) -> bool:
    candidate_numeric = _numeric_key(candidate.numeric_residual)
    current_numeric = _numeric_key(current.numeric_residual)
    if candidate_numeric != current_numeric:
        return candidate_numeric < current_numeric
    if candidate.exact_symbolic_match != current.exact_symbolic_match:
        return candidate.exact_symbolic_match
    if candidate.score != current.score:
        return candidate.score < current.score
    return tree_size(candidate.antiderivative) < tree_size(current.antiderivative)


def _best_numeric_from_states(states: Sequence[BeamSearchState]) -> float | None:
    best: float | None = None
    for state in states:
        best = _best_numeric_residual(best, state.numeric_residual)
    return best


__all__ = [
    "BeamSearchResult",
    "BeamSearchScoringConfig",
    "BeamSearchState",
    "BeamSearchStopConfig",
    "beam_search_repair",
    "beam_search_repair_from_seeds",
    "score_beam_state",
]
