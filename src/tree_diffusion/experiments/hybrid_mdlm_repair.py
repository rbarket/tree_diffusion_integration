from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from src.mathlang.ast import Const, Expr, Var
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion._common import (
    json_safe as _json_safe,
    mean_or_none as _mean_or_none,
    resolve_device as _resolve_device,
    write_json as _write_json,
)
from src.tree_diffusion.beam_search import (
    BeamSearchResult,
    BeamSearchStopConfig,
    beam_search_repair_from_seeds,
)
from src.tree_diffusion.evaluation_common import (
    residual_executor_context as _residual_executor_context,
)
from src.tree_diffusion.eval_one_step import numeric_residual_score
from src.tree_diffusion.repair import derivative_matches_target
from src.tree_diffusion.runtime import (
    load_model_and_tokenizer_for_inference as _load_cli_model_and_tokenizer,
)


_VALID_SEED_SELECTIONS = {"all_parseable", "first_parseable", "best_numeric_seed"}
_MASK_OR_UNK_TOKENS = {"<mask>", "<unk>"}


@dataclass(frozen=True)
class MdlmSeedParseResult:
    attempt_index: int
    ok: bool
    seed: Expr | None
    normalized_prefix: str | None
    error: str | None
    contains_mask_or_unk: bool
    contains_complex_constant: bool


@dataclass(frozen=True)
class HybridRepairExampleResult:
    row_index: int
    pair_index: int | None
    target_integrand_prefix: str
    target_antiderivative_prefix: str | None
    num_mdlm_attempts: int
    num_parseable_mdlm_seeds: int
    first_parseable_attempt_index: int | None
    parseable_attempt_indices: list[int]
    mdlm_attempt_prefixes: list[str]
    mdlm_parse_errors: list[str | None]
    mdlm_any_seed_parse_ok: bool
    mdlm_no_parseable_seed: bool
    mdlm_any_seed_exact_symbolic_match: bool
    mdlm_first_parseable_exact_symbolic_match: bool
    repair_attempted: bool
    tree_repair_success: bool
    hybrid_success: bool
    hybrid_exact_symbolic_match: bool
    failure_stage: str | None
    repair_gain: bool
    regression: bool
    initial_best_mdlm_numeric_residual: float | None
    final_numeric_residual: float | None
    best_numeric_residual: float | None
    beam_stop_reason: str | None
    beam_steps_taken: int | None
    expanded_states: int | None
    fallback_used: bool = False


@dataclass(frozen=True)
class HybridMdlmRepairSummary:
    examples: int
    mdlm_first_attempt_parseable_rate: float
    mdlm_any_attempt_parseable_rate: float
    mdlm_no_parseable_seed_rate: float
    mean_parseable_attempts_per_example: float
    mean_first_parseable_attempt_index: float | None
    mdlm_first_parseable_exact_rate_over_all: float
    mdlm_first_parseable_exact_rate_over_parseable: float | None
    mdlm_any_parseable_seed_exact_rate_over_all: float
    mdlm_any_parseable_seed_exact_rate_over_parseable: float | None
    repair_attempt_rate: float
    tree_repair_failure_rate_over_all: float
    tree_repair_failure_rate_over_parseable: float | None
    hybrid_success_rate_over_all: float
    hybrid_success_rate_over_parseable: float | None
    repair_gain_rate_over_all: float
    repair_gain_rate_over_initially_incorrect_parseable: float | None
    regression_rate_over_initially_correct: float | None
    mean_initial_best_mdlm_numeric_residual_parseable: float | None
    mean_final_numeric_residual_repaired: float | None
    mean_best_numeric_residual_repaired: float | None
    numeric_residual_improvement_rate_repaired: float | None
    failure_stage_counts: dict[str, int]
    beam_stop_reason_counts: dict[str, int]
    parse_error_counts: dict[str, int]
    complex_seed_rate: float


def parse_mdlm_seed(
    pred_prefix: str,
    *,
    attempt_index: int = 0,
    reject_mask_or_unk: bool = True,
    allow_complex_constant: bool = True,
) -> MdlmSeedParseResult:
    tokens = str(pred_prefix).split()
    contains_mask_or_unk = any(token in _MASK_OR_UNK_TOKENS for token in tokens)
    contains_complex_constant = "I" in tokens
    if reject_mask_or_unk and contains_mask_or_unk:
        return MdlmSeedParseResult(
            attempt_index=int(attempt_index),
            ok=False,
            seed=None,
            normalized_prefix=None,
            error="contains_mask_or_unk",
            contains_mask_or_unk=True,
            contains_complex_constant=contains_complex_constant,
        )
    if contains_complex_constant and not allow_complex_constant:
        return MdlmSeedParseResult(
            attempt_index=int(attempt_index),
            ok=False,
            seed=None,
            normalized_prefix=None,
            error="contains_complex_constant",
            contains_mask_or_unk=contains_mask_or_unk,
            contains_complex_constant=True,
        )

    try:
        seed = canonicalize(parse_prefix_string(" ".join(tokens)))
        normalized_prefix = serialize_prefix_string(seed)
    except Exception as exc:  # noqa: BLE001 - parse failures are exported as data.
        return MdlmSeedParseResult(
            attempt_index=int(attempt_index),
            ok=False,
            seed=None,
            normalized_prefix=None,
            error=f"{type(exc).__name__}: {exc}",
            contains_mask_or_unk=contains_mask_or_unk,
            contains_complex_constant=contains_complex_constant,
        )
    return MdlmSeedParseResult(
        attempt_index=int(attempt_index),
        ok=True,
        seed=seed,
        normalized_prefix=normalized_prefix,
        error=None,
        contains_mask_or_unk=contains_mask_or_unk,
        contains_complex_constant=contains_complex_constant,
    )


def parse_mdlm_seed_attempts(
    attempts: Sequence[Mapping[str, Any]],
    *,
    reject_mask_or_unk: bool = True,
    allow_complex_constant: bool = True,
) -> list[MdlmSeedParseResult]:
    results: list[MdlmSeedParseResult] = []
    for fallback_index, attempt in enumerate(attempts):
        attempt_index = _attempt_index(attempt, fallback=fallback_index)
        results.append(
            parse_mdlm_seed(
                str(attempt.get("pred_prefix", "")),
                attempt_index=attempt_index,
                reject_mask_or_unk=reject_mask_or_unk,
                allow_complex_constant=allow_complex_constant,
            )
        )
    return results


@torch.no_grad()
def evaluate_hybrid_mdlm_repair(
    *,
    predictions_path: str | Path,
    tree_checkpoint: str | Path,
    output_path: str | Path | None = None,
    examples_out_path: str | Path | None = None,
    examples_parts_dir: str | Path | None = None,
    device: str = "auto",
    limit: int | None = None,
    start: int = 0,
    beam_size: int = 8,
    candidate_k: int = 8,
    max_steps: int = 10,
    numeric_patience: int | None = 5,
    structural_patience: int | None = None,
    numeric_tol: float = 1e-10,
    numeric_residual_timeout_seconds: float | None = 2.0,
    symbolic_check_timeout_seconds: float | None = 2.0,
    reject_mask_or_unk: bool = True,
    allow_complex_constant: bool = True,
    seed_selection: str = "all_parseable",
    use_fallback_seeds: bool = False,
    part_size: int = 500,
    progress_every: int = 25,
    residual_workers: int = 16,
    resume: bool = False,
    overwrite: bool = False,
    progress: bool = True,
) -> HybridMdlmRepairSummary:
    _validate_eval_args(
        start=start,
        limit=limit,
        beam_size=beam_size,
        candidate_k=candidate_k,
        max_steps=max_steps,
        numeric_patience=numeric_patience,
        structural_patience=structural_patience,
        numeric_tol=numeric_tol,
        numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
        symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
        seed_selection=seed_selection,
        part_size=part_size,
        progress_every=progress_every,
        residual_workers=residual_workers,
        resume=resume,
        overwrite=overwrite,
    )
    target_device = _resolve_device(str(device))
    tokenizer, model = _load_cli_model_and_tokenizer(
        checkpoint=str(tree_checkpoint),
        precomputed_data_dir=None,
        allow_random_init_model=False,
    )
    model.to(target_device)
    model.eval()

    grouped = _prediction_groups(_load_prediction_rows(predictions_path))
    end = None if limit is None else start + int(limit)
    selected_groups = grouped[start:end]
    target_examples = len(selected_groups)
    stopping = BeamSearchStopConfig(
        max_steps=int(max_steps),
        numeric_tol=float(numeric_tol),
        numeric_patience=numeric_patience,
        structural_patience=structural_patience,
    )

    parts_dir = None if examples_parts_dir is None else Path(examples_parts_dir)
    existing_results: list[HybridRepairExampleResult] = []
    completed_examples = 0
    next_part_index = 0
    if parts_dir is not None:
        if overwrite and parts_dir.exists():
            shutil.rmtree(parts_dir)
        parts_dir.mkdir(parents=True, exist_ok=True)
        existing_part_paths = _examples_part_paths(parts_dir)
        if resume:
            existing_results = _load_examples_part_results(parts_dir)
            completed_examples = len(existing_results)
            if completed_examples > target_examples:
                raise ValueError(
                    "Cannot resume: completed part rows exceed requested target examples "
                    f"({completed_examples} > {target_examples})."
                )
            next_part_index = _next_examples_part_index(parts_dir)
        elif existing_part_paths:
            raise FileExistsError(
                f"Examples parts already exist in {parts_dir}. Use --resume to continue "
                "or --overwrite to replace them."
            )
    current_part: list[HybridRepairExampleResult] = []
    results: list[HybridRepairExampleResult] = list(existing_results)
    _progress(
        "hybrid_mdlm_repair_start "
        f"target={target_examples} start={int(start)} limit={limit} "
        f"part_size={int(part_size)} progress_every={int(progress_every)} "
        f"residual_workers={int(residual_workers)} resume={bool(resume)} "
        f"completed={completed_examples} next_part={next_part_index:06d}",
        enabled=progress,
    )

    with _residual_executor_context(int(residual_workers)) as residual_executor:
        for local_offset, group in enumerate(selected_groups[completed_examples:]):
            group_ordinal = int(start) + completed_examples + local_offset
            result = _evaluate_group(
                group=group,
                group_ordinal=group_ordinal,
                model=model,
                tokenizer=tokenizer,
                device=target_device,
                beam_size=int(beam_size),
                candidate_k=int(candidate_k),
                stopping=stopping,
                reject_mask_or_unk=bool(reject_mask_or_unk),
                allow_complex_constant=bool(allow_complex_constant),
                numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
                symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
                seed_selection=str(seed_selection),
                use_fallback_seeds=bool(use_fallback_seeds),
                residual_executor=residual_executor,
            )
            results.append(result)
            current_part.append(result)
            completed = len(results)

            if progress_every > 0 and completed % int(progress_every) == 0:
                _progress(
                    "hybrid_mdlm_repair_progress "
                    f"completed={completed}/{target_examples} "
                    f"current_part_rows={len(current_part)} "
                    f"last_failure_stage={result.failure_stage} "
                    f"last_hybrid_success={result.hybrid_success} "
                    f"parseable_seeds={result.num_parseable_mdlm_seeds} "
                    f"expanded_states={result.expanded_states}",
                    enabled=progress,
                )

            if parts_dir is not None and len(current_part) >= int(part_size):
                _write_examples_part(parts_dir, next_part_index, current_part)
                _progress(
                    "hybrid_mdlm_repair_part_written "
                    f"part={next_part_index:06d} rows={len(current_part)} "
                    f"completed={completed}/{target_examples}",
                    enabled=progress,
                )
                next_part_index += 1
                current_part = []

    if parts_dir is not None and current_part:
        _write_examples_part(parts_dir, next_part_index, current_part)
        _progress(
            "hybrid_mdlm_repair_part_written "
            f"part={next_part_index:06d} rows={len(current_part)} "
            f"completed={len(results)}/{target_examples}",
            enabled=progress,
        )
        next_part_index += 1

    summary = summarize_hybrid_mdlm_repair_results(results)

    if examples_out_path is not None:
        _write_examples_jsonl(Path(examples_out_path), results)
    if parts_dir is not None:
        _write_examples_parts_manifest(
            parts_dir,
            target_examples=target_examples,
            completed_examples=len(results),
            part_count=next_part_index,
        )
    if output_path is not None:
        _write_json(Path(output_path), asdict(summary))
    _progress(
        "hybrid_mdlm_repair_complete "
        f"completed={len(results)}/{target_examples} "
        f"hybrid_success_rate={summary.hybrid_success_rate_over_all:.6f} "
        f"mdlm_no_parseable_seed_rate={summary.mdlm_no_parseable_seed_rate:.6f}",
        enabled=progress,
    )
    return summary


def summarize_hybrid_mdlm_repair_results(
    results: Sequence[HybridRepairExampleResult],
) -> HybridMdlmRepairSummary:
    examples = len(results)
    parseable = [result for result in results if result.mdlm_any_seed_parse_ok]
    parseable_count = len(parseable)
    initially_correct = [
        result for result in parseable if result.mdlm_any_seed_exact_symbolic_match
    ]
    initially_incorrect_parseable = [
        result for result in parseable if not result.mdlm_any_seed_exact_symbolic_match
    ]
    repaired_numeric_pairs = [
        (result.initial_best_mdlm_numeric_residual, result.final_numeric_residual)
        for result in results
        if result.repair_attempted
        and result.initial_best_mdlm_numeric_residual is not None
        and result.final_numeric_residual is not None
    ]
    improved_numeric = sum(
        1
        for before, after in repaired_numeric_pairs
        if _finite(before) and _finite(after) and float(after) < float(before)
    )

    failure_stage_counts: Counter[str] = Counter(
        result.failure_stage for result in results if result.failure_stage is not None
    )
    beam_stop_reason_counts: Counter[str] = Counter(
        result.beam_stop_reason for result in results if result.beam_stop_reason is not None
    )
    parse_error_counts: Counter[str] = Counter()
    for result in results:
        for error in result.mdlm_parse_errors:
            if error is not None:
                parse_error_counts[_parse_error_bucket(error)] += 1

    return HybridMdlmRepairSummary(
        examples=int(examples),
        mdlm_first_attempt_parseable_rate=_rate(
            sum(1 for result in results if 0 in result.parseable_attempt_indices),
            examples,
        ),
        mdlm_any_attempt_parseable_rate=_rate(parseable_count, examples),
        mdlm_no_parseable_seed_rate=_rate(
            sum(1 for result in results if result.mdlm_no_parseable_seed),
            examples,
        ),
        mean_parseable_attempts_per_example=(
            float(sum(result.num_parseable_mdlm_seeds for result in results)) / float(examples)
            if examples
            else 0.0
        ),
        mean_first_parseable_attempt_index=_mean_or_none(
            [
                int(result.first_parseable_attempt_index)
                for result in parseable
                if result.first_parseable_attempt_index is not None
            ]
        ),
        mdlm_first_parseable_exact_rate_over_all=_rate(
            sum(1 for result in results if result.mdlm_first_parseable_exact_symbolic_match),
            examples,
        ),
        mdlm_first_parseable_exact_rate_over_parseable=_optional_rate(
            sum(1 for result in parseable if result.mdlm_first_parseable_exact_symbolic_match),
            parseable_count,
        ),
        mdlm_any_parseable_seed_exact_rate_over_all=_rate(
            sum(1 for result in results if result.mdlm_any_seed_exact_symbolic_match),
            examples,
        ),
        mdlm_any_parseable_seed_exact_rate_over_parseable=_optional_rate(
            sum(1 for result in parseable if result.mdlm_any_seed_exact_symbolic_match),
            parseable_count,
        ),
        repair_attempt_rate=_rate(
            sum(1 for result in results if result.repair_attempted),
            examples,
        ),
        tree_repair_failure_rate_over_all=_rate(
            sum(1 for result in results if result.failure_stage == "tree_repair_failed"),
            examples,
        ),
        tree_repair_failure_rate_over_parseable=_optional_rate(
            sum(1 for result in parseable if result.failure_stage == "tree_repair_failed"),
            parseable_count,
        ),
        hybrid_success_rate_over_all=_rate(
            sum(1 for result in results if result.hybrid_success),
            examples,
        ),
        hybrid_success_rate_over_parseable=_optional_rate(
            sum(1 for result in parseable if result.hybrid_success),
            parseable_count,
        ),
        repair_gain_rate_over_all=_rate(
            sum(1 for result in results if result.repair_gain),
            examples,
        ),
        repair_gain_rate_over_initially_incorrect_parseable=_optional_rate(
            sum(1 for result in initially_incorrect_parseable if result.repair_gain),
            len(initially_incorrect_parseable),
        ),
        regression_rate_over_initially_correct=_optional_rate(
            sum(1 for result in initially_correct if result.regression),
            len(initially_correct),
        ),
        mean_initial_best_mdlm_numeric_residual_parseable=_mean_or_none(
            [
                float(result.initial_best_mdlm_numeric_residual)
                for result in parseable
                if result.initial_best_mdlm_numeric_residual is not None
            ]
        ),
        mean_final_numeric_residual_repaired=_mean_or_none(
            [
                float(result.final_numeric_residual)
                for result in results
                if result.repair_attempted and result.final_numeric_residual is not None
            ]
        ),
        mean_best_numeric_residual_repaired=_mean_or_none(
            [
                float(result.best_numeric_residual)
                for result in results
                if result.repair_attempted and result.best_numeric_residual is not None
            ]
        ),
        numeric_residual_improvement_rate_repaired=_optional_rate(
            improved_numeric,
            len(repaired_numeric_pairs),
        ),
        failure_stage_counts=dict(failure_stage_counts),
        beam_stop_reason_counts=dict(beam_stop_reason_counts),
        parse_error_counts=dict(parse_error_counts),
        complex_seed_rate=_rate(
            sum(1 for result in results if _example_has_complex_seed(result)),
            examples,
        ),
    )


def _evaluate_group(
    *,
    group: Sequence[Mapping[str, Any]],
    group_ordinal: int,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    beam_size: int,
    candidate_k: int,
    stopping: BeamSearchStopConfig,
    reject_mask_or_unk: bool,
    allow_complex_constant: bool,
    numeric_residual_timeout_seconds: float | None,
    symbolic_check_timeout_seconds: float | None,
    seed_selection: str,
    use_fallback_seeds: bool,
    residual_executor: Any,
) -> HybridRepairExampleResult:
    attempts = sorted(
        group,
        key=lambda row: _attempt_index(row, fallback=0),
    )
    first = attempts[0]
    row_index = _row_index(first, fallback=group_ordinal)
    pair_index = _optional_int(first.get("pair_index"))
    integrand_prefix = str(first.get("integrand_prefix", ""))
    target_antiderivative_prefix = _optional_nonempty_str(
        first.get("target_antiderivative_prefix")
    )
    parse_results = parse_mdlm_seed_attempts(
        attempts,
        reject_mask_or_unk=reject_mask_or_unk,
        allow_complex_constant=allow_complex_constant,
    )
    parseable_results = [result for result in parse_results if result.ok and result.seed is not None]
    parseable_attempt_indices = [int(result.attempt_index) for result in parseable_results]
    first_parseable_attempt_index = (
        parseable_attempt_indices[0] if parseable_attempt_indices else None
    )
    attempt_prefixes = [str(attempt.get("pred_prefix", "")) for attempt in attempts]
    parse_errors = [result.error for result in parse_results]

    target_integrand = _parse_target_integrand(integrand_prefix)
    if target_integrand is None:
        return _example_result(
            row_index=row_index,
            pair_index=pair_index,
            integrand_prefix=integrand_prefix,
            target_antiderivative_prefix=target_antiderivative_prefix,
            parse_results=parse_results,
            attempt_prefixes=attempt_prefixes,
            parse_errors=parse_errors,
            failure_stage="integrand_parse_failed",
        )

    target_antiderivative: Expr | None = None
    if target_antiderivative_prefix is not None:
        try:
            target_antiderivative = canonicalize(parse_prefix_string(target_antiderivative_prefix))
        except Exception:
            return _example_result(
                row_index=row_index,
                pair_index=pair_index,
                integrand_prefix=integrand_prefix,
                target_antiderivative_prefix=target_antiderivative_prefix,
                parse_results=parse_results,
                attempt_prefixes=attempt_prefixes,
                parse_errors=parse_errors,
                failure_stage="target_parse_failed",
            )

    if not parseable_results:
        return _example_result(
            row_index=row_index,
            pair_index=pair_index,
            integrand_prefix=integrand_prefix,
            target_antiderivative_prefix=target_antiderivative_prefix,
            parse_results=parse_results,
            attempt_prefixes=attempt_prefixes,
            parse_errors=parse_errors,
            failure_stage="mdlm_no_parseable_seed",
        )

    initial_exact_by_attempt = {
        int(result.attempt_index): _safe_exact_match(
            result.seed,
            target_integrand,
            timeout_seconds=symbolic_check_timeout_seconds,
        )
        for result in parseable_results
        if result.seed is not None
    }
    initial_numeric_by_attempt = {
        int(result.attempt_index): _safe_numeric_residual(
            result.seed,
            target_integrand,
            timeout_seconds=numeric_residual_timeout_seconds,
        )
        for result in parseable_results
        if result.seed is not None
    }
    any_initial_exact = any(initial_exact_by_attempt.values())
    first_initial_exact = (
        bool(initial_exact_by_attempt.get(first_parseable_attempt_index, False))
        if first_parseable_attempt_index is not None
        else False
    )
    selected_results = _select_seed_results(
        parseable_results,
        seed_selection=seed_selection,
        numeric_by_attempt=initial_numeric_by_attempt,
    )
    selected_seeds = [result.seed for result in selected_results if result.seed is not None]
    fallback_used = bool(use_fallback_seeds and selected_seeds)
    if fallback_used:
        selected_seeds = [*selected_seeds, Const(value=0), Var(name="x")]

    beam_result: BeamSearchResult | None = None
    beam_exception: str | None = None
    try:
        beam_result = beam_search_repair_from_seeds(
            model,
            target_integrand,
            selected_seeds,
            tokenizer=tokenizer,
            device=device,
            beam_size=beam_size,
            candidate_k=candidate_k,
            stopping=stopping,
            target_antiderivative=target_antiderivative,
            numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
            symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
            residual_executor=residual_executor,
        )
    except Exception as exc:  # noqa: BLE001 - one repair failure should not drop the example.
        beam_exception = f"{type(exc).__name__}: {exc}"

    tree_repair_success = bool(beam_result is not None and beam_result.success)
    beam_exact = bool(beam_result is not None and beam_result.exact_symbolic_match)
    hybrid_exact = bool(any_initial_exact or beam_exact)
    hybrid_success = bool(any_initial_exact or tree_repair_success)
    failure_stage = None if hybrid_success else "tree_repair_failed"
    return _example_result(
        row_index=row_index,
        pair_index=pair_index,
        integrand_prefix=integrand_prefix,
        target_antiderivative_prefix=target_antiderivative_prefix,
        parse_results=parse_results,
        attempt_prefixes=attempt_prefixes,
        parse_errors=parse_errors,
        mdlm_any_seed_exact_symbolic_match=any_initial_exact,
        mdlm_first_parseable_exact_symbolic_match=first_initial_exact,
        repair_attempted=True,
        tree_repair_success=tree_repair_success,
        hybrid_success=hybrid_success,
        hybrid_exact_symbolic_match=hybrid_exact,
        failure_stage=failure_stage,
        repair_gain=bool((not any_initial_exact) and tree_repair_success),
        regression=bool(any_initial_exact and not hybrid_exact),
        initial_best_mdlm_numeric_residual=_best_numeric(
            initial_numeric_by_attempt.values()
        ),
        final_numeric_residual=(
            None if beam_result is None else beam_result.final_best_numeric_residual
        ),
        best_numeric_residual=None if beam_result is None else beam_result.best_numeric_residual,
        beam_stop_reason=(
            None
            if beam_result is None and beam_exception is None
            else beam_exception if beam_result is None else beam_result.stop_reason
        ),
        beam_steps_taken=None if beam_result is None else beam_result.steps_taken,
        expanded_states=None if beam_result is None else beam_result.expanded_states,
        fallback_used=fallback_used,
    )


def _example_result(
    *,
    row_index: int,
    pair_index: int | None,
    integrand_prefix: str,
    target_antiderivative_prefix: str | None,
    parse_results: Sequence[MdlmSeedParseResult],
    attempt_prefixes: Sequence[str],
    parse_errors: Sequence[str | None],
    mdlm_any_seed_exact_symbolic_match: bool = False,
    mdlm_first_parseable_exact_symbolic_match: bool = False,
    repair_attempted: bool = False,
    tree_repair_success: bool = False,
    hybrid_success: bool = False,
    hybrid_exact_symbolic_match: bool = False,
    failure_stage: str | None = None,
    repair_gain: bool = False,
    regression: bool = False,
    initial_best_mdlm_numeric_residual: float | None = None,
    final_numeric_residual: float | None = None,
    best_numeric_residual: float | None = None,
    beam_stop_reason: str | None = None,
    beam_steps_taken: int | None = None,
    expanded_states: int | None = None,
    fallback_used: bool = False,
) -> HybridRepairExampleResult:
    parseable_indices = [
        int(result.attempt_index) for result in parse_results if result.ok
    ]
    return HybridRepairExampleResult(
        row_index=int(row_index),
        pair_index=pair_index,
        target_integrand_prefix=str(integrand_prefix),
        target_antiderivative_prefix=target_antiderivative_prefix,
        num_mdlm_attempts=int(len(parse_results)),
        num_parseable_mdlm_seeds=int(len(parseable_indices)),
        first_parseable_attempt_index=parseable_indices[0] if parseable_indices else None,
        parseable_attempt_indices=parseable_indices,
        mdlm_attempt_prefixes=list(attempt_prefixes),
        mdlm_parse_errors=list(parse_errors),
        mdlm_any_seed_parse_ok=bool(parseable_indices),
        mdlm_no_parseable_seed=not bool(parseable_indices),
        mdlm_any_seed_exact_symbolic_match=bool(mdlm_any_seed_exact_symbolic_match),
        mdlm_first_parseable_exact_symbolic_match=bool(
            mdlm_first_parseable_exact_symbolic_match
        ),
        repair_attempted=bool(repair_attempted),
        tree_repair_success=bool(tree_repair_success),
        hybrid_success=bool(hybrid_success),
        hybrid_exact_symbolic_match=bool(hybrid_exact_symbolic_match),
        failure_stage=failure_stage,
        repair_gain=bool(repair_gain),
        regression=bool(regression),
        initial_best_mdlm_numeric_residual=initial_best_mdlm_numeric_residual,
        final_numeric_residual=final_numeric_residual,
        best_numeric_residual=best_numeric_residual,
        beam_stop_reason=beam_stop_reason,
        beam_steps_taken=beam_steps_taken,
        expanded_states=expanded_states,
        fallback_used=bool(fallback_used),
    )


def _load_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    prediction_path = Path(path)
    if prediction_path.suffix.lower() == ".parquet":
        rows = pd.read_parquet(prediction_path).to_dict(orient="records")
    else:
        rows = []
        with prediction_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise TypeError(
                        f"Expected JSON object at {prediction_path}:{line_number}, got {type(payload).__name__}."
                    )
                rows.append(payload)
    for index, row in enumerate(rows):
        _validate_prediction_row(row, row_number=index)
    return rows


def _prediction_groups(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    order: list[tuple[Any, ...]] = []
    for row in rows:
        key = _prediction_group_key(row)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)
    return [
        sorted(grouped[key], key=lambda row: _attempt_index(row, fallback=0))
        for key in order
    ]


def _prediction_group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    if not _is_missing(row.get("row_index")):
        return ("row_index", int(row["row_index"]))
    if not _is_missing(row.get("pair_index")):
        return ("pair_index", int(row["pair_index"]))
    return (
        "prefixes",
        str(row.get("integrand_prefix", "")),
        _optional_nonempty_str(row.get("target_antiderivative_prefix")),
    )


def _validate_prediction_row(row: Mapping[str, Any], *, row_number: int) -> None:
    for key in ("integrand_prefix", "pred_prefix"):
        if key not in row:
            raise KeyError(f"Prediction row {row_number} is missing required field {key!r}.")
    if "attempt_index" not in row:
        raise KeyError("Prediction rows must include attempt_index.")


def _select_seed_results(
    parseable_results: Sequence[MdlmSeedParseResult],
    *,
    seed_selection: str,
    numeric_by_attempt: Mapping[int, float | None],
) -> list[MdlmSeedParseResult]:
    if seed_selection == "all_parseable":
        return list(parseable_results)
    if seed_selection == "first_parseable":
        return [parseable_results[0]]
    if seed_selection == "best_numeric_seed":
        finite = [
            result
            for result in parseable_results
            if _finite(numeric_by_attempt.get(int(result.attempt_index)))
        ]
        if finite:
            return [
                min(
                    finite,
                    key=lambda result: float(numeric_by_attempt[int(result.attempt_index)]),
                )
            ]
        return [parseable_results[0]]
    raise ValueError(f"Unsupported seed_selection: {seed_selection}")


def _parse_target_integrand(prefix: str) -> Expr | None:
    try:
        return canonicalize(
            parse_prefix_string(prefix),
            strip_additive_constants=False,
        )
    except Exception:
        return None


def _safe_exact_match(
    seed: Expr | None,
    target_integrand: Expr,
    *,
    timeout_seconds: float | None = 2.0,
) -> bool:
    if seed is None:
        return False
    try:
        return bool(
            derivative_matches_target(
                seed,
                target_integrand,
                timeout_seconds=timeout_seconds,
            )
        )
    except Exception:
        return False


def _safe_numeric_residual(seed: Expr | None, target_integrand: Expr, timeout_seconds: float | None = 2.0) -> float | None:
    if seed is None:
        return None
    try:
        return numeric_residual_score(seed, target_integrand, timeout_seconds=timeout_seconds)
    except Exception:
        return None


def _best_numeric(values: Sequence[float | None]) -> float | None:
    finite_values = [float(value) for value in values if _finite(value)]
    if not finite_values:
        return None
    return min(finite_values)


def _write_examples_jsonl(
    path: Path,
    results: Sequence[HybridRepairExampleResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(_json_safe(asdict(result)), sort_keys=True) + "\n")


def _write_examples_part(
    parts_dir: Path,
    part_index: int,
    results: Sequence[HybridRepairExampleResult],
) -> None:
    part_path = parts_dir / f"part_{part_index:06d}.jsonl"
    tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(_json_safe(asdict(result)), sort_keys=True) + "\n")
    tmp_path.replace(part_path)
    part_summary = {
        "part_index": int(part_index),
        "path": str(part_path),
        "examples": int(len(results)),
        "first_row_index": None if not results else int(results[0].row_index),
        "last_row_index": None if not results else int(results[-1].row_index),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(parts_dir / f"part_{part_index:06d}.summary.json", part_summary)


def _examples_part_paths(parts_dir: Path) -> list[Path]:
    return sorted(parts_dir.glob("part_*.jsonl"))


def _next_examples_part_index(parts_dir: Path) -> int:
    indices = [_examples_part_index(path) for path in _examples_part_paths(parts_dir)]
    indices = [index for index in indices if index is not None]
    return 0 if not indices else max(indices) + 1


def _examples_part_index(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("part_"):
        return None
    try:
        return int(stem.removeprefix("part_"))
    except ValueError:
        return None


def _load_examples_part_results(parts_dir: Path) -> list[HybridRepairExampleResult]:
    results: list[HybridRepairExampleResult] = []
    for path in _examples_part_paths(parts_dir):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, Mapping):
                    raise TypeError(
                        f"Expected JSON object in {path}:{line_number}, got {type(payload).__name__}."
                    )
                results.append(_example_result_from_mapping(payload, path=path, line_number=line_number))
    return results


def _example_result_from_mapping(
    payload: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
) -> HybridRepairExampleResult:
    field_names = {field.name for field in fields(HybridRepairExampleResult)}
    missing = [
        field.name
        for field in fields(HybridRepairExampleResult)
        if field.default is MISSING and field.default_factory is MISSING and field.name not in payload
    ]
    if missing:
        raise KeyError(
            f"Existing result row {path}:{line_number} is missing required field(s): "
            + ", ".join(missing)
        )
    values = {name: payload[name] for name in field_names if name in payload}
    return HybridRepairExampleResult(**values)


def _write_examples_parts_manifest(
    parts_dir: Path,
    *,
    target_examples: int,
    completed_examples: int,
    part_count: int,
) -> None:
    _write_json(
        parts_dir / "manifest.json",
        {
            "target_examples": int(target_examples),
            "completed_examples": int(completed_examples),
            "part_count": int(part_count),
            "complete": int(completed_examples) == int(target_examples),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def _progress(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def _attempt_index(row: Mapping[str, Any], *, fallback: int) -> int:
    value = row.get("attempt_index")
    if _is_missing(value):
        return int(fallback)
    return int(value)


def _row_index(row: Mapping[str, Any], *, fallback: int) -> int:
    value = row.get("row_index")
    if _is_missing(value):
        return int(fallback)
    return int(value)


def _optional_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    return int(value)


def _optional_nonempty_str(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value)
    if text.strip() == "":
        return None
    return text


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _optional_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _parse_error_bucket(error: str) -> str:
    return str(error).split(":", 1)[0]


def _example_has_complex_seed(result: HybridRepairExampleResult) -> bool:
    for prefix in result.mdlm_attempt_prefixes:
        if "I" in str(prefix).split():
            return True
    return False


def _validate_eval_args(
    *,
    start: int,
    limit: int | None,
    beam_size: int,
    candidate_k: int,
    max_steps: int,
    numeric_patience: int | None,
    structural_patience: int | None,
    numeric_tol: float,
    numeric_residual_timeout_seconds: float | None,
    symbolic_check_timeout_seconds: float | None,
    seed_selection: str,
    part_size: int,
    progress_every: int,
    residual_workers: int,
    resume: bool,
    overwrite: bool,
) -> None:
    if int(start) < 0:
        raise ValueError("start must be >= 0.")
    if limit is not None and int(limit) < 0:
        raise ValueError("limit must be None or >= 0.")
    if int(beam_size) < 1:
        raise ValueError("beam_size must be >= 1.")
    if int(candidate_k) < 1:
        raise ValueError("candidate_k must be >= 1.")
    if int(max_steps) < 0:
        raise ValueError("max_steps must be >= 0.")
    if numeric_patience is not None and int(numeric_patience) < 0:
        raise ValueError("numeric_patience must be None or >= 0.")
    if structural_patience is not None and int(structural_patience) < 0:
        raise ValueError("structural_patience must be None or >= 0.")
    if float(numeric_tol) < 0.0:
        raise ValueError("numeric_tol must be >= 0.")
    if numeric_residual_timeout_seconds is not None and float(numeric_residual_timeout_seconds) <= 0.0:
        raise ValueError("numeric_residual_timeout_seconds must be None or > 0.")
    if symbolic_check_timeout_seconds is not None and float(symbolic_check_timeout_seconds) <= 0.0:
        raise ValueError("symbolic_check_timeout_seconds must be None or > 0.")
    if seed_selection not in _VALID_SEED_SELECTIONS:
        raise ValueError(
            "seed_selection must be one of: " + ", ".join(sorted(_VALID_SEED_SELECTIONS))
        )
    if int(part_size) < 1:
        raise ValueError("part_size must be >= 1.")
    if int(progress_every) < 0:
        raise ValueError("progress_every must be >= 0.")
    if int(residual_workers) < 0:
        raise ValueError("residual_workers must be >= 0.")
    if resume and overwrite:
        raise ValueError("resume and overwrite cannot both be true.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run tree-diffusion beam repair from parseable MDLM prediction attempts."
    )
    parser.add_argument("--predictions", required=True, type=str)
    parser.add_argument("--tree-checkpoint", required=True, type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--examples-out", type=str, default=None)
    parser.add_argument("--examples-parts-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--numeric-patience", type=int, default=5)
    parser.add_argument("--structural-patience", type=int, default=None)
    parser.add_argument("--numeric-tol", type=float, default=1e-10)
    parser.add_argument("--numeric-residual-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--symbolic-check-timeout-seconds", type=float, default=2.0)
    parser.add_argument(
        "--seed-selection",
        choices=sorted(_VALID_SEED_SELECTIONS),
        default="all_parseable",
    )
    parser.add_argument("--use-fallback-seeds", action="store_true")
    parser.add_argument("--part-size", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--residual-workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--reject-mask-or-unk", dest="reject_mask_or_unk", action="store_true", default=True)
    parser.add_argument("--allow-mask-or-unk", dest="reject_mask_or_unk", action="store_false")
    parser.add_argument("--allow-complex-constant", dest="allow_complex_constant", action="store_true", default=True)
    parser.add_argument("--reject-complex-constant", dest="allow_complex_constant", action="store_false")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_hybrid_mdlm_repair(
        predictions_path=args.predictions,
        tree_checkpoint=args.tree_checkpoint,
        output_path=args.output,
        examples_out_path=args.examples_out,
        examples_parts_dir=args.examples_parts_dir,
        device=args.device,
        limit=args.limit,
        start=args.start,
        beam_size=args.beam_size,
        candidate_k=args.candidate_k,
        max_steps=args.max_steps,
        numeric_patience=args.numeric_patience,
        structural_patience=args.structural_patience,
        numeric_tol=args.numeric_tol,
        numeric_residual_timeout_seconds=args.numeric_residual_timeout_seconds,
        symbolic_check_timeout_seconds=args.symbolic_check_timeout_seconds,
        reject_mask_or_unk=args.reject_mask_or_unk,
        allow_complex_constant=args.allow_complex_constant,
        seed_selection=args.seed_selection,
        use_fallback_seeds=args.use_fallback_seeds,
        part_size=args.part_size,
        progress_every=args.progress_every,
        residual_workers=args.residual_workers,
        resume=args.resume,
        overwrite=args.overwrite,
        progress=not bool(args.quiet),
    )
    print(json.dumps(_json_safe(asdict(summary)), indent=2, sort_keys=True))
    return 0


__all__ = [
    "HybridMdlmRepairSummary",
    "HybridRepairExampleResult",
    "MdlmSeedParseResult",
    "evaluate_hybrid_mdlm_repair",
    "main",
    "parse_mdlm_seed",
    "parse_mdlm_seed_attempts",
    "summarize_hybrid_mdlm_repair_results",
]


if __name__ == "__main__":
    raise SystemExit(main())
