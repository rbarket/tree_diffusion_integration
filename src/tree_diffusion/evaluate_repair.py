from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch

from src.mathlang.ast import Expr
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion._common import (
    json_safe as _json_safe,
    mean_or_none as _mean_or_none,
    rate as _rate,
    resolve_device as _resolve_device,
    write_json as _write_json,
)
from src.tree_diffusion.edit_path import structural_distance
from src.tree_diffusion.evaluation_common import (
    batch_size as _batch_size,
    metadata_item as _metadata_item,
    mutation_trace_record as _mutation_trace_record,
    repair_inputs_from_batch as _repair_inputs,
    required_metadata as _required_metadata,
    residual_executor_context as _residual_executor_context,
)
from src.tree_diffusion.eval_metrics import (
    RepairGroupSummary,
    is_finite as _is_finite,
    median_or_none as _median_or_none,
    meets_numeric_tol as _meets_numeric_tol,
    num_mutations_group as _num_mutations_group,
    numeric_values as _numeric_values,
    optional_bool_metadata as _optional_bool_metadata,
    optional_int_metadata as _optional_int_metadata,
    repair_group_summary as _repair_group_summary,
    residual_improvement_rate as _residual_improvement_rate,
    summarize_repair_groups as _summarize_repair_groups,
    used_random_init_group as _used_random_init_group,
)
from src.tree_diffusion.model import TreeDiffusionPolicyModel
from src.tree_diffusion.repair import RepairResult, greedy_repair
from src.tree_diffusion.runtime import (
    build_evaluation_dataloader as _build_cli_dataloader,
    load_model_and_tokenizer_for_inference as _load_cli_model_and_tokenizer,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


@dataclass(frozen=True)
class RepairEvaluationRecord:
    result: RepairResult
    used_random_init: bool | None = None
    num_mutations: int | None = None
    structural_distance_initial: float | None = None
    structural_distance_final: float | None = None


@dataclass(frozen=True)
class RepairEvaluationSummary:
    examples: int
    selection_strategy: str
    candidate_k: int
    success_rate: float
    exact_symbolic_match_rate: float
    numeric_success_rate: float
    mean_steps_to_success: float | None
    median_steps_to_success: float | None
    no_candidate_rate: float
    repeated_state_rate: float
    max_steps_rate: float
    no_numeric_improvement_rate: float
    mean_initial_numeric_residual: float | None
    mean_final_numeric_residual: float | None
    numeric_residual_improvement_rate: float | None
    mean_best_numeric_residual: float | None
    best_numeric_residual_improvement_rate: float | None
    best_numeric_success_rate: float
    mean_structural_distance_initial: float | None
    mean_structural_distance_final: float | None
    structural_distance_improvement_rate: float | None
    per_step_numeric_residual_mean: dict[str, float | None]
    per_step_numeric_residual_median: dict[str, float | None]
    per_step_active_examples: dict[str, int]
    per_step_exact_match_rate: dict[str, float]
    mean_chosen_candidate_rank: float | None
    rank1_chosen_rate: float | None
    by_used_random_init: dict[str, RepairGroupSummary]
    by_num_mutations: dict[str, RepairGroupSummary]
    stop_reason_counts: dict[str, int]


@torch.no_grad()
def evaluate_greedy_repair(
    model: TreeDiffusionPolicyModel,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    tokenizer: TreeDiffusionTokenizer,
    device: torch.device | str,
    num_batches: int,
    max_steps: int = 10,
    candidate_k: int = 8,
    numeric_tol: float = 1e-10,
    patience: int = 2,
    constrain_position: bool = True,
    max_decode_length: int | None = None,
    residual_mode: str = "both",
    simplify_symbolic_residual: bool = True,
    observation_timeout_seconds: float | None = 2.0,
    numeric_residual_timeout_seconds: float | None = 2.0,
    symbolic_check_timeout_seconds: float | None = 2.0,
    compute_structural_metrics: bool = True,
    selection_strategy: Literal["rank1", "residual_scored"] = "residual_scored",
    residual_workers: int = 0,
) -> RepairEvaluationSummary:
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1.")
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0.")
    if candidate_k < 1:
        raise ValueError("candidate_k must be >= 1.")
    if numeric_tol < 0.0:
        raise ValueError("numeric_tol must be >= 0.")
    if patience < 0:
        raise ValueError("patience must be >= 0.")
    if residual_workers < 0:
        raise ValueError("residual_workers must be >= 0.")
    if selection_strategy not in {"rank1", "residual_scored"}:
        raise ValueError("selection_strategy must be 'rank1' or 'residual_scored'.")

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    records: list[RepairEvaluationRecord] = []
    iterator = iter(dataloader)
    with _residual_executor_context(int(residual_workers)) as residual_executor:
        for _ in range(num_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
            if not isinstance(batch, Mapping):
                raise TypeError(f"Expected dataloader batches to be mappings, got {type(batch).__name__}.")

            for row_index in range(_batch_size(batch)):
                target_integrand, target_antiderivative, current = _repair_inputs(batch, row_index)
                result = greedy_repair(
                    model,
                    target_integrand,
                    current,
                    tokenizer=tokenizer,
                    device=target_device,
                    max_steps=max_steps,
                    candidate_k=candidate_k,
                    numeric_tol=numeric_tol,
                    patience=patience,
                    constrain_position=constrain_position,
                    max_decode_length=max_decode_length,
                    residual_mode=residual_mode,
                    simplify_symbolic_residual=simplify_symbolic_residual,
                    observation_timeout_seconds=observation_timeout_seconds,
                    numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
                    symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
                    target_antiderivative=(
                        target_antiderivative if compute_structural_metrics else None
                    ),
                    selection_strategy=selection_strategy,
                    residual_executor=residual_executor,
                )

                initial_distance = None
                final_distance = None
                if compute_structural_metrics:
                    final_tree = canonicalize(parse_prefix_string(result.final_prefix))
                    initial_distance = float(structural_distance(current, target_antiderivative))
                    final_distance = float(structural_distance(final_tree, target_antiderivative))

                records.append(
                    RepairEvaluationRecord(
                        result=result,
                        used_random_init=_optional_bool_metadata(
                            _metadata_item(batch, "used_random_init", row_index, default=None)
                        ),
                        num_mutations=_optional_int_metadata(
                            _metadata_item(batch, "num_mutations", row_index, default=None)
                        ),
                        structural_distance_initial=initial_distance,
                        structural_distance_final=final_distance,
                    )
                )

    return summarize_repair_results(
        records,
        numeric_tol=numeric_tol,
        max_steps=max_steps,
        candidate_k=candidate_k,
        selection_strategy=selection_strategy,
    )


def summarize_repair_results(
    records: Sequence[RepairEvaluationRecord],
    *,
    numeric_tol: float,
    max_steps: int,
    candidate_k: int,
    selection_strategy: str,
) -> RepairEvaluationSummary:
    if numeric_tol < 0.0:
        raise ValueError("numeric_tol must be >= 0.")
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0.")
    if candidate_k < 1:
        raise ValueError("candidate_k must be >= 1.")
    if selection_strategy not in {"rank1", "residual_scored"}:
        raise ValueError("selection_strategy must be 'rank1' or 'residual_scored'.")

    rows = list(records)
    examples = len(rows)
    results = [row.result for row in rows]
    initial_numeric_values = _numeric_values(
        result.initial_numeric_residual for result in results
    )
    final_numeric_values = _numeric_values(
        result.final_numeric_residual for result in results
    )
    best_numeric_values = _numeric_values(
        result.best_numeric_residual for result in results
    )
    steps_to_success = [float(result.steps_taken) for result in results if result.success]
    chosen_ranks = [
        float(step.candidate_rank)
        for result in results
        for step in result.steps
        if step.chosen_prefix is not None and step.candidate_rank is not None
    ]
    structural_initial_values = _numeric_values(
        row.structural_distance_initial for row in rows
    )
    structural_final_values = _numeric_values(
        row.structural_distance_final for row in rows
    )

    per_step = _per_step_metrics(
        results,
        examples=examples,
        max_steps=max_steps,
    )

    return RepairEvaluationSummary(
        examples=examples,
        selection_strategy=str(selection_strategy),
        candidate_k=int(candidate_k),
        success_rate=_rate(sum(int(result.success) for result in results), examples),
        exact_symbolic_match_rate=_rate(
            sum(int(result.exact_symbolic_match) for result in results),
            examples,
        ),
        numeric_success_rate=_rate(
            sum(int(_meets_numeric_tol(result.final_numeric_residual, numeric_tol)) for result in results),
            examples,
        ),
        mean_steps_to_success=_mean_or_none(steps_to_success),
        median_steps_to_success=_median_or_none(steps_to_success),
        no_candidate_rate=_rate(sum(int(result.no_candidate) for result in results), examples),
        repeated_state_rate=_rate(sum(int(result.repeated_state) for result in results), examples),
        max_steps_rate=_rate(
            sum(int(result.stop_reason == "max_steps") for result in results),
            examples,
        ),
        no_numeric_improvement_rate=_rate(
            sum(int(result.stop_reason == "no_numeric_improvement") for result in results),
            examples,
        ),
        mean_initial_numeric_residual=_mean_or_none(initial_numeric_values),
        mean_final_numeric_residual=_mean_or_none(final_numeric_values),
        numeric_residual_improvement_rate=_residual_improvement_rate(
            (result.initial_numeric_residual, result.final_numeric_residual)
            for result in results
        ),
        mean_best_numeric_residual=_mean_or_none(best_numeric_values),
        best_numeric_residual_improvement_rate=_residual_improvement_rate(
            (result.initial_numeric_residual, result.best_numeric_residual)
            for result in results
        ),
        best_numeric_success_rate=_rate(
            sum(int(_meets_numeric_tol(result.best_numeric_residual, numeric_tol)) for result in results),
            examples,
        ),
        mean_structural_distance_initial=_mean_or_none(structural_initial_values),
        mean_structural_distance_final=_mean_or_none(structural_final_values),
        structural_distance_improvement_rate=_residual_improvement_rate(
            (row.structural_distance_initial, row.structural_distance_final)
            for row in rows
        ),
        per_step_numeric_residual_mean=per_step["mean"],
        per_step_numeric_residual_median=per_step["median"],
        per_step_active_examples=per_step["active"],
        per_step_exact_match_rate=per_step["exact"],
        mean_chosen_candidate_rank=_mean_or_none(chosen_ranks),
        rank1_chosen_rate=(
            None
            if not chosen_ranks
            else sum(1 for rank in chosen_ranks if int(rank) == 1) / len(chosen_ranks)
        ),
        by_used_random_init=_group_by(
            rows,
            key_fn=lambda row: _used_random_init_group(row.used_random_init),
            numeric_tol=numeric_tol,
        ),
        by_num_mutations=_group_by(
            rows,
            key_fn=lambda row: _num_mutations_group(row.num_mutations),
            numeric_tol=numeric_tol,
        ),
        stop_reason_counts=dict(Counter(result.stop_reason for result in results)),
    )


def repair_evaluation_summary_to_json(summary: RepairEvaluationSummary) -> dict[str, Any]:
    overall_fields = (
        "examples",
        "success_rate",
        "exact_symbolic_match_rate",
        "numeric_success_rate",
        "mean_steps_to_success",
        "median_steps_to_success",
        "no_candidate_rate",
        "repeated_state_rate",
        "max_steps_rate",
        "no_numeric_improvement_rate",
        "mean_initial_numeric_residual",
        "mean_final_numeric_residual",
        "numeric_residual_improvement_rate",
        "mean_structural_distance_initial",
        "mean_structural_distance_final",
        "structural_distance_improvement_rate",
    )
    return _json_safe(
        {
            "examples": summary.examples,
            "selection_strategy": summary.selection_strategy,
            "candidate_k": summary.candidate_k,
            "overall": {
                name: getattr(summary, name)
                for name in overall_fields
            },
            "by_used_random_init": {
                name: asdict(group)
                for name, group in summary.by_used_random_init.items()
            },
            "by_num_mutations": {
                name: asdict(group)
                for name, group in summary.by_num_mutations.items()
            },
            "best_so_far": {
                "mean_best_numeric_residual": summary.mean_best_numeric_residual,
                "best_numeric_residual_improvement_rate": summary.best_numeric_residual_improvement_rate,
                "best_numeric_success_rate": summary.best_numeric_success_rate,
            },
            "per_step": {
                "numeric_residual_mean": summary.per_step_numeric_residual_mean,
                "numeric_residual_median": summary.per_step_numeric_residual_median,
                "active_examples": summary.per_step_active_examples,
                "exact_match_rate": summary.per_step_exact_match_rate,
            },
            "candidate_rank": {
                "mean_chosen_candidate_rank": summary.mean_chosen_candidate_rank,
                "rank1_chosen_rate": summary.rank1_chosen_rate,
            },
            "stop_reason_counts": dict(summary.stop_reason_counts),
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate greedy tree-diffusion repair.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--precomputed-data-dir", default=None)
    parser.add_argument("--precomputed-split", choices=("train", "val"), default="val")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-pairs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--numeric-tol", type=float, default=1e-10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--selection-strategy",
        choices=("rank1", "residual_scored"),
        default="residual_scored",
    )
    parser.add_argument("--constrain-position", dest="constrain_position", action="store_true", default=True)
    parser.add_argument("--no-constrain-position", dest="constrain_position", action="store_false")
    parser.add_argument("--max-decode-length", type=int, default=None)
    parser.add_argument("--observation-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--numeric-residual-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--symbolic-check-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--residual-workers", type=int, default=0)
    parser.add_argument("--no-structural-metrics", dest="compute_structural_metrics", action="store_false")
    parser.add_argument("--dump-examples", default=None)
    parser.add_argument("--num-dump-examples", type=int, default=50)
    parser.add_argument("--dump-failures-only", action="store_true")
    args = parser.parse_args(argv)

    _validate_cli_args(args)
    torch.manual_seed(int(args.seed))
    device = _resolve_device(str(args.device))
    tokenizer, model = _load_cli_model_and_tokenizer(
        checkpoint=str(args.checkpoint),
        precomputed_data_dir=args.precomputed_data_dir,
        allow_random_init_model=False,
    )
    model.to(device)

    dataloader = _build_cli_dataloader(
        data=args.data,
        precomputed_data_dir=args.precomputed_data_dir,
        precomputed_split=str(args.precomputed_split),
        tokenizer=tokenizer,
        model=model,
        num_pairs=int(args.num_pairs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    summary = evaluate_greedy_repair(
        model,
        dataloader,
        tokenizer=tokenizer,
        device=device,
        num_batches=int(args.num_batches),
        max_steps=int(args.max_steps),
        candidate_k=int(args.candidate_k),
        numeric_tol=float(args.numeric_tol),
        patience=int(args.patience),
        constrain_position=bool(args.constrain_position),
        max_decode_length=args.max_decode_length,
        observation_timeout_seconds=args.observation_timeout_seconds,
        numeric_residual_timeout_seconds=args.numeric_residual_timeout_seconds,
        symbolic_check_timeout_seconds=args.symbolic_check_timeout_seconds,
        compute_structural_metrics=bool(args.compute_structural_metrics),
        selection_strategy=str(args.selection_strategy),
        residual_workers=int(args.residual_workers),
    )
    output_json = repair_evaluation_summary_to_json(summary)
    output_json["residual_workers"] = int(args.residual_workers)

    if args.dump_examples is not None:
        dump_loader = _build_cli_dataloader(
            data=args.data,
            precomputed_data_dir=args.precomputed_data_dir,
            precomputed_split=str(args.precomputed_split),
            tokenizer=tokenizer,
            model=model,
            num_pairs=int(args.num_pairs),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
        )
        output_json["dump_examples"] = _dump_repair_examples(
            model,
            dump_loader,
            tokenizer=tokenizer,
            device=device,
            path=Path(args.dump_examples),
            num_examples=int(args.num_dump_examples),
            dump_failures_only=bool(args.dump_failures_only),
            max_steps=int(args.max_steps),
            candidate_k=int(args.candidate_k),
            numeric_tol=float(args.numeric_tol),
            patience=int(args.patience),
            constrain_position=bool(args.constrain_position),
            max_decode_length=args.max_decode_length,
            observation_timeout_seconds=args.observation_timeout_seconds,
            numeric_residual_timeout_seconds=args.numeric_residual_timeout_seconds,
            symbolic_check_timeout_seconds=args.symbolic_check_timeout_seconds,
            selection_strategy=str(args.selection_strategy),
            residual_workers=int(args.residual_workers),
        )

    print(json.dumps(output_json, indent=2, sort_keys=True))
    if args.output is not None:
        _write_json(Path(args.output), output_json)
    return 0


def _dump_repair_examples(
    model: TreeDiffusionPolicyModel,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    tokenizer: TreeDiffusionTokenizer,
    device: torch.device | str,
    path: Path,
    num_examples: int,
    dump_failures_only: bool,
    max_steps: int,
    candidate_k: int,
    numeric_tol: float,
    patience: int,
    constrain_position: bool,
    max_decode_length: int | None,
    observation_timeout_seconds: float | None,
    numeric_residual_timeout_seconds: float | None,
    symbolic_check_timeout_seconds: float | None,
    selection_strategy: Literal["rank1", "residual_scored"],
    residual_workers: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    scanned = 0
    iterator = iter(dataloader)
    with _residual_executor_context(int(residual_workers)) as residual_executor, path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        while written < num_examples:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if not isinstance(batch, Mapping):
                raise TypeError(f"Expected dataloader batches to be mappings, got {type(batch).__name__}.")
            for row_index in range(_batch_size(batch)):
                if written >= num_examples:
                    break
                target_integrand, target_antiderivative, current = _repair_inputs(batch, row_index)
                result = greedy_repair(
                    model,
                    target_integrand,
                    current,
                    tokenizer=tokenizer,
                    device=device,
                    max_steps=max_steps,
                    candidate_k=candidate_k,
                    numeric_tol=numeric_tol,
                    patience=patience,
                    constrain_position=constrain_position,
                    max_decode_length=max_decode_length,
                    observation_timeout_seconds=observation_timeout_seconds,
                    numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
                    symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
                    target_antiderivative=target_antiderivative,
                    selection_strategy=selection_strategy,
                    residual_executor=residual_executor,
                )
                scanned += 1
                if dump_failures_only and result.success:
                    continue
                record = _repair_example_record(
                    result,
                    target_antiderivative=target_antiderivative,
                    source_mutation_trace=_mutation_trace_record(batch, row_index),
                )
                handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
                written += 1
    return {
        "path": str(path),
        "records_written": written,
        "records_scanned": scanned,
        "residual_workers": int(residual_workers),
    }


def _repair_example_record(
    result: RepairResult,
    *,
    target_antiderivative: Expr,
    source_mutation_trace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "target_integrand_prefix": result.target_integrand_prefix,
        "target_antiderivative_prefix": serialize_prefix_string(target_antiderivative),
        "source_mutation_trace": source_mutation_trace,
        "initial_prefix": result.initial_prefix,
        "final_prefix": result.final_prefix,
        "success": result.success,
        "stop_reason": result.stop_reason,
        "initial_numeric_residual": result.initial_numeric_residual,
        "final_numeric_residual": result.final_numeric_residual,
        "best_numeric_residual": result.best_numeric_residual,
        "best_prefix": result.best_prefix,
        "best_step_index": result.best_step_index,
        "steps_taken": result.steps_taken,
        "steps": [asdict(step) for step in result.steps],
    }


def _group_by(
    records: Sequence[RepairEvaluationRecord],
    *,
    key_fn,
    numeric_tol: float,
) -> dict[str, RepairGroupSummary]:
    return _summarize_repair_groups(
        records,
        key_fn=key_fn,
        result_fn=lambda row: row.result,
        final_numeric_residual_fn=lambda result: result.final_numeric_residual,
        structural_distance_initial_fn=lambda row: row.structural_distance_initial,
        structural_distance_final_fn=lambda row: row.structural_distance_final,
        numeric_tol=numeric_tol,
    )


def _group_summary(
    records: Sequence[RepairEvaluationRecord],
    *,
    numeric_tol: float,
) -> RepairGroupSummary:
    return _repair_group_summary(
        records,
        result_fn=lambda row: row.result,
        final_numeric_residual_fn=lambda result: result.final_numeric_residual,
        structural_distance_initial_fn=lambda row: row.structural_distance_initial,
        structural_distance_final_fn=lambda row: row.structural_distance_final,
        numeric_tol=numeric_tol,
    )


def _per_step_metrics(
    results: Sequence[RepairResult],
    *,
    examples: int,
    max_steps: int,
) -> dict[str, dict[str, Any]]:
    max_observed_step = max(
        [max(_numeric_residual_by_step(result), default=0) for result in results],
        default=0,
    )
    final_step = max(max_steps, max_observed_step)
    residuals_by_step: dict[int, list[float]] = {step: [] for step in range(final_step + 1)}
    exact_counts: dict[int, int] = {step: 0 for step in range(final_step + 1)}

    for result in results:
        residuals = _numeric_residual_by_step(result)
        exact_step = _exact_match_step(result)
        for step in range(final_step + 1):
            value = residuals.get(step)
            if value is not None and _is_finite(value):
                residuals_by_step[step].append(float(value))
            if exact_step is not None and exact_step <= step:
                exact_counts[step] += 1

    mean: dict[str, float | None] = {}
    median: dict[str, float | None] = {}
    active: dict[str, int] = {}
    exact: dict[str, float] = {}
    for step in range(final_step + 1):
        key = f"step_{step}"
        values = residuals_by_step[step]
        mean[key] = _mean_or_none(values)
        median[key] = _median_or_none(values)
        active[key] = len(values)
        exact[key] = _rate(exact_counts[step], examples)
    return {
        "mean": mean,
        "median": median,
        "active": active,
        "exact": exact,
    }


def _numeric_residual_by_step(result: RepairResult) -> dict[int, float | None]:
    values: dict[int, float | None] = {0: result.initial_numeric_residual}
    for step in result.steps:
        if step.chosen_prefix is not None:
            values[int(step.step_index) + 1] = step.numeric_residual_after
    return values


def _exact_match_step(result: RepairResult) -> int | None:
    if not result.exact_symbolic_match:
        return None
    if result.steps_taken == 0:
        return 0
    for step in result.steps:
        if step.chosen_prefix is not None and step.exact_symbolic_match:
            return int(step.step_index) + 1
    return int(result.steps_taken)


def _validate_cli_args(args: argparse.Namespace) -> None:
    if (args.data is None) == (args.precomputed_data_dir is None):
        raise ValueError("Provide exactly one data source: --data or --precomputed-data-dir.")
    if args.num_pairs < 1:
        raise ValueError("--num-pairs must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.num_batches < 1:
        raise ValueError("--num-batches must be >= 1.")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0.")
    if args.candidate_k < 1:
        raise ValueError("--candidate-k must be >= 1.")
    if args.numeric_tol < 0.0:
        raise ValueError("--numeric-tol must be >= 0.")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0.")
    if args.numeric_residual_timeout_seconds is not None and args.numeric_residual_timeout_seconds <= 0.0:
        raise ValueError("--numeric-residual-timeout-seconds must be > 0.")
    if args.observation_timeout_seconds is not None and args.observation_timeout_seconds <= 0.0:
        raise ValueError("--observation-timeout-seconds must be > 0.")
    if args.symbolic_check_timeout_seconds is not None and args.symbolic_check_timeout_seconds <= 0.0:
        raise ValueError("--symbolic-check-timeout-seconds must be > 0.")
    if args.residual_workers < 0:
        raise ValueError("--residual-workers must be >= 0.")
    if args.num_dump_examples < 0:
        raise ValueError("--num-dump-examples must be >= 0.")


__all__ = [
    "RepairEvaluationRecord",
    "RepairEvaluationSummary",
    "RepairGroupSummary",
    "evaluate_greedy_repair",
    "main",
    "repair_evaluation_summary_to_json",
    "summarize_repair_results",
]


if __name__ == "__main__":
    raise SystemExit(main())
