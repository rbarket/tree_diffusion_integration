from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch

from src.mathlang.ast import Expr
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion._common import (
    json_safe as _json_safe,
    mean_or_none as _mean_or_none,
    rate as _rate,
    resolve_device as _resolve_device,
    write_json as _write_json,
)
from src.tree_diffusion.beam_search import (
    BeamSearchResult,
    BeamSearchScoringConfig,
    BeamSearchStopConfig,
    beam_search_repair,
)
from src.tree_diffusion.cli_common import (
    optional_float_arg as _optional_float_arg,
    optional_int_arg as _optional_int_arg,
)
from src.tree_diffusion.evaluation_common import (
    batch_size as _batch_size,
    load_config_values as _load_config_values_common,
    metadata_item as _metadata_item,
    mutation_trace_record as _mutation_trace_record,
    repair_inputs_from_batch as _repair_inputs,
    residual_executor_context as _residual_executor_context,
    summarize_repair_record_groups as _summarize_repair_record_groups,
)
from src.tree_diffusion.eval_metrics import (
    RepairGroupSummary,
    mean_or_zero as _mean_or_zero,
    median_or_none as _median_or_none,
    meets_numeric_tol as _meets_numeric_tol,
    num_mutations_group as _num_mutations_group,
    numeric_values as _numeric_values,
    optional_bool_metadata as _optional_bool_metadata,
    optional_int_metadata as _optional_int_metadata,
    residual_improvement_rate as _residual_improvement_rate,
    used_random_init_group as _used_random_init_group,
)
from src.tree_diffusion.numeric import finite_numeric as _finite_numeric
from src.tree_diffusion.model import TreeDiffusionPolicyModel
from src.tree_diffusion.runtime import (
    build_evaluation_dataloader as _build_cli_dataloader,
    load_model_and_tokenizer_for_inference as _load_cli_model_and_tokenizer,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


@dataclass(frozen=True)
class BeamRepairEvaluationRecord:
    result: BeamSearchResult
    used_random_init: bool | None = None
    num_mutations: int | None = None
    structural_distance_initial: float | None = None
    structural_distance_best: float | None = None


@dataclass(frozen=True)
class BeamRepairEvaluationSummary:
    examples: int
    beam_size: int
    candidate_k: int
    scoring_config: dict[str, Any]
    stop_config: dict[str, Any]
    success_rate: float
    exact_symbolic_match_rate: float
    numeric_success_rate: float
    mean_steps_to_success: float | None
    median_steps_to_success: float | None
    mean_expanded_states: float
    mean_generated_candidates: float
    mean_applicable_candidates: float
    mean_repeated_candidates: float
    mean_pruned_candidates: float
    beam_empty_rate: float
    max_steps_rate: float
    numeric_patience_rate: float
    structural_patience_rate: float
    timeout_rate: float
    mean_initial_numeric_residual: float | None
    mean_best_numeric_residual: float | None
    best_numeric_residual_improvement_rate: float | None
    mean_final_best_numeric_residual: float | None
    mean_initial_structural_distance: float | None
    mean_best_structural_distance: float | None
    structural_distance_improvement_rate: float | None
    stop_reason_counts: dict[str, int]
    by_used_random_init: dict[str, RepairGroupSummary]
    by_num_mutations: dict[str, RepairGroupSummary]


@torch.no_grad()
def evaluate_beam_repair(
    model: TreeDiffusionPolicyModel,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    tokenizer: TreeDiffusionTokenizer,
    device: torch.device | str,
    num_batches: int,
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
    compute_structural_metrics: bool = True,
    residual_workers: int = 0,
    progress_every: int = 0,
    progress: bool = False,
) -> BeamRepairEvaluationSummary:
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1.")
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1.")
    if candidate_k < 1:
        raise ValueError("candidate_k must be >= 1.")
    if residual_workers < 0:
        raise ValueError("residual_workers must be >= 0.")
    if progress_every < 0:
        raise ValueError("progress_every must be >= 0.")

    scoring_config = scoring or BeamSearchScoringConfig()
    stop_config = stopping or BeamSearchStopConfig()
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    records: list[BeamRepairEvaluationRecord] = []
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
                result = beam_search_repair(
                    model,
                    target_integrand,
                    current,
                    tokenizer=tokenizer,
                    device=target_device,
                    beam_size=beam_size,
                    candidate_k=candidate_k,
                    scoring=scoring_config,
                    stopping=stop_config,
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
                    residual_executor=residual_executor,
                )
                records.append(
                    BeamRepairEvaluationRecord(
                        result=result,
                        used_random_init=_optional_bool_metadata(
                            _metadata_item(batch, "used_random_init", row_index, default=None)
                        ),
                        num_mutations=_optional_int_metadata(
                            _metadata_item(batch, "num_mutations", row_index, default=None)
                        ),
                        structural_distance_initial=(
                            None
                            if not compute_structural_metrics
                            else _float_or_none(result.path[0].structural_distance_before)
                            if result.path
                            else _float_or_none(result.best_structural_distance)
                        ),
                        structural_distance_best=(
                            None
                            if not compute_structural_metrics
                            else _float_or_none(result.best_structural_distance)
                        ),
                    )
                )
                if progress and progress_every > 0 and len(records) % int(progress_every) == 0:
                    print(
                        "beam_eval_progress "
                        f"examples={len(records)} "
                        f"last_stop_reason={result.stop_reason} "
                        f"last_success={result.success} "
                        f"expanded_states={result.expanded_states}",
                        file=sys.stderr,
                        flush=True,
                    )

    return summarize_beam_repair_results(
        records,
        beam_size=beam_size,
        candidate_k=candidate_k,
        scoring=scoring_config,
        stopping=stop_config,
    )


def summarize_beam_repair_results(
    records: Sequence[BeamRepairEvaluationRecord],
    *,
    beam_size: int,
    candidate_k: int,
    scoring: BeamSearchScoringConfig,
    stopping: BeamSearchStopConfig,
) -> BeamRepairEvaluationSummary:
    rows = list(records)
    examples = len(rows)
    results = [row.result for row in rows]
    steps_to_success = [float(result.steps_taken) for result in results if result.success]

    return BeamRepairEvaluationSummary(
        examples=examples,
        beam_size=int(beam_size),
        candidate_k=int(candidate_k),
        scoring_config=asdict(scoring),
        stop_config=asdict(stopping),
        success_rate=_rate(sum(int(result.success) for result in results), examples),
        exact_symbolic_match_rate=_rate(
            sum(int(result.exact_symbolic_match) for result in results),
            examples,
        ),
        numeric_success_rate=_rate(
            sum(int(_meets_numeric_tol(result.best_numeric_residual, stopping.numeric_tol)) for result in results),
            examples,
        ),
        mean_steps_to_success=_mean_or_none(steps_to_success),
        median_steps_to_success=_median_or_none(steps_to_success),
        mean_expanded_states=_mean_or_zero(result.expanded_states for result in results),
        mean_generated_candidates=_mean_or_zero(result.generated_candidates for result in results),
        mean_applicable_candidates=_mean_or_zero(result.applicable_candidates for result in results),
        mean_repeated_candidates=_mean_or_zero(result.repeated_candidates for result in results),
        mean_pruned_candidates=_mean_or_zero(result.pruned_candidates for result in results),
        beam_empty_rate=_rate(sum(int(result.stop_reason == "beam_empty") for result in results), examples),
        max_steps_rate=_rate(sum(int(result.stop_reason == "max_steps") for result in results), examples),
        numeric_patience_rate=_rate(
            sum(int(result.stop_reason == "numeric_patience") for result in results),
            examples,
        ),
        structural_patience_rate=_rate(
            sum(int(result.stop_reason == "structural_patience") for result in results),
            examples,
        ),
        timeout_rate=_rate(sum(int(result.stop_reason == "timeout") for result in results), examples),
        mean_initial_numeric_residual=_mean_or_none(
            _numeric_values(result.initial_numeric_residual for result in results)
        ),
        mean_best_numeric_residual=_mean_or_none(
            _numeric_values(result.best_numeric_residual for result in results)
        ),
        best_numeric_residual_improvement_rate=_residual_improvement_rate(
            (result.initial_numeric_residual, result.best_numeric_residual)
            for result in results
        ),
        mean_final_best_numeric_residual=_mean_or_none(
            _numeric_values(result.final_best_numeric_residual for result in results)
        ),
        mean_initial_structural_distance=_mean_or_none(
            _numeric_values(row.structural_distance_initial for row in rows)
        ),
        mean_best_structural_distance=_mean_or_none(
            _numeric_values(row.structural_distance_best for row in rows)
        ),
        structural_distance_improvement_rate=_residual_improvement_rate(
            (row.structural_distance_initial, row.structural_distance_best)
            for row in rows
        ),
        stop_reason_counts=dict(Counter(result.stop_reason for result in results)),
        by_used_random_init=_summarize_repair_record_groups(
            rows,
            key_fn=lambda row: _used_random_init_group(row.used_random_init),
            result_fn=lambda row: row.result,
            final_numeric_residual_fn=lambda result: result.best_numeric_residual,
            structural_distance_initial_fn=lambda row: row.structural_distance_initial,
            structural_distance_final_fn=lambda row: row.structural_distance_best,
            numeric_tol=stopping.numeric_tol,
        ),
        by_num_mutations=_summarize_repair_record_groups(
            rows,
            key_fn=lambda row: _num_mutations_group(row.num_mutations),
            result_fn=lambda row: row.result,
            final_numeric_residual_fn=lambda result: result.best_numeric_residual,
            structural_distance_initial_fn=lambda row: row.structural_distance_initial,
            structural_distance_final_fn=lambda row: row.structural_distance_best,
            numeric_tol=stopping.numeric_tol,
        ),
    )


def beam_repair_evaluation_summary_to_json(
    summary: BeamRepairEvaluationSummary,
) -> dict[str, Any]:
    overall_fields = (
        "examples",
        "success_rate",
        "exact_symbolic_match_rate",
        "numeric_success_rate",
        "mean_steps_to_success",
        "median_steps_to_success",
        "mean_expanded_states",
        "mean_generated_candidates",
        "mean_applicable_candidates",
        "mean_repeated_candidates",
        "mean_pruned_candidates",
        "beam_empty_rate",
        "max_steps_rate",
        "numeric_patience_rate",
        "structural_patience_rate",
        "timeout_rate",
        "mean_initial_numeric_residual",
        "mean_best_numeric_residual",
        "best_numeric_residual_improvement_rate",
        "mean_final_best_numeric_residual",
        "mean_initial_structural_distance",
        "mean_best_structural_distance",
        "structural_distance_improvement_rate",
    )
    return _json_safe(
        {
            "examples": summary.examples,
            "beam_size": summary.beam_size,
            "candidate_k": summary.candidate_k,
            "scoring_config": dict(summary.scoring_config),
            "stop_config": dict(summary.stop_config),
            "overall": {name: getattr(summary, name) for name in overall_fields},
            "by_used_random_init": {
                name: asdict(group)
                for name, group in summary.by_used_random_init.items()
            },
            "by_num_mutations": {
                name: asdict(group)
                for name, group in summary.by_num_mutations.items()
            },
            "stop_reason_counts": dict(summary.stop_reason_counts),
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    config_values = _load_config_values(config_args.config)

    parser = argparse.ArgumentParser(
        description="Evaluate beam-search tree-diffusion repair.",
        parents=[config_parser],
    )
    parser.add_argument("--checkpoint", default=config_values.get("checkpoint"))
    parser.add_argument("--precomputed-data-dir", default=config_values.get("precomputed_data_dir"))
    parser.add_argument(
        "--precomputed-split",
        choices=("train", "val"),
        default=config_values.get("precomputed_split", "val"),
    )
    parser.add_argument("--data", default=config_values.get("data"))
    parser.add_argument("--output", default=config_values.get("output"))
    parser.add_argument("--dump-examples", default=config_values.get("dump_examples"))
    parser.add_argument("--num-dump-examples", type=int, default=config_values.get("num_dump_examples", 50))
    parser.add_argument(
        "--dump-failures-only",
        action="store_true",
        default=bool(config_values.get("dump_failures_only", False)),
    )
    parser.add_argument("--num-pairs", type=int, default=config_values.get("num_pairs", 128))
    parser.add_argument("--batch-size", type=int, default=config_values.get("batch_size", 32))
    parser.add_argument("--num-batches", type=int, default=config_values.get("num_batches", 5))
    parser.add_argument("--device", default=config_values.get("device", "auto"))
    parser.add_argument("--seed", type=int, default=config_values.get("seed", 123))
    parser.add_argument("--beam-size", type=int, default=config_values.get("beam_size", 8))
    parser.add_argument("--candidate-k", type=int, default=config_values.get("candidate_k", 8))
    parser.add_argument("--max-steps", type=int, default=config_values.get("max_steps", 10))
    parser.add_argument("--numeric-tol", type=float, default=config_values.get("numeric_tol", 1e-10))
    parser.add_argument(
        "--numeric-patience",
        type=_optional_int_arg,
        default=config_values.get("numeric_patience", 5),
    )
    parser.add_argument(
        "--structural-patience",
        type=_optional_int_arg,
        default=config_values.get("structural_patience"),
    )
    parser.add_argument(
        "--max-expanded-states",
        type=_optional_int_arg,
        default=config_values.get("max_expanded_states"),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_optional_float_arg,
        default=config_values.get("timeout_seconds"),
    )
    parser.add_argument("--lambda-residual", type=float, default=config_values.get("lambda_residual", 1.0))
    parser.add_argument("--lambda-size", type=float, default=config_values.get("lambda_size", 1e-3))
    parser.add_argument("--lambda-steps", type=float, default=config_values.get("lambda_steps", 1e-3))
    parser.add_argument("--lambda-policy", type=float, default=config_values.get("lambda_policy", 1e-2))
    parser.add_argument(
        "--use-log-residual",
        dest="use_log_residual",
        action="store_true",
        default=bool(config_values.get("use_log_residual", True)),
    )
    parser.add_argument("--no-use-log-residual", dest="use_log_residual", action="store_false")
    parser.add_argument(
        "--constrain-position",
        dest="constrain_position",
        action="store_true",
        default=bool(config_values.get("constrain_position", True)),
    )
    parser.add_argument("--no-constrain-position", dest="constrain_position", action="store_false")
    parser.add_argument("--max-decode-length", type=int, default=config_values.get("max_decode_length"))
    parser.add_argument(
        "--observation-timeout-seconds",
        type=float,
        default=config_values.get("observation_timeout_seconds", 2.0),
    )
    parser.add_argument(
        "--numeric-residual-timeout-seconds",
        type=float,
        default=config_values.get("numeric_residual_timeout_seconds", 2.0),
    )
    parser.add_argument(
        "--symbolic-check-timeout-seconds",
        type=float,
        default=config_values.get("symbolic_check_timeout_seconds", 2.0),
    )
    parser.add_argument("--residual-workers", type=int, default=config_values.get("residual_workers", 0))
    parser.add_argument("--progress-every", type=int, default=config_values.get("progress_every", 0))
    parser.add_argument("--quiet", action="store_true", default=bool(config_values.get("quiet", False)))
    parser.add_argument(
        "--compute-structural-metrics",
        dest="compute_structural_metrics",
        action="store_true",
        default=bool(config_values.get("compute_structural_metrics", True)),
    )
    parser.add_argument("--no-structural-metrics", dest="compute_structural_metrics", action="store_false")
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
    scoring = BeamSearchScoringConfig(
        lambda_residual=float(args.lambda_residual),
        lambda_size=float(args.lambda_size),
        lambda_steps=float(args.lambda_steps),
        lambda_policy=float(args.lambda_policy),
        use_log_residual=bool(args.use_log_residual),
    )
    stopping = BeamSearchStopConfig(
        max_steps=int(args.max_steps),
        numeric_tol=float(args.numeric_tol),
        numeric_patience=_optional_int_arg(args.numeric_patience),
        structural_patience=_optional_int_arg(args.structural_patience),
        max_expanded_states=_optional_int_arg(args.max_expanded_states),
        timeout_seconds=_optional_float_arg(args.timeout_seconds),
    )
    summary = evaluate_beam_repair(
        model,
        dataloader,
        tokenizer=tokenizer,
        device=device,
        num_batches=int(args.num_batches),
        beam_size=int(args.beam_size),
        candidate_k=int(args.candidate_k),
        scoring=scoring,
        stopping=stopping,
        constrain_position=bool(args.constrain_position),
        max_decode_length=args.max_decode_length,
        observation_timeout_seconds=args.observation_timeout_seconds,
        numeric_residual_timeout_seconds=args.numeric_residual_timeout_seconds,
        symbolic_check_timeout_seconds=args.symbolic_check_timeout_seconds,
        compute_structural_metrics=bool(args.compute_structural_metrics),
        residual_workers=int(args.residual_workers),
        progress_every=int(args.progress_every),
        progress=not bool(args.quiet),
    )
    output_json = beam_repair_evaluation_summary_to_json(summary)
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
        output_json["dump_examples"] = _dump_beam_examples(
            model,
            dump_loader,
            tokenizer=tokenizer,
            device=device,
            path=Path(args.dump_examples),
            num_examples=int(args.num_dump_examples),
            dump_failures_only=bool(args.dump_failures_only),
            beam_size=int(args.beam_size),
            candidate_k=int(args.candidate_k),
            scoring=scoring,
            stopping=stopping,
            constrain_position=bool(args.constrain_position),
            max_decode_length=args.max_decode_length,
            observation_timeout_seconds=args.observation_timeout_seconds,
            numeric_residual_timeout_seconds=args.numeric_residual_timeout_seconds,
            symbolic_check_timeout_seconds=args.symbolic_check_timeout_seconds,
            residual_workers=int(args.residual_workers),
        )

    print(json.dumps(output_json, indent=2, sort_keys=True))
    if args.output is not None:
        _write_json(Path(args.output), output_json)
    return 0


def _dump_beam_examples(
    model: TreeDiffusionPolicyModel,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    tokenizer: TreeDiffusionTokenizer,
    device: torch.device | str,
    path: Path,
    num_examples: int,
    dump_failures_only: bool,
    beam_size: int,
    candidate_k: int,
    scoring: BeamSearchScoringConfig,
    stopping: BeamSearchStopConfig,
    constrain_position: bool,
    max_decode_length: int | None,
    observation_timeout_seconds: float | None,
    numeric_residual_timeout_seconds: float | None,
    symbolic_check_timeout_seconds: float | None,
    residual_workers: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    scanned = 0
    with _residual_executor_context(int(residual_workers)) as residual_executor, path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for batch in dataloader:
            if written >= num_examples:
                break
            if not isinstance(batch, Mapping):
                raise TypeError(f"Expected dataloader batches to be mappings, got {type(batch).__name__}.")
            for row_index in range(_batch_size(batch)):
                if written >= num_examples:
                    break
                target_integrand, target_antiderivative, current = _repair_inputs(batch, row_index)
                result = beam_search_repair(
                    model,
                    target_integrand,
                    current,
                    tokenizer=tokenizer,
                    device=device,
                    beam_size=beam_size,
                    candidate_k=candidate_k,
                    scoring=scoring,
                    stopping=stopping,
                    constrain_position=constrain_position,
                    max_decode_length=max_decode_length,
                    observation_timeout_seconds=observation_timeout_seconds,
                    numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
                    symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
                    target_antiderivative=target_antiderivative,
                    residual_executor=residual_executor,
                )
                scanned += 1
                if dump_failures_only and result.success:
                    continue
                handle.write(
                    json.dumps(
                        _json_safe(
                            _beam_example_record(
                                result,
                                target_antiderivative=target_antiderivative,
                                source_mutation_trace=_mutation_trace_record(batch, row_index),
                            )
                        ),
                        sort_keys=True,
                    )
                    + "\n"
                )
                written += 1
    return {
        "path": str(path),
        "records_written": written,
        "records_scanned": scanned,
        "residual_workers": int(residual_workers),
    }


def _beam_example_record(
    result: BeamSearchResult,
    *,
    target_antiderivative: Expr,
    source_mutation_trace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "target_integrand_prefix": result.target_integrand_prefix,
        "target_antiderivative_prefix": serialize_prefix_string(target_antiderivative),
        "source_mutation_trace": source_mutation_trace,
        "initial_prefix": result.initial_prefix,
        "best_prefix": result.best_prefix,
        "success": result.success,
        "stop_reason": result.stop_reason,
        "initial_numeric_residual": result.initial_numeric_residual,
        "best_numeric_residual": result.best_numeric_residual,
        "final_best_numeric_residual": result.final_best_numeric_residual,
        "expanded_states": result.expanded_states,
        "generated_candidates": result.generated_candidates,
        "applicable_candidates": result.applicable_candidates,
        "repeated_candidates": result.repeated_candidates,
        "pruned_candidates": result.pruned_candidates,
        "path": [asdict(step) for step in result.path],
        "per_depth_best_numeric_residual": result.per_depth_best_numeric_residual,
        "per_depth_best_structural_distance": result.per_depth_best_structural_distance,
        "stop_diagnostics": result.stop_diagnostics,
    }


_BEAM_EVAL_CONFIG_FIELDS = {
    "checkpoint",
    "precomputed_data_dir",
    "precomputed_split",
    "data",
    "output",
    "dump_examples",
    "num_dump_examples",
    "dump_failures_only",
    "num_pairs",
    "batch_size",
    "num_batches",
    "device",
    "seed",
    "beam_size",
    "candidate_k",
    "max_steps",
    "numeric_tol",
    "numeric_patience",
    "structural_patience",
    "max_expanded_states",
    "timeout_seconds",
    "lambda_residual",
    "lambda_size",
    "lambda_steps",
    "lambda_policy",
    "use_log_residual",
    "constrain_position",
    "max_decode_length",
    "observation_timeout_seconds",
    "numeric_residual_timeout_seconds",
    "symbolic_check_timeout_seconds",
    "residual_workers",
    "progress_every",
    "quiet",
    "compute_structural_metrics",
}


def _load_config_values(config_path: str | None) -> dict[str, Any]:
    return _load_config_values_common(
        config_path,
        known_fields=_BEAM_EVAL_CONFIG_FIELDS,
        label="Beam eval",
    )


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.checkpoint is None:
        raise ValueError("Provide --checkpoint or set checkpoint in --config.")
    if (args.data is None) == (args.precomputed_data_dir is None):
        raise ValueError("Provide exactly one data source: --data or --precomputed-data-dir.")
    if args.num_pairs < 1:
        raise ValueError("--num-pairs must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.num_batches < 1:
        raise ValueError("--num-batches must be >= 1.")
    if args.beam_size < 1:
        raise ValueError("--beam-size must be >= 1.")
    if args.candidate_k < 1:
        raise ValueError("--candidate-k must be >= 1.")
    if args.residual_workers < 0:
        raise ValueError("--residual-workers must be >= 0.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be >= 0.")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0.")
    if args.numeric_tol < 0.0:
        raise ValueError("--numeric-tol must be >= 0.")
    for name in (
        "observation_timeout_seconds",
        "numeric_residual_timeout_seconds",
        "symbolic_check_timeout_seconds",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0.")
    if args.num_dump_examples < 0:
        raise ValueError("--num-dump-examples must be >= 0.")


def _float_or_none(value: float | int | None) -> float | None:
    return _finite_numeric(value)


__all__ = [
    "BeamRepairEvaluationRecord",
    "BeamRepairEvaluationSummary",
    "beam_repair_evaluation_summary_to_json",
    "evaluate_beam_repair",
    "main",
    "summarize_beam_repair_results",
]


if __name__ == "__main__":
    raise SystemExit(main())
