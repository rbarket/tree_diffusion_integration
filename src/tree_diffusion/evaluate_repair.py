from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

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
from src.tree_diffusion.eval_one_step import (
    _build_cli_dataloader,
    _load_cli_model_and_tokenizer,
)
from src.tree_diffusion.model import TreeDiffusionPolicyModel
from src.tree_diffusion.repair import RepairResult, greedy_repair
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


@dataclass(frozen=True)
class RepairEvaluationSummary:
    examples: int
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
    mean_structural_distance_initial: float | None
    mean_structural_distance_final: float | None
    structural_distance_improvement_rate: float | None
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
    constrain_position: bool = True,
    max_decode_length: int | None = None,
    residual_mode: str = "both",
    simplify_symbolic_residual: bool = True,
    numeric_residual_timeout_seconds: float | None = 2.0,
    symbolic_check_timeout_seconds: float | None = 2.0,
    compute_structural_metrics: bool = True,
) -> RepairEvaluationSummary:
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1.")
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0.")
    if candidate_k < 1:
        raise ValueError("candidate_k must be >= 1.")
    if numeric_tol < 0.0:
        raise ValueError("numeric_tol must be >= 0.")

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    examples = 0
    successes = 0
    exact_matches = 0
    numeric_successes = 0
    steps_to_success: list[float] = []
    no_candidates = 0
    repeated_states = 0
    max_steps_count = 0
    no_numeric_improvements = 0
    initial_numeric_values: list[float] = []
    final_numeric_values: list[float] = []
    numeric_improvements = 0
    numeric_pairs = 0
    initial_structural_values: list[float] = []
    final_structural_values: list[float] = []
    structural_improvements = 0
    structural_examples = 0
    stop_reason_counts: Counter[str] = Counter()

    iterator = iter(dataloader)
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
                constrain_position=constrain_position,
                max_decode_length=max_decode_length,
                residual_mode=residual_mode,
                simplify_symbolic_residual=simplify_symbolic_residual,
                numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
                symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
                target_antiderivative=(
                    target_antiderivative if compute_structural_metrics else None
                ),
            )

            examples += 1
            stop_reason_counts[result.stop_reason] += 1
            successes += int(result.success)
            exact_matches += int(result.exact_symbolic_match)
            numeric_successes += int(
                result.final_numeric_residual is not None
                and float(result.final_numeric_residual) <= numeric_tol
            )
            no_candidates += int(result.no_candidate)
            repeated_states += int(result.repeated_state)
            max_steps_count += int(result.stop_reason == "max_steps")
            no_numeric_improvements += int(result.stop_reason == "no_numeric_improvement")
            if result.success:
                steps_to_success.append(float(result.steps_taken))

            if result.initial_numeric_residual is not None:
                initial_numeric_values.append(float(result.initial_numeric_residual))
            if result.final_numeric_residual is not None:
                final_numeric_values.append(float(result.final_numeric_residual))
            if (
                result.initial_numeric_residual is not None
                and result.final_numeric_residual is not None
            ):
                numeric_pairs += 1
                if float(result.final_numeric_residual) < float(result.initial_numeric_residual):
                    numeric_improvements += 1

            if compute_structural_metrics:
                final_tree = canonicalize(parse_prefix_string(result.final_prefix))
                initial_distance = float(structural_distance(current, target_antiderivative))
                final_distance = float(structural_distance(final_tree, target_antiderivative))
                structural_examples += 1
                initial_structural_values.append(initial_distance)
                final_structural_values.append(final_distance)
                if final_distance < initial_distance:
                    structural_improvements += 1

    return RepairEvaluationSummary(
        examples=examples,
        success_rate=_rate(successes, examples),
        exact_symbolic_match_rate=_rate(exact_matches, examples),
        numeric_success_rate=_rate(numeric_successes, examples),
        mean_steps_to_success=_mean_or_none(steps_to_success),
        median_steps_to_success=_median_or_none(steps_to_success),
        no_candidate_rate=_rate(no_candidates, examples),
        repeated_state_rate=_rate(repeated_states, examples),
        max_steps_rate=_rate(max_steps_count, examples),
        no_numeric_improvement_rate=_rate(no_numeric_improvements, examples),
        mean_initial_numeric_residual=_mean_or_none(initial_numeric_values),
        mean_final_numeric_residual=_mean_or_none(final_numeric_values),
        numeric_residual_improvement_rate=(
            None if numeric_pairs == 0 else numeric_improvements / numeric_pairs
        ),
        mean_structural_distance_initial=_mean_or_none(initial_structural_values),
        mean_structural_distance_final=_mean_or_none(final_structural_values),
        structural_distance_improvement_rate=(
            None if structural_examples == 0 else structural_improvements / structural_examples
        ),
        stop_reason_counts=dict(stop_reason_counts),
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
    parser.add_argument("--constrain-position", dest="constrain_position", action="store_true", default=True)
    parser.add_argument("--no-constrain-position", dest="constrain_position", action="store_false")
    parser.add_argument("--max-decode-length", type=int, default=None)
    parser.add_argument("--numeric-residual-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--symbolic-check-timeout-seconds", type=float, default=2.0)
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
        constrain_position=bool(args.constrain_position),
        max_decode_length=args.max_decode_length,
        numeric_residual_timeout_seconds=args.numeric_residual_timeout_seconds,
        symbolic_check_timeout_seconds=args.symbolic_check_timeout_seconds,
        compute_structural_metrics=bool(args.compute_structural_metrics),
    )
    output_json = _json_safe(asdict(summary))

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
            constrain_position=bool(args.constrain_position),
            max_decode_length=args.max_decode_length,
            numeric_residual_timeout_seconds=args.numeric_residual_timeout_seconds,
            symbolic_check_timeout_seconds=args.symbolic_check_timeout_seconds,
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
    constrain_position: bool,
    max_decode_length: int | None,
    numeric_residual_timeout_seconds: float | None,
    symbolic_check_timeout_seconds: float | None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    scanned = 0
    iterator = iter(dataloader)
    with path.open("w", encoding="utf-8") as handle:
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
                    constrain_position=constrain_position,
                    max_decode_length=max_decode_length,
                    numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
                    symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
                    target_antiderivative=target_antiderivative,
                )
                scanned += 1
                if dump_failures_only and result.success:
                    continue
                record = _repair_example_record(
                    result,
                    target_antiderivative=target_antiderivative,
                )
                handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
                written += 1
    return {
        "path": str(path),
        "records_written": written,
        "records_scanned": scanned,
    }


def _repair_example_record(
    result: RepairResult,
    *,
    target_antiderivative: Expr,
) -> dict[str, Any]:
    return {
        "target_integrand_prefix": result.target_integrand_prefix,
        "target_antiderivative_prefix": serialize_prefix_string(target_antiderivative),
        "initial_prefix": result.initial_prefix,
        "final_prefix": result.final_prefix,
        "success": result.success,
        "stop_reason": result.stop_reason,
        "initial_numeric_residual": result.initial_numeric_residual,
        "final_numeric_residual": result.final_numeric_residual,
        "steps_taken": result.steps_taken,
        "steps": [asdict(step) for step in result.steps],
    }


def _repair_inputs(batch: Mapping[str, Any], row_index: int) -> tuple[Expr, Expr, Expr]:
    current_prefix = _required_metadata(batch, "current_prefix", row_index)
    target_integrand_prefix = _required_metadata(batch, "target_integrand_prefix", row_index)
    target_antiderivative_prefix = _required_metadata(
        batch,
        "target_antiderivative_prefix",
        row_index,
    )
    return (
        canonicalize(
            parse_prefix_string(target_integrand_prefix),
            strip_additive_constants=False,
        ),
        canonicalize(parse_prefix_string(target_antiderivative_prefix)),
        canonicalize(parse_prefix_string(current_prefix)),
    )


def _batch_size(batch: Mapping[str, Any]) -> int:
    for key in ("current_prefix", "target_integrand_prefix", "target_antiderivative_prefix"):
        value = batch.get(key)
        if isinstance(value, (list, tuple)):
            return len(value)
    input_ids = batch.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 1:
            return 1
        if input_ids.ndim >= 2:
            return int(input_ids.size(0))
    return 1


def _required_metadata(batch: Mapping[str, Any], key: str, row_index: int) -> str:
    if key not in batch:
        raise ValueError(f"Batch is missing required metadata field {key!r}.")
    value = batch[key]
    if isinstance(value, (list, tuple)):
        try:
            item = value[row_index]
        except IndexError as exc:
            raise ValueError(f"Metadata field {key!r} is shorter than the batch.") from exc
    else:
        item = value
    if item is None:
        raise ValueError(f"Metadata field {key!r} contains None.")
    return str(item)


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


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
    if args.numeric_residual_timeout_seconds is not None and args.numeric_residual_timeout_seconds <= 0.0:
        raise ValueError("--numeric-residual-timeout-seconds must be > 0.")
    if args.symbolic_check_timeout_seconds is not None and args.symbolic_check_timeout_seconds <= 0.0:
        raise ValueError("--symbolic-check-timeout-seconds must be > 0.")
    if args.num_dump_examples < 0:
        raise ValueError("--num-dump-examples must be >= 0.")


__all__ = [
    "RepairEvaluationSummary",
    "evaluate_greedy_repair",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
