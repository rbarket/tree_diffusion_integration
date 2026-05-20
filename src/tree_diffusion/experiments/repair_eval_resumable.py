from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion._common import (
    json_safe as _json_safe,
    resolve_device as _resolve_device,
    write_json as _write_json,
)
from src.tree_diffusion.edit_path import structural_distance
from src.tree_diffusion.evaluation_common import (
    batch_size as _batch_size,
    metadata_item as _metadata_item,
    mutation_trace_record as _mutation_trace_record,
    repair_inputs_from_batch as _repair_inputs,
    residual_executor_context as _residual_executor_context,
)
from src.tree_diffusion.eval_metrics import (
    optional_bool_metadata as _optional_bool_metadata,
    optional_int_metadata as _optional_int_metadata,
)
from src.tree_diffusion.evaluate_repair import (
    RepairEvaluationRecord,
    repair_evaluation_summary_to_json,
    summarize_repair_results,
)
from src.tree_diffusion.experiments.resumable import (
    build_resumable_dataloader as _build_resumable_dataloader,
    completed_example_count as _completed_example_count,
    data_source_summary as _data_source_summary,
    load_config as _load_config,
    load_part_records as _load_generic_part_records,
    merge_cli_config as _merge_cli_config,
    next_part_index as _next_part_index,
    prepare_output_dir as _prepare_output_dir,
    progress as _progress,
    run_config as _run_config,
    target_example_count as _target_example_count,
    write_manifest as _write_manifest,
    write_part_records as _write_part,
)
from src.tree_diffusion.repair import RepairResult, RepairStep, greedy_repair
from src.tree_diffusion.runtime import (
    load_model_and_tokenizer_for_inference as _load_cli_model_and_tokenizer,
)


def run_resumable_greedy_repair_eval(
    *,
    checkpoint: str,
    output_dir: str | Path,
    precomputed_data_dir: str | None = None,
    precomputed_split: str = "val",
    data: str | None = None,
    num_pairs: int | None = None,
    num_batches: int | None = None,
    batch_size: int = 32,
    device: torch.device | str = "auto",
    seed: int = 123,
    max_steps: int = 10,
    candidate_k: int = 8,
    numeric_tol: float = 1e-10,
    patience: int = 2,
    constrain_position: bool = True,
    max_decode_length: int | None = None,
    observation_timeout_seconds: float | None = 2.0,
    numeric_residual_timeout_seconds: float | None = 2.0,
    symbolic_check_timeout_seconds: float | None = 2.0,
    compute_structural_metrics: bool = True,
    selection_strategy: str = "residual_scored",
    residual_workers: int = 0,
    part_size: int = 500,
    resume: bool = False,
    overwrite: bool = False,
    max_examples_this_run: int | None = None,
    progress: bool = True,
    progress_every: int = 25,
    flush_every: int | None = None,
) -> dict[str, Any]:
    _validate_args(
        data=data,
        precomputed_data_dir=precomputed_data_dir,
        precomputed_split=precomputed_split,
        num_pairs=num_pairs,
        num_batches=num_batches,
        batch_size=batch_size,
        max_steps=max_steps,
        candidate_k=candidate_k,
        numeric_tol=numeric_tol,
        patience=patience,
        residual_workers=residual_workers,
        part_size=part_size,
        resume=resume,
        overwrite=overwrite,
        max_examples_this_run=max_examples_this_run,
        selection_strategy=selection_strategy,
        progress_every=progress_every,
        flush_every=flush_every,
        observation_timeout_seconds=observation_timeout_seconds,
    )

    output_path = Path(output_dir)
    parts_dir = output_path / "parts"
    config = _run_config(
        checkpoint=checkpoint,
        precomputed_data_dir=precomputed_data_dir,
        precomputed_split=precomputed_split,
        data=data,
        num_pairs=num_pairs,
        num_batches=num_batches,
        batch_size=batch_size,
        device=device,
        seed=seed,
        max_steps=max_steps,
        candidate_k=candidate_k,
        numeric_tol=numeric_tol,
        patience=patience,
        constrain_position=constrain_position,
        max_decode_length=max_decode_length,
        observation_timeout_seconds=observation_timeout_seconds,
        numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
        symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
        compute_structural_metrics=compute_structural_metrics,
        selection_strategy=selection_strategy,
        residual_workers=residual_workers,
        part_size=part_size,
        progress_every=progress_every,
        flush_every=flush_every,
    )
    _prepare_output_dir(output_path, config=config, resume=resume, overwrite=overwrite)
    parts_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(seed))
    target_device = _resolve_device(str(device))
    tokenizer, model = _load_cli_model_and_tokenizer(
        checkpoint=str(checkpoint),
        precomputed_data_dir=precomputed_data_dir,
        allow_random_init_model=False,
    )
    model.to(target_device)
    model.eval()

    dataloader = _build_resumable_dataloader(
        data=data,
        precomputed_data_dir=precomputed_data_dir,
        precomputed_split=precomputed_split,
        tokenizer=tokenizer,
        model=model,
        num_pairs=num_pairs,
        batch_size=batch_size,
        seed=seed,
    )
    target_examples = _target_example_count(
        dataloader=dataloader,
        num_pairs=num_pairs,
        num_batches=num_batches,
        batch_size=batch_size,
    )
    completed_examples = _completed_example_count(parts_dir)
    if completed_examples > target_examples:
        raise ValueError(
            "Cannot resume: completed part rows exceed requested target examples "
            f"({completed_examples} > {target_examples})."
        )
    _progress(
        "repair_eval_resume_state "
        f"completed={completed_examples} target={target_examples} "
        f"next_part={_next_part_index(parts_dir):06d} resume={bool(resume)}",
        enabled=progress,
    )

    next_part_index = _next_part_index(parts_dir)
    current_part: list[dict[str, Any]] = []
    seen_examples = 0
    written_this_run = 0
    batch_count = 0

    with _residual_executor_context(int(residual_workers)) as residual_executor:
        for batch in dataloader:
            if num_batches is not None and batch_count >= int(num_batches):
                break
            batch_count += 1
            if not isinstance(batch, Mapping):
                raise TypeError(f"Expected dataloader batches to be mappings, got {type(batch).__name__}.")
            for row_index in range(_batch_size(batch)):
                if seen_examples >= target_examples:
                    break
                if seen_examples < completed_examples:
                    seen_examples += 1
                    continue
                if max_examples_this_run is not None and written_this_run >= max_examples_this_run:
                    break

                record_payload = _evaluate_one_record(
                    model=model,
                    batch=batch,
                    row_index=row_index,
                    example_index=seen_examples,
                    tokenizer=tokenizer,
                    device=target_device,
                    max_steps=max_steps,
                    candidate_k=candidate_k,
                    numeric_tol=numeric_tol,
                    patience=patience,
                    constrain_position=constrain_position,
                    max_decode_length=max_decode_length,
                    observation_timeout_seconds=observation_timeout_seconds,
                    numeric_residual_timeout_seconds=numeric_residual_timeout_seconds,
                    symbolic_check_timeout_seconds=symbolic_check_timeout_seconds,
                    compute_structural_metrics=compute_structural_metrics,
                    selection_strategy=selection_strategy,
                    residual_executor=residual_executor,
                )
                current_part.append(record_payload)
                seen_examples += 1
                written_this_run += 1
                completed_including_buffer = completed_examples + written_this_run
                if (
                    progress_every > 0
                    and completed_including_buffer % int(progress_every) == 0
                ):
                    _progress(
                        "repair_eval_progress "
                        f"completed={completed_including_buffer}/{target_examples} "
                        f"current_part_rows={len(current_part)} "
                        f"last_stop_reason={record_payload['result']['stop_reason']} "
                        f"last_success={record_payload['result']['success']}",
                        enabled=progress,
                    )

                if len(current_part) >= int(part_size) or (
                    flush_every is not None
                    and int(flush_every) > 0
                    and completed_including_buffer % int(flush_every) == 0
                ):
                    _write_part(parts_dir, next_part_index, current_part)
                    completed_now = _completed_example_count(parts_dir)
                    _progress(
                        "repair_eval_part_written "
                        f"part={next_part_index:06d} rows={len(current_part)} "
                        f"completed={completed_now}/{target_examples}",
                        enabled=progress,
                    )
                    next_part_index += 1
                    current_part = []
                    _write_manifest(
                        output_path,
                        config=config,
                        target_examples=target_examples,
                        completed_examples=_completed_example_count(parts_dir),
                        complete=False,
                    )

            if seen_examples >= target_examples:
                break
            if max_examples_this_run is not None and written_this_run >= max_examples_this_run:
                break

    if current_part:
        _write_part(parts_dir, next_part_index, current_part)
        completed_now = _completed_example_count(parts_dir)
        _progress(
            "repair_eval_part_written "
            f"part={next_part_index:06d} rows={len(current_part)} "
            f"completed={completed_now}/{target_examples}",
            enabled=progress,
        )

    completed_after = _completed_example_count(parts_dir)
    complete = completed_after >= target_examples
    summary_json = combine_resumable_repair_eval(
        output_path,
        numeric_tol=float(numeric_tol),
        max_steps=int(max_steps),
        candidate_k=int(candidate_k),
        selection_strategy=str(selection_strategy),
        target_examples=target_examples,
        complete=complete,
    )
    _write_manifest(
        output_path,
        config=config,
        target_examples=target_examples,
        completed_examples=completed_after,
        complete=complete,
    )
    _progress(
        "repair_eval_summary_written "
        f"path={output_path / 'repair_eval_summary.json'} "
        f"completed={completed_after}/{target_examples} complete={complete}",
        enabled=progress,
    )
    return summary_json


def combine_resumable_repair_eval(
    output_dir: str | Path,
    *,
    numeric_tol: float | None = None,
    max_steps: int | None = None,
    candidate_k: int | None = None,
    selection_strategy: str | None = None,
    target_examples: int | None = None,
    complete: bool | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    config = _load_config(output_path)
    numeric_tol = float(config["numeric_tol"] if numeric_tol is None else numeric_tol)
    max_steps = int(config["max_steps"] if max_steps is None else max_steps)
    candidate_k = int(config["candidate_k"] if candidate_k is None else candidate_k)
    selection_strategy = str(
        config["selection_strategy"] if selection_strategy is None else selection_strategy
    )
    records = _load_part_records(output_path / "parts")
    summary = summarize_repair_results(
        records,
        numeric_tol=numeric_tol,
        max_steps=max_steps,
        candidate_k=candidate_k,
        selection_strategy=selection_strategy,
    )
    summary_json = repair_evaluation_summary_to_json(summary)
    summary_json.update(
        {
            "checkpoint": config["checkpoint"],
            "data_source": _data_source_summary(config),
            "output_dir": str(output_path),
            "part_size": int(config["part_size"]),
            "part_count": len(list((output_path / "parts").glob("part_*.jsonl"))),
            "completed_examples": len(records),
            "target_examples": target_examples,
            "complete": bool(complete) if complete is not None else None,
            "residual_workers": int(config["residual_workers"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json(output_path / "repair_eval_summary.json", summary_json)
    return summary_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run resumable greedy repair evaluation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--precomputed-data-dir", default=None)
    parser.add_argument("--precomputed-split", choices=("train", "val"), default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-pairs", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--numeric-tol", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--selection-strategy",
        choices=("rank1", "residual_scored"),
        default=None,
    )
    parser.add_argument("--constrain-position", dest="constrain_position", action="store_true", default=None)
    parser.add_argument("--no-constrain-position", dest="constrain_position", action="store_false")
    parser.add_argument("--max-decode-length", type=int, default=None)
    parser.add_argument("--observation-timeout-seconds", type=float, default=None)
    parser.add_argument("--numeric-residual-timeout-seconds", type=float, default=None)
    parser.add_argument("--symbolic-check-timeout-seconds", type=float, default=None)
    parser.add_argument("--residual-workers", type=int, default=None)
    parser.add_argument("--part-size", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-examples-this-run", type=int, default=None)
    parser.add_argument("--compute-structural-metrics", dest="compute_structural_metrics", action="store_true", default=None)
    parser.add_argument("--no-structural-metrics", dest="compute_structural_metrics", action="store_false")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    values = _merged_cli_config(args)

    summary = run_resumable_greedy_repair_eval(
        checkpoint=str(values["checkpoint"]),
        output_dir=values["output_dir"],
        precomputed_data_dir=values["precomputed_data_dir"],
        precomputed_split=str(values["precomputed_split"]),
        data=values["data"],
        num_pairs=values["num_pairs"],
        num_batches=values["num_batches"],
        batch_size=int(values["batch_size"]),
        device=str(values["device"]),
        seed=int(values["seed"]),
        max_steps=int(values["max_steps"]),
        candidate_k=int(values["candidate_k"]),
        numeric_tol=float(values["numeric_tol"]),
        patience=int(values["patience"]),
        constrain_position=bool(values["constrain_position"]),
        max_decode_length=values["max_decode_length"],
        observation_timeout_seconds=values["observation_timeout_seconds"],
        numeric_residual_timeout_seconds=values["numeric_residual_timeout_seconds"],
        symbolic_check_timeout_seconds=values["symbolic_check_timeout_seconds"],
        compute_structural_metrics=bool(values["compute_structural_metrics"]),
        selection_strategy=str(values["selection_strategy"]),
        residual_workers=int(values["residual_workers"]),
        part_size=int(values["part_size"]),
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        max_examples_this_run=values["max_examples_this_run"],
        progress=not bool(args.quiet),
        progress_every=int(values["progress_every"]),
        flush_every=values["flush_every"],
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


def _evaluate_one_record(
    *,
    model,
    batch: Mapping[str, Any],
    row_index: int,
    example_index: int,
    tokenizer,
    device: torch.device,
    max_steps: int,
    candidate_k: int,
    numeric_tol: float,
    patience: int,
    constrain_position: bool,
    max_decode_length: int | None,
    observation_timeout_seconds: float | None,
    numeric_residual_timeout_seconds: float | None,
    symbolic_check_timeout_seconds: float | None,
    compute_structural_metrics: bool,
    selection_strategy: str,
    residual_executor,
) -> dict[str, Any]:
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
        target_antiderivative=target_antiderivative if compute_structural_metrics else None,
        selection_strategy=selection_strategy,
        residual_executor=residual_executor,
    )

    initial_distance = None
    final_distance = None
    if compute_structural_metrics:
        final_tree = canonicalize(parse_prefix_string(result.final_prefix))
        initial_distance = float(structural_distance(current, target_antiderivative))
        final_distance = float(structural_distance(final_tree, target_antiderivative))

    return _record_to_json(
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
        ),
        example_index=example_index,
        target_antiderivative_prefix=serialize_prefix_string(target_antiderivative),
        source_mutation_trace=_mutation_trace_record(batch, row_index),
    )


def _record_to_json(
    record: RepairEvaluationRecord,
    *,
    example_index: int,
    target_antiderivative_prefix: str,
    source_mutation_trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _json_safe(
        {
            "example_index": int(example_index),
            "target_antiderivative_prefix": target_antiderivative_prefix,
            "source_mutation_trace": source_mutation_trace,
            "result": asdict(record.result),
            "used_random_init": record.used_random_init,
            "num_mutations": record.num_mutations,
            "structural_distance_initial": record.structural_distance_initial,
            "structural_distance_final": record.structural_distance_final,
        }
    )


def _merged_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "checkpoint": None,
        "precomputed_data_dir": None,
        "precomputed_split": "val",
        "data": None,
        "output_dir": None,
        "num_pairs": None,
        "num_batches": None,
        "batch_size": 32,
        "device": "auto",
        "seed": 123,
        "max_steps": 10,
        "candidate_k": 8,
        "numeric_tol": 1e-10,
        "patience": 2,
        "selection_strategy": "residual_scored",
        "constrain_position": True,
        "max_decode_length": None,
        "observation_timeout_seconds": 2.0,
        "numeric_residual_timeout_seconds": 2.0,
        "symbolic_check_timeout_seconds": 2.0,
        "residual_workers": 0,
        "part_size": 500,
        "progress_every": 25,
        "flush_every": None,
        "max_examples_this_run": None,
        "compute_structural_metrics": True,
    }
    return _merge_cli_config(
        args,
        defaults=defaults,
        required=("checkpoint", "output_dir"),
        label="repair eval",
    )


def _record_from_json(payload: Mapping[str, Any]) -> RepairEvaluationRecord:
    raw_result = dict(payload["result"])
    raw_steps = raw_result.pop("steps", [])
    result = RepairResult(
        **raw_result,
        steps=[RepairStep(**dict(step)) for step in raw_steps],
    )
    return RepairEvaluationRecord(
        result=result,
        used_random_init=_optional_bool_metadata(payload.get("used_random_init")),
        num_mutations=_optional_int_metadata(payload.get("num_mutations")),
        structural_distance_initial=_optional_float(payload.get("structural_distance_initial")),
        structural_distance_final=_optional_float(payload.get("structural_distance_final")),
    )


def _load_part_records(parts_dir: Path) -> list[RepairEvaluationRecord]:
    return _load_generic_part_records(
        parts_dir,
        record_from_json=_record_from_json,
        label="repair",
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if torch.isfinite(torch.tensor(parsed)).item() else None


def _validate_args(
    *,
    data: str | None,
    precomputed_data_dir: str | None,
    precomputed_split: str,
    num_pairs: int | None,
    num_batches: int | None,
    batch_size: int,
    max_steps: int,
    candidate_k: int,
    numeric_tol: float,
    patience: int,
    residual_workers: int,
    part_size: int,
    resume: bool,
    overwrite: bool,
    max_examples_this_run: int | None,
    selection_strategy: str,
    progress_every: int,
    flush_every: int | None,
    observation_timeout_seconds: float | None,
) -> None:
    if (data is None) == (precomputed_data_dir is None):
        raise ValueError("Provide exactly one data source: --data or --precomputed-data-dir.")
    if precomputed_split not in {"train", "val"}:
        raise ValueError("precomputed_split must be 'train' or 'val'.")
    if data is not None and num_pairs is None:
        raise ValueError("--num-pairs is required for online --data evaluation.")
    if num_pairs is not None and num_pairs < 1:
        raise ValueError("--num-pairs must be >= 1 when provided.")
    if num_batches is not None and num_batches < 1:
        raise ValueError("--num-batches must be >= 1 when provided.")
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if max_steps < 0:
        raise ValueError("--max-steps must be >= 0.")
    if candidate_k < 1:
        raise ValueError("--candidate-k must be >= 1.")
    if numeric_tol < 0.0:
        raise ValueError("--numeric-tol must be >= 0.")
    if progress_every < 0:
        raise ValueError("--progress-every must be >= 0.")
    if flush_every is not None and flush_every < 1:
        raise ValueError("--flush-every must be >= 1 when provided.")
    if observation_timeout_seconds is not None and observation_timeout_seconds <= 0.0:
        raise ValueError("--observation-timeout-seconds must be > 0.")
    if patience < 0:
        raise ValueError("--patience must be >= 0.")
    if residual_workers < 0:
        raise ValueError("--residual-workers must be >= 0.")
    if part_size < 1:
        raise ValueError("--part-size must be >= 1.")
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if max_examples_this_run is not None and max_examples_this_run < 1:
        raise ValueError("--max-examples-this-run must be >= 1 when provided.")
    if selection_strategy not in {"rank1", "residual_scored"}:
        raise ValueError("selection_strategy must be 'rank1' or 'residual_scored'.")


__all__ = [
    "combine_resumable_repair_eval",
    "main",
    "run_resumable_greedy_repair_eval",
]


if __name__ == "__main__":
    raise SystemExit(main())
