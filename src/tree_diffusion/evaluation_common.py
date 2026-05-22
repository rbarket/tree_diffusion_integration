from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Iterator, Mapping, Sequence

from src.mathlang.ast import Expr
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion._common import json_safe
from src.tree_diffusion.eval_metrics import (
    RepairGroupSummary,
    repair_group_summary,
    summarize_repair_groups,
)
from src.tree_diffusion.runtime import (
    batch_size,
    metadata_item,
    required_metadata as _required_metadata,
)


@dataclass(frozen=True)
class EvaluationDataRow:
    row_index: int
    target_integrand: Expr
    target_antiderivative: Expr
    current: Expr
    used_random_init: bool | None
    num_mutations: int | None


def iter_evaluation_rows(
    dataloader: Iterable[Mapping[str, Any]],
    *,
    num_batches: int,
    used_random_init_fn: Callable[[Any], bool | None] | None = None,
    num_mutations_fn: Callable[[Any], int | None] | None = None,
) -> Iterator[EvaluationDataRow]:
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= int(num_batches):
            break
        if not isinstance(batch, Mapping):
            raise TypeError(f"Expected dataloader batches to be mappings, got {type(batch).__name__}.")
        for row_index in range(batch_size(batch)):
            target_integrand, target_antiderivative, current = repair_inputs_from_batch(batch, row_index)
            used_random_init_raw = metadata_item(batch, "used_random_init", row_index, default=None)
            num_mutations_raw = metadata_item(batch, "num_mutations", row_index, default=None)
            yield EvaluationDataRow(
                row_index=row_index,
                target_integrand=target_integrand,
                target_antiderivative=target_antiderivative,
                current=current,
                used_random_init=(
                    None if used_random_init_fn is None else used_random_init_fn(used_random_init_raw)
                ),
                num_mutations=None if num_mutations_fn is None else num_mutations_fn(num_mutations_raw),
            )


def repair_inputs_from_batch(batch: Mapping[str, Any], row_index: int) -> tuple[Expr, Expr, Expr]:
    current_prefix = required_metadata(batch, "current_prefix", row_index)
    target_integrand_prefix = required_metadata(batch, "target_integrand_prefix", row_index)
    target_antiderivative_prefix = required_metadata(
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


def mutation_trace_record(batch: Mapping[str, Any], row_index: int) -> dict[str, Any] | None:
    mode = metadata_item(batch, "trajectory_mode", row_index, default=None)
    trajectory = metadata_item(batch, "trajectory", row_index, default=None)
    if mode is None and trajectory is None:
        return None
    return {
        "mode": mode,
        "forward": {
            "complete": metadata_item(batch, "forward_complete", row_index, default=None),
            "num_mutations": metadata_item(batch, "forward_num_mutations", row_index, default=None),
            "mutation_kinds": metadata_item(batch, "forward_mutation_kinds", row_index, default=None),
            "start_prefix": metadata_item(batch, "forward_start_prefix", row_index, default=None),
            "end_prefix": metadata_item(batch, "forward_end_prefix", row_index, default=None),
        },
        "gold_repair_step": {
            "step_index": metadata_item(batch, "repair_step_index", row_index, default=None),
            "mutation_kind": metadata_item(batch, "repair_mutation_kind", row_index, default=None),
            "reason": metadata_item(batch, "repair_reason", row_index, default=None),
            "selected_node_id": metadata_item(batch, "repair_selected_node_id", row_index, default=None),
            "selected_node_span": metadata_item(batch, "repair_selected_node_span", row_index, default=None),
            "original_subtree_prefix": metadata_item(
                batch,
                "repair_original_subtree_prefix",
                row_index,
                default=None,
            ),
            "replacement_subtree_prefix": metadata_item(
                batch,
                "repair_replacement_subtree_prefix",
                row_index,
                default=None,
            ),
            "distance_before": metadata_item(batch, "repair_distance_before", row_index, default=None),
            "distance_after": metadata_item(batch, "repair_distance_after", row_index, default=None),
        },
        "repair_reached_target": metadata_item(batch, "repair_reached_target", row_index, default=None),
        "repair_step_count": metadata_item(batch, "repair_step_count", row_index, default=None),
    }


def required_metadata(batch: Mapping[str, Any], key: str, row_index: int) -> str:
    return str(_required_metadata(batch, key, row_index))


def dump_jsonl_records(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_safe(record), sort_keys=True) + "\n")
            written += 1
    return written


def load_config_values(path: str | Path | None, *, known_fields: set[str], label: str) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} config must be a JSON object, got {type(payload).__name__}.")
    values = dict(payload)
    unknown = set(values) - set(known_fields)
    if unknown:
        raise ValueError(f"Unknown {label} config field(s): " + ", ".join(sorted(unknown)))
    return values


def summarize_repair_record_groups(
    records: Sequence[Any],
    *,
    key_fn: Callable[[Any], str],
    result_fn: Callable[[Any], Any],
    final_numeric_residual_fn: Callable[[Any], float | None],
    structural_distance_initial_fn: Callable[[Any], float | None],
    structural_distance_final_fn: Callable[[Any], float | None],
    numeric_tol: float,
) -> dict[str, RepairGroupSummary]:
    return summarize_repair_groups(
        records,
        key_fn=key_fn,
        result_fn=result_fn,
        final_numeric_residual_fn=final_numeric_residual_fn,
        structural_distance_initial_fn=structural_distance_initial_fn,
        structural_distance_final_fn=structural_distance_final_fn,
        numeric_tol=numeric_tol,
    )


def repair_record_group_summary(
    records: Sequence[Any],
    *,
    result_fn: Callable[[Any], Any],
    final_numeric_residual_fn: Callable[[Any], float | None],
    structural_distance_initial_fn: Callable[[Any], float | None],
    structural_distance_final_fn: Callable[[Any], float | None],
    numeric_tol: float,
) -> RepairGroupSummary:
    return repair_group_summary(
        records,
        result_fn=result_fn,
        final_numeric_residual_fn=final_numeric_residual_fn,
        structural_distance_initial_fn=structural_distance_initial_fn,
        structural_distance_final_fn=structural_distance_final_fn,
        numeric_tol=numeric_tol,
    )


def residual_executor_context(
    residual_workers: int,
) -> ContextManager[ProcessPoolExecutor | None]:
    if residual_workers <= 0:
        return nullcontext(None)
    # Use spawn so CPU-only SymPy workers do not inherit CUDA state from the
    # main decoding process.
    context = mp.get_context("spawn")
    return ProcessPoolExecutor(max_workers=int(residual_workers), mp_context=context)


__all__ = [
    "EvaluationDataRow",
    "batch_size",
    "dump_jsonl_records",
    "iter_evaluation_rows",
    "load_config_values",
    "metadata_item",
    "mutation_trace_record",
    "repair_inputs_from_batch",
    "required_metadata",
    "repair_record_group_summary",
    "residual_executor_context",
    "summarize_repair_record_groups",
]
