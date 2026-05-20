from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import statistics
from typing import Any, Callable, Iterable, Sequence, TypeVar

from src.tree_diffusion._common import mean_or_none, rate


@dataclass(frozen=True)
class RepairGroupSummary:
    examples: int
    success_rate: float
    exact_symbolic_match_rate: float
    numeric_success_rate: float
    mean_steps_to_success: float | None
    mean_initial_numeric_residual: float | None
    mean_final_numeric_residual: float | None
    numeric_residual_improvement_rate: float | None
    structural_distance_improvement_rate: float | None
    stop_reason_counts: dict[str, int]


_RecordT = TypeVar("_RecordT")
_ResultT = TypeVar("_ResultT")


def numeric_values(values: Iterable[float | int | None]) -> list[float]:
    return [numeric for value in values if (numeric := finite_numeric(value)) is not None]


def finite_numeric(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def is_finite(value: Any) -> bool:
    return finite_numeric(value) is not None


def meets_numeric_tol(value: float | int | None, numeric_tol: float) -> bool:
    numeric = finite_numeric(value)
    return numeric is not None and numeric <= float(numeric_tol)


def residual_improvement_rate(
    pairs: Iterable[tuple[float | int | None, float | int | None]],
) -> float | None:
    total = 0
    improved = 0
    for before, after in pairs:
        before_numeric = finite_numeric(before)
        after_numeric = finite_numeric(after)
        if before_numeric is None or after_numeric is None:
            continue
        total += 1
        improved += int(after_numeric < before_numeric)
    return None if total == 0 else improved / total


def median_or_none(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def mean_or_zero(values: Iterable[float | int | None]) -> float:
    rows = numeric_values(values)
    if not rows:
        return 0.0
    return float(sum(rows) / len(rows))


def used_random_init_group(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "random_init" if bool(value) else "local_corruption"


def num_mutations_group(value: int | None) -> str:
    if value is None:
        return "unknown"
    numeric_value = int(value)
    if numeric_value > 5:
        return "s>5"
    return f"s={numeric_value}"


def optional_int_metadata(value: Any) -> int | None:
    if value is None:
        return None
    if finite_numeric(value) is None and isinstance(value, float):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_bool_metadata(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
        return None
    if finite_numeric(value) is None and isinstance(value, float):
        return None
    return bool(value)


def summarize_repair_groups(
    records: Sequence[_RecordT],
    *,
    key_fn: Callable[[_RecordT], str],
    result_fn: Callable[[_RecordT], _ResultT],
    final_numeric_residual_fn: Callable[[_ResultT], float | None],
    structural_distance_initial_fn: Callable[[_RecordT], float | None],
    structural_distance_final_fn: Callable[[_RecordT], float | None],
    numeric_tol: float,
) -> dict[str, RepairGroupSummary]:
    grouped: dict[str, list[_RecordT]] = {}
    for record in records:
        grouped.setdefault(str(key_fn(record)), []).append(record)
    return {
        key: repair_group_summary(
            group_records,
            result_fn=result_fn,
            final_numeric_residual_fn=final_numeric_residual_fn,
            structural_distance_initial_fn=structural_distance_initial_fn,
            structural_distance_final_fn=structural_distance_final_fn,
            numeric_tol=numeric_tol,
        )
        for key, group_records in sorted(grouped.items(), key=lambda item: item[0])
    }


def repair_group_summary(
    records: Sequence[_RecordT],
    *,
    result_fn: Callable[[_RecordT], _ResultT],
    final_numeric_residual_fn: Callable[[_ResultT], float | None],
    structural_distance_initial_fn: Callable[[_RecordT], float | None],
    structural_distance_final_fn: Callable[[_RecordT], float | None],
    numeric_tol: float,
) -> RepairGroupSummary:
    rows = list(records)
    examples = len(rows)
    results = [result_fn(row) for row in rows]
    steps_to_success = [
        float(getattr(result, "steps_taken"))
        for result in results
        if bool(getattr(result, "success"))
    ]
    return RepairGroupSummary(
        examples=examples,
        success_rate=rate(sum(int(bool(getattr(result, "success"))) for result in results), examples),
        exact_symbolic_match_rate=rate(
            sum(int(bool(getattr(result, "exact_symbolic_match"))) for result in results),
            examples,
        ),
        numeric_success_rate=rate(
            sum(
                int(meets_numeric_tol(final_numeric_residual_fn(result), numeric_tol))
                for result in results
            ),
            examples,
        ),
        mean_steps_to_success=mean_or_none(steps_to_success),
        mean_initial_numeric_residual=mean_or_none(
            numeric_values(getattr(result, "initial_numeric_residual") for result in results)
        ),
        mean_final_numeric_residual=mean_or_none(
            numeric_values(final_numeric_residual_fn(result) for result in results)
        ),
        numeric_residual_improvement_rate=residual_improvement_rate(
            (
                getattr(result, "initial_numeric_residual"),
                final_numeric_residual_fn(result),
            )
            for result in results
        ),
        structural_distance_improvement_rate=residual_improvement_rate(
            (
                structural_distance_initial_fn(row),
                structural_distance_final_fn(row),
            )
            for row in rows
        ),
        stop_reason_counts=dict(Counter(str(getattr(result, "stop_reason")) for result in results)),
    )


__all__ = [
    "RepairGroupSummary",
    "finite_numeric",
    "is_finite",
    "mean_or_zero",
    "median_or_none",
    "meets_numeric_tol",
    "num_mutations_group",
    "numeric_values",
    "optional_bool_metadata",
    "optional_int_metadata",
    "repair_group_summary",
    "residual_improvement_rate",
    "summarize_repair_groups",
    "used_random_init_group",
]
