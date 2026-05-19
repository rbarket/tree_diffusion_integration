from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import torch


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is not available.")
    return resolved


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def mean_or_none(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def percentile(
    ordered_values: Sequence[int | float],
    quantile: float,
    *,
    mode: str = "linear",
) -> float | int | None:
    if not ordered_values:
        return None
    if len(ordered_values) == 1:
        return ordered_values[0]
    if mode == "nearest":
        index = int(round((len(ordered_values) - 1) * quantile))
        return int(ordered_values[index])
    if mode != "linear":
        raise ValueError("percentile mode must be 'linear' or 'nearest'.")

    index = quantile * (len(ordered_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered_values) - 1)
    fraction = index - lower
    return ordered_values[lower] * (1.0 - fraction) + ordered_values[upper] * fraction


def length_summary(
    values: Sequence[int | float],
    *,
    percentile_mode: str = "linear",
) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(int(value) for value in values)
    return {
        "mean": float(statistics.mean(ordered)),
        "p50": percentile(ordered, 0.50, mode=percentile_mode),
        "p95": percentile(ordered, 0.95, mode=percentile_mode),
        "max": ordered[-1],
    }


def selected_node_summary(
    values: Sequence[int],
    *,
    include_min: bool = True,
    include_top_counts: bool = False,
    percentile_mode: str = "linear",
) -> dict[str, Any]:
    if include_min:
        if values:
            ordered = sorted(int(value) for value in values)
            summary: dict[str, Any] = {
                "min": ordered[0],
                "p50": percentile(ordered, 0.50, mode=percentile_mode),
                "p95": percentile(ordered, 0.95, mode=percentile_mode),
                "max": ordered[-1],
            }
        else:
            summary = {"min": None, "p50": None, "p95": None, "max": None}
    else:
        summary = length_summary(values, percentile_mode=percentile_mode)

    if include_top_counts:
        summary["top_counts"] = Counter(values).most_common(10)
    return summary


def move_tensor_batch(
    batch: Mapping[str, Any],
    *,
    device: torch.device | str | None,
) -> dict[str, Any]:
    if device is None:
        return dict(batch)
    target_device = torch.device(device)
    return {
        key: value.to(target_device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def diagnostic_metrics(summary: Any) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "examples": summary.examples,
        "valid_position_rate": summary.valid_position_rate,
        "parseable_replacement_rate": summary.parseable_replacement_rate,
        "applicable_edit_rate": summary.applicable_edit_rate,
        "structural_improvement_rate": summary.structural_improvement_rate,
        "numeric_residual_improvement_rate": summary.numeric_residual_improvement_rate,
        "exact_target_rate": summary.exact_target_rate,
        "mean_structural_distance_before": summary.mean_structural_distance_before,
        "mean_structural_distance_after": summary.mean_structural_distance_after,
    }
    for name in (
        "decoded_ok_rate",
        "nonincreasing_structural_rate",
        "mean_numeric_residual_before",
        "mean_numeric_residual_after",
        "diagnostic_example_timeout_count",
        "diagnostic_total_timeout_count",
        "numeric_residual_timeout_count",
        "any_decoded_ok_rate",
        "any_applicable_edit_rate",
        "any_structural_improvement_rate",
        "first_applicable_rank_mean",
    ):
        value = getattr(summary, name, None)
        if value is not None:
            metrics[name] = value
    return metrics
