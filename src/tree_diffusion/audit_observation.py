from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any

import pandas as pd

from src.mathlang.ast import Expr
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string, serialize_prefix_tokens
from src.tree_diffusion.mutation import mutate_once
from src.tree_diffusion.observation import Observation, build_observation


DEFAULT_SUMMARY_FILE = Path("artifacts/observation_timing_summary.json")


@dataclass(frozen=True)
class ObservationTimingCase:
    case_id: str
    row_index: int
    target_integrand: Expr
    current_antiderivative: Expr
    mutation_count: int = 0
    seed: int | None = None


@dataclass(frozen=True)
class ObservationTimingRecord:
    case_id: str
    row_index: int
    repeat_index: int
    mutation_count: int
    seed: int | None
    elapsed_seconds: float
    status: str
    warnings: tuple[str, ...]
    derivative_token_length: int | None
    residual_token_length: int | None
    fraction_finite: float | None
    fraction_complex: float | None
    target_integrand_prefix: str
    current_antiderivative_prefix: str


def load_dataset_frame(
    parquet_path: str | Path,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_parquet(
        path,
        columns=["integrand_prefix", "integral_prefix"],
    )
    if limit is not None:
        frame = frame.head(limit).copy()
    return frame


def build_gold_timing_cases(frame: pd.DataFrame) -> list[ObservationTimingCase]:
    cases: list[ObservationTimingCase] = []
    for row_index, row in enumerate(frame.itertuples(index=False)):
        target = parse_prefix_string(str(row.integrand_prefix))
        current = parse_prefix_string(str(row.integral_prefix))
        cases.append(
            ObservationTimingCase(
                case_id=f"gold_row_{row_index}",
                row_index=row_index,
                target_integrand=target,
                current_antiderivative=current,
            )
        )
    return cases


def build_corrupted_timing_cases(
    frame: pd.DataFrame,
    scenarios: list[dict[str, int]],
    *,
    sigma_small: int = 0,
) -> list[ObservationTimingCase]:
    cases: list[ObservationTimingCase] = []

    for scenario in scenarios:
        row_index = int(scenario["row_index"])
        mutation_count = int(scenario["mutation_count"])
        seed = int(scenario["seed"])
        row = frame.iloc[row_index]
        target = parse_prefix_string(str(row.integrand_prefix))
        current = parse_prefix_string(str(row.integral_prefix))
        rng = random.Random(seed)

        for _ in range(mutation_count):
            mutation = mutate_once(current, sigma_small=sigma_small, rng=rng)
            if mutation is None:
                raise ValueError(
                    "Expected deterministic mutation sequence to be fully applicable: "
                    f"{scenario}"
                )
            current = mutation.mutated_expr

        cases.append(
            ObservationTimingCase(
                case_id=(
                    f"corrupted_row_{row_index}_mutations_{mutation_count}_seed_{seed}"
                ),
                row_index=row_index,
                target_integrand=target,
                current_antiderivative=current,
                mutation_count=mutation_count,
                seed=seed,
            )
        )

    return cases


def run_timing_cases(
    cases: list[ObservationTimingCase],
    *,
    residual_mode: str = "both",
    repeats: int = 1,
) -> list[ObservationTimingRecord]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    records: list[ObservationTimingRecord] = []
    for case in cases:
        for repeat_index in range(repeats):
            start_time = time.perf_counter()
            observation = build_observation(
                target_integrand=case.target_integrand,
                current_antiderivative=case.current_antiderivative,
                residual_mode=residual_mode,
            )
            elapsed = time.perf_counter() - start_time
            records.append(_record_from_observation(case, repeat_index, elapsed, observation))
    return records


def summarize_timing_records(
    records: list[ObservationTimingRecord],
    *,
    summary_name: str,
    residual_mode: str,
    repeats: int,
) -> dict[str, Any]:
    elapsed_values = [record.elapsed_seconds for record in records]
    warning_prefixes: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    derivative_token_lengths: list[int] = []
    residual_token_lengths: list[int] = []
    fraction_finite_values: list[float] = []
    fraction_complex_values: list[float] = []

    for record in records:
        status_counts[record.status] += 1
        for warning in record.warnings:
            warning_prefixes[warning.split(":", 1)[0]] += 1
        if record.derivative_token_length is not None:
            derivative_token_lengths.append(record.derivative_token_length)
        if record.residual_token_length is not None:
            residual_token_lengths.append(record.residual_token_length)
        if record.fraction_finite is not None:
            fraction_finite_values.append(record.fraction_finite)
        if record.fraction_complex is not None:
            fraction_complex_values.append(record.fraction_complex)

    by_case: dict[str, list[ObservationTimingRecord]] = defaultdict(list)
    for record in records:
        by_case[record.case_id].append(record)

    per_case = []
    for case_id, case_records in sorted(by_case.items()):
        case_timings = [record.elapsed_seconds for record in case_records]
        exemplar = case_records[0]
        per_case.append(
            {
                "case_id": case_id,
                "row_index": exemplar.row_index,
                "mutation_count": exemplar.mutation_count,
                "seed": exemplar.seed,
                "timings_seconds": [round(value, 6) for value in case_timings],
                "average_seconds": round(statistics.mean(case_timings), 6),
                "mean_seconds": round(statistics.mean(case_timings), 6),
                "median_seconds": round(statistics.median(case_timings), 6),
                "min_seconds": round(min(case_timings), 6),
                "max_seconds": round(max(case_timings), 6),
                "statuses": [record.status for record in case_records],
                "warnings": [list(record.warnings) for record in case_records],
                "fraction_finite": exemplar.fraction_finite,
                "fraction_complex": exemplar.fraction_complex,
            }
        )

    return {
        "summary_name": summary_name,
        "residual_mode": residual_mode,
        "repeats": repeats,
        "total_records": len(records),
        "total_cases": len(by_case),
        "overall_timing": _timing_stats(elapsed_values),
        "status_counts": dict(status_counts),
        "derivative_success_rate": _status_rate(records, lambda record: record.derivative_token_length is not None),
        "symbolic_residual_success_rate": _status_rate(records, lambda record: record.residual_token_length is not None),
        "numeric_probe_success_rate": _status_rate(records, lambda record: record.fraction_finite is not None),
        "fraction_with_complex_probes": _mean_or_none(fraction_complex_values),
        "fraction_with_all_probes_finite": _status_rate(
            records,
            lambda record: record.fraction_finite == 1.0,
        ),
        "derivative_token_length_stats": _length_stats(derivative_token_lengths),
        "residual_token_length_stats": _length_stats(residual_token_lengths),
        "top_warning_types": warning_prefixes.most_common(10),
        "per_case": per_case,
        "records": [asdict(record) for record in records],
    }


def write_timing_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit observation construction on a dataset slice.")
    parser.add_argument(
        "--parquet-path",
        default="data/processed/train_prefix_filtered.parquet",
        help="Parquet file containing integrand_prefix and integral_prefix columns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of examples to audit.",
    )
    parser.add_argument(
        "--num-mutations",
        type=int,
        default=0,
        help="Number of mutate_once steps to apply to each gold antiderivative before observation build.",
    )
    parser.add_argument(
        "--sigma-small",
        type=int,
        default=0,
        help="sigma_small argument passed to mutate_once when mutations are requested.",
    )
    parser.add_argument(
        "--residual-mode",
        choices=("none", "symbolic", "numeric", "both"),
        default="both",
        help="Residual mode passed to build_observation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed. Each row uses seed + row_index.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of timing repeats to run for each example.",
    )
    parser.add_argument(
        "--summary-file",
        default=str(DEFAULT_SUMMARY_FILE),
        help="JSON file where the timing summary should be written.",
    )
    args = parser.parse_args(argv)

    frame = load_dataset_frame(args.parquet_path, limit=args.limit)
    if frame.empty:
        print("No rows selected.")
        return 0

    if args.num_mutations > 0:
        scenarios = [
            {
                "row_index": row_index,
                "mutation_count": args.num_mutations,
                "seed": args.seed + row_index,
            }
            for row_index in range(len(frame))
        ]
        cases = build_corrupted_timing_cases(
            frame,
            scenarios,
            sigma_small=args.sigma_small,
        )
        summary_name = "corrupted_observation_timing"
    else:
        cases = build_gold_timing_cases(frame)
        summary_name = "gold_observation_timing"

    records = run_timing_cases(
        cases,
        residual_mode=args.residual_mode,
        repeats=args.repeats,
    )
    summary = summarize_timing_records(
        records,
        summary_name=summary_name,
        residual_mode=args.residual_mode,
        repeats=args.repeats,
    )
    output_path = write_timing_summary(summary, args.summary_file)

    print(f"summary_name: {summary_name}")
    print(f"summary_file: {output_path}")
    print(f"total_cases: {summary['total_cases']}")
    print(f"total_records: {summary['total_records']}")
    print(f"overall_timing: {summary['overall_timing']}")
    print(f"status_counts: {summary['status_counts']}")
    print(f"derivative_success_rate: {summary['derivative_success_rate']}")
    print(f"symbolic_residual_success_rate: {summary['symbolic_residual_success_rate']}")
    print(f"numeric_probe_success_rate: {summary['numeric_probe_success_rate']}")
    print(f"fraction_with_complex_probes: {summary['fraction_with_complex_probes']}")
    print(f"fraction_with_all_probes_finite: {summary['fraction_with_all_probes_finite']}")
    print(f"top_warning_types: {summary['top_warning_types']}")
    return 0


def _record_from_observation(
    case: ObservationTimingCase,
    repeat_index: int,
    elapsed_seconds: float,
    observation: Observation,
) -> ObservationTimingRecord:
    derivative_token_length = None
    residual_token_length = None
    fraction_finite = None
    fraction_complex = None

    if observation.current_derivative is not None:
        derivative_token_length = len(serialize_prefix_tokens(observation.current_derivative))
    if observation.symbolic_residual is not None:
        residual_token_length = len(serialize_prefix_tokens(observation.symbolic_residual))
    if observation.numeric_probes is not None:
        fraction_finite = observation.numeric_probes.fraction_finite
        fraction_complex = observation.numeric_probes.fraction_complex

    return ObservationTimingRecord(
        case_id=case.case_id,
        row_index=case.row_index,
        repeat_index=repeat_index,
        mutation_count=case.mutation_count,
        seed=case.seed,
        elapsed_seconds=elapsed_seconds,
        status=observation.status,
        warnings=observation.warnings,
        derivative_token_length=derivative_token_length,
        residual_token_length=residual_token_length,
        fraction_finite=fraction_finite,
        fraction_complex=fraction_complex,
        target_integrand_prefix=serialize_prefix_string(observation.target_integrand),
        current_antiderivative_prefix=serialize_prefix_string(observation.current_antiderivative),
    )


def _status_rate(
    records: list[ObservationTimingRecord],
    predicate: Any,
) -> float:
    if not records:
        return 0.0
    successes = sum(1 for record in records if predicate(record))
    return round(float(successes) / float(len(records)), 6)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.mean(values)), 6)


def _timing_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "average_seconds": None,
            "mean_seconds": None,
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    mean_value = statistics.mean(values)
    return {
        "count": len(values),
        "average_seconds": round(mean_value, 6),
        "mean_seconds": round(mean_value, 6),
        "median_seconds": round(statistics.median(values), 6),
        "min_seconds": round(min(values), 6),
        "max_seconds": round(max(values), 6),
    }


def _length_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 3),
    }


if __name__ == "__main__":
    raise SystemExit(main())
