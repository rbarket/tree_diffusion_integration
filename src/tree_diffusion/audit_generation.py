from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any, Sequence

from src.tree_diffusion._common import (
    length_summary as _length_summary,
    mean_or_none as _mean_or_none,
    parse_bool as _parse_bool,
    rate as _rate,
    selected_node_summary,
)
from src.tree_diffusion.dataset import load_integration_pairs_from_parquet
from src.tree_diffusion.edit_path import structural_distance
from src.tree_diffusion.label_validation import validate_edit_label_progress
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.training_examples import generate_training_example


def run_generation_audit(
    *,
    data: str | Path,
    num_examples: int,
    output: str | Path,
    seed: int = 123,
    sigma_small: int = 2,
    smax: int = 5,
    rho: float = 0.2,
    residual_mode: str = "both",
    simplify_symbolic_residual: bool = True,
    validate_labels: bool = True,
    allow_complex_constants: bool = False,
    allow_distributional_unary_ops: bool = False,
    excluded_random_tokens: Sequence[str] = (),
    max_derivative_tokens: int | None = None,
    max_residual_tokens: int | None = None,
    observation_timeout_seconds: float | None = None,
    max_random_size: int | None = None,
    max_attempts: int = 32,
    max_failed_examples: int = 20,
    allow_label_validation_failures: bool = False,
) -> dict[str, Any]:
    if num_examples < 1:
        raise ValueError("num_examples must be >= 1.")
    if max_failed_examples < 0:
        raise ValueError("max_failed_examples must be >= 0.")

    pairs = load_integration_pairs_from_parquet(data)
    if not pairs:
        raise ValueError("No integration pairs were loaded.")

    rng = random.Random(seed)
    tokenizer = TreeDiffusionTokenizer()

    failures: Counter[str] = Counter()
    observation_status_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    failed_examples: list[dict[str, Any]] = []
    input_lengths: list[int] = []
    target_lengths: list[int] = []
    selected_node_ids: list[int] = []
    distance_before_values: list[int] = []
    distance_after_values: list[int] = []
    used_random_init_count = 0
    num_mutations_values: list[int] = []
    root_edit_count = 0
    nonincreasing_count = 0
    strict_improvement_count = 0
    label_validation_failure_count = 0
    derivative_missing_count = 0
    residual_missing_count = 0
    numeric_missing_count = 0

    for attempted_index in range(num_examples):
        pair_offset = rng.randrange(len(pairs))
        pair = pairs[pair_offset]
        try:
            example = generate_training_example(
                pair.target_integrand,
                pair.target_antiderivative,
                tokenizer=tokenizer,
                rng=rng,
                sigma_small=sigma_small,
                smax=smax,
                rho=rho,
                residual_mode=residual_mode,
                max_random_size=max_random_size,
                max_attempts=max_attempts,
                observation_timeout_seconds=observation_timeout_seconds,
                simplify_symbolic_residual=simplify_symbolic_residual,
                allow_complex_constants=allow_complex_constants,
                allow_distributional_unary_ops=allow_distributional_unary_ops,
                excluded_random_tokens=excluded_random_tokens,
                validate_label=False,
                max_derivative_tokens=max_derivative_tokens,
                max_residual_tokens=max_residual_tokens,
            )
        except Exception as exc:
            failures[type(exc).__name__] += 1
            if len(failed_examples) < max_failed_examples:
                failed_examples.append(
                    {
                        "attempted_index": attempted_index,
                        "pair_offset": pair_offset,
                        "pair_index": pair.index,
                        "source": pair.source,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            continue

        observation_status_counts[example.observation.status] += 1
        for warning in example.warnings:
            warning_counts[warning.split(":", 1)[0]] += 1

        input_lengths.append(len(example.input_tokens))
        target_lengths.append(len(example.target_tokens))
        selected_node_ids.append(example.edit_target.selected_node_id)
        used_random_init_count += int(example.used_random_init)
        num_mutations_values.append(example.num_mutations)
        root_edit_count += int(example.edit_target.selected_node_id == 0)

        before = structural_distance(example.current_antiderivative, example.target_antiderivative)
        after = structural_distance(example.edit_target.resulting_tree, example.target_antiderivative)
        distance_before_values.append(before)
        distance_after_values.append(after)
        nonincreasing_count += int(after <= before)
        strict_improvement_count += int(after < before)

        if validate_labels:
            validation = validate_edit_label_progress(
                example.current_antiderivative,
                example.target_antiderivative,
                example.edit_target,
            )
            if not validation.ok:
                label_validation_failure_count += 1

        derivative_missing_count += int(example.observation.current_derivative is None)
        residual_missing_count += int(example.observation.symbolic_residual is None)
        numeric_missing_count += int(example.observation.numeric_probes is None)

    total_success = len(input_lengths)
    total_failed = num_examples - total_success
    summary = {
        "data": str(data),
        "seed": seed,
        "num_examples": num_examples,
        "total_attempted": num_examples,
        "total_success": total_success,
        "total_failed": total_failed,
        "failure_rate": _rate(total_failed, num_examples),
        "failure_by_exception_type": dict(failures),
        "observation_status_counts": dict(observation_status_counts),
        "warning_counts": dict(warning_counts),
        "used_random_init_fraction": _rate(used_random_init_count, total_success),
        "num_mutations_mean": _mean_or_none(num_mutations_values),
        "input_length": _length_summary(input_lengths),
        "target_length": _length_summary(target_lengths),
        "selected_node_id": _selected_node_summary(selected_node_ids),
        "root_edit_fraction": _rate(root_edit_count, total_success),
        "distance_before": _length_summary(distance_before_values),
        "distance_after": _length_summary(distance_after_values),
        "validate_labels": validate_labels,
        "label_validation_failure_count": label_validation_failure_count,
        "nonincreasing_distance_rate": _rate(nonincreasing_count, total_success),
        "strict_improvement_rate": _rate(strict_improvement_count, total_success),
        "derivative_missing_rate": _rate(derivative_missing_count, total_success),
        "residual_missing_rate": _rate(residual_missing_count, total_success),
        "numeric_missing_rate": _rate(numeric_missing_count, total_success),
        "failed_examples": failed_examples,
        "generation_config": {
            "sigma_small": sigma_small,
            "smax": smax,
            "rho": rho,
            "residual_mode": residual_mode,
            "simplify_symbolic_residual": simplify_symbolic_residual,
            "allow_complex_constants": allow_complex_constants,
            "allow_distributional_unary_ops": allow_distributional_unary_ops,
            "excluded_random_tokens": list(excluded_random_tokens),
            "max_derivative_tokens": max_derivative_tokens,
            "max_residual_tokens": max_residual_tokens,
            "observation_timeout_seconds": observation_timeout_seconds,
            "max_random_size": max_random_size,
            "max_attempts": max_attempts,
        },
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if total_success == 0:
        raise RuntimeError(f"Generation audit produced no successful examples; summary written to {output_path}.")
    if validate_labels and label_validation_failure_count > 0 and not allow_label_validation_failures:
        raise RuntimeError(
            "Generation audit found label validation failures: "
            f"{label_validation_failure_count}; summary written to {output_path}."
        )

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit online tree-diffusion example generation.")
    parser.add_argument("--data", required=True, help="Parquet file containing integration pairs.")
    parser.add_argument("--num-examples", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sigma-small", type=int, default=2)
    parser.add_argument("--smax", type=int, default=5)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--residual-mode", choices=("none", "symbolic", "numeric", "both"), default="both")
    parser.add_argument("--simplify-symbolic-residual", type=_parse_bool, default=True)
    parser.add_argument("--validate-labels", type=_parse_bool, default=True)
    parser.add_argument("--allow-complex-constants", type=_parse_bool, default=False)
    parser.add_argument("--allow-distributional-unary-ops", type=_parse_bool, default=False)
    parser.add_argument("--excluded-random-tokens", nargs="*", default=())
    parser.add_argument("--max-derivative-tokens", type=int, default=None)
    parser.add_argument("--max-residual-tokens", type=int, default=None)
    parser.add_argument("--observation-timeout-seconds", type=float, default=None)
    parser.add_argument("--max-random-size", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=32)
    parser.add_argument("--allow-label-validation-failures", action="store_true")
    args = parser.parse_args(argv)

    summary = run_generation_audit(
        data=args.data,
        num_examples=args.num_examples,
        output=args.output,
        seed=args.seed,
        sigma_small=args.sigma_small,
        smax=args.smax,
        rho=args.rho,
        residual_mode=args.residual_mode,
        simplify_symbolic_residual=args.simplify_symbolic_residual,
        validate_labels=args.validate_labels,
        allow_complex_constants=args.allow_complex_constants,
        allow_distributional_unary_ops=args.allow_distributional_unary_ops,
        excluded_random_tokens=tuple(args.excluded_random_tokens),
        max_derivative_tokens=args.max_derivative_tokens,
        max_residual_tokens=args.max_residual_tokens,
        observation_timeout_seconds=args.observation_timeout_seconds,
        max_random_size=args.max_random_size,
        max_attempts=args.max_attempts,
        allow_label_validation_failures=args.allow_label_validation_failures,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _selected_node_summary(values: Sequence[int]) -> dict[str, Any]:
    return selected_node_summary(
        values,
        include_min=False,
        include_top_counts=True,
        percentile_mode="nearest",
    )


__all__ = ["main", "run_generation_audit"]


if __name__ == "__main__":
    raise SystemExit(main())
