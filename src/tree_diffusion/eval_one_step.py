from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import torch

from src.mathlang.ast import Expr
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion._common import (
    json_safe,
    mean_or_none as _mean_or_none,
    move_tensor_batch as _move_tensor_batch,
    resolve_device as _resolve_device,
    write_json as _write_json,
)
from src.tree_diffusion.dataset import (
    load_integration_pairs_from_parquet,
    make_tree_diffusion_dataloader,
)
from src.tree_diffusion.decoding import (
    apply_decoded_edit,
    predict_greedy_edit,
)
from src.tree_diffusion.edit_path import structural_distance
from src.tree_diffusion.model import TreeDiffusionModelConfig, TreeDiffusionPolicyModel
from src.tree_diffusion.observation import (
    ObservationTimeoutError,
    _observation_timeout,
    compute_current_derivative,
    compute_numeric_probes,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


@dataclass(frozen=True)
class OneStepEditEvaluationSummary:
    examples: int
    decoded_ok_rate: float
    valid_position_rate: float
    parseable_replacement_rate: float
    applicable_edit_rate: float
    structural_improvement_rate: float
    nonincreasing_structural_rate: float
    exact_target_rate: float
    numeric_residual_improvement_rate: float | None
    mean_structural_distance_before: float | None
    mean_structural_distance_after: float | None
    mean_numeric_residual_before: float | None
    mean_numeric_residual_after: float | None
    status_counts: dict[str, int]
    diagnostic_example_timeout_count: int = 0
    diagnostic_total_timeout_count: int = 0
    numeric_residual_timeout_count: int = 0


@torch.no_grad()
def evaluate_one_step_edits(
    model: TreeDiffusionPolicyModel,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    tokenizer: TreeDiffusionTokenizer,
    device: torch.device | str,
    num_batches: int,
    constrain_position: bool = True,
    max_decode_length: int | None = None,
    compute_numeric_residual: bool = True,
    diagnostic_timeout_seconds: float | None = None,
    diagnostic_example_timeout_seconds: float | None = None,
    numeric_residual_timeout_seconds: float | None = None,
) -> OneStepEditEvaluationSummary:
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1.")
    _validate_optional_timeout("diagnostic_timeout_seconds", diagnostic_timeout_seconds)
    _validate_optional_timeout("diagnostic_example_timeout_seconds", diagnostic_example_timeout_seconds)
    _validate_optional_timeout("numeric_residual_timeout_seconds", numeric_residual_timeout_seconds)

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    examples = 0
    decoded_ok = 0
    valid_positions = 0
    parseable_replacements = 0
    applicable_edits = 0
    structural_improvements = 0
    nonincreasing_structural = 0
    exact_targets = 0
    numeric_improvements = 0
    numeric_examples = 0
    before_distances: list[float] = []
    after_distances: list[float] = []
    before_numeric_scores: list[float] = []
    after_numeric_scores: list[float] = []
    status_counts: Counter[str] = Counter()
    diagnostic_example_timeout_count = 0
    diagnostic_total_timeout_count = 0
    numeric_residual_timeout_count = 0
    deadline = (
        None
        if diagnostic_timeout_seconds is None
        else time.monotonic() + float(diagnostic_timeout_seconds)
    )

    iterator = iter(dataloader)
    for _ in range(num_batches):
        if deadline is not None and time.monotonic() >= deadline:
            diagnostic_total_timeout_count += 1
            status_counts["diagnostic_total_timeout"] += 1
            break
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

        working_batch = _move_tensor_batch(batch, device=target_device)
        input_ids = _required_tensor(working_batch, "input_ids")
        input_attention_mask = working_batch.get("input_attention_mask")
        if input_attention_mask is not None and not isinstance(input_attention_mask, torch.Tensor):
            raise TypeError("input_attention_mask must be a tensor when provided.")

        for row_index in range(_batch_size(input_ids)):
            if deadline is not None and time.monotonic() >= deadline:
                diagnostic_total_timeout_count += 1
                status_counts["diagnostic_total_timeout"] += 1
                return _one_step_edit_summary(
                    examples=examples,
                    decoded_ok=decoded_ok,
                    valid_positions=valid_positions,
                    parseable_replacements=parseable_replacements,
                    applicable_edits=applicable_edits,
                    structural_improvements=structural_improvements,
                    nonincreasing_structural=nonincreasing_structural,
                    exact_targets=exact_targets,
                    numeric_improvements=numeric_improvements,
                    numeric_examples=numeric_examples,
                    before_distances=before_distances,
                    after_distances=after_distances,
                    before_numeric_scores=before_numeric_scores,
                    after_numeric_scores=after_numeric_scores,
                    status_counts=status_counts,
                    diagnostic_example_timeout_count=diagnostic_example_timeout_count,
                    diagnostic_total_timeout_count=diagnostic_total_timeout_count,
                    numeric_residual_timeout_count=numeric_residual_timeout_count,
                )
            examples += 1
            try:
                with _observation_timeout(
                    _effective_example_timeout(
                        diagnostic_example_timeout_seconds,
                        deadline=deadline,
                    )
                ):
                    current_prefix = _required_metadata(batch, "current_prefix", row_index)
                    target_antiderivative_prefix = _required_metadata(
                        batch,
                        "target_antiderivative_prefix",
                        row_index,
                    )
                    target_integrand_prefix = _required_metadata(
                        batch,
                        "target_integrand_prefix",
                        row_index,
                    )

                    try:
                        current_tree = canonicalize(parse_prefix_string(current_prefix))
                        target_tree = canonicalize(parse_prefix_string(target_antiderivative_prefix))
                        target_integrand = canonicalize(
                            parse_prefix_string(target_integrand_prefix),
                            strip_additive_constants=False,
                        )
                    except Exception:
                        status_counts["metadata_parse_failed"] += 1
                        continue

                    decoded = predict_greedy_edit(
                        model,
                        _tensor_row(input_ids, row_index),
                        tokenizer=tokenizer,
                        current_tree=current_tree,
                        input_attention_mask=(
                            None
                            if input_attention_mask is None
                            else _tensor_row(input_attention_mask, row_index)
                        ),
                        max_length=max_decode_length,
                        constrain_position=constrain_position,
                        device=target_device,
                    )

                    if _has_valid_position(decoded_status=decoded.status):
                        valid_positions += 1
                    if decoded.status == "ok":
                        decoded_ok += 1
                        parseable_replacements += 1
                    else:
                        status_counts[decoded.status] += 1
                        continue

                    try:
                        edited_tree = apply_decoded_edit(current_tree, decoded)
                    except Exception:
                        status_counts["apply_failed"] += 1
                        continue

                    applicable_edits += 1
                    status_counts["ok"] += 1

                    before = float(structural_distance(current_tree, target_tree))
                    after = float(structural_distance(edited_tree, target_tree))
                    before_distances.append(before)
                    after_distances.append(after)
                    if after < before:
                        structural_improvements += 1
                    if after <= before:
                        nonincreasing_structural += 1
                    if canonicalize(edited_tree) == target_tree:
                        exact_targets += 1

                    if compute_numeric_residual:
                        before_score, before_timed_out = _numeric_residual_score_with_status(
                            current_tree,
                            target_integrand,
                            timeout_seconds=numeric_residual_timeout_seconds,
                        )
                        after_score, after_timed_out = _numeric_residual_score_with_status(
                            edited_tree,
                            target_integrand,
                            timeout_seconds=numeric_residual_timeout_seconds,
                        )
                        numeric_residual_timeout_count += int(before_timed_out) + int(after_timed_out)
                        if before_timed_out:
                            status_counts["numeric_residual_before_timeout"] += 1
                        if after_timed_out:
                            status_counts["numeric_residual_after_timeout"] += 1
                        if before_score is not None and after_score is not None:
                            numeric_examples += 1
                            before_numeric_scores.append(before_score)
                            after_numeric_scores.append(after_score)
                            if after_score < before_score:
                                numeric_improvements += 1
            except ObservationTimeoutError:
                diagnostic_example_timeout_count += 1
                status_counts["diagnostic_example_timeout"] += 1
                continue

    return _one_step_edit_summary(
        examples=examples,
        decoded_ok=decoded_ok,
        valid_positions=valid_positions,
        parseable_replacements=parseable_replacements,
        applicable_edits=applicable_edits,
        structural_improvements=structural_improvements,
        nonincreasing_structural=nonincreasing_structural,
        exact_targets=exact_targets,
        numeric_improvements=numeric_improvements,
        numeric_examples=numeric_examples,
        before_distances=before_distances,
        after_distances=after_distances,
        before_numeric_scores=before_numeric_scores,
        after_numeric_scores=after_numeric_scores,
        status_counts=status_counts,
        diagnostic_example_timeout_count=diagnostic_example_timeout_count,
        diagnostic_total_timeout_count=diagnostic_total_timeout_count,
        numeric_residual_timeout_count=numeric_residual_timeout_count,
    )


def numeric_residual_score(
    antiderivative: Expr,
    target_integrand: Expr,
    *,
    probe_points: Sequence[float] | None = None,
    timeout_seconds: float | None = None,
) -> float | None:
    score, _ = _numeric_residual_score_with_status(
        antiderivative,
        target_integrand,
        probe_points=probe_points,
        timeout_seconds=timeout_seconds,
    )
    return score


def _numeric_residual_score_with_status(
    antiderivative: Expr,
    target_integrand: Expr,
    *,
    probe_points: Sequence[float] | None = None,
    timeout_seconds: float | None = None,
) -> tuple[float | None, bool]:
    try:
        with _observation_timeout(timeout_seconds):
            current_derivative = compute_current_derivative(antiderivative)
            probes = compute_numeric_probes(
                current_derivative,
                target_integrand,
                probe_points=probe_points,
            )
    except ObservationTimeoutError:
        return None, True
    except Exception:
        return None, False

    finite_squared_abs = [
        float(value)
        for is_finite, value in zip(probes.finite_mask, probes.residual_abs_squared)
        if is_finite and value is not None and math.isfinite(float(value))
    ]
    if not finite_squared_abs:
        return None, False
    return sum(finite_squared_abs) / len(finite_squared_abs), False


def _one_step_edit_summary(
    *,
    examples: int,
    decoded_ok: int,
    valid_positions: int,
    parseable_replacements: int,
    applicable_edits: int,
    structural_improvements: int,
    nonincreasing_structural: int,
    exact_targets: int,
    numeric_improvements: int,
    numeric_examples: int,
    before_distances: Sequence[float],
    after_distances: Sequence[float],
    before_numeric_scores: Sequence[float],
    after_numeric_scores: Sequence[float],
    status_counts: Counter[str],
    diagnostic_example_timeout_count: int,
    diagnostic_total_timeout_count: int,
    numeric_residual_timeout_count: int,
) -> OneStepEditEvaluationSummary:
    return OneStepEditEvaluationSummary(
        examples=examples,
        decoded_ok_rate=_rate(decoded_ok, examples),
        valid_position_rate=_rate(valid_positions, examples),
        parseable_replacement_rate=_rate(parseable_replacements, examples),
        applicable_edit_rate=_rate(applicable_edits, examples),
        structural_improvement_rate=_rate(structural_improvements, examples),
        nonincreasing_structural_rate=_rate(nonincreasing_structural, examples),
        exact_target_rate=_rate(exact_targets, examples),
        numeric_residual_improvement_rate=(
            None if numeric_examples == 0 else numeric_improvements / numeric_examples
        ),
        mean_structural_distance_before=_mean_or_none(before_distances),
        mean_structural_distance_after=_mean_or_none(after_distances),
        mean_numeric_residual_before=_mean_or_none(before_numeric_scores),
        mean_numeric_residual_after=_mean_or_none(after_numeric_scores),
        status_counts=dict(status_counts),
        diagnostic_example_timeout_count=diagnostic_example_timeout_count,
        diagnostic_total_timeout_count=diagnostic_total_timeout_count,
        numeric_residual_timeout_count=numeric_residual_timeout_count,
    )


def _effective_example_timeout(
    timeout_seconds: float | None,
    *,
    deadline: float | None,
) -> float | None:
    if deadline is None:
        return timeout_seconds
    remaining = max(0.001, deadline - time.monotonic())
    if timeout_seconds is None:
        return remaining
    return min(float(timeout_seconds), remaining)


def _validate_optional_timeout(name: str, value: float | None) -> None:
    if value is not None and value <= 0.0:
        raise ValueError(f"{name} must be > 0 when provided.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate one-step tree-diffusion edit decoding.")
    parser.add_argument("--checkpoint", default=None, help="Tree-diffusion checkpoint path.")
    parser.add_argument("--data", default=None, help="Parquet file containing integration pairs.")
    parser.add_argument("--precomputed-data-dir", default=None, help="Precomputed tree-diffusion data dir.")
    parser.add_argument("--output", default=None, help="Optional summary JSON output path.")
    parser.add_argument("--num-pairs", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--constrain-position", dest="constrain_position", action="store_true", default=True)
    parser.add_argument("--no-constrain-position", dest="constrain_position", action="store_false")
    parser.add_argument("--max-decode-length", type=int, default=None)
    parser.add_argument(
        "--compute-numeric-residual",
        dest="compute_numeric_residual",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-compute-numeric-residual", dest="compute_numeric_residual", action="store_false")
    parser.add_argument(
        "--allow-random-init-model",
        action="store_true",
        help="Allow evaluation without --checkpoint using a randomly initialized model.",
    )
    args = parser.parse_args(argv)

    if args.checkpoint is None and not args.allow_random_init_model:
        raise ValueError("Provide --checkpoint or pass --allow-random-init-model explicitly.")
    if (args.data is None) == (args.precomputed_data_dir is None):
        raise ValueError("Provide exactly one data source: --data or --precomputed-data-dir.")
    if args.num_pairs < 1:
        raise ValueError("--num-pairs must be >= 1.")
    if args.num_batches < 1:
        raise ValueError("--num-batches must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    torch.manual_seed(int(args.seed))
    device = _resolve_device(str(args.device))
    tokenizer, model = _load_cli_model_and_tokenizer(
        checkpoint=args.checkpoint,
        precomputed_data_dir=args.precomputed_data_dir,
        allow_random_init_model=bool(args.allow_random_init_model),
    )
    model.to(device)

    dataloader = _build_cli_dataloader(
        data=args.data,
        precomputed_data_dir=args.precomputed_data_dir,
        tokenizer=tokenizer,
        model=model,
        num_pairs=int(args.num_pairs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    summary = evaluate_one_step_edits(
        model,
        dataloader,
        tokenizer=tokenizer,
        device=device,
        num_batches=int(args.num_batches),
        constrain_position=bool(args.constrain_position),
        max_decode_length=args.max_decode_length,
        compute_numeric_residual=bool(args.compute_numeric_residual),
    )
    summary_json = json_safe(asdict(summary))
    print(json.dumps(summary_json, indent=2, sort_keys=True))
    if args.output is not None:
        _write_json(Path(args.output), summary_json)
    return 0


def _load_cli_model_and_tokenizer(
    *,
    checkpoint: str | None,
    precomputed_data_dir: str | None,
    allow_random_init_model: bool,
) -> tuple[TreeDiffusionTokenizer, TreeDiffusionPolicyModel]:
    if checkpoint is None:
        assert allow_random_init_model
        tokenizer = _tokenizer_from_precomputed(precomputed_data_dir) or TreeDiffusionTokenizer()
        from src.training.workflows.tree_diffusion import (
            TreeDiffusionTrainingConfig,
            build_policy_model_for_config,
        )

        model = build_policy_model_for_config(TreeDiffusionTrainingConfig(), tokenizer)
        return tokenizer, model

    checkpoint_path = Path(checkpoint)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(payload).__name__}.")

    tokenizer = _tokenizer_from_checkpoint(payload) or _tokenizer_from_precomputed(precomputed_data_dir)
    if tokenizer is None:
        tokenizer = TreeDiffusionTokenizer()

    model_config = _model_config_from_checkpoint(payload, tokenizer=tokenizer)
    model = TreeDiffusionPolicyModel(model_config)
    _load_model_state(model, payload, checkpoint_path=checkpoint_path)
    return tokenizer, model


def _build_cli_dataloader(
    *,
    data: str | None,
    precomputed_data_dir: str | None,
    tokenizer: TreeDiffusionTokenizer,
    model: TreeDiffusionPolicyModel,
    num_pairs: int,
    batch_size: int,
    seed: int,
):
    if precomputed_data_dir is not None:
        return make_tree_diffusion_dataloader(
            tokenizer=tokenizer,
            precomputed_data_dir=precomputed_data_dir,
            precomputed_limit=num_pairs,
            batch_size=batch_size,
            shuffle_pairs=False,
            include_metadata=True,
        )

    assert data is not None
    pairs = load_integration_pairs_from_parquet(data, limit=num_pairs)
    return make_tree_diffusion_dataloader(
        pairs,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=0,
        max_input_length=model.config.max_input_length,
        max_target_length=model.config.max_target_length,
        base_seed=seed,
        shuffle_pairs=False,
        include_metadata=True,
    )


def _tokenizer_from_checkpoint(payload: Mapping[str, Any]) -> TreeDiffusionTokenizer | None:
    metadata = payload.get("tokenizer")
    if not isinstance(metadata, Mapping):
        return None
    return TreeDiffusionTokenizer(
        max_positions=int(metadata.get("max_positions", 512)),
        numeric_log_min=int(metadata.get("numeric_log_min", -12)),
        numeric_log_max=int(metadata.get("numeric_log_max", 12)),
    )


def _tokenizer_from_precomputed(data_dir: str | None) -> TreeDiffusionTokenizer | None:
    if data_dir is None:
        return None
    from src.tree_diffusion.precomputed_dataset import load_precomputed_tokenizer_metadata

    metadata = load_precomputed_tokenizer_metadata(data_dir)
    return TreeDiffusionTokenizer(
        max_positions=int(metadata.get("max_positions", 512)),
        numeric_log_min=int(metadata.get("numeric_log_min", -12)),
        numeric_log_max=int(metadata.get("numeric_log_max", 12)),
    )


def _model_config_from_checkpoint(
    payload: Mapping[str, Any],
    *,
    tokenizer: TreeDiffusionTokenizer,
) -> TreeDiffusionModelConfig:
    raw_model_cfg = payload.get("model_cfg")
    if isinstance(raw_model_cfg, Mapping):
        allowed = {field.name for field in fields(TreeDiffusionModelConfig)}
        values = {str(key): value for key, value in raw_model_cfg.items() if str(key) in allowed}
        values["vocab_size"] = tokenizer.vocab_size
        values["pad_token_id"] = tokenizer.pad_id
        values["bos_token_id"] = tokenizer.bos_id
        values["eos_token_id"] = tokenizer.eos_id
        return TreeDiffusionModelConfig(**values)

    raw_training_cfg = payload.get("config")
    if isinstance(raw_training_cfg, Mapping):
        from src.training.workflows.tree_diffusion import (
            TreeDiffusionTrainingConfig,
            build_policy_model_for_config,
        )

        model = build_policy_model_for_config(
            TreeDiffusionTrainingConfig(**dict(raw_training_cfg)),
            tokenizer,
        )
        return model.config

    raise ValueError("Checkpoint does not contain model_cfg or training config metadata.")


def _load_model_state(
    model: TreeDiffusionPolicyModel,
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> None:
    if "state_dict" in payload and "model_state_dict" not in payload:
        state_dict = payload["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Lightning checkpoint state_dict must be a mapping: {checkpoint_path}")
        model_state = {
            str(key).removeprefix("model."): value
            for key, value in state_dict.items()
            if str(key).startswith("model.")
        }
        if not model_state:
            raise KeyError(f"Lightning checkpoint missing model.* state_dict keys: {checkpoint_path}")
        model.load_state_dict(model_state)
        return

    if "model_state_dict" not in payload:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")
    model.load_state_dict(payload["model_state_dict"])


def _required_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Batch is missing required tensor field {key!r}.")
    return value


def _required_metadata(batch: Mapping[str, Any], key: str, row_index: int) -> str:
    if key not in batch:
        raise ValueError(f"Batch is missing required metadata field {key!r}.")
    value = batch[key]
    if isinstance(value, (list, tuple)):
        try:
            item = value[row_index]
        except IndexError as exc:
            raise ValueError(f"Metadata field {key!r} is shorter than the tensor batch.") from exc
    else:
        item = value
    if item is None:
        raise ValueError(f"Metadata field {key!r} contains None.")
    return str(item)


def _batch_size(input_ids: torch.Tensor) -> int:
    if input_ids.ndim == 1:
        return 1
    if input_ids.ndim == 2:
        return int(input_ids.size(0))
    raise ValueError("input_ids must have shape (L,) or (B, L).")


def _tensor_row(value: torch.Tensor, row_index: int) -> torch.Tensor:
    if value.ndim == 1:
        if row_index != 0:
            raise IndexError("Cannot index more than one row from a 1-D tensor.")
        return value
    return value[row_index]


def _has_valid_position(*, decoded_status: str) -> bool:
    return decoded_status in {"ok", "missing_replacement", "replacement_parse_failed"}


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


__all__ = [
    "OneStepEditEvaluationSummary",
    "evaluate_one_step_edits",
    "main",
    "numeric_residual_score",
]


if __name__ == "__main__":
    raise SystemExit(main())
