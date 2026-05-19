from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
    move_tensor_batch as _move_tensor_batch,
    resolve_device as _resolve_device,
    write_json as _write_json,
)
from src.tree_diffusion.dataset import (
    IntegrationPair,
    load_integration_pairs_from_parquet,
    make_tree_diffusion_dataloader,
)
from src.tree_diffusion.decoding import (
    DecodedEdit,
    apply_decoded_edit,
    decode_edit_candidates,
)
from src.tree_diffusion.edit_path import structural_distance
from src.tree_diffusion.eval_one_step import (
    evaluate_one_step_edits,
    numeric_residual_score,
    _load_cli_model_and_tokenizer,
)
from src.tree_diffusion.model import TreeDiffusionPolicyModel
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


DEFAULT_TOP_K_VALUES = (4, 8, 16)
BASELINE_MODE_NAME = "position_constrained_greedy"


@dataclass(frozen=True)
class OneStepInferenceEvalMode:
    name: str
    constrain_position: bool
    candidate_k: int
    use_first_applicable_candidate: bool


def run_one_step_inference_eval(
    *,
    checkpoint: str,
    output_dir: str | Path,
    precomputed_data_dir: str | None = None,
    data: str | None = None,
    num_pairs: int = 128,
    num_batches: int = 5,
    batch_size: int = 32,
    device: torch.device | str = "auto",
    seed: int = 123,
    max_decode_length: int | None = None,
    compute_numeric_residual: bool = True,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    num_dump_examples: int = 50,
    dump_failures_only: bool = False,
    dump_improvements_only: bool = False,
) -> dict[str, Any]:
    _validate_args(
        data=data,
        precomputed_data_dir=precomputed_data_dir,
        num_pairs=num_pairs,
        num_batches=num_batches,
        batch_size=batch_size,
        top_k_values=top_k_values,
        num_dump_examples=num_dump_examples,
        dump_failures_only=dump_failures_only,
        dump_improvements_only=dump_improvements_only,
    )

    torch.manual_seed(int(seed))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_device = _resolve_device(str(device))

    tokenizer, model = _load_cli_model_and_tokenizer(
        checkpoint=checkpoint,
        precomputed_data_dir=precomputed_data_dir,
        allow_random_init_model=False,
    )
    model.to(target_device)
    model.eval()

    online_pairs = _load_online_pairs(data=data, num_pairs=num_pairs)
    modes = _eval_modes(top_k_values)
    mode_results: dict[str, dict[str, Any]] = {}

    for mode in modes:
        dataloader = _build_eval_dataloader(
            data=data,
            precomputed_data_dir=precomputed_data_dir,
            online_pairs=online_pairs,
            tokenizer=tokenizer,
            model=model,
            num_pairs=num_pairs,
            batch_size=batch_size,
            seed=seed,
        )
        summary = evaluate_one_step_edits(
            model,
            dataloader,
            tokenizer=tokenizer,
            device=target_device,
            num_batches=num_batches,
            constrain_position=mode.constrain_position,
            max_decode_length=max_decode_length,
            compute_numeric_residual=compute_numeric_residual,
            candidate_k=mode.candidate_k,
            use_first_applicable_candidate=mode.use_first_applicable_candidate,
        )
        mode_results[mode.name] = {
            "mode": asdict(mode),
            "metrics": _json_safe(asdict(summary)),
            "derived_comparison": {},
        }

    baseline_metrics = mode_results[BASELINE_MODE_NAME]["metrics"]
    for mode in modes:
        result = mode_results[mode.name]
        if mode.candidate_k > 1:
            result["derived_comparison"] = _derived_comparison(
                result["metrics"],
                baseline_metrics=baseline_metrics,
            )
        _write_json(output_path / f"{mode.name}.json", result)

    qualitative_examples = None
    if num_dump_examples > 0:
        qualitative_examples = _dump_examples(
            model=model,
            tokenizer=tokenizer,
            data=data,
            precomputed_data_dir=precomputed_data_dir,
            online_pairs=online_pairs,
            output_dir=output_path,
            num_pairs=num_pairs,
            num_batches=num_batches,
            batch_size=batch_size,
            seed=seed,
            device=target_device,
            max_decode_length=max_decode_length,
            compute_numeric_residual=compute_numeric_residual,
            candidate_k=max(int(value) for value in top_k_values),
            num_dump_examples=num_dump_examples,
            dump_failures_only=dump_failures_only,
            dump_improvements_only=dump_improvements_only,
        )

    summary = {
        "checkpoint": str(checkpoint),
        "data_source": _data_source_summary(data=data, precomputed_data_dir=precomputed_data_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "num_pairs": int(num_pairs),
        "num_batches": int(num_batches),
        "batch_size": int(batch_size),
        "device": {"requested": str(device), "resolved": str(target_device)},
        "max_decode_length": max_decode_length,
        "compute_numeric_residual": bool(compute_numeric_residual),
        "modes": [asdict(mode) for mode in modes],
        "metrics_by_mode": mode_results,
        "qualitative_examples": qualitative_examples,
    }
    safe_summary = _json_safe(summary)
    _write_json(output_path / "one_step_eval_summary.json", safe_summary)
    return safe_summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one-step inference-mode evaluation for a tree-diffusion policy."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--precomputed-data-dir", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-pairs", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-decode-length", type=int, default=None)
    parser.add_argument("--top-k-values", type=int, nargs="+", default=list(DEFAULT_TOP_K_VALUES))
    parser.add_argument("--num-dump-examples", type=int, default=50)
    parser.add_argument("--dump-failures-only", action="store_true")
    parser.add_argument("--dump-improvements-only", action="store_true")
    parser.add_argument(
        "--compute-numeric-residual",
        dest="compute_numeric_residual",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-compute-numeric-residual", dest="compute_numeric_residual", action="store_false")
    args = parser.parse_args(argv)

    summary = run_one_step_inference_eval(
        checkpoint=str(args.checkpoint),
        precomputed_data_dir=args.precomputed_data_dir,
        data=args.data,
        output_dir=args.output_dir,
        num_pairs=int(args.num_pairs),
        num_batches=int(args.num_batches),
        batch_size=int(args.batch_size),
        device=str(args.device),
        seed=int(args.seed),
        max_decode_length=args.max_decode_length,
        compute_numeric_residual=bool(args.compute_numeric_residual),
        top_k_values=tuple(int(value) for value in args.top_k_values),
        num_dump_examples=int(args.num_dump_examples),
        dump_failures_only=bool(args.dump_failures_only),
        dump_improvements_only=bool(args.dump_improvements_only),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _eval_modes(top_k_values: Sequence[int]) -> list[OneStepInferenceEvalMode]:
    modes = [
        OneStepInferenceEvalMode(
            name="unconstrained_greedy",
            constrain_position=False,
            candidate_k=1,
            use_first_applicable_candidate=False,
        ),
        OneStepInferenceEvalMode(
            name=BASELINE_MODE_NAME,
            constrain_position=True,
            candidate_k=1,
            use_first_applicable_candidate=False,
        ),
    ]
    modes.extend(
        OneStepInferenceEvalMode(
            name=f"position_constrained_topk_{int(k)}",
            constrain_position=True,
            candidate_k=int(k),
            use_first_applicable_candidate=True,
        )
        for k in top_k_values
    )
    return modes


def _build_eval_dataloader(
    *,
    data: str | None,
    precomputed_data_dir: str | None,
    online_pairs: Sequence[IntegrationPair] | None,
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
            precomputed_split="val",
            precomputed_limit=num_pairs,
            batch_size=batch_size,
            num_workers=0,
            shuffle_pairs=False,
            include_metadata=True,
        )

    assert data is not None
    assert online_pairs is not None
    return make_tree_diffusion_dataloader(
        online_pairs,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=0,
        max_input_length=model.config.max_input_length,
        max_target_length=model.config.max_target_length,
        base_seed=int(seed),
        shuffle_pairs=False,
        include_metadata=True,
    )


def _load_online_pairs(
    *,
    data: str | None,
    num_pairs: int,
) -> list[IntegrationPair] | None:
    if data is None:
        return None
    return load_integration_pairs_from_parquet(data, limit=num_pairs)


def _dump_examples(
    *,
    model: TreeDiffusionPolicyModel,
    tokenizer: TreeDiffusionTokenizer,
    data: str | None,
    precomputed_data_dir: str | None,
    online_pairs: Sequence[IntegrationPair] | None,
    output_dir: Path,
    num_pairs: int,
    num_batches: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    max_decode_length: int | None,
    compute_numeric_residual: bool,
    candidate_k: int,
    num_dump_examples: int,
    dump_failures_only: bool,
    dump_improvements_only: bool,
) -> dict[str, Any]:
    path = output_dir / "examples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataloader = _build_eval_dataloader(
        data=data,
        precomputed_data_dir=precomputed_data_dir,
        online_pairs=online_pairs,
        tokenizer=tokenizer,
        model=model,
        num_pairs=num_pairs,
        batch_size=batch_size,
        seed=seed,
    )

    model.to(device)
    model.eval()
    failure_type_counts: dict[str, int] = {}
    written = 0
    scanned = 0
    iterator = iter(dataloader)
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(num_batches):
            if written >= num_dump_examples:
                break
            try:
                batch = next(iterator)
            except StopIteration:
                break
            working_batch = _move_tensor_batch(batch, device=device)
            input_ids = _required_tensor(working_batch, "input_ids")
            input_attention_mask = working_batch.get("input_attention_mask")
            if input_attention_mask is not None and not isinstance(input_attention_mask, torch.Tensor):
                raise TypeError("input_attention_mask must be a tensor when provided.")

            for row_index in range(_batch_size(input_ids)):
                if written >= num_dump_examples:
                    break
                record = _example_record(
                    model=model,
                    tokenizer=tokenizer,
                    batch=batch,
                    input_ids=input_ids,
                    input_attention_mask=input_attention_mask,
                    row_index=row_index,
                    example_index=scanned,
                    device=device,
                    candidate_k=candidate_k,
                    max_decode_length=max_decode_length,
                    compute_numeric_residual=compute_numeric_residual,
                )
                scanned += 1
                if dump_failures_only and bool(record["any_structural_improved"]):
                    continue
                if dump_improvements_only and not bool(record["chosen_structural_improved"]):
                    continue
                failure_type = str(record["failure_type"])
                failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
                handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
                written += 1

    return {
        "path": str(path),
        "records_written": written,
        "records_scanned": scanned,
        "candidate_k": int(candidate_k),
        "failure_type_counts": failure_type_counts,
    }


def _example_record(
    *,
    model: TreeDiffusionPolicyModel,
    tokenizer: TreeDiffusionTokenizer,
    batch: Mapping[str, Any],
    input_ids: torch.Tensor,
    input_attention_mask: torch.Tensor | None,
    row_index: int,
    example_index: int,
    device: torch.device,
    candidate_k: int,
    max_decode_length: int | None,
    compute_numeric_residual: bool,
) -> dict[str, Any]:
    current_prefix = _required_metadata(batch, "current_prefix", row_index)
    target_antiderivative_prefix = _required_metadata(batch, "target_antiderivative_prefix", row_index)
    target_integrand_prefix = _required_metadata(batch, "target_integrand_prefix", row_index)
    input_tokens = _metadata_item(batch, "input_tokens", row_index, default=None)
    target_tokens = _metadata_item(batch, "target_tokens", row_index, default=None)

    current_tree = canonicalize(parse_prefix_string(current_prefix))
    target_tree = canonicalize(parse_prefix_string(target_antiderivative_prefix))
    target_integrand = canonicalize(
        parse_prefix_string(target_integrand_prefix),
        strip_additive_constants=False,
    )
    distance_before = float(structural_distance(current_tree, target_tree))
    numeric_before = (
        numeric_residual_score(current_tree, target_integrand)
        if compute_numeric_residual
        else None
    )

    row_attention_mask = None if input_attention_mask is None else _tensor_row(input_attention_mask, row_index)
    candidates = decode_edit_candidates(
        model,
        _tensor_row(input_ids, row_index),
        tokenizer=tokenizer,
        current_tree=current_tree,
        input_attention_mask=row_attention_mask,
        k=int(candidate_k),
        max_length=max_decode_length,
        constrain_position=True,
        device=device,
    )

    candidate_records: list[dict[str, Any]] = []
    chosen_record: dict[str, Any] | None = None
    any_structural_improved = False
    for rank, candidate in enumerate(candidates, start=1):
        candidate_record = _candidate_record(
            candidate,
            rank=rank,
            current_tree=current_tree,
            target_tree=target_tree,
            target_integrand=target_integrand,
            distance_before=distance_before,
            numeric_before=numeric_before,
            compute_numeric_residual=compute_numeric_residual,
        )
        candidate_records.append(candidate_record)
        if candidate_record["structural_improved"]:
            any_structural_improved = True
        if chosen_record is None and candidate_record["apply_success"]:
            chosen_record = candidate_record

    chosen_structural_improved = bool(chosen_record and chosen_record["structural_improved"])
    return {
        "example_index": int(example_index),
        "target_integrand_prefix": target_integrand_prefix,
        "target_antiderivative_prefix": target_antiderivative_prefix,
        "current_antiderivative_prefix": current_prefix,
        "current_prefix": current_prefix,
        "current_derivative_prefix": _section_prefix(input_tokens, "<DER>", "</DER>"),
        "symbolic_residual_prefix": _section_prefix(input_tokens, "<RES>", "</RES>"),
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "gold_edit_tokens": target_tokens,
        "distance_before": distance_before,
        "numeric_residual_before": numeric_before,
        "predicted_candidates": candidate_records,
        "chosen_candidate_rank": None if chosen_record is None else chosen_record["rank"],
        "chosen_resulting_tree_prefix": None if chosen_record is None else chosen_record["resulting_tree_prefix"],
        "chosen_distance_after": None if chosen_record is None else chosen_record["distance_after"],
        "chosen_numeric_residual_after": None if chosen_record is None else chosen_record["numeric_residual_after"],
        "chosen_structural_improved": chosen_structural_improved,
        "any_structural_improved": any_structural_improved,
        "failure_type": _failure_type(candidate_records, chosen_record),
    }


def _candidate_record(
    candidate: DecodedEdit,
    *,
    rank: int,
    current_tree: Any,
    target_tree: Any,
    target_integrand: Any,
    distance_before: float,
    numeric_before: float | None,
    compute_numeric_residual: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "rank": int(rank),
        "generated_tokens": list(candidate.generated_tokens),
        "raw_generated_tokens": list(candidate.generated_tokens),
        "status": candidate.status,
        "selected_node_id": candidate.selected_node_id,
        "replacement_tokens": list(candidate.replacement_tokens),
        "replacement_subtree_prefix": (
            None
            if candidate.replacement_subtree is None
            else serialize_prefix_string(candidate.replacement_subtree)
        ),
        "logprob": candidate.logprob,
        "apply_success": False,
        "resulting_tree_prefix": None,
        "distance_after": None,
        "numeric_residual_after": None,
        "structural_improved": False,
        "numeric_improved": None,
    }
    if candidate.status != "ok":
        return record

    try:
        edited_tree = apply_decoded_edit(current_tree, candidate)
    except Exception as exc:
        record["apply_error"] = str(exc)
        return record

    distance_after = float(structural_distance(edited_tree, target_tree))
    numeric_after = (
        numeric_residual_score(edited_tree, target_integrand)
        if compute_numeric_residual
        else None
    )
    record.update(
        {
            "apply_success": True,
            "resulting_tree_prefix": serialize_prefix_string(edited_tree),
            "distance_after": distance_after,
            "numeric_residual_after": numeric_after,
            "structural_improved": distance_after < distance_before,
            "numeric_improved": (
                None
                if numeric_before is None or numeric_after is None
                else numeric_after < numeric_before
            ),
        }
    )
    return record


def _failure_type(
    candidate_records: Sequence[Mapping[str, Any]],
    chosen_record: Mapping[str, Any] | None,
) -> str:
    if chosen_record is not None:
        if bool(chosen_record.get("structural_improved")):
            return "structural_improved"
        return "valid_unhelpful_edit"
    statuses = {str(record.get("status")) for record in candidate_records}
    if statuses & {"invalid_position_token", "position_out_of_range", "empty"}:
        return "wrong_or_missing_position"
    if statuses & {"replacement_parse_failed", "missing_replacement"}:
        return "invalid_replacement"
    return "no_applicable_candidate"


def _derived_comparison(
    metrics: Mapping[str, Any],
    *,
    baseline_metrics: Mapping[str, Any],
) -> dict[str, float | None]:
    return {
        "applicable_edit_rate_delta_vs_position_constrained_greedy": _optional_delta(
            metrics.get("applicable_edit_rate"),
            baseline_metrics.get("applicable_edit_rate"),
        ),
        "structural_improvement_rate_delta_vs_position_constrained_greedy": _optional_delta(
            metrics.get("structural_improvement_rate"),
            baseline_metrics.get("structural_improvement_rate"),
        ),
        "numeric_residual_improvement_rate_delta_vs_position_constrained_greedy": _optional_delta(
            metrics.get("numeric_residual_improvement_rate"),
            baseline_metrics.get("numeric_residual_improvement_rate"),
        ),
    }


def _optional_delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _data_source_summary(
    *,
    data: str | None,
    precomputed_data_dir: str | None,
) -> dict[str, Any]:
    if precomputed_data_dir is not None:
        return {"kind": "precomputed", "path": str(precomputed_data_dir), "split": "val"}
    return {"kind": "online_parquet", "path": str(data)}


def _validate_args(
    *,
    data: str | None,
    precomputed_data_dir: str | None,
    num_pairs: int,
    num_batches: int,
    batch_size: int,
    top_k_values: Sequence[int],
    num_dump_examples: int,
    dump_failures_only: bool,
    dump_improvements_only: bool,
) -> None:
    if (data is None) == (precomputed_data_dir is None):
        raise ValueError("Provide exactly one data source: --data or --precomputed-data-dir.")
    if num_pairs < 1:
        raise ValueError("num_pairs must be >= 1.")
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if not top_k_values:
        raise ValueError("At least one top-k value is required.")
    if any(int(value) <= 1 for value in top_k_values):
        raise ValueError("top-k values must be > 1.")
    if num_dump_examples < 0:
        raise ValueError("num_dump_examples must be >= 0.")
    if dump_failures_only and dump_improvements_only:
        raise ValueError("Use at most one of dump_failures_only or dump_improvements_only.")


def _section_prefix(
    tokens: Any,
    start_token: str,
    end_token: str,
) -> str | None:
    if not isinstance(tokens, list):
        return None
    try:
        start_index = tokens.index(start_token)
        end_index = tokens.index(end_token, start_index + 1)
    except ValueError:
        return None
    section = [str(token) for token in tokens[start_index + 1 : end_index]]
    if not section or section[0] in {"<NO_DER>", "<NO_RES>", "<NO_NUM>"}:
        return None
    return " ".join(section)


def _metadata_item(
    batch: Mapping[str, Any],
    key: str,
    row_index: int,
    *,
    default: Any = None,
) -> Any:
    if key not in batch:
        return default
    value = batch[key]
    if isinstance(value, (list, tuple)):
        try:
            return value[row_index]
        except IndexError:
            return default
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value[row_index].detach().cpu().tolist()
    return value


def _required_metadata(batch: Mapping[str, Any], key: str, row_index: int) -> str:
    value = _metadata_item(batch, key, row_index)
    if value is None:
        raise ValueError(f"Batch is missing required metadata field {key!r}.")
    return str(value)


def _required_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Batch is missing required tensor field {key!r}.")
    return value


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


__all__ = [
    "DEFAULT_TOP_K_VALUES",
    "OneStepInferenceEvalMode",
    "main",
    "run_one_step_inference_eval",
]


if __name__ == "__main__":
    raise SystemExit(main())
