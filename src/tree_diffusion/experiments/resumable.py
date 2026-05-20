from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping, Sequence, TypeVar

from src.tree_diffusion._common import json_safe, write_json
from src.tree_diffusion.dataset import (
    load_integration_pairs_from_parquet,
    make_tree_diffusion_dataloader,
)


@dataclass(frozen=True)
class ResumableRunConfig:
    output_dir: Path
    part_size: int
    start_part: int | None = None
    max_parts: int | None = None
    overwrite: bool = False


_RecordT = TypeVar("_RecordT")


def build_resumable_dataloader(
    *,
    data: str | None,
    precomputed_data_dir: str | None,
    precomputed_split: str,
    tokenizer,
    model,
    num_pairs: int | None,
    batch_size: int,
    seed: int,
):
    if precomputed_data_dir is not None:
        return make_tree_diffusion_dataloader(
            tokenizer=tokenizer,
            precomputed_data_dir=precomputed_data_dir,
            precomputed_split=precomputed_split,
            precomputed_limit=num_pairs,
            batch_size=batch_size,
            shuffle_pairs=False,
            include_metadata=True,
        )
    if data is None:
        raise ValueError("Provide data or precomputed_data_dir.")
    if num_pairs is None:
        raise ValueError("Provide num_pairs for parquet-backed resumable evaluation.")
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


def target_example_count(
    *,
    dataloader,
    num_pairs: int | None,
    num_batches: int | None,
    batch_size: int,
) -> int:
    dataset = getattr(dataloader, "dataset", None)
    dataset_len = None if dataset is None else int(len(dataset))
    target = int(num_pairs) if num_pairs is not None else dataset_len
    if target is None:
        raise ValueError("Cannot infer target example count; provide --num-pairs.")
    if num_batches is not None:
        batch_limit = int(num_batches) * int(batch_size)
        target = min(target, batch_limit)
    if dataset_len is not None:
        target = min(target, dataset_len)
    return int(target)


def prepare_output_dir(
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    resume: bool,
    overwrite: bool,
) -> None:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    if resume:
        if not config_path.exists():
            raise ValueError(f"Cannot resume: missing config.json in {output_dir}.")
        prior = load_json(config_path)
        if prior != json_safe(dict(config)):
            completed_rows = completed_example_count(output_dir / "parts")
            if completed_rows == 0:
                write_json(config_path, dict(config))
                return
            if resume_compatible_config(prior, config):
                write_json(config_path, dict(config))
                return
            raise ValueError("Cannot resume: current arguments do not match config.json.")
        return
    existing = [path for path in output_dir.iterdir() if path.name != "config.json"]
    if existing:
        raise ValueError(f"Output directory is not empty; use --resume or --overwrite: {output_dir}")
    write_json(config_path, dict(config))


def run_config(**values: Any) -> dict[str, Any]:
    return json_safe(dict(values))


def resume_compatible_config(
    prior: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    allowed_to_change = {"progress_every", "flush_every", "part_size"}
    prior_filtered = {
        key: value for key, value in prior.items() if key not in allowed_to_change
    }
    current_filtered = {
        key: value for key, value in current.items() if key not in allowed_to_change
    }
    return prior_filtered == json_safe(current_filtered)


def data_source_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("precomputed_data_dir") is not None:
        return {
            "kind": "precomputed",
            "path": config.get("precomputed_data_dir"),
            "split": config.get("precomputed_split"),
        }
    return {"kind": "parquet", "path": config.get("data")}


def write_part_records(
    parts_dir: Path,
    part_index: int,
    records: Sequence[Mapping[str, Any]],
) -> None:
    part_path = parts_dir / f"part_{part_index:06d}.jsonl"
    tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_safe(dict(record)), sort_keys=True) + "\n")
    tmp_path.replace(part_path)
    part_summary = {
        "part_index": int(part_index),
        "path": str(part_path),
        "examples": len(records),
        "first_example_index": None if not records else records[0]["example_index"],
        "last_example_index": None if not records else records[-1]["example_index"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(parts_dir / f"part_{part_index:06d}.summary.json", part_summary)


def load_part_records(
    parts_dir: Path,
    *,
    record_from_json: Callable[[Mapping[str, Any]], _RecordT],
    label: str,
) -> list[_RecordT]:
    records: list[_RecordT] = []
    expected_index = 0
    for path in sorted(parts_dir.glob("part_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                example_index = int(payload.get("example_index", -1))
                if example_index != expected_index:
                    raise ValueError(
                        f"Non-contiguous {label} part index at {path}:{line_number}: "
                        f"expected {expected_index}, got {example_index}."
                    )
                records.append(record_from_json(payload))
                expected_index += 1
    return records


def existing_part_indices(parts_dir: Path) -> list[int]:
    indices: list[int] = []
    for path in parts_dir.glob("part_*.jsonl"):
        try:
            indices.append(int(path.stem.removeprefix("part_")))
        except ValueError:
            continue
    return sorted(indices)


def next_part_index(parts_dir: Path) -> int:
    indices = existing_part_indices(parts_dir)
    return 0 if not indices else max(indices) + 1


def resume_offset_from_parts(parts_dir: Path) -> int:
    return completed_example_count(parts_dir)


def completed_example_count(parts_dir: Path) -> int:
    return sum(jsonl_line_count(path) for path in sorted(parts_dir.glob("part_*.jsonl")))


def jsonl_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_manifest(
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    target_examples: int,
    completed_examples: int,
    complete: bool,
) -> None:
    payload = {
        "config": dict(config),
        "target_examples": int(target_examples),
        "completed_examples": int(completed_examples),
        "complete": bool(complete),
        "part_count": len(list((output_dir / "parts").glob("part_*.jsonl"))),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "manifest.json", json_safe(payload))


def write_combined_summary(output_dir: Path, payload: Mapping[str, Any]) -> None:
    write_json(output_dir / "summary.json", json_safe(dict(payload)))


def append_or_write_status(path: Path, payload: Mapping[str, Any], *, append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(dict(payload)), sort_keys=True) + "\n")


def progress(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def merge_cli_config(
    args: argparse.Namespace,
    *,
    defaults: Mapping[str, Any],
    required: Sequence[str],
    label: str,
) -> dict[str, Any]:
    values = dict(defaults)
    if getattr(args, "config", None) is not None:
        config_values = load_json(Path(args.config))
        unknown = set(config_values) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown {label} config field(s): " + ", ".join(sorted(unknown)))
        values.update(config_values)
    for key in defaults:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    missing = [key for key in required if values[key] is None]
    if missing:
        raise ValueError(
            f"Missing required {label} setting(s): "
            + ", ".join(missing)
            + ". Provide them in --config or on the command line."
        )
    return values


def load_config(output_dir: Path) -> dict[str, Any]:
    return load_json(output_dir / "config.json")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


__all__ = [
    "ResumableRunConfig",
    "append_or_write_status",
    "build_resumable_dataloader",
    "completed_example_count",
    "data_source_summary",
    "existing_part_indices",
    "jsonl_line_count",
    "load_config",
    "load_json",
    "load_part_records",
    "merge_cli_config",
    "next_part_index",
    "prepare_output_dir",
    "progress",
    "resume_offset_from_parts",
    "run_config",
    "write_combined_summary",
    "write_manifest",
    "write_part_records",
]
