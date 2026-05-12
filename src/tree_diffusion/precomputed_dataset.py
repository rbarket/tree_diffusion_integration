from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset


class PrecomputedTreeDiffusionDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: str = "train",
        include_metadata: bool = True,
        limit: int | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'.")
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1 when provided.")

        self.data_dir = Path(data_dir)
        self.split = split
        self.include_metadata = include_metadata
        self.tokenizer_metadata = _load_json(self.data_dir / "tokenizer_metadata.json")
        try:
            self.pad_id = int(self.tokenizer_metadata["pad_id"])
        except KeyError as exc:
            raise ValueError(f"tokenizer_metadata.json is missing {exc.args[0]!r}.") from exc

        shard_paths = _resolve_shard_paths(self.data_dir, split)
        if not shard_paths:
            raise ValueError(f"No precomputed parquet shards found for split {split!r} in {self.data_dir}.")

        import pandas as pd

        frames = [pd.read_parquet(path) for path in shard_paths]
        frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].reset_index(drop=True)
        if limit is not None:
            frame = frame.iloc[:limit].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No precomputed rows loaded for split {split!r} from {self.data_dir}.")
        self._rows = frame.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        input_ids = torch.tensor(_json_list(row["input_ids_json"]), dtype=torch.long)
        target_ids = torch.tensor(_json_list(row["target_ids_json"]), dtype=torch.long)
        labels = (
            torch.tensor(_json_list(row["labels_json"]), dtype=torch.long)
            if _has_value(row.get("labels_json"))
            else _labels_from_target_ids(target_ids, self.pad_id)
        )
        input_attention_mask = input_ids.ne(self.pad_id).to(dtype=torch.long)
        target_attention_mask = target_ids.ne(self.pad_id).to(dtype=torch.long)

        item: dict[str, Any] = {
            "input_ids": input_ids,
            "input_attention_mask": input_attention_mask,
            "target_ids": target_ids,
            "target_attention_mask": target_attention_mask,
            "labels": labels,
            "num_mutations": torch.tensor(_int_value(row.get("num_mutations"), default=0), dtype=torch.long),
            "used_random_init": torch.tensor(bool(row.get("used_random_init", False)), dtype=torch.bool),
            "pair_index": torch.tensor(_int_value(row.get("pair_index"), default=-1), dtype=torch.long),
            "input_length": torch.tensor(_int_value(row.get("input_length"), default=int(input_attention_mask.sum())), dtype=torch.long),
            "target_length": torch.tensor(_int_value(row.get("target_length"), default=int(target_attention_mask.sum())), dtype=torch.long),
        }

        if self.include_metadata:
            warnings = _json_list(row.get("warnings_json", "[]"))
            item.update(
                {
                    "input_tokens": _json_list(row["input_tokens_json"]),
                    "target_tokens": _json_list(row["target_tokens_json"]),
                    "current_prefix": _optional_string(row.get("current_antiderivative_prefix")),
                    "target_integrand_prefix": _optional_string(row.get("target_integrand_prefix")),
                    "target_antiderivative_prefix": _optional_string(row.get("target_antiderivative_prefix")),
                    "selected_node_id": _int_value(row.get("selected_node_id"), default=-1),
                    "replacement_subtree_prefix": _optional_string(row.get("replacement_subtree_prefix")),
                    "resulting_tree_prefix": _optional_string(row.get("resulting_tree_prefix")),
                    "distance_before": _optional_int(row.get("distance_before")),
                    "distance_after": _optional_int(row.get("distance_after")),
                    "observation_status": _optional_string(row.get("observation_status")),
                    "warnings": warnings,
                    "warning_count": len(warnings),
                    "split": _optional_string(row.get("split")),
                    "global_example_index": _int_value(row.get("global_example_index"), default=index),
                    "source": _optional_string(row.get("source")),
                }
            )

        return item


def load_precomputed_tokenizer_metadata(data_dir: str | Path) -> dict[str, Any]:
    return _load_json(Path(data_dir) / "tokenizer_metadata.json")


def _resolve_shard_paths(data_dir: Path, split: str) -> list[Path]:
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = _load_json(metadata_path)
        candidates: Sequence[str] | None = None
        if isinstance(metadata.get(f"{split}_shard_paths"), list):
            candidates = metadata[f"{split}_shard_paths"]
        elif isinstance(metadata.get(split), Mapping) and isinstance(metadata[split].get("output_files"), list):
            candidates = metadata[split]["output_files"]
        elif isinstance(metadata.get(f"{split}_summary"), Mapping) and isinstance(
            metadata[f"{split}_summary"].get("output_files"), list
        ):
            candidates = metadata[f"{split}_summary"]["output_files"]
        if candidates is not None:
            return [_resolve_maybe_relative(data_dir, value) for value in candidates]

    return sorted((data_dir / split).glob("shard_*.parquet"))


def _resolve_maybe_relative(data_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_dir / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required precomputed metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return raw


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not _has_value(value):
        return []
    decoded = json.loads(str(value))
    if not isinstance(decoded, list):
        raise ValueError("Expected a JSON list.")
    return decoded


def _labels_from_target_ids(target_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    labels = target_ids.clone()
    labels[target_ids == pad_id] = -100
    return labels


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _optional_string(value: Any) -> str | None:
    if not _has_value(value):
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if not _has_value(value):
        return None
    return int(value)


def _int_value(value: Any, *, default: int) -> int:
    parsed = _optional_int(value)
    return default if parsed is None else parsed


__all__ = [
    "PrecomputedTreeDiffusionDataset",
    "load_precomputed_tokenizer_metadata",
]
