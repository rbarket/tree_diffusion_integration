from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from src.mathlang.ast import Expr
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.training_examples import (
    TreeDiffusionTrainingExample,
    generate_training_example,
)

_WORKER_SEED_PRIME = 1_000_003
_TENSOR_FIELDS = (
    "input_ids",
    "input_attention_mask",
    "target_ids",
    "target_attention_mask",
    "labels",
    "num_mutations",
    "used_random_init",
    "pair_index",
    "input_length",
    "target_length",
)
_METADATA_FIELDS = (
    "input_tokens",
    "target_tokens",
    "current_prefix",
    "target_integrand_prefix",
    "target_antiderivative_prefix",
    "selected_node_id",
    "replacement_subtree_prefix",
    "resulting_tree_prefix",
    "distance_before",
    "distance_after",
    "observation_status",
    "warnings",
    "warning_count",
    "split",
    "global_example_index",
    "source",
)


@dataclass(frozen=True)
class IntegrationPair:
    target_integrand: Expr
    target_antiderivative: Expr
    source: str | None = None
    index: int | None = None


def pairs_from_prefix_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    integrand_column: str = "integrand_prefix",
    integral_column: str = "integral_prefix",
    source: str | None = None,
    canonicalize_pairs: bool = True,
) -> list[IntegrationPair]:
    pairs: list[IntegrationPair] = []
    for offset, row in enumerate(rows):
        row_index = _row_index(row, fallback=offset)
        try:
            integrand_text = str(row[integrand_column])
            integral_text = str(row[integral_column])
        except KeyError as exc:
            raise ValueError(f"Missing required column {exc.args[0]!r} in row {row_index}.") from exc

        pairs.append(
            _parse_pair(
                integrand_text,
                integral_text,
                source=source,
                index=row_index,
                canonicalize_pairs=canonicalize_pairs,
            )
        )
    return pairs


def load_integration_pairs_from_parquet(
    path: str | Path,
    *,
    integrand_column: str = "integrand_prefix",
    integral_column: str = "integral_prefix",
    limit: int | None = None,
    start: int = 0,
    canonicalize_pairs: bool = True,
) -> list[IntegrationPair]:
    if start < 0:
        raise ValueError("start must be non-negative.")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative when provided.")

    import pandas as pd

    parquet_path = Path(path)
    frame = pd.read_parquet(
        parquet_path,
        columns=[integrand_column, integral_column],
    )
    end = None if limit is None else start + limit
    sliced = frame.iloc[start:end]

    pairs: list[IntegrationPair] = []
    for row_index, row in sliced.iterrows():
        pairs.append(
            _parse_pair(
                str(row[integrand_column]),
                str(row[integral_column]),
                source=str(parquet_path),
                index=int(row_index) if _is_int_like(row_index) else len(pairs) + start,
                canonicalize_pairs=canonicalize_pairs,
            )
        )
    return pairs


class TreeDiffusionIterableDataset(IterableDataset):
    def __init__(
        self,
        pairs: Sequence[IntegrationPair],
        *,
        tokenizer: TreeDiffusionTokenizer | None = None,
        sigma_small: int = 2,
        smax: int = 5,
        rho: float = 0.2,
        residual_mode: str = "both",
        max_input_length: int = 1024,
        max_target_length: int = 128,
        base_seed: int = 0,
        shuffle_pairs: bool = True,
        max_attempts: int = 32,
        max_random_size: int | None = None,
        observation_timeout_seconds: float | None = None,
        simplify_symbolic_residual: bool = True,
        allow_complex_constants: bool = False,
        allow_distributional_unary_ops: bool = False,
        excluded_random_tokens: Sequence[str] = (),
        validate_generated_labels: bool = False,
        max_derivative_tokens: int | None = None,
        max_residual_tokens: int | None = None,
        include_metadata: bool = True,
    ) -> None:
        _validate_dataset_args(
            pairs=pairs,
            sigma_small=sigma_small,
            smax=smax,
            rho=rho,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            max_attempts=max_attempts,
            observation_timeout_seconds=observation_timeout_seconds,
            max_derivative_tokens=max_derivative_tokens,
            max_residual_tokens=max_residual_tokens,
        )

        self.pairs = tuple(pairs)
        self.tokenizer = tokenizer or TreeDiffusionTokenizer()
        self.sigma_small = sigma_small
        self.smax = smax
        self.rho = rho
        self.residual_mode = residual_mode
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.base_seed = base_seed
        self.shuffle_pairs = shuffle_pairs
        self.max_attempts = max_attempts
        self.max_random_size = max_random_size
        self.observation_timeout_seconds = observation_timeout_seconds
        self.simplify_symbolic_residual = simplify_symbolic_residual
        self.allow_complex_constants = allow_complex_constants
        self.allow_distributional_unary_ops = allow_distributional_unary_ops
        self.excluded_random_tokens = tuple(str(token) for token in excluded_random_tokens)
        self.validate_generated_labels = validate_generated_labels
        self.max_derivative_tokens = max_derivative_tokens
        self.max_residual_tokens = max_residual_tokens
        self.include_metadata = include_metadata

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        worker_count = 1 if worker_info is None else worker_info.num_workers
        rng = random.Random(self.base_seed + worker_id * _WORKER_SEED_PRIME)
        ordered_index = worker_id

        while True:
            last_error: BaseException | None = None
            last_pair: IntegrationPair | None = None
            last_pair_offset: int | None = None

            for _ in range(self.max_attempts):
                if self.shuffle_pairs:
                    pair_offset = rng.randrange(len(self.pairs))
                else:
                    pair_offset = ordered_index % len(self.pairs)
                    ordered_index += worker_count

                pair = self.pairs[pair_offset]
                last_pair = pair
                last_pair_offset = pair_offset
                try:
                    example = generate_training_example(
                        pair.target_integrand,
                        pair.target_antiderivative,
                        tokenizer=self.tokenizer,
                        rng=rng,
                        sigma_small=self.sigma_small,
                        smax=self.smax,
                        rho=self.rho,
                        residual_mode=self.residual_mode,
                        encode=True,
                        max_input_length=self.max_input_length,
                        max_target_length=self.max_target_length,
                        max_random_size=self.max_random_size,
                        max_attempts=self.max_attempts,
                        observation_timeout_seconds=self.observation_timeout_seconds,
                        simplify_symbolic_residual=self.simplify_symbolic_residual,
                        allow_complex_constants=self.allow_complex_constants,
                        allow_distributional_unary_ops=self.allow_distributional_unary_ops,
                        excluded_random_tokens=self.excluded_random_tokens,
                        validate_label=self.validate_generated_labels,
                        max_derivative_tokens=self.max_derivative_tokens,
                        max_residual_tokens=self.max_residual_tokens,
                    )
                except Exception as exc:
                    last_error = exc
                    continue

                yield _example_to_item(
                    example,
                    tokenizer=self.tokenizer,
                    pair=pair,
                    pair_offset=pair_offset,
                    include_metadata=self.include_metadata,
                )
                break
            else:
                raise _generation_failure(
                    max_attempts=self.max_attempts,
                    last_error=last_error,
                    last_pair=last_pair,
                    last_pair_offset=last_pair_offset,
                )


@dataclass
class TreeDiffusionBatchCollator:
    tokenizer: TreeDiffusionTokenizer
    include_metadata: bool = True

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty example list.")

        batch: dict[str, Any] = {}
        for field in _TENSOR_FIELDS:
            batch[field] = torch.stack([_as_tensor(example[field]) for example in examples])

        if self.include_metadata:
            for field in _METADATA_FIELDS:
                if field in examples[0]:
                    batch[field] = [example[field] for example in examples]

        return batch


def make_tree_diffusion_dataloader(
    pairs: Sequence[IntegrationPair] | None = None,
    *,
    tokenizer: TreeDiffusionTokenizer | None = None,
    precomputed_data_dir: str | Path | None = None,
    precomputed_split: str = "train",
    precomputed_limit: int | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    sigma_small: int = 2,
    smax: int = 5,
    rho: float = 0.2,
    residual_mode: str = "both",
    max_input_length: int = 1024,
    max_target_length: int = 128,
    base_seed: int = 0,
    shuffle_pairs: bool = True,
    max_attempts: int = 32,
    max_random_size: int | None = None,
    observation_timeout_seconds: float | None = None,
    simplify_symbolic_residual: bool = True,
    allow_complex_constants: bool = False,
    allow_distributional_unary_ops: bool = False,
    excluded_random_tokens: Sequence[str] = (),
    validate_generated_labels: bool = False,
    max_derivative_tokens: int | None = None,
    max_residual_tokens: int | None = None,
    include_metadata: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if (pairs is None) == (precomputed_data_dir is None):
        raise ValueError("Provide exactly one data source: pairs or precomputed_data_dir.")
    if precomputed_limit is not None and precomputed_limit < 1:
        raise ValueError("precomputed_limit must be >= 1 when provided.")

    if precomputed_data_dir is not None:
        from src.tree_diffusion.precomputed_dataset import (
            PrecomputedTreeDiffusionDataset,
            load_precomputed_tokenizer_metadata,
        )

        metadata = load_precomputed_tokenizer_metadata(precomputed_data_dir)
        tokenizer = tokenizer or _tokenizer_from_metadata(metadata)
        _validate_tokenizer_matches_metadata(tokenizer, metadata)
        dataset = PrecomputedTreeDiffusionDataset(
            precomputed_data_dir,
            split=precomputed_split,
            include_metadata=include_metadata,
            limit=precomputed_limit,
        )
        collator = TreeDiffusionBatchCollator(
            tokenizer=tokenizer,
            include_metadata=include_metadata,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_pairs,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin_memory,
        )

    assert pairs is not None
    tokenizer = tokenizer or TreeDiffusionTokenizer()
    dataset = TreeDiffusionIterableDataset(
        pairs,
        tokenizer=tokenizer,
        sigma_small=sigma_small,
        smax=smax,
        rho=rho,
        residual_mode=residual_mode,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        base_seed=base_seed,
        shuffle_pairs=shuffle_pairs,
        max_attempts=max_attempts,
        max_random_size=max_random_size,
        observation_timeout_seconds=observation_timeout_seconds,
        simplify_symbolic_residual=simplify_symbolic_residual,
        allow_complex_constants=allow_complex_constants,
        allow_distributional_unary_ops=allow_distributional_unary_ops,
        excluded_random_tokens=excluded_random_tokens,
        validate_generated_labels=validate_generated_labels,
        max_derivative_tokens=max_derivative_tokens,
        max_residual_tokens=max_residual_tokens,
        include_metadata=include_metadata,
    )
    collator = TreeDiffusionBatchCollator(
        tokenizer=tokenizer,
        include_metadata=include_metadata,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
    )


def _parse_pair(
    integrand_text: str,
    integral_text: str,
    *,
    source: str | None,
    index: int,
    canonicalize_pairs: bool,
) -> IntegrationPair:
    try:
        target_integrand = parse_prefix_string(integrand_text)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse integrand row {index}: expression={integrand_text!r}."
        ) from exc

    try:
        target_antiderivative = parse_prefix_string(integral_text)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse integral row {index}: expression={integral_text!r}."
        ) from exc

    if canonicalize_pairs:
        target_integrand = canonicalize(target_integrand, strip_additive_constants=False)
        target_antiderivative = canonicalize(target_antiderivative)

    return IntegrationPair(
        target_integrand=target_integrand,
        target_antiderivative=target_antiderivative,
        source=source,
        index=index,
    )


def _row_index(row: Mapping[str, Any], *, fallback: int) -> int:
    raw_index = row.get("index", fallback)
    if _is_int_like(raw_index):
        return int(raw_index)
    return fallback


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _validate_dataset_args(
    *,
    pairs: Sequence[IntegrationPair],
    sigma_small: int,
    smax: int,
    rho: float,
    max_input_length: int,
    max_target_length: int,
    max_attempts: int,
    observation_timeout_seconds: float | None,
    max_derivative_tokens: int | None,
    max_residual_tokens: int | None,
) -> None:
    if not pairs:
        raise ValueError("pairs must be non-empty.")
    if max_input_length < 1:
        raise ValueError("max_input_length must be >= 1.")
    if max_target_length < 1:
        raise ValueError("max_target_length must be >= 1.")
    if sigma_small < 1:
        raise ValueError("sigma_small must be >= 1.")
    if smax < 1:
        raise ValueError("smax must be >= 1.")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must satisfy 0.0 <= rho <= 1.0.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")
    if observation_timeout_seconds is not None and observation_timeout_seconds <= 0.0:
        raise ValueError("observation_timeout_seconds must be > 0 when provided.")
    if max_derivative_tokens is not None and max_derivative_tokens < 1:
        raise ValueError("max_derivative_tokens must be >= 1 when provided.")
    if max_residual_tokens is not None and max_residual_tokens < 1:
        raise ValueError("max_residual_tokens must be >= 1 when provided.")


def _example_to_item(
    example: TreeDiffusionTrainingExample,
    *,
    tokenizer: TreeDiffusionTokenizer,
    pair: IntegrationPair,
    pair_offset: int,
    include_metadata: bool,
) -> dict[str, Any]:
    if example.input_ids is None or example.target_ids is None:
        raise ValueError("Encoded training example is missing input_ids or target_ids.")

    input_ids = torch.tensor(example.input_ids, dtype=torch.long)
    target_ids = torch.tensor(example.target_ids, dtype=torch.long)
    input_attention_mask = (input_ids != tokenizer.pad_id).to(dtype=torch.long)
    target_attention_mask = (target_ids != tokenizer.pad_id).to(dtype=torch.long)
    labels = target_ids.clone()
    labels[target_ids == tokenizer.pad_id] = -100

    item: dict[str, Any] = {
        "input_ids": input_ids,
        "input_attention_mask": input_attention_mask,
        "target_ids": target_ids,
        "target_attention_mask": target_attention_mask,
        "labels": labels,
        "num_mutations": torch.tensor(example.num_mutations, dtype=torch.long),
        "used_random_init": torch.tensor(example.used_random_init, dtype=torch.bool),
        "pair_index": torch.tensor(pair.index if pair.index is not None else pair_offset, dtype=torch.long),
        "input_length": torch.tensor(len(example.input_tokens), dtype=torch.long),
        "target_length": torch.tensor(len(example.target_tokens), dtype=torch.long),
    }

    if include_metadata:
        item.update(
            {
                "input_tokens": example.input_tokens,
                "target_tokens": example.target_tokens,
                "current_prefix": serialize_prefix_string(example.current_antiderivative),
                "target_integrand_prefix": serialize_prefix_string(example.target_integrand),
                "target_antiderivative_prefix": serialize_prefix_string(example.target_antiderivative),
                "warning_count": len(example.warnings),
            }
        )

    return item


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, bool):
        return torch.tensor(value, dtype=torch.bool)
    return torch.tensor(value, dtype=torch.long)


def _tokenizer_from_metadata(metadata: Mapping[str, Any]) -> TreeDiffusionTokenizer:
    return TreeDiffusionTokenizer(
        max_positions=int(metadata.get("max_positions", 512)),
        numeric_log_min=int(metadata.get("numeric_log_min", -12)),
        numeric_log_max=int(metadata.get("numeric_log_max", 12)),
    )


def _validate_tokenizer_matches_metadata(
    tokenizer: TreeDiffusionTokenizer,
    metadata: Mapping[str, Any],
) -> None:
    expected = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "unk_id": tokenizer.unk_id,
        "max_positions": tokenizer.max_positions,
        "numeric_log_min": tokenizer.numeric_log_min,
        "numeric_log_max": tokenizer.numeric_log_max,
    }
    mismatches = [
        f"{name}: tokenizer={actual} metadata={metadata[name]}"
        for name, actual in expected.items()
        if name in metadata and int(metadata[name]) != int(actual)
    ]
    if mismatches:
        raise ValueError(
            "Tokenizer is incompatible with precomputed tokenizer_metadata.json: "
            + "; ".join(mismatches)
        )


def _generation_failure(
    *,
    max_attempts: int,
    last_error: BaseException | None,
    last_pair: IntegrationPair | None,
    last_pair_offset: int | None,
) -> RuntimeError:
    pair_index = None
    source = None
    if last_pair is not None:
        pair_index = last_pair.index if last_pair.index is not None else last_pair_offset
        source = last_pair.source

    error_text = "none"
    if last_error is not None:
        error_text = f"{type(last_error).__name__}: {last_error}"

    return RuntimeError(
        "Failed to generate a tree-diffusion dataset item "
        f"after {max_attempts} attempts; pair_index={pair_index}, "
        f"source={source!r}, last_error={error_text}."
    )


__all__ = [
    "IntegrationPair",
    "TreeDiffusionBatchCollator",
    "TreeDiffusionIterableDataset",
    "load_integration_pairs_from_parquet",
    "make_tree_diffusion_dataloader",
    "pairs_from_prefix_rows",
]
