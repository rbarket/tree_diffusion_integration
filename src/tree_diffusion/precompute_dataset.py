from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from collections import Counter, deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import importlib.metadata
import json
from itertools import islice
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any, Iterator, Mapping, Sequence

from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion._common import (
    json_safe as _json_safe,
    length_summary as _length_summary,
    mean_or_none as _mean_or_none,
    rate as _rate,
    selected_node_summary as _selected_node_summary,
)
from src.tree_diffusion.dataset import IntegrationPair, load_integration_pairs_from_parquet
from src.tree_diffusion.edit_path import structural_distance
from src.tree_diffusion.label_validation import validate_edit_label_progress
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.training_examples import TreeDiffusionTrainingExample, generate_training_example


PROGRESS_EVERY_EXAMPLES = 1_000

_RESUME_SHARD_COLUMNS = (
    "example_index_for_pair",
    "rng_seed",
    "input_length",
    "target_length",
    "selected_node_id",
    "used_random_init",
    "num_mutations",
    "distance_before",
    "distance_after",
    "warnings_json",
    "observation_status",
)

_RESUME_REQUIRED_CONFIG_FIELDS = (
    "input_data",
    "integrand_column",
    "integral_column",
    "train_limit",
    "val_limit",
    "val_fraction",
    "seed",
    "shuffle_before_limit",
    "examples_per_pair_train",
    "examples_per_pair_val",
    "sigma_small",
    "smax",
    "rho",
    "residual_mode",
    "simplify_symbolic_residual",
    "max_input_length",
    "max_target_length",
    "max_positions",
    "max_random_size",
    "max_attempts",
    "observation_timeout_seconds",
    "observation_timeout_retries",
    "allow_complex_constants",
    "allow_distributional_unary_ops",
    "excluded_random_tokens",
    "validate_labels",
    "require_strict_label_improvement",
)


class ObservationTimeoutRetriesExhausted(RuntimeError):
    pass


@dataclass
class TreeDiffusionPrecomputeConfig:
    input_data: str
    output_dir: str = "data/precomputed/tree_diffusion"

    integrand_column: str = "integrand_prefix"
    integral_column: str = "integral_prefix"

    train_limit: int | None = None
    val_limit: int | None = 10000
    val_fraction: float = 0.05
    seed: int = 123
    shuffle_before_limit: bool = True

    examples_per_pair_train: int = 2
    examples_per_pair_val: int = 2

    shard_size: int = 50000
    overwrite: bool = False
    resume: bool = False

    sigma_small: int = 2
    smax: int = 5
    rho: float = 0.2
    residual_mode: str = "both"
    simplify_symbolic_residual: bool = True

    max_input_length: int = 1024
    max_target_length: int = 128
    max_positions: int = 512
    max_random_size: int | None = None
    max_attempts: int = 32
    observation_timeout_seconds: float | None = 5.0
    observation_timeout_retries: int = 3

    allow_complex_constants: bool = False
    allow_distributional_unary_ops: bool = False
    excluded_random_tokens: tuple[str, ...] = ()

    validate_labels: bool = True
    require_strict_label_improvement: bool = False

    max_failures: int | None = None
    write_failed_examples: bool = True
    failed_examples_limit: int = 100
    num_workers: int = 1
    worker_restart_interval: int | None = 1000
    worker_pool_retries: int = 3

    def __post_init__(self) -> None:
        self.excluded_random_tokens = tuple(str(token) for token in self.excluded_random_tokens)
        validate_precompute_config(self)


@dataclass(frozen=True)
class PrecomputedTreeDiffusionExampleRecord:
    split: str
    global_example_index: int
    pair_index: int | None
    source: str | None
    example_index_for_pair: int
    rng_seed: int

    target_integrand_prefix: str
    target_antiderivative_prefix: str
    current_antiderivative_prefix: str
    current_derivative_prefix: str | None
    symbolic_residual_prefix: str | None

    input_tokens_json: str
    target_tokens_json: str
    input_ids_json: str
    target_ids_json: str
    labels_json: str
    input_length: int
    target_length: int

    selected_node_id: int
    replacement_subtree_prefix: str
    resulting_tree_prefix: str | None

    num_mutations: int
    used_random_init: bool
    sampled_s: int | None

    distance_before: int | None
    distance_after: int | None
    label_validation_ok: bool
    label_strict_improvement: bool

    observation_status: str | None
    warnings_json: str


@dataclass(frozen=True)
class _PrecomputeTask:
    split: str
    pair_counter: int
    pair: IntegrationPair
    example_index_for_pair: int
    rng_seed: int


@dataclass(frozen=True)
class _PrecomputeWorkerResult:
    task: _PrecomputeTask
    record: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    timeout_exhausted: bool = False


_WORKER_CONFIG: TreeDiffusionPrecomputeConfig | None = None
_WORKER_TOKENIZER: TreeDiffusionTokenizer | None = None


def load_precompute_config(path: str | Path) -> TreeDiffusionPrecomputeConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Precompute config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Precompute config root must be a JSON object.")

    known_fields = {field.name for field in fields(TreeDiffusionPrecomputeConfig)}
    unknown = sorted(set(raw) - known_fields)
    if unknown:
        raise ValueError(f"Unknown precompute config field(s): {', '.join(unknown)}.")
    if "excluded_random_tokens" in raw:
        raw["excluded_random_tokens"] = tuple(raw["excluded_random_tokens"])
    return TreeDiffusionPrecomputeConfig(**raw)


def validate_precompute_config(config: TreeDiffusionPrecomputeConfig) -> None:
    if not config.input_data:
        raise ValueError("input_data is required.")
    if not Path(config.input_data).exists():
        raise ValueError(f"input_data does not exist: {config.input_data}")
    if config.overwrite and config.resume:
        raise ValueError("overwrite and resume are mutually exclusive.")
    output_dir = Path(config.output_dir)
    if config.resume and not output_dir.exists():
        raise ValueError(f"Cannot resume because output_dir does not exist: {output_dir}")
    if output_dir.exists() and not config.overwrite and not config.resume:
        raise ValueError(f"output_dir already exists and overwrite=False: {output_dir}")
    if config.train_limit is not None and config.train_limit < 1:
        raise ValueError("train_limit must be >= 1 when provided.")
    if config.val_limit is not None and config.val_limit < 0:
        raise ValueError("val_limit must be >= 0 when provided.")
    if not 0.0 <= config.val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1.")
    if config.examples_per_pair_train < 1:
        raise ValueError("examples_per_pair_train must be >= 1.")
    if config.examples_per_pair_val < 1:
        raise ValueError("examples_per_pair_val must be >= 1.")
    if config.shard_size < 1:
        raise ValueError("shard_size must be >= 1.")
    if config.sigma_small < 1:
        raise ValueError("sigma_small must be >= 1.")
    if config.smax < 1:
        raise ValueError("smax must be >= 1.")
    if not 0.0 <= config.rho <= 1.0:
        raise ValueError("rho must satisfy 0 <= rho <= 1.")
    if config.residual_mode not in {"none", "symbolic", "numeric", "both"}:
        raise ValueError("residual_mode must be one of: none, symbolic, numeric, both.")
    if config.max_input_length < 1:
        raise ValueError("max_input_length must be >= 1.")
    if config.max_target_length < 1:
        raise ValueError("max_target_length must be >= 1.")
    if config.max_positions < 1:
        raise ValueError("max_positions must be >= 1.")
    if config.max_random_size is not None and config.max_random_size < 0:
        raise ValueError("max_random_size must be >= 0 when provided.")
    if config.max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")
    if config.observation_timeout_seconds is not None and config.observation_timeout_seconds <= 0.0:
        raise ValueError("observation_timeout_seconds must be > 0 when provided.")
    if config.observation_timeout_retries < 0:
        raise ValueError("observation_timeout_retries must be >= 0.")
    if config.max_failures is not None and config.max_failures < 0:
        raise ValueError("max_failures must be >= 0 when provided.")
    if config.failed_examples_limit < 0:
        raise ValueError("failed_examples_limit must be >= 0.")
    if config.num_workers < 1:
        raise ValueError("num_workers must be >= 1.")
    if config.worker_restart_interval is not None and config.worker_restart_interval < 1:
        raise ValueError("worker_restart_interval must be >= 1 when provided.")
    if config.worker_pool_retries < 0:
        raise ValueError("worker_pool_retries must be >= 0.")


def split_pairs_for_precompute(
    pairs: Sequence[IntegrationPair],
    *,
    val_fraction: float,
    seed: int,
    train_limit: int | None = None,
    val_limit: int | None = None,
    shuffle_before_limit: bool = True,
) -> tuple[list[IntegrationPair], list[IntegrationPair]]:
    if train_limit is not None and train_limit < 1:
        raise ValueError("train_limit must be >= 1 when provided.")
    if val_limit is not None and val_limit < 0:
        raise ValueError("val_limit must be >= 0 when provided.")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1.")
    if not pairs:
        raise ValueError("pairs must be non-empty.")

    ordered = list(pairs)
    if shuffle_before_limit:
        rng = random.Random(seed)
        rng.shuffle(ordered)

    if train_limit is not None and val_limit is None and val_fraction == 0.0:
        train_pairs = ordered[:train_limit]
        val_pairs = ordered[train_limit:]
        if not train_pairs:
            raise ValueError("No training pairs were available after split and limits.")
        _validate_no_pair_index_overlap(train_pairs, val_pairs)
        return train_pairs, val_pairs

    if val_fraction > 0.0 and len(ordered) >= 2:
        val_count = max(1, int(round(len(ordered) * val_fraction)))
        val_count = min(val_count, len(ordered) - 1)
        if val_limit is not None:
            val_count = min(val_count, val_limit)
    elif val_limit is not None and val_limit > 0 and len(ordered) >= 2:
        val_count = min(val_limit, len(ordered) - 1)
    else:
        val_count = 0

    val_pairs = ordered[:val_count]
    train_pairs = ordered[val_count:]
    if train_limit is not None:
        train_pairs = train_pairs[:train_limit]
    if val_limit is not None and val_fraction == 0.0:
        val_pairs = val_pairs[:val_limit]

    if not train_pairs:
        raise ValueError("No training pairs were available after split and limits.")
    _validate_no_pair_index_overlap(train_pairs, val_pairs)
    return train_pairs, val_pairs


def precomputed_record_from_training_example(
    example: TreeDiffusionTrainingExample,
    *,
    split: str,
    global_example_index: int,
    pair: IntegrationPair,
    example_index_for_pair: int,
    rng_seed: int,
    tokenizer: TreeDiffusionTokenizer,
    validate_labels: bool = True,
    require_strict_label_improvement: bool = False,
) -> PrecomputedTreeDiffusionExampleRecord:
    input_ids = example.input_ids
    target_ids = example.target_ids
    if input_ids is None:
        input_ids = tokenizer.encode_tokens(example.input_tokens)
    if target_ids is None:
        target_ids = tokenizer.encode_tokens(example.target_tokens)
    labels = [token_id if token_id != tokenizer.pad_id else -100 for token_id in target_ids]

    distance_before: int | None = None
    distance_after: int | None = None
    label_validation_ok = True
    label_strict_improvement = False
    if validate_labels:
        validation = validate_edit_label_progress(
            example.current_antiderivative,
            example.target_antiderivative,
            example.edit_target,
            require_strict_improvement=require_strict_label_improvement,
        )
        distance_before = validation.distance_before
        distance_after = validation.distance_after
        label_validation_ok = validation.ok
        label_strict_improvement = validation.strict_improvement
        if not validation.ok:
            raise ValueError(
                "label_validation_failed:"
                f"{validation.error or 'unknown'}; split={split}; "
                f"global_example_index={global_example_index}; pair_index={pair.index}; "
                f"rng_seed={rng_seed}"
            )
    else:
        try:
            distance_before = structural_distance(
                example.current_antiderivative,
                example.target_antiderivative,
            )
            distance_after = structural_distance(
                example.edit_target.resulting_tree,
                example.target_antiderivative,
            )
            label_strict_improvement = distance_after < distance_before
            label_validation_ok = distance_after <= distance_before
        except Exception:
            distance_before = None
            distance_after = None
            label_validation_ok = False
            label_strict_improvement = False

    return PrecomputedTreeDiffusionExampleRecord(
        split=split,
        global_example_index=global_example_index,
        pair_index=pair.index,
        source=pair.source,
        example_index_for_pair=example_index_for_pair,
        rng_seed=rng_seed,
        target_integrand_prefix=serialize_prefix_string(example.target_integrand),
        target_antiderivative_prefix=serialize_prefix_string(example.target_antiderivative),
        current_antiderivative_prefix=serialize_prefix_string(example.current_antiderivative),
        current_derivative_prefix=(
            None
            if example.observation.current_derivative is None
            else serialize_prefix_string(example.observation.current_derivative)
        ),
        symbolic_residual_prefix=(
            None
            if example.observation.symbolic_residual is None
            else serialize_prefix_string(example.observation.symbolic_residual)
        ),
        input_tokens_json=json.dumps(example.input_tokens),
        target_tokens_json=json.dumps(example.target_tokens),
        input_ids_json=json.dumps(list(input_ids)),
        target_ids_json=json.dumps(list(target_ids)),
        labels_json=json.dumps(labels),
        input_length=len(example.input_tokens),
        target_length=len(example.target_tokens),
        selected_node_id=example.edit_target.selected_node_id,
        replacement_subtree_prefix=serialize_prefix_string(example.edit_target.replacement_subtree),
        resulting_tree_prefix=serialize_prefix_string(example.edit_target.resulting_tree),
        num_mutations=example.num_mutations,
        used_random_init=example.used_random_init,
        sampled_s=None,
        distance_before=distance_before,
        distance_after=distance_after,
        label_validation_ok=label_validation_ok,
        label_strict_improvement=label_strict_improvement,
        observation_status=example.observation.status,
        warnings_json=json.dumps(list(example.warnings)),
    )


def _iter_precompute_tasks(
    pairs: Sequence[IntegrationPair],
    *,
    split: str,
    config: TreeDiffusionPrecomputeConfig,
    examples_per_pair: int,
    seed_offset: int,
) -> Iterator[_PrecomputeTask]:
    for pair_counter, pair in enumerate(pairs):
        for example_index_for_pair in range(examples_per_pair):
            yield _PrecomputeTask(
                split=split,
                pair_counter=pair_counter,
                pair=pair,
                example_index_for_pair=example_index_for_pair,
                rng_seed=config.seed + seed_offset + pair_counter * 1_000_003 + example_index_for_pair,
            )


def _iter_precompute_worker_results(
    tasks: Iterator[_PrecomputeTask],
    *,
    config: TreeDiffusionPrecomputeConfig,
    tokenizer: TreeDiffusionTokenizer,
) -> Iterator[_PrecomputeWorkerResult]:
    if config.num_workers == 1:
        for task in tasks:
            yield _run_precompute_task(task, config=config, tokenizer=tokenizer)
        return

    task_iter = iter(tasks)
    if config.worker_restart_interval is None:
        yield from _iter_precompute_worker_results_in_pool(task_iter, config=config)
        return

    batch_index = 0
    while True:
        batch = list(islice(task_iter, config.worker_restart_interval))
        if not batch:
            return
        print(
            "precompute_worker_pool_start "
            f"batch={batch_index} tasks={len(batch)} num_workers={config.num_workers}",
            flush=True,
        )
        yield from _iter_precompute_worker_batch_results(
            batch,
            config=config,
            batch_index=batch_index,
        )
        batch_index += 1


def _iter_precompute_worker_batch_results(
    batch: Sequence[_PrecomputeTask],
    *,
    config: TreeDiffusionPrecomputeConfig,
    batch_index: int,
) -> Iterator[_PrecomputeWorkerResult]:
    remaining = list(batch)
    retry_index = 0
    while remaining:
        yielded_count = 0
        try:
            for result in _iter_precompute_worker_results_in_pool(iter(remaining), config=config):
                yielded_count += 1
                yield result
            return
        except BrokenProcessPool:
            remaining = remaining[yielded_count:]
            retry_index += 1
            if retry_index > config.worker_pool_retries:
                raise
            print(
                "precompute_worker_pool_broken "
                f"batch={batch_index} retry={retry_index}/{config.worker_pool_retries} "
                f"remaining_tasks={len(remaining)} num_workers={config.num_workers}",
                flush=True,
            )


def _iter_precompute_worker_results_in_pool(
    tasks: Iterator[_PrecomputeTask],
    *,
    config: TreeDiffusionPrecomputeConfig,
) -> Iterator[_PrecomputeWorkerResult]:
    max_pending = max(config.num_workers * 2, 1)
    task_iter = iter(tasks)
    with ProcessPoolExecutor(
        max_workers=config.num_workers,
        initializer=_init_precompute_worker,
        initargs=(config,),
    ) as executor:
        pending = deque()

        def submit_next() -> bool:
            try:
                task = next(task_iter)
            except StopIteration:
                return False
            pending.append(executor.submit(_run_precompute_task, task))
            return True

        while len(pending) < max_pending and submit_next():
            pass

        while pending:
            yield pending.popleft().result()
            while len(pending) < max_pending and submit_next():
                pass


def _init_precompute_worker(config: TreeDiffusionPrecomputeConfig) -> None:
    global _WORKER_CONFIG, _WORKER_TOKENIZER
    _WORKER_CONFIG = config
    _WORKER_TOKENIZER = TreeDiffusionTokenizer(max_positions=config.max_positions)


def _run_precompute_task(
    task: _PrecomputeTask,
    *,
    config: TreeDiffusionPrecomputeConfig | None = None,
    tokenizer: TreeDiffusionTokenizer | None = None,
) -> _PrecomputeWorkerResult:
    worker_config = config or _WORKER_CONFIG
    worker_tokenizer = tokenizer or _WORKER_TOKENIZER
    if worker_config is None or worker_tokenizer is None:
        raise RuntimeError("Precompute worker was not initialized.")

    try:
        example, rng_seed = _generate_example_with_timeout_retries(
            task.pair,
            tokenizer=worker_tokenizer,
            config=worker_config,
            base_rng_seed=task.rng_seed,
        )
        record = precomputed_record_from_training_example(
            example,
            split=task.split,
            global_example_index=-1,
            pair=task.pair,
            example_index_for_pair=task.example_index_for_pair,
            rng_seed=rng_seed,
            tokenizer=worker_tokenizer,
            validate_labels=worker_config.validate_labels,
            require_strict_label_improvement=worker_config.require_strict_label_improvement,
        )
        return _PrecomputeWorkerResult(task=task, record=asdict(record))
    except Exception as exc:
        return _PrecomputeWorkerResult(
            task=task,
            failure=_failure_record(
                split=task.split,
                pair=task.pair,
                pair_counter=task.pair_counter,
                example_index_for_pair=task.example_index_for_pair,
                rng_seed=task.rng_seed,
                exc=exc,
            ),
            timeout_exhausted=isinstance(exc, ObservationTimeoutRetriesExhausted),
        )


def precompute_split(
    pairs: Sequence[IntegrationPair],
    *,
    split: str,
    output_dir: str | Path,
    tokenizer: TreeDiffusionTokenizer,
    config: TreeDiffusionPrecomputeConfig,
    examples_per_pair: int,
    seed_offset: int = 0,
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'.")
    if examples_per_pair < 1:
        raise ValueError("examples_per_pair must be >= 1.")

    root = Path(output_dir)
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    total_expected = len(pairs) * examples_per_pair
    print(
        "precompute_split_start "
        f"split={split} pairs={len(pairs)} examples_per_pair={examples_per_pair} "
        f"attempted_target={total_expected} shard_size={config.shard_size} "
        f"num_workers={config.num_workers} "
        f"worker_restart_interval={config.worker_restart_interval}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    output_files: list[str] = []
    failed_examples: list[dict[str, Any]] = []
    failure_by_exception_type: Counter[str] = Counter()
    failure_by_category: Counter[str] = Counter()
    observation_status_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    input_lengths: list[int] = []
    target_lengths: list[int] = []
    distance_before_values: list[int] = []
    distance_after_values: list[int] = []
    selected_node_ids: list[int] = []
    used_random_init_count = 0
    num_mutations_values: list[int] = []
    root_edit_count = 0
    nonincreasing_count = 0
    strict_improvement_count = 0
    label_validation_failure_count = 0
    timeout_retry_exhaustion_count = 0
    attempted = 0
    success = 0
    failed = 0
    shard_index = 0
    resume_skipped_unsaved_success_count = 0
    resume_regenerated_unflushed_attempt_count = 0
    resume_split = config.resume and _split_has_resume_state(split_dir)

    if config.resume and not resume_split and any(split_dir.glob("shard_*.parquet")):
        raise ValueError(
            f"Cannot resume split {split!r}: found existing shards but no progress/audit summary "
            f"in {split_dir}."
        )
    if config.resume and not resume_split:
        print(f"precompute_resume_split split={split} no_existing_state=start_fresh", flush=True)

    if resume_split:
        resume_state = _load_resume_split_state(
            split_dir=split_dir,
            root=root,
            split=split,
            failed_examples_limit=config.failed_examples_limit,
            config=config,
            examples_per_pair=examples_per_pair,
            seed_offset=seed_offset,
        )
        attempted = int(resume_state["attempted"])
        success = int(resume_state["success"])
        failed = int(resume_state["failed"])
        shard_index = int(resume_state["shard_index"])
        resume_skipped_unsaved_success_count = int(
            resume_state["resume_skipped_unsaved_success_count"]
        )
        resume_regenerated_unflushed_attempt_count = int(
            resume_state["resume_regenerated_unflushed_attempt_count"]
        )
        output_files = list(resume_state["output_files"])
        failed_examples = list(resume_state["failed_examples"])
        failure_by_exception_type.update(resume_state["failure_by_exception_type"])
        failure_by_category.update(resume_state["failure_by_category"])
        observation_status_counts.update(resume_state["observation_status_counts"])
        warning_counts.update(resume_state["warning_counts"])
        input_lengths.extend(resume_state["input_lengths"])
        target_lengths.extend(resume_state["target_lengths"])
        distance_before_values.extend(resume_state["distance_before_values"])
        distance_after_values.extend(resume_state["distance_after_values"])
        selected_node_ids.extend(resume_state["selected_node_ids"])
        used_random_init_count = int(resume_state["used_random_init_count"])
        num_mutations_values.extend(resume_state["num_mutations_values"])
        root_edit_count = int(resume_state["root_edit_count"])
        nonincreasing_count = int(resume_state["nonincreasing_count"])
        strict_improvement_count = int(resume_state["strict_improvement_count"])
        label_validation_failure_count = int(resume_state["label_validation_failure_count"])
        timeout_retry_exhaustion_count = int(resume_state["timeout_retry_exhaustion_count"])
        print(
            "precompute_resume_split "
            f"split={split} resume_from_task={attempted} saved_success={success} "
            f"prior_failed={failed} "
            f"regenerate_unflushed_attempts={resume_regenerated_unflushed_attempt_count} "
            f"next_shard={shard_index:05d}",
            flush=True,
        )

    def flush_rows() -> None:
        nonlocal rows, shard_index
        if not rows:
            return
        import pandas as pd

        shard_path = split_dir / f"shard_{shard_index:05d}.parquet"
        pd.DataFrame(rows).to_parquet(shard_path, index=False)
        output_files.append(str(shard_path.relative_to(root)))
        print(
            "precompute_shard_written "
            f"split={split} shard={shard_index:05d} rows={len(rows)} path={shard_path}",
            flush=True,
        )
        rows = []
        shard_index += 1

    def write_failed_examples() -> None:
        if not config.write_failed_examples or not failed_examples:
            return
        failed_path = split_dir / "failed_examples.jsonl"
        with failed_path.open("w", encoding="utf-8") as handle:
            for failure in failed_examples:
                handle.write(json.dumps(_json_safe(failure), sort_keys=True) + "\n")

    def append_failed_example(failure: Mapping[str, Any]) -> None:
        if not config.write_failed_examples:
            return
        failed_path = split_dir / "failed_examples.jsonl"
        with failed_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(failure), sort_keys=True) + "\n")

    def append_timeout_example(failure: Mapping[str, Any]) -> None:
        timeout_path = split_dir / "timeout_examples.jsonl"
        with timeout_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(failure), sort_keys=True) + "\n")

    def build_summary() -> dict[str, Any]:
        return {
            "split": split,
            "num_pairs": len(pairs),
            "examples_per_pair": examples_per_pair,
            "attempted": attempted,
            "success": success,
            "failed": failed,
            "resume_skipped_unsaved_success_count": resume_skipped_unsaved_success_count,
            "resume_regenerated_unflushed_attempt_count": resume_regenerated_unflushed_attempt_count,
            "failure_rate": _rate(failed, attempted),
            "failure_by_exception_type": dict(failure_by_exception_type),
            "failure_by_category": dict(failure_by_category),
            "shard_count": len(output_files),
            "output_files": output_files,
            "used_random_init_fraction": _rate(used_random_init_count, success),
            "mean_num_mutations": _mean_or_none(num_mutations_values),
            "input_length": _length_summary(input_lengths),
            "target_length": _length_summary(target_lengths),
            "selected_node_id": _selected_node_summary(selected_node_ids),
            "root_edit_fraction": _rate(root_edit_count, success),
            "label_validation_failure_count": label_validation_failure_count,
            "timeout_retry_exhaustion_count": timeout_retry_exhaustion_count,
            "timeout_examples_file": "timeout_examples.jsonl"
            if timeout_retry_exhaustion_count > 0
            else None,
            "nonincreasing_distance_rate": _rate(nonincreasing_count, success),
            "strict_improvement_rate": _rate(strict_improvement_count, success),
            "distance_before": _length_summary(distance_before_values),
            "distance_after": _length_summary(distance_after_values),
            "observation_status_counts": dict(observation_status_counts),
            "warning_counts": dict(warning_counts),
        }

    def write_summary(summary: Mapping[str, Any]) -> None:
        (split_dir / "audit_summary.json").write_text(
            json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_progress_summary() -> None:
        progress = build_summary()
        progress["progress_snapshot"] = True
        progress["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        (split_dir / "progress_summary.json").write_text(
            json.dumps(_json_safe(progress), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    tasks = _iter_precompute_tasks(
        pairs,
        split=split,
        config=config,
        examples_per_pair=examples_per_pair,
        seed_offset=seed_offset,
    )
    if attempted > 0:
        tasks = islice(tasks, attempted, None)
    try:
        for result in _iter_precompute_worker_results(tasks, config=config, tokenizer=tokenizer):
            attempted += 1
            if result.failure is not None:
                failure = result.failure
                failure_category = str(failure["failure_category"])
                failed += 1
                failure_by_exception_type[str(failure["exception_type"])] += 1
                failure_by_category[failure_category] += 1
                if "label_validation_failed" in str(failure["exception_message"]):
                    label_validation_failure_count += 1
                if result.timeout_exhausted:
                    timeout_retry_exhaustion_count += 1
                    append_timeout_example(failure)
                if len(failed_examples) < config.failed_examples_limit:
                    failed_examples.append(failure)
                    append_failed_example(failure)
                if config.max_failures is not None and failed > config.max_failures:
                    flush_rows()
                    summary = build_summary()
                    write_failed_examples()
                    write_summary(summary)
                    raise RuntimeError(
                        f"Precompute split {split!r} exceeded max_failures={config.max_failures}; "
                        f"audit summary written to {split_dir / 'audit_summary.json'}."
                    )
            else:
                if result.record is None:
                    raise RuntimeError("Precompute worker returned neither a record nor a failure.")
                record = dict(result.record)
                record["global_example_index"] = success
                rows.append(record)
                success += 1
                observation_status_counts[str(record["observation_status"])] += 1
                warnings = json.loads(record["warnings_json"])
                for warning in warnings:
                    warning_counts[str(warning).split(":", 1)[0]] += 1
                input_lengths.append(int(record["input_length"]))
                target_lengths.append(int(record["target_length"]))
                selected_node_ids.append(int(record["selected_node_id"]))
                used_random_init_count += int(record["used_random_init"])
                num_mutations_values.append(int(record["num_mutations"]))
                root_edit_count += int(record["selected_node_id"] == 0)
                if record["distance_before"] is not None:
                    distance_before_values.append(int(record["distance_before"]))
                if record["distance_after"] is not None:
                    distance_after_values.append(int(record["distance_after"]))
                if record["distance_before"] is not None and record["distance_after"] is not None:
                    distance_before = int(record["distance_before"])
                    distance_after = int(record["distance_after"])
                    nonincreasing_count += int(distance_after <= distance_before)
                    strict_improvement_count += int(distance_after < distance_before)

            if len(rows) >= config.shard_size:
                flush_rows()
            if attempted % PROGRESS_EVERY_EXAMPLES == 0:
                write_progress_summary()
                print(
                    "precompute_progress "
                    f"split={split} attempted={attempted}/{total_expected} "
                    f"success={success} failed={failed} shards={len(output_files)} "
                    f"failure_categories={dict(failure_by_category.most_common(5))} "
                    f"failure_types={dict(failure_by_exception_type.most_common(5))}",
                    flush=True,
                )
    finally:
        flush_rows()

    summary = build_summary()
    write_failed_examples()
    write_summary(summary)
    print(
        "precompute_split_done "
        f"split={split} attempted={attempted} success={success} failed={failed} "
        f"failure_rate={summary['failure_rate']:.4f} shards={summary['shard_count']}",
        flush=True,
    )
    return summary


def precompute_tree_diffusion_dataset(
    config: TreeDiffusionPrecomputeConfig,
) -> dict[str, Any]:
    validate_precompute_config(config)
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        if not config.overwrite:
            if not config.resume:
                raise ValueError(f"output_dir already exists and overwrite=False: {output_dir}")
            _validate_resume_config(output_dir=output_dir, config=config)
        else:
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=False)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    config_snapshot = _config_dict(config)
    if not config.resume:
        (output_dir / "precompute_config.json").write_text(
            json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"precompute_load_pairs path={config.input_data}", flush=True)
    pairs = load_integration_pairs_from_parquet(
        config.input_data,
        integrand_column=config.integrand_column,
        integral_column=config.integral_column,
    )
    train_pairs, val_pairs = split_pairs_for_precompute(
        pairs,
        val_fraction=config.val_fraction,
        seed=config.seed,
        train_limit=config.train_limit,
        val_limit=config.val_limit,
        shuffle_before_limit=config.shuffle_before_limit,
    )
    print(
        "precompute_pairs_ready "
        f"loaded={len(pairs)} train_pairs={len(train_pairs)} val_pairs={len(val_pairs)}",
        flush=True,
    )

    tokenizer = TreeDiffusionTokenizer(max_positions=config.max_positions)
    tokenizer_metadata = _tokenizer_metadata(tokenizer)
    tokenizer_metadata_path = output_dir / "tokenizer_metadata.json"
    if not config.resume or not tokenizer_metadata_path.exists():
        tokenizer_metadata_path.write_text(
            json.dumps(tokenizer_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    train_summary = precompute_split(
        train_pairs,
        split="train",
        output_dir=output_dir,
        tokenizer=tokenizer,
        config=config,
        examples_per_pair=config.examples_per_pair_train,
        seed_offset=0,
    )
    val_summary = (
        precompute_split(
            val_pairs,
            split="val",
            output_dir=output_dir,
            tokenizer=tokenizer,
            config=config,
            examples_per_pair=config.examples_per_pair_val,
            seed_offset=10_000_000,
        )
        if val_pairs
        else _empty_split_summary(split="val", examples_per_pair=config.examples_per_pair_val)
    )
    if not val_pairs:
        val_dir = output_dir / "val"
        val_dir.mkdir(parents=True, exist_ok=True)
        (val_dir / "audit_summary.json").write_text(
            json.dumps(val_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_data": config.input_data,
        "pair_split_counts": {
            "loaded": len(pairs),
            "train": len(train_pairs),
            "val": len(val_pairs),
        },
        "examples_per_pair_train": config.examples_per_pair_train,
        "examples_per_pair_val": config.examples_per_pair_val,
        "total_train_examples": train_summary["success"],
        "total_val_examples": val_summary["success"],
        "config": config_snapshot,
        "tokenizer_metadata": tokenizer_metadata,
        "train_shard_paths": train_summary["output_files"],
        "val_shard_paths": val_summary["output_files"],
        "train_summary": train_summary,
        "val_summary": val_summary,
        "git_commit": _git_commit(),
        "code_version": _package_version(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    audit_summary = {
        "train": train_summary,
        "val": val_summary,
        "total_attempted": train_summary["attempted"] + val_summary["attempted"],
        "total_success": train_summary["success"] + val_summary["success"],
        "total_failed": train_summary["failed"] + val_summary["failed"],
        "failure_rate": _rate(
            train_summary["failed"] + val_summary["failed"],
            train_summary["attempted"] + val_summary["attempted"],
        ),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(_json_safe(audit_summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "precompute_done "
        f"output_dir={output_dir} train_success={train_summary['success']} "
        f"val_success={val_summary['success']} total_failed={audit_summary['total_failed']}",
        flush=True,
    )
    return audit_summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Precompute tree-diffusion supervised edit examples.")
    parser.add_argument("--config", required=True, help="JSON precompute config path.")
    parser.add_argument("--input-data", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--examples-per-pair-train", type=int, default=None)
    parser.add_argument("--examples-per-pair-val", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--observation-timeout-seconds", type=float, default=None)
    parser.add_argument("--observation-timeout-retries", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--worker-restart-interval", type=int, default=None)
    parser.add_argument("--worker-pool-retries", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-failures", type=int, default=None)
    args = parser.parse_args(argv)
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive.")

    values = _load_config_values(args.config)
    overrides = {
        "input_data": args.input_data,
        "output_dir": args.output_dir,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "examples_per_pair_train": args.examples_per_pair_train,
        "examples_per_pair_val": args.examples_per_pair_val,
        "shard_size": args.shard_size,
        "seed": args.seed,
        "observation_timeout_seconds": args.observation_timeout_seconds,
        "observation_timeout_retries": args.observation_timeout_retries,
        "num_workers": args.num_workers,
        "worker_restart_interval": args.worker_restart_interval,
        "worker_pool_retries": args.worker_pool_retries,
        "max_failures": args.max_failures,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    if args.overwrite:
        values["overwrite"] = True
    if args.resume:
        values["overwrite"] = False
        values["resume"] = True

    config = TreeDiffusionPrecomputeConfig(**values)
    summary = precompute_tree_diffusion_dataset(config)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


def _load_config_values(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Precompute config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Precompute config root must be a JSON object.")
    known_fields = {field.name for field in fields(TreeDiffusionPrecomputeConfig)}
    unknown = sorted(set(raw) - known_fields)
    if unknown:
        raise ValueError(f"Unknown precompute config field(s): {', '.join(unknown)}.")
    return raw


def _validate_no_pair_index_overlap(
    train_pairs: Sequence[IntegrationPair],
    val_pairs: Sequence[IntegrationPair],
) -> None:
    train_indices = {pair.index for pair in train_pairs if pair.index is not None}
    val_indices = {pair.index for pair in val_pairs if pair.index is not None}
    overlap = train_indices & val_indices
    if overlap:
        raise ValueError(f"Train/validation pair index overlap: {sorted(overlap)[:5]}")


def _load_resume_split_state(
    *,
    split_dir: Path,
    root: Path,
    split: str,
    failed_examples_limit: int,
    config: TreeDiffusionPrecomputeConfig,
    examples_per_pair: int,
    seed_offset: int,
) -> dict[str, Any]:
    progress_path = split_dir / "progress_summary.json"
    if not progress_path.exists():
        progress_path = split_dir / "audit_summary.json"
    if not progress_path.exists():
        raise ValueError(
            f"Cannot resume split {split!r}: expected {split_dir / 'progress_summary.json'} "
            f"or {split_dir / 'audit_summary.json'}."
        )
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("split") not in {None, split}:
        raise ValueError(
            f"Cannot resume split {split!r}: progress file belongs to split "
            f"{progress.get('split')!r}."
        )

    shard_paths = sorted(split_dir.glob("shard_*.parquet"))
    output_files = [str(path.relative_to(root)) for path in shard_paths]
    shard_index = _next_shard_index(shard_paths)
    resume_task_offset = _resume_task_offset_from_last_saved_success(
        shard_paths=shard_paths,
        config=config,
        examples_per_pair=examples_per_pair,
        seed_offset=seed_offset,
    )

    input_lengths: list[int] = []
    target_lengths: list[int] = []
    distance_before_values: list[int] = []
    distance_after_values: list[int] = []
    selected_node_ids: list[int] = []
    num_mutations_values: list[int] = []
    observation_status_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    used_random_init_count = 0
    root_edit_count = 0
    nonincreasing_count = 0
    strict_improvement_count = 0
    saved_success = 0

    if shard_paths:
        import pandas as pd

        for shard_path in shard_paths:
            frame = pd.read_parquet(shard_path, columns=list(_RESUME_SHARD_COLUMNS))
            saved_success += len(frame)
            for row in frame.itertuples(index=False):
                input_lengths.append(int(row.input_length))
                target_lengths.append(int(row.target_length))
                selected_node_id = int(row.selected_node_id)
                selected_node_ids.append(selected_node_id)
                used_random_init_count += int(bool(row.used_random_init))
                num_mutations_values.append(int(row.num_mutations))
                root_edit_count += int(selected_node_id == 0)
                observation_status_counts[str(row.observation_status)] += 1
                try:
                    warnings = json.loads(row.warnings_json)
                except Exception:
                    warnings = []
                for warning in warnings:
                    warning_counts[str(warning).split(":", 1)[0]] += 1
                if not pd.isna(row.distance_before):
                    distance_before_values.append(int(row.distance_before))
                if not pd.isna(row.distance_after):
                    distance_after_values.append(int(row.distance_after))
                if not pd.isna(row.distance_before) and not pd.isna(row.distance_after):
                    distance_before = int(row.distance_before)
                    distance_after = int(row.distance_after)
                    nonincreasing_count += int(distance_after <= distance_before)
                    strict_improvement_count += int(distance_after < distance_before)

    progress_attempted = int(progress.get("attempted", 0))
    prior_failed = max(0, resume_task_offset - saved_success)
    failure_by_exception_type: Counter[str] = Counter()
    failure_by_category: Counter[str] = Counter()
    if prior_failed > 0:
        failure_by_exception_type["ResumedPriorFailure"] = prior_failed
        failure_by_category["resumed_prior_failure"] = prior_failed
    return {
        "attempted": resume_task_offset,
        "success": saved_success,
        "failed": prior_failed,
        "resume_skipped_unsaved_success_count": 0,
        "resume_regenerated_unflushed_attempt_count": max(
            0,
            progress_attempted - resume_task_offset,
        ),
        "output_files": output_files,
        "shard_index": shard_index,
        "failed_examples": _read_failed_examples_sample(
            split_dir / "failed_examples.jsonl",
            limit=failed_examples_limit,
        ),
        "failure_by_exception_type": failure_by_exception_type,
        "failure_by_category": failure_by_category,
        "observation_status_counts": observation_status_counts,
        "warning_counts": warning_counts,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "distance_before_values": distance_before_values,
        "distance_after_values": distance_after_values,
        "selected_node_ids": selected_node_ids,
        "used_random_init_count": used_random_init_count,
        "num_mutations_values": num_mutations_values,
        "root_edit_count": root_edit_count,
        "nonincreasing_count": nonincreasing_count,
        "strict_improvement_count": strict_improvement_count,
        "label_validation_failure_count": 0,
        "timeout_retry_exhaustion_count": 0,
    }


def _split_has_resume_state(split_dir: Path) -> bool:
    return (split_dir / "progress_summary.json").exists() or (
        split_dir / "audit_summary.json"
    ).exists()


def _resume_task_offset_from_last_saved_success(
    *,
    shard_paths: Sequence[Path],
    config: TreeDiffusionPrecomputeConfig,
    examples_per_pair: int,
    seed_offset: int,
) -> int:
    if not shard_paths:
        return 0

    import pandas as pd

    last_shard = sorted(shard_paths)[-1]
    frame = pd.read_parquet(last_shard, columns=["rng_seed", "example_index_for_pair"])
    if frame.empty:
        return 0
    last_row = frame.iloc[-1]
    example_index_for_pair = int(last_row["example_index_for_pair"])
    if not 0 <= example_index_for_pair < examples_per_pair:
        raise ValueError(
            f"Cannot resume from {last_shard}: example_index_for_pair="
            f"{example_index_for_pair} is incompatible with examples_per_pair={examples_per_pair}."
        )

    pair_counter = _pair_counter_from_record_seed(
        rng_seed=int(last_row["rng_seed"]),
        example_index_for_pair=example_index_for_pair,
        config=config,
        seed_offset=seed_offset,
    )
    return pair_counter * examples_per_pair + example_index_for_pair + 1


def _pair_counter_from_record_seed(
    *,
    rng_seed: int,
    example_index_for_pair: int,
    config: TreeDiffusionPrecomputeConfig,
    seed_offset: int,
) -> int:
    for retry_index in range(config.observation_timeout_retries + 1):
        base_seed = rng_seed - retry_index * 17_000_017
        offset = base_seed - config.seed - seed_offset - example_index_for_pair
        if offset >= 0 and offset % 1_000_003 == 0:
            return offset // 1_000_003
    raise ValueError(
        "Cannot derive resume task offset from last saved rng_seed="
        f"{rng_seed}, example_index_for_pair={example_index_for_pair}."
    )


def _next_shard_index(shard_paths: Sequence[Path]) -> int:
    max_index = -1
    for path in shard_paths:
        try:
            max_index = max(max_index, int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max_index + 1


def _read_failed_examples_sample(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not path.exists():
        return []
    examples: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if len(examples) >= limit:
            break
        if not line.strip():
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return examples


def _validate_resume_config(*, output_dir: Path, config: TreeDiffusionPrecomputeConfig) -> None:
    config_path = output_dir / "precompute_config.json"
    if not config_path.exists():
        raise ValueError(f"Cannot resume: missing existing config snapshot at {config_path}.")
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    current = _config_dict(config)
    mismatches = [
        field
        for field in _RESUME_REQUIRED_CONFIG_FIELDS
        if existing.get(field) != current.get(field)
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: existing={existing.get(field)!r} current={current.get(field)!r}"
            for field in mismatches[:8]
        )
        raise ValueError(
            "Cannot resume because config fields that affect split/generation changed. "
            f"Mismatched field(s): {details}"
        )


def _failure_record(
    *,
    split: str,
    pair: IntegrationPair,
    pair_counter: int | None = None,
    example_index_for_pair: int,
    rng_seed: int,
    exc: BaseException,
) -> dict[str, Any]:
    exception_type = type(exc).__name__
    exception_message = str(exc)
    return {
        "split": split,
        "pair_index": pair.index,
        "pair_counter": pair_counter,
        "source": pair.source,
        "example_index_for_pair": example_index_for_pair,
        "rng_seed": rng_seed,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "failure_category": _failure_category(
            exception_type=exception_type,
            exception_message=exception_message,
        ),
        "target_integrand_prefix": serialize_prefix_string(pair.target_integrand),
        "target_antiderivative_prefix": serialize_prefix_string(pair.target_antiderivative),
    }


def _failure_category(*, exception_type: str, exception_message: str) -> str:
    message = exception_message.lower()
    if exception_type == ObservationTimeoutRetriesExhausted.__name__:
        return "timeout_retry_exhausted"
    if "label_validation_failed" in message or "edit_label_validation_failed" in message:
        return "label_validation_failed"
    if "failed to encode input tokens" in message or "max_input_length" in message:
        return "input_too_long"
    if "failed to encode target tokens" in message or "max_target_length" in message:
        return "target_too_long"
    if "failed to generate a supervised tree-diffusion training example" in message:
        if "first_edit_toward_target returned none" in message:
            return "no_edit_path"
        if "increased structural distance" in message:
            return "edit_increased_distance"
        return "generation_attempts_exhausted"
    if "failed to generate a current candidate" in message:
        return "candidate_generation_failed"
    if "unsupported sympy expression" in message:
        return "unsupported_sympy_expression"
    if "unsupported constant symbol" in message:
        return "unsupported_constant"
    return "other"


def _generate_example_with_timeout_retries(
    pair: IntegrationPair,
    *,
    tokenizer: TreeDiffusionTokenizer,
    config: TreeDiffusionPrecomputeConfig,
    base_rng_seed: int,
) -> tuple[TreeDiffusionTrainingExample, int]:
    last_timeout_warnings: tuple[str, ...] = ()
    for retry_index in range(config.observation_timeout_retries + 1):
        rng_seed = base_rng_seed + retry_index * 17_000_017
        example = generate_training_example(
            pair.target_integrand,
            pair.target_antiderivative,
            tokenizer=tokenizer,
            rng=random.Random(rng_seed),
            sigma_small=config.sigma_small,
            smax=config.smax,
            rho=config.rho,
            residual_mode=config.residual_mode,
            simplify_symbolic_residual=config.simplify_symbolic_residual,
            encode=True,
            max_input_length=config.max_input_length,
            max_target_length=config.max_target_length,
            max_random_size=config.max_random_size,
            max_attempts=config.max_attempts,
            observation_timeout_seconds=config.observation_timeout_seconds,
            validate_label=False,
            allow_complex_constants=config.allow_complex_constants,
            allow_distributional_unary_ops=config.allow_distributional_unary_ops,
            excluded_random_tokens=config.excluded_random_tokens,
        )
        if not _has_timeout_warning(example.warnings):
            return example, rng_seed
        last_timeout_warnings = example.warnings

    raise ObservationTimeoutRetriesExhausted(
        "observation_timeout_retries_exhausted:"
        f"{config.observation_timeout_retries + 1} attempts; "
        f"warnings={list(last_timeout_warnings)!r}"
    )


def _has_timeout_warning(warnings: Sequence[str]) -> bool:
    return any("timeout" in warning for warning in warnings)


def _empty_split_summary(*, split: str, examples_per_pair: int) -> dict[str, Any]:
    return {
        "split": split,
        "num_pairs": 0,
        "examples_per_pair": examples_per_pair,
        "attempted": 0,
        "success": 0,
        "failed": 0,
        "resume_skipped_unsaved_success_count": 0,
        "resume_regenerated_unflushed_attempt_count": 0,
        "failure_rate": 0.0,
        "failure_by_exception_type": {},
        "failure_by_category": {},
        "shard_count": 0,
        "output_files": [],
        "used_random_init_fraction": 0.0,
        "mean_num_mutations": None,
        "input_length": _length_summary([]),
        "target_length": _length_summary([]),
        "selected_node_id": _selected_node_summary([]),
        "root_edit_fraction": 0.0,
        "label_validation_failure_count": 0,
        "timeout_retry_exhaustion_count": 0,
        "timeout_examples_file": None,
        "nonincreasing_distance_rate": 0.0,
        "strict_improvement_rate": 0.0,
        "distance_before": _length_summary([]),
        "distance_after": _length_summary([]),
        "observation_status_counts": {},
        "warning_counts": {},
    }


def _tokenizer_metadata(tokenizer: TreeDiffusionTokenizer) -> dict[str, Any]:
    return {
        "vocab_size": tokenizer.vocab_size,
        "max_positions": tokenizer.max_positions,
        "pad_id": tokenizer.pad_id,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "unk_id": tokenizer.unk_id,
        "numeric_log_min": tokenizer.numeric_log_min,
        "numeric_log_max": tokenizer.numeric_log_max,
    }


def _config_dict(config: TreeDiffusionPrecomputeConfig) -> dict[str, Any]:
    return _json_safe(asdict(config))


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("tree_diffusion_integration")
    except importlib.metadata.PackageNotFoundError:
        return None


__all__ = [
    "ObservationTimeoutRetriesExhausted",
    "PrecomputedTreeDiffusionExampleRecord",
    "TreeDiffusionPrecomputeConfig",
    "load_precompute_config",
    "main",
    "precompute_split",
    "precompute_tree_diffusion_dataset",
    "precomputed_record_from_training_example",
    "split_pairs_for_precompute",
    "validate_precompute_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
