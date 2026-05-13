from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from concurrent.futures.process import BrokenProcessPool

from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion.dataset import IntegrationPair
from src.tree_diffusion.precompute_dataset import (
    TreeDiffusionPrecomputeConfig,
    _PrecomputeTask,
    _PrecomputeWorkerResult,
    _iter_precompute_worker_batch_results,
    _iter_precompute_isolated_worker_results,
    _generate_example_with_timeout_retries,
    load_precompute_config,
    precompute_split,
    precompute_tree_diffusion_dataset,
    split_pairs_for_precompute,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from tests.tree_diffusion_test_utils import run_tiny_precompute, write_extended_toy_parquet


class TreeDiffusionPrecomputeDatasetTests(unittest.TestCase):
    def test_config_loading_and_validation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            config_path = _write_config(
                work_dir / "config.json",
                input_data=parquet,
                output_dir=work_dir / "precomputed",
                values={"train_limit": 4},
            )

            config = load_precompute_config(config_path)

            self.assertEqual(config.input_data, str(parquet))
            self.assertEqual(config.output_dir, str(work_dir / "precomputed"))
            self.assertEqual(config.examples_per_pair_train, 2)
            self.assertEqual(config.excluded_random_tokens, ())
            self.assertEqual(config.observation_timeout_seconds, 5.0)
            self.assertEqual(config.observation_timeout_retries, 3)
            self.assertEqual(config.num_workers, 1)
            self.assertEqual(config.worker_restart_interval, 1000)
            self.assertEqual(config.worker_pool_retries, 3)
            self.assertTrue(config.isolate_broken_pool_tasks)

            missing = _write_config(
                work_dir / "missing.json",
                input_data=work_dir / "missing.parquet",
                output_dir=work_dir / "missing_out",
            )
            with self.assertRaisesRegex(ValueError, "input_data"):
                load_precompute_config(missing)

            invalid_cases = (
                ("bad_examples", {"examples_per_pair_train": 0}, "examples_per_pair_train"),
                ("bad_shard", {"shard_size": 0}, "shard_size"),
                ("bad_rho", {"rho": 1.5}, "rho"),
                ("bad_timeout", {"observation_timeout_seconds": 0.0}, "observation_timeout_seconds"),
                ("bad_timeout_retries", {"observation_timeout_retries": -1}, "observation_timeout_retries"),
                ("bad_num_workers", {"num_workers": 0}, "num_workers"),
                ("bad_worker_restart_interval", {"worker_restart_interval": 0}, "worker_restart_interval"),
                ("bad_worker_pool_retries", {"worker_pool_retries": -1}, "worker_pool_retries"),
            )
            for name, overrides, pattern in invalid_cases:
                with self.subTest(name=name):
                    path = _write_config(
                        work_dir / f"{name}.json",
                        input_data=parquet,
                        output_dir=work_dir / f"{name}_out",
                        values=overrides,
                    )
                    with self.assertRaisesRegex(ValueError, pattern):
                        load_precompute_config(path)

    def test_split_pairs_for_precompute_is_deterministic_and_disjoint(self) -> None:
        pairs = _fake_pairs(20)

        first_train, first_val = split_pairs_for_precompute(
            pairs,
            val_fraction=0.2,
            seed=123,
            train_limit=10,
            val_limit=5,
        )
        second_train, second_val = split_pairs_for_precompute(
            pairs,
            val_fraction=0.2,
            seed=123,
            train_limit=10,
            val_limit=5,
        )
        other_train, other_val = split_pairs_for_precompute(
            pairs,
            val_fraction=0.2,
            seed=456,
            train_limit=10,
            val_limit=5,
        )

        self.assertEqual([pair.index for pair in first_train], [pair.index for pair in second_train])
        self.assertEqual([pair.index for pair in first_val], [pair.index for pair in second_val])
        self.assertNotEqual(
            [pair.index for pair in first_train + first_val],
            [pair.index for pair in other_train + other_val],
        )
        self.assertEqual(len(first_train), 10)
        self.assertEqual(len(first_val), 4)
        self.assertFalse({pair.index for pair in first_train} & {pair.index for pair in first_val})

    def test_split_pairs_can_use_remaining_pairs_as_validation(self) -> None:
        pairs = _fake_pairs(20)

        train_pairs, val_pairs = split_pairs_for_precompute(
            pairs,
            val_fraction=0.0,
            seed=123,
            train_limit=10,
            val_limit=None,
            shuffle_before_limit=True,
        )

        self.assertEqual(len(train_pairs), 10)
        self.assertEqual(len(val_pairs), 10)
        self.assertFalse({pair.index for pair in train_pairs} & {pair.index for pair in val_pairs})
        self.assertNotEqual([pair.index for pair in train_pairs], list(range(10)))

    def test_tiny_precompute_creates_expected_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            output_dir = _run_tiny_precompute(work_dir)

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            audit = json.loads((output_dir / "audit_summary.json").read_text(encoding="utf-8"))

            self.assertTrue((output_dir / "tokenizer_metadata.json").exists())
            self.assertTrue((output_dir / "train" / "audit_summary.json").exists())
            self.assertTrue((output_dir / "val" / "audit_summary.json").exists())
            self.assertGreater(metadata["total_train_examples"], 0)
            self.assertGreater(metadata["total_val_examples"], 0)
            self.assertEqual(audit["train"]["label_validation_failure_count"], 0)
            self.assertEqual(audit["val"]["label_validation_failure_count"], 0)
            self.assertTrue(list((output_dir / "train").glob("shard_*.parquet")))
            self.assertTrue(list((output_dir / "val").glob("shard_*.parquet")))

            train_rows = _read_split_rows(output_dir, "train")
            val_rows = _read_split_rows(output_dir, "val")
            self.assertEqual(len(train_rows), metadata["total_train_examples"])
            self.assertEqual(len(val_rows), metadata["total_val_examples"])

    def test_parquet_schema_and_json_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = _run_tiny_precompute(Path(temp_dir))
            frame = pd.read_parquet(sorted((output_dir / "train").glob("shard_*.parquet"))[0])

            required_columns = {
                "split",
                "global_example_index",
                "pair_index",
                "target_integrand_prefix",
                "target_antiderivative_prefix",
                "current_antiderivative_prefix",
                "input_tokens_json",
                "target_tokens_json",
                "input_ids_json",
                "target_ids_json",
                "labels_json",
                "selected_node_id",
                "replacement_subtree_prefix",
                "distance_before",
                "distance_after",
                "label_validation_ok",
            }
            self.assertTrue(required_columns.issubset(set(frame.columns)))

            row = frame.iloc[0]
            input_tokens = json.loads(row["input_tokens_json"])
            target_tokens = json.loads(row["target_tokens_json"])
            input_ids = json.loads(row["input_ids_json"])
            target_ids = json.loads(row["target_ids_json"])
            labels = json.loads(row["labels_json"])

            self.assertEqual(input_tokens[-1], "<EDIT>")
            self.assertTrue(target_tokens[0].startswith("<POS_"))
            self.assertEqual(target_tokens[-1], "<eos>")
            self.assertEqual(len(input_ids), 256)
            self.assertEqual(len(target_ids), 64)
            self.assertEqual(len(labels), 64)
            self.assertLessEqual(int(row["distance_after"]), int(row["distance_before"]))

    def test_failure_handling_writes_summary_before_raising(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            output_dir = work_dir / "precomputed_fail"
            config = TreeDiffusionPrecomputeConfig(
                input_data=str(parquet),
                output_dir=str(output_dir),
                train_limit=2,
                val_limit=1,
                val_fraction=0.2,
                examples_per_pair_train=1,
                examples_per_pair_val=1,
                shard_size=2,
                overwrite=True,
                max_input_length=1,
                max_target_length=1,
                max_failures=0,
            )

            with self.assertRaisesRegex(RuntimeError, "max_failures"):
                precompute_tree_diffusion_dataset(config)

            summary_path = output_dir / "train" / "audit_summary.json"
            failed_path = output_dir / "train" / "failed_examples.jsonl"
            self.assertTrue(summary_path.exists())
            self.assertTrue(failed_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertGreater(summary["failed"], 0)

    def test_timeout_warnings_retry_same_precompute_example(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            config = TreeDiffusionPrecomputeConfig(
                input_data=str(parquet),
                output_dir=str(work_dir / "precomputed"),
                overwrite=True,
                observation_timeout_seconds=5.0,
                observation_timeout_retries=2,
            )
            pair = _fake_pairs(1)[0]
            examples = [
                SimpleNamespace(warnings=("derivative_timeout",)),
                SimpleNamespace(warnings=("numeric_probe_timeout",)),
                SimpleNamespace(warnings=()),
            ]

            with patch(
                "src.tree_diffusion.precompute_dataset.generate_training_example",
                side_effect=examples,
            ) as generate:
                example, seed = _generate_example_with_timeout_retries(
                    pair,
                    tokenizer=TreeDiffusionTokenizer(),
                    config=config,
                    base_rng_seed=11,
                )

            self.assertIs(example, examples[-1])
            self.assertEqual(seed, 11 + 2 * 17_000_017)
            self.assertEqual(generate.call_count, 3)
            for call in generate.call_args_list:
                self.assertEqual(call.kwargs["observation_timeout_seconds"], 5.0)

    def test_timeout_retry_exhaustion_marks_example_failed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            config = TreeDiffusionPrecomputeConfig(
                input_data=str(parquet),
                output_dir=str(work_dir / "precomputed"),
                overwrite=True,
                observation_timeout_retries=1,
            )

            with patch(
                "src.tree_diffusion.precompute_dataset.generate_training_example",
                return_value=SimpleNamespace(warnings=("symbolic_residual_timeout",)),
            ):
                with self.assertRaisesRegex(RuntimeError, "observation_timeout_retries_exhausted"):
                    _generate_example_with_timeout_retries(
                        _fake_pairs(1)[0],
                        tokenizer=TreeDiffusionTokenizer(),
                        config=config,
                        base_rng_seed=11,
                    )

    def test_timeout_retry_exhaustion_writes_separate_file_and_moves_on(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            output_dir = work_dir / "precomputed"
            config = TreeDiffusionPrecomputeConfig(
                input_data=str(parquet),
                output_dir=str(output_dir),
                overwrite=True,
                observation_timeout_retries=1,
                write_failed_examples=True,
            )

            with patch(
                "src.tree_diffusion.precompute_dataset.generate_training_example",
                return_value=SimpleNamespace(warnings=("derivative_timeout",)),
            ):
                summary = precompute_split(
                    _fake_pairs(1),
                    split="train",
                    output_dir=output_dir,
                    tokenizer=TreeDiffusionTokenizer(),
                    config=config,
                    examples_per_pair=1,
                )

            timeout_path = output_dir / "train" / "timeout_examples.jsonl"
            self.assertEqual(summary["attempted"], 1)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["timeout_retry_exhaustion_count"], 1)
            self.assertEqual(summary["timeout_examples_file"], "timeout_examples.jsonl")
            self.assertTrue(timeout_path.exists())
            timeout_record = json.loads(timeout_path.read_text(encoding="utf-8").strip())
            self.assertEqual(timeout_record["exception_type"], "ObservationTimeoutRetriesExhausted")
            self.assertIn("observation_timeout_retries_exhausted", timeout_record["exception_message"])

    def test_resume_regenerates_unflushed_attempts_after_last_saved_shard(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            output_dir = work_dir / "precomputed"
            split_dir = output_dir / "train"
            split_dir.mkdir(parents=True)
            _write_resume_shard(split_dir / "shard_00000.parquet")
            (split_dir / "progress_summary.json").write_text(
                json.dumps(
                    {
                        "split": "train",
                        "attempted": 3,
                        "success": 2,
                        "failed": 1,
                        "failure_by_category": {"input_too_long": 1},
                        "failure_by_exception_type": {"ValueError": 1},
                    }
                ),
                encoding="utf-8",
            )
            config = TreeDiffusionPrecomputeConfig(
                input_data=str(parquet),
                output_dir=str(output_dir),
                resume=True,
                train_limit=4,
                val_limit=0,
                val_fraction=0.0,
            )
            seen_tasks = []

            def fake_worker_results(tasks, *, config, tokenizer):
                seen_tasks.extend(list(tasks))
                if False:
                    yield

            with patch(
                "src.tree_diffusion.precompute_dataset._iter_precompute_worker_results",
                side_effect=fake_worker_results,
            ):
                summary = precompute_split(
                    _fake_pairs(3),
                    split="train",
                    output_dir=output_dir,
                    tokenizer=TreeDiffusionTokenizer(),
                    config=config,
                    examples_per_pair=2,
                )

            self.assertEqual(summary["attempted"], 1)
            self.assertEqual(summary["success"], 1)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["resume_skipped_unsaved_success_count"], 0)
            self.assertEqual(summary["resume_regenerated_unflushed_attempt_count"], 2)
            self.assertEqual(summary["shard_count"], 1)
            self.assertEqual(seen_tasks[0].pair_counter, 0)
            self.assertEqual(seen_tasks[0].example_index_for_pair, 1)

    def test_worker_batch_retries_unemitted_tasks_after_broken_pool(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            config = TreeDiffusionPrecomputeConfig(
                input_data=str(parquet),
                output_dir=str(work_dir / "precomputed"),
                overwrite=True,
                worker_pool_retries=1,
            )
            tasks = [
                _PrecomputeTask("train", 0, _fake_pairs(1)[0], 0, 123),
                _PrecomputeTask("train", 0, _fake_pairs(1)[0], 1, 124),
            ]
            first_result = _PrecomputeWorkerResult(task=tasks[0], record={"ok": 0})
            second_result = _PrecomputeWorkerResult(task=tasks[1], record={"ok": 1})
            calls: list[list[int]] = []

            def fake_pool(task_iter, *, config):
                current = list(task_iter)
                calls.append([task.example_index_for_pair for task in current])
                if len(calls) == 1:
                    yield first_result
                    raise BrokenProcessPool("boom")
                yield second_result

            with patch(
                "src.tree_diffusion.precompute_dataset._iter_precompute_worker_results_in_pool",
                side_effect=fake_pool,
            ):
                results = list(
                    _iter_precompute_worker_batch_results(
                        tasks,
                        config=config,
                        batch_index=0,
                    )
                )

            self.assertEqual(results, [first_result, second_result])
            self.assertEqual(calls, [[0, 1], [1]])

    def test_worker_batch_isolates_tasks_after_retries_are_exhausted(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            config = TreeDiffusionPrecomputeConfig(
                input_data=str(parquet),
                output_dir=str(work_dir / "precomputed"),
                overwrite=True,
                worker_pool_retries=0,
            )
            tasks = [
                _PrecomputeTask("train", 0, _fake_pairs(1)[0], 0, 123),
                _PrecomputeTask("train", 0, _fake_pairs(1)[0], 1, 124),
            ]
            fallback_result = _PrecomputeWorkerResult(task=tasks[0], record={"ok": 0})

            def broken_pool(task_iter, *, config):
                list(task_iter)
                raise BrokenProcessPool("boom")
                if False:
                    yield

            with patch(
                "src.tree_diffusion.precompute_dataset._iter_precompute_worker_results_in_pool",
                side_effect=broken_pool,
            ), patch(
                "src.tree_diffusion.precompute_dataset._iter_precompute_isolated_worker_results",
                return_value=iter([fallback_result]),
            ) as isolated:
                results = list(
                    _iter_precompute_worker_batch_results(
                        tasks,
                        config=config,
                        batch_index=0,
                    )
                )

            self.assertEqual(results, [fallback_result])
            isolated.assert_called_once()

    def test_isolated_worker_records_worker_termination_as_failure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = _write_parquet(work_dir / "toy.parquet")
            config = TreeDiffusionPrecomputeConfig(
                input_data=str(parquet),
                output_dir=str(work_dir / "precomputed"),
                overwrite=True,
            )
            task = _PrecomputeTask("train", 0, _fake_pairs(1)[0], 0, 123)

            class _BrokenExecutor:
                def __init__(self, *args, **kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def submit(self, *args, **kwargs):
                    raise BrokenProcessPool("boom")

            with patch(
                "src.tree_diffusion.precompute_dataset.ProcessPoolExecutor",
                _BrokenExecutor,
            ):
                results = list(
                    _iter_precompute_isolated_worker_results(
                        [task],
                        config=config,
                        batch_index=0,
                    )
                )

            self.assertEqual(len(results), 1)
            self.assertIsNotNone(results[0].failure)
            assert results[0].failure is not None
            self.assertEqual(results[0].failure["failure_category"], "worker_process_terminated")


def _run_tiny_precompute(work_dir: Path) -> Path:
    return run_tiny_precompute(work_dir)


def _write_config(
    path: Path,
    *,
    input_data: Path,
    output_dir: Path,
    values: dict | None = None,
) -> Path:
    data = {
        "input_data": str(input_data),
        "output_dir": str(output_dir),
    }
    data.update(values or {})
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_parquet(path: Path) -> Path:
    return write_extended_toy_parquet(path)


def _write_resume_shard(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "input_length": 8,
                "target_length": 4,
                "example_index_for_pair": 0,
                "rng_seed": 123,
                "selected_node_id": 0,
                "used_random_init": False,
                "num_mutations": 1,
                "distance_before": 2,
                "distance_after": 1,
                "warnings_json": "[]",
                "observation_status": "ok",
            }
        ]
    ).to_parquet(path, index=False)


def _fake_pairs(count: int) -> list[IntegrationPair]:
    return [
        IntegrationPair(
            target_integrand=parse_prefix_string("x"),
            target_antiderivative=parse_prefix_string("div pow x INT+ 2 INT+ 2"),
            source="fake",
            index=index,
        )
        for index in range(count)
    ]


def _read_split_rows(output_dir: Path, split: str) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in sorted((output_dir / split).glob("shard_*.parquet"))]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    unittest.main()
