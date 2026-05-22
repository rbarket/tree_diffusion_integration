from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader

from src.tree_diffusion.evaluate_repair import (
    evaluate_greedy_repair,
    main as evaluate_repair_main,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from tests.test_tree_diffusion_repair import (
    _correct_exponent_model,
    _invalid_replacement_model,
    _worse_exponent_model,
)
from tests.tree_diffusion_test_utils import (
    small_policy_model,
    tiny_training_config_values,
    write_toy_parquet,
)


class TreeDiffusionRepairEvaluationTests(unittest.TestCase):
    def test_evaluate_greedy_repair_summarizes_success(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        loader = DataLoader([_known_repair_batch()], batch_size=None)
        model = _correct_exponent_model(tokenizer)

        summary = evaluate_greedy_repair(
            model,  # type: ignore[arg-type]
            loader,
            tokenizer=tokenizer,
            device="cpu",
            num_batches=1,
            max_steps=3,
            candidate_k=1,
            max_decode_length=4,
        )

        self.assertEqual(summary.examples, 1)
        self.assertEqual(summary.success_rate, 1.0)
        self.assertEqual(summary.exact_symbolic_match_rate, 1.0)
        self.assertEqual(summary.numeric_success_rate, 1.0)
        self.assertEqual(summary.mean_steps_to_success, 1.0)
        self.assertEqual(summary.median_steps_to_success, 1.0)
        self.assertEqual(summary.structural_distance_improvement_rate, 1.0)
        self.assertEqual(summary.numeric_residual_improvement_rate, 1.0)
        self.assertEqual(summary.stop_reason_counts.get("exact_symbolic_match"), 1)
        _assert_rates_in_bounds(self, summary)

    def test_evaluate_greedy_repair_counts_no_candidate(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        loader = DataLoader([_known_repair_batch()], batch_size=None)
        model = _invalid_replacement_model(tokenizer)

        summary = evaluate_greedy_repair(
            model,  # type: ignore[arg-type]
            loader,
            tokenizer=tokenizer,
            device="cpu",
            num_batches=1,
            max_steps=3,
            candidate_k=1,
            max_decode_length=4,
        )

        self.assertEqual(summary.examples, 1)
        self.assertEqual(summary.success_rate, 0.0)
        self.assertEqual(summary.no_candidate_rate, 1.0)
        self.assertEqual(summary.stop_reason_counts.get("no_applicable_candidate"), 1)
        _assert_rates_in_bounds(self, summary)

    def test_evaluate_greedy_repair_counts_no_numeric_improvement(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        loader = DataLoader([_known_repair_batch()], batch_size=None)
        model = _worse_exponent_model(tokenizer)

        summary = evaluate_greedy_repair(
            model,  # type: ignore[arg-type]
            loader,
            tokenizer=tokenizer,
            device="cpu",
            num_batches=1,
            max_steps=3,
            candidate_k=1,
            max_decode_length=4,
            patience=1,
            selection_strategy="rank1",
        )

        self.assertEqual(summary.examples, 1)
        self.assertEqual(summary.success_rate, 0.0)
        self.assertEqual(summary.no_numeric_improvement_rate, 1.0)
        self.assertEqual(summary.stop_reason_counts.get("no_numeric_improvement"), 1)
        _assert_rates_in_bounds(self, summary)

    def test_cli_writes_summary_and_dump_examples(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = write_toy_parquet(work_dir / "toy.parquet")
            checkpoint = _write_tiny_checkpoint(work_dir / "checkpoint.pt", parquet)
            output = work_dir / "repair.json"
            dump = work_dir / "repair_examples.jsonl"

            result = evaluate_repair_main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--data",
                    str(parquet),
                    "--output",
                    str(output),
                    "--num-pairs",
                    "2",
                    "--num-batches",
                    "1",
                    "--batch-size",
                    "1",
                    "--device",
                    "cpu",
                    "--max-steps",
                    "1",
                    "--candidate-k",
                    "1",
                    "--patience",
                    "1",
                    "--selection-strategy",
                    "rank1",
                    "--residual-workers",
                    "0",
                    "--no-structural-metrics",
                    "--dump-examples",
                    str(dump),
                    "--num-dump-examples",
                    "1",
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue(output.exists())
            payload = json.loads(output.read_text(encoding="utf-8"))
            for key in (
                "examples",
                "selection_strategy",
                "candidate_k",
                "overall",
                "by_used_random_init",
                "by_num_mutations",
                "best_so_far",
                "per_step",
                "candidate_rank",
                "dump_examples",
                "residual_workers",
            ):
                self.assertIn(key, payload)
            self.assertEqual(payload["residual_workers"], 0)
            self.assertIn("success_rate", payload["overall"])
            self.assertIn("stop_reason_counts", payload)
            self.assertTrue(dump.exists())
            row = json.loads(dump.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("target_integrand_prefix", row)
            self.assertIn("target_antiderivative_prefix", row)
            self.assertIn("initial_prefix", row)
            self.assertIn("final_prefix", row)
            self.assertIn("best_numeric_residual", row)
            self.assertIn("best_prefix", row)
            self.assertIn("best_step_index", row)
            self.assertIn("steps", row)
            if row["steps"]:
                self.assertIn("candidate_rank", row["steps"][0])
                self.assertIn("best_numeric_residual_so_far", row["steps"][0])

    def test_cli_config_values_can_be_overridden(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = write_toy_parquet(work_dir / "toy.parquet")
            checkpoint = _write_tiny_checkpoint(work_dir / "checkpoint.pt", parquet)
            config_path = work_dir / "greedy_config.json"
            output = work_dir / "from_config.json"
            override_output = work_dir / "override.json"
            config_path.write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "data": str(parquet),
                        "output": str(output),
                        "num_pairs": 2,
                        "num_batches": 1,
                        "batch_size": 1,
                        "device": "cpu",
                        "max_steps": 1,
                        "candidate_k": 1,
                        "patience": 1,
                        "selection_strategy": "rank1",
                        "residual_workers": 0,
                        "compute_structural_metrics": False,
                    }
                ),
                encoding="utf-8",
            )

            result = evaluate_repair_main(
                [
                    "--config",
                    str(config_path),
                    "--output",
                    str(override_output),
                ]
            )

            self.assertEqual(result, 0)
            self.assertFalse(output.exists())
            self.assertTrue(override_output.exists())


def _known_repair_batch() -> dict:
    return {
        "current_prefix": ["pow x INT+ 5"],
        "target_integrand_prefix": ["mul INT+ 3 pow x INT+ 2"],
        "target_antiderivative_prefix": ["pow x INT+ 3"],
    }


def _assert_rates_in_bounds(testcase: unittest.TestCase, summary) -> None:
    for name in (
        "success_rate",
        "exact_symbolic_match_rate",
        "numeric_success_rate",
        "no_candidate_rate",
        "repeated_state_rate",
        "max_steps_rate",
        "no_numeric_improvement_rate",
    ):
        value = getattr(summary, name)
        testcase.assertGreaterEqual(value, 0.0, name)
        testcase.assertLessEqual(value, 1.0, name)
    for name in (
        "numeric_residual_improvement_rate",
        "structural_distance_improvement_rate",
    ):
        value = getattr(summary, name)
        if value is not None:
            testcase.assertGreaterEqual(value, 0.0, name)
            testcase.assertLessEqual(value, 1.0, name)


def _write_tiny_checkpoint(path: Path, parquet: Path) -> Path:
    torch.manual_seed(123)
    tokenizer = TreeDiffusionTokenizer(max_positions=128)
    model = small_policy_model(tokenizer)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": tiny_training_config_values(parquet),
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "max_positions": tokenizer.max_positions,
            "pad_id": tokenizer.pad_id,
            "bos_id": tokenizer.bos_id,
            "eos_id": tokenizer.eos_id,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


if __name__ == "__main__":
    unittest.main()
