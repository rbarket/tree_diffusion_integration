from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader

from src.tree_diffusion.beam_search import BeamSearchScoringConfig, BeamSearchStopConfig
from src.tree_diffusion.evaluate_beam_search import (
    BeamRepairEvaluationRecord,
    beam_repair_evaluation_summary_to_json,
    evaluate_beam_repair,
    main as evaluate_beam_main,
    summarize_beam_repair_results,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from tests.test_tree_diffusion_repair import (
    _correct_exponent_model,
    _invalid_replacement_model,
)
from tests.tree_diffusion_test_utils import (
    small_policy_model,
    tiny_training_config_values,
    write_toy_parquet,
)


class TreeDiffusionBeamRepairEvaluationTests(unittest.TestCase):
    def test_evaluate_beam_repair_summarizes_success_and_groups(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        loader = DataLoader([_known_repair_batch()], batch_size=None)
        model = _correct_exponent_model(tokenizer)

        summary = evaluate_beam_repair(
            model,  # type: ignore[arg-type]
            loader,
            tokenizer=tokenizer,
            device="cpu",
            num_batches=1,
            beam_size=2,
            candidate_k=1,
            max_decode_length=4,
        )

        self.assertEqual(summary.examples, 2)
        self.assertEqual(summary.success_rate, 1.0)
        self.assertEqual(summary.exact_symbolic_match_rate, 1.0)
        self.assertEqual(summary.numeric_success_rate, 1.0)
        self.assertEqual(summary.mean_steps_to_success, 1.0)
        self.assertGreater(summary.mean_expanded_states, 0.0)
        self.assertGreater(summary.mean_generated_candidates, 0.0)
        self.assertGreater(summary.mean_applicable_candidates, 0.0)
        self.assertEqual(summary.by_used_random_init["local_corruption"].examples, 1)
        self.assertEqual(summary.by_used_random_init["random_init"].examples, 1)
        self.assertEqual(summary.by_num_mutations["s=1"].examples, 1)
        self.assertEqual(summary.by_num_mutations["s=2"].examples, 1)
        self.assertEqual(
            sum(group.examples for group in summary.by_used_random_init.values()),
            summary.examples,
        )
        self.assertEqual(
            sum(group.examples for group in summary.by_num_mutations.values()),
            summary.examples,
        )
        _assert_rates_in_bounds(self, summary)

    def test_evaluate_beam_repair_counts_beam_empty(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        loader = DataLoader([_single_known_repair_batch()], batch_size=None)
        model = _invalid_replacement_model(tokenizer)

        summary = evaluate_beam_repair(
            model,  # type: ignore[arg-type]
            loader,
            tokenizer=tokenizer,
            device="cpu",
            num_batches=1,
            beam_size=1,
            candidate_k=1,
            max_decode_length=4,
        )

        self.assertEqual(summary.examples, 1)
        self.assertEqual(summary.success_rate, 0.0)
        self.assertEqual(summary.beam_empty_rate, 1.0)
        self.assertEqual(summary.stop_reason_counts.get("beam_empty"), 1)
        _assert_rates_in_bounds(self, summary)

    def test_summary_json_serializes_core_sections(self) -> None:
        summary = summarize_beam_repair_results(
            [
                BeamRepairEvaluationRecord(
                    result=_synthetic_beam_result(),
                    used_random_init=False,
                    num_mutations=1,
                    structural_distance_initial=2.0,
                    structural_distance_best=1.0,
                )
            ],
            beam_size=8,
            candidate_k=8,
            scoring=BeamSearchScoringConfig(),
            stopping=BeamSearchStopConfig(),
        )

        payload = beam_repair_evaluation_summary_to_json(summary)

        for key in (
            "examples",
            "beam_size",
            "candidate_k",
            "scoring_config",
            "stop_config",
            "overall",
            "by_used_random_init",
            "by_num_mutations",
            "stop_reason_counts",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["beam_size"], 8)
        self.assertEqual(payload["candidate_k"], 8)

    def test_cli_writes_summary_and_dump_examples(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = write_toy_parquet(work_dir / "toy.parquet")
            checkpoint = _write_tiny_checkpoint(work_dir / "checkpoint.pt", parquet)
            output = work_dir / "beam.json"
            dump = work_dir / "beam_examples.jsonl"

            result = evaluate_beam_main(
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
                    "--beam-size",
                    "1",
                    "--candidate-k",
                    "1",
                    "--numeric-patience",
                    "none",
                    "--structural-patience",
                    "none",
                    "--max-expanded-states",
                    "none",
                    "--timeout-seconds",
                    "none",
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
            self.assertEqual(payload["stop_config"]["numeric_patience"], None)
            self.assertEqual(payload["stop_config"]["structural_patience"], None)
            self.assertIn("overall", payload)
            self.assertIn("mean_expanded_states", payload["overall"])
            self.assertTrue(dump.exists())
            row = json.loads(dump.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("target_integrand_prefix", row)
            self.assertIn("best_prefix", row)
            self.assertIn("path", row)
            self.assertIn("per_depth_best_numeric_residual", row)


def _known_repair_batch() -> dict:
    return {
        "current_prefix": ["pow x INT+ 5", "pow x INT+ 5"],
        "target_integrand_prefix": [
            "mul INT+ 3 pow x INT+ 2",
            "mul INT+ 3 pow x INT+ 2",
        ],
        "target_antiderivative_prefix": ["pow x INT+ 3", "pow x INT+ 3"],
        "used_random_init": [False, True],
        "num_mutations": [1, 2],
    }


def _single_known_repair_batch() -> dict:
    return {
        "current_prefix": ["pow x INT+ 5"],
        "target_integrand_prefix": ["mul INT+ 3 pow x INT+ 2"],
        "target_antiderivative_prefix": ["pow x INT+ 3"],
    }


def _synthetic_beam_result():
    from src.tree_diffusion.beam_search import BeamSearchResult

    return BeamSearchResult(
        target_integrand_prefix="x",
        initial_prefix="initial",
        best_prefix="best",
        final_beam_prefixes=["best"],
        success=True,
        stop_reason="numeric_tol",
        steps_taken=1,
        expanded_states=1,
        generated_candidates=2,
        applicable_candidates=1,
        repeated_candidates=0,
        pruned_candidates=0,
        initial_numeric_residual=10.0,
        best_numeric_residual=1e-12,
        final_best_numeric_residual=1e-12,
        best_structural_distance=1,
        exact_symbolic_match=False,
        best_step_index=1,
        path=[],
        per_depth_best_numeric_residual=[10.0, 1e-12],
        per_depth_best_structural_distance=[2, 1],
        stop_diagnostics={},
    )


def _assert_rates_in_bounds(testcase: unittest.TestCase, summary) -> None:
    for name in (
        "success_rate",
        "exact_symbolic_match_rate",
        "numeric_success_rate",
        "beam_empty_rate",
        "max_steps_rate",
        "numeric_patience_rate",
        "structural_patience_rate",
        "timeout_rate",
    ):
        value = getattr(summary, name)
        testcase.assertGreaterEqual(value, 0.0, name)
        testcase.assertLessEqual(value, 1.0, name)
    for name in (
        "best_numeric_residual_improvement_rate",
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
