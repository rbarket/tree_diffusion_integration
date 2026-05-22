from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.tree_diffusion.experiments.one_step_inference_eval import (
    main as one_step_inference_main,
    run_one_step_inference_eval,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from tests.tree_diffusion_test_utils import (
    small_policy_model,
    tiny_training_config_values,
    write_toy_parquet,
)


class OneStepInferenceEvalRunnerTests(unittest.TestCase):
    def test_runner_writes_summary_modes_and_examples(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = write_toy_parquet(work_dir / "toy.parquet")
            checkpoint = _write_tiny_checkpoint(work_dir / "checkpoint.pt", parquet)
            output_dir = work_dir / "eval"

            summary = run_one_step_inference_eval(
                checkpoint=str(checkpoint),
                data=str(parquet),
                output_dir=output_dir,
                num_pairs=2,
                num_batches=1,
                batch_size=1,
                device="cpu",
                seed=123,
                max_decode_length=4,
                compute_numeric_residual=False,
                top_k_values=(2,),
                num_dump_examples=1,
            )

            summary_path = output_dir / "one_step_eval_summary.json"
            self.assertTrue(summary_path.exists())
            on_disk = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["checkpoint"], str(checkpoint))
            self.assertEqual(summary["checkpoint"], str(checkpoint))

            expected_modes = {
                "unconstrained_greedy",
                "position_constrained_greedy",
                "position_constrained_topk_2",
            }
            self.assertEqual(set(on_disk["metrics_by_mode"]), expected_modes)
            for mode_name in expected_modes:
                self.assertTrue((output_dir / f"{mode_name}.json").exists(), mode_name)
                metrics = on_disk["metrics_by_mode"][mode_name]["metrics"]
                for metric_name in (
                    "examples",
                    "decoded_ok_rate",
                    "valid_position_rate",
                    "parseable_replacement_rate",
                    "applicable_edit_rate",
                    "status_counts",
                    "structural_improvement_rate",
                    "nonincreasing_structural_rate",
                    "exact_target_rate",
                    "mean_structural_distance_before",
                    "mean_structural_distance_after",
                    "numeric_residual_improvement_rate",
                    "mean_numeric_residual_before",
                    "mean_numeric_residual_after",
                ):
                    self.assertIn(metric_name, metrics)
                for rate_name in (
                    "decoded_ok_rate",
                    "valid_position_rate",
                    "parseable_replacement_rate",
                    "applicable_edit_rate",
                    "structural_improvement_rate",
                    "nonincreasing_structural_rate",
                    "exact_target_rate",
                ):
                    self.assertGreaterEqual(metrics[rate_name], 0.0)
                    self.assertLessEqual(metrics[rate_name], 1.0)

            topk = on_disk["metrics_by_mode"]["position_constrained_topk_2"]
            for metric_name in (
                "any_decoded_ok_rate",
                "any_applicable_edit_rate",
                "first_applicable_rank_mean",
            ):
                self.assertIn(metric_name, topk["metrics"])
            for derived_name in (
                "applicable_edit_rate_delta_vs_position_constrained_greedy",
                "structural_improvement_rate_delta_vs_position_constrained_greedy",
                "numeric_residual_improvement_rate_delta_vs_position_constrained_greedy",
            ):
                self.assertIn(derived_name, topk["derived_comparison"])

            examples_path = output_dir / "examples.jsonl"
            self.assertTrue(examples_path.exists())
            first_line = examples_path.read_text(encoding="utf-8").splitlines()[0]
            example = json.loads(first_line)
            self.assertIn("current_antiderivative_prefix", example)
            self.assertIn("target_antiderivative_prefix", example)
            self.assertIn("predicted_candidates", example)
            self.assertGreater(len(example["predicted_candidates"]), 0)
            self.assertIn("rank", example["predicted_candidates"][0])
            self.assertIn("generated_tokens", example["predicted_candidates"][0])
            self.assertIn("status", example["predicted_candidates"][0])

    def test_cli_main_writes_combined_summary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = write_toy_parquet(work_dir / "toy.parquet")
            checkpoint = _write_tiny_checkpoint(work_dir / "checkpoint.pt", parquet)
            output_dir = work_dir / "eval"

            result = one_step_inference_main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--data",
                    str(parquet),
                    "--output-dir",
                    str(output_dir),
                    "--num-pairs",
                    "2",
                    "--num-batches",
                    "1",
                    "--batch-size",
                    "1",
                    "--device",
                    "cpu",
                    "--max-decode-length",
                    "4",
                    "--top-k-values",
                    "2",
                    "--num-dump-examples",
                    "0",
                    "--no-compute-numeric-residual",
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "one_step_eval_summary.json").exists())

    def test_cli_main_accepts_config_and_cli_overrides(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = write_toy_parquet(work_dir / "toy.parquet")
            checkpoint = _write_tiny_checkpoint(work_dir / "checkpoint.pt", parquet)
            config_output_dir = work_dir / "config_eval"
            override_output_dir = work_dir / "override_eval"
            config_path = work_dir / "one_step_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "data": str(parquet),
                        "output_dir": str(config_output_dir),
                        "num_pairs": 2,
                        "num_batches": 1,
                        "batch_size": 1,
                        "device": "cpu",
                        "max_decode_length": 4,
                        "top_k_values": [2],
                        "num_dump_examples": 0,
                        "compute_numeric_residual": False,
                    }
                ),
                encoding="utf-8",
            )

            result = one_step_inference_main(
                [
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(override_output_dir),
                ]
            )

            self.assertEqual(result, 0)
            self.assertFalse((config_output_dir / "one_step_eval_summary.json").exists())
            self.assertTrue((override_output_dir / "one_step_eval_summary.json").exists())


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
