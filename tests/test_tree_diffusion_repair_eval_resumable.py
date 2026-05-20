from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.tree_diffusion.experiments.repair_eval_resumable import (
    main as repair_eval_resumable_main,
    run_resumable_greedy_repair_eval,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from tests.tree_diffusion_test_utils import (
    small_policy_model,
    tiny_training_config_values,
    write_toy_parquet,
)


class ResumableRepairEvalTests(unittest.TestCase):
    def test_runner_writes_parts_manifest_and_combined_summary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = write_toy_parquet(work_dir / "toy.parquet")
            checkpoint = _write_tiny_checkpoint(work_dir / "checkpoint.pt", parquet)
            output_dir = work_dir / "repair_eval"

            summary = run_resumable_greedy_repair_eval(
                checkpoint=str(checkpoint),
                data=str(parquet),
                output_dir=output_dir,
                num_pairs=3,
                batch_size=1,
                device="cpu",
                max_steps=0,
                candidate_k=1,
                part_size=2,
                progress=False,
                progress_every=0,
            )

            self.assertEqual(summary["examples"], 3)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["completed_examples"], 3)
            self.assertEqual(summary["target_examples"], 3)
            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "repair_eval_summary.json").exists())

            parts = sorted((output_dir / "parts").glob("part_*.jsonl"))
            self.assertEqual(len(parts), 2)
            self.assertEqual(_line_count(parts[0]), 2)
            self.assertEqual(_line_count(parts[1]), 1)

            first_row = json.loads(parts[0].read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first_row["example_index"], 0)
            self.assertIn("result", first_row)
            self.assertIn("used_random_init", first_row)
            self.assertIn("num_mutations", first_row)

    def test_resume_skips_completed_part_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = write_toy_parquet(work_dir / "toy.parquet")
            checkpoint = _write_tiny_checkpoint(work_dir / "checkpoint.pt", parquet)
            output_dir = work_dir / "repair_eval"

            partial = run_resumable_greedy_repair_eval(
                checkpoint=str(checkpoint),
                data=str(parquet),
                output_dir=output_dir,
                num_pairs=2,
                batch_size=1,
                device="cpu",
                max_steps=0,
                candidate_k=1,
                part_size=1,
                max_examples_this_run=1,
                progress=False,
                progress_every=1,
            )
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["completed_examples"], 1)

            completed = run_resumable_greedy_repair_eval(
                checkpoint=str(checkpoint),
                data=str(parquet),
                output_dir=output_dir,
                num_pairs=2,
                batch_size=1,
                device="cpu",
                max_steps=0,
                candidate_k=1,
                part_size=1,
                resume=True,
                progress=False,
                progress_every=1,
            )

            self.assertTrue(completed["complete"])
            self.assertEqual(completed["examples"], 2)
            self.assertEqual(completed["completed_examples"], 2)
            rows = [
                json.loads(line)
                for path in sorted((output_dir / "parts").glob("part_*.jsonl"))
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["example_index"] for row in rows], [0, 1])

    def test_cli_runs_resumable_eval(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = write_toy_parquet(work_dir / "toy.parquet")
            checkpoint = _write_tiny_checkpoint(work_dir / "checkpoint.pt", parquet)
            output_dir = work_dir / "repair_eval"

            result = repair_eval_resumable_main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--data",
                    str(parquet),
                    "--output-dir",
                    str(output_dir),
                    "--num-pairs",
                    "2",
                    "--batch-size",
                    "1",
                    "--device",
                    "cpu",
                    "--max-steps",
                    "0",
                    "--candidate-k",
                    "1",
                    "--part-size",
                    "1",
                    "--progress-every",
                    "1",
                    "--flush-every",
                    "1",
                    "--quiet",
                ]
            )

            self.assertEqual(result, 0)
            payload = json.loads((output_dir / "repair_eval_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["examples"], 2)
            self.assertTrue(payload["complete"])

    def test_cli_accepts_config_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = write_toy_parquet(work_dir / "toy.parquet")
            checkpoint = _write_tiny_checkpoint(work_dir / "checkpoint.pt", parquet)
            output_dir = work_dir / "repair_eval"
            config = work_dir / "repair_config.json"
            config.write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "data": str(parquet),
                        "output_dir": str(output_dir),
                        "num_pairs": 2,
                        "batch_size": 1,
                        "device": "cpu",
                        "max_steps": 0,
                        "candidate_k": 1,
                        "part_size": 1,
                        "progress_every": 1,
                        "flush_every": 1,
                    }
                ),
                encoding="utf-8",
            )

            result = repair_eval_resumable_main(["--config", str(config), "--quiet"])

            self.assertEqual(result, 0)
            payload = json.loads((output_dir / "repair_eval_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["examples"], 2)
            self.assertTrue(payload["complete"])

    def test_resume_allows_part_and_progress_cadence_changes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            parquet = write_toy_parquet(work_dir / "toy.parquet")
            checkpoint = _write_tiny_checkpoint(work_dir / "checkpoint.pt", parquet)
            output_dir = work_dir / "repair_eval"

            run_resumable_greedy_repair_eval(
                checkpoint=str(checkpoint),
                data=str(parquet),
                output_dir=output_dir,
                num_pairs=2,
                batch_size=1,
                device="cpu",
                max_steps=0,
                candidate_k=1,
                part_size=1,
                max_examples_this_run=1,
                progress=False,
                progress_every=1,
                flush_every=1,
            )
            completed = run_resumable_greedy_repair_eval(
                checkpoint=str(checkpoint),
                data=str(parquet),
                output_dir=output_dir,
                num_pairs=2,
                batch_size=1,
                device="cpu",
                max_steps=0,
                candidate_k=1,
                part_size=500,
                resume=True,
                progress=False,
                progress_every=25,
                flush_every=500,
            )

            self.assertTrue(completed["complete"])
            self.assertEqual(completed["examples"], 2)


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


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    unittest.main()
