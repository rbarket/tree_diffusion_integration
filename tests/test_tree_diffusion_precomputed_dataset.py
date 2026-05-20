from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.tree_diffusion.dataset import make_tree_diffusion_dataloader
from src.tree_diffusion.precomputed_dataset import PrecomputedTreeDiffusionDataset
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.training.workflows.tree_diffusion import TreeDiffusionTrainingConfig, train_tree_diffusion_policy
from tests.tree_diffusion_test_utils import run_tiny_precompute, small_policy_model


class PrecomputedTreeDiffusionDatasetTests(unittest.TestCase):
    def test_dataset_reader_returns_online_batch_contract(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = _run_tiny_precompute(Path(temp_dir))
            dataset = PrecomputedTreeDiffusionDataset(output_dir, split="train", include_metadata=True)
            item = dataset[0]

            self.assertGreater(len(dataset), 0)
            for key in (
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
            ):
                self.assertIn(key, item)
                self.assertIsInstance(item[key], torch.Tensor)

            self.assertEqual(item["input_ids"].shape, (256,))
            self.assertEqual(item["target_ids"].shape, (64,))
            self.assertEqual(item["input_ids"].dtype, torch.long)
            self.assertEqual(item["labels"].dtype, torch.long)
            self.assertTrue(torch.equal(item["input_attention_mask"], item["input_ids"].ne(dataset.pad_id).long()))
            self.assertTrue(torch.equal(item["target_attention_mask"], item["target_ids"].ne(dataset.pad_id).long()))

            expected_labels = item["target_ids"].clone()
            expected_labels[item["target_ids"] == dataset.pad_id] = -100
            self.assertTrue(torch.equal(item["labels"], expected_labels))

    def test_shared_dataloader_batches_precomputed_rows_and_model_forward_works(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = _run_tiny_precompute(Path(temp_dir))
            tokenizer = TreeDiffusionTokenizer(max_positions=128)
            loader = make_tree_diffusion_dataloader(
                precomputed_data_dir=output_dir,
                precomputed_split="train",
                tokenizer=tokenizer,
                batch_size=2,
                num_workers=0,
                shuffle_pairs=False,
                include_metadata=True,
            )
            batch = next(iter(loader))

            self.assertEqual(batch["input_ids"].shape, (2, 256))
            self.assertEqual(batch["target_ids"].shape, (2, 64))
            self.assertEqual(batch["labels"].shape, (2, 64))
            self.assertEqual(len(batch["input_tokens"]), 2)

            model = _small_model(tokenizer)
            output = model(
                batch["input_ids"],
                batch["target_ids"],
                input_attention_mask=batch["input_attention_mask"],
                target_attention_mask=batch["target_attention_mask"],
                labels=batch["labels"],
            )

            self.assertIsNotNone(output.loss)
            assert output.loss is not None
            self.assertTrue(torch.isfinite(output.loss))

    def test_metadata_mode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = _run_tiny_precompute(Path(temp_dir))
            with_metadata = PrecomputedTreeDiffusionDataset(output_dir, split="train", include_metadata=True)[0]
            without_metadata = PrecomputedTreeDiffusionDataset(output_dir, split="train", include_metadata=False)[0]

            self.assertIn("input_tokens", with_metadata)
            self.assertIn("target_tokens", with_metadata)
            self.assertIn("current_prefix", with_metadata)
            self.assertIn("selected_node_id", with_metadata)
            self.assertIn("replacement_subtree_prefix", with_metadata)
            self.assertIn("distance_before", with_metadata)
            self.assertIn("warnings", with_metadata)
            self.assertTrue(with_metadata["target_tokens"][0].startswith("<POS_"))
            self.assertEqual(with_metadata["target_tokens"][-1], "<eos>")

            self.assertNotIn("input_tokens", without_metadata)
            self.assertNotIn("target_tokens", without_metadata)
            self.assertIn("input_ids", without_metadata)
            self.assertIn("labels", without_metadata)

    def test_trajectory_metadata_from_validation_shards_is_exposed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = _write_tiny_trajectory_precomputed(Path(temp_dir))
            dataset = PrecomputedTreeDiffusionDataset(output_dir, split="val", include_metadata=True)
            item = dataset[0]

            self.assertEqual(item["trajectory_mode"], "forward_and_repair")
            self.assertEqual(item["forward_num_mutations"], 1)
            self.assertEqual(item["forward_mutation_kinds"], ["local_const_edit"])
            self.assertTrue(item["repair_reached_target"])
            self.assertEqual(item["repair_step_count"], 1)
            self.assertEqual(item["repair_step_index"], 0)
            self.assertEqual(item["repair_mutation_kind"], "local_const_edit")
            self.assertEqual(item["repair_selected_node_id"], 2)
            self.assertEqual(item["repair_selected_node_span"], [2, 4])
            self.assertEqual(item["repair_replacement_subtree_prefix"], "INT+ 3")

            loader = make_tree_diffusion_dataloader(
                precomputed_data_dir=output_dir,
                precomputed_split="val",
                tokenizer=TreeDiffusionTokenizer(max_positions=128),
                batch_size=1,
                num_workers=0,
                shuffle_pairs=False,
                include_metadata=True,
            )
            batch = next(iter(loader))
            self.assertEqual(batch["trajectory_mode"], ["forward_and_repair"])
            self.assertEqual(batch["forward_mutation_kinds"], [["local_const_edit"]])
            self.assertEqual(batch["repair_mutation_kind"], ["local_const_edit"])
            self.assertEqual(batch["repair_selected_node_span"], [[2, 4]])

    def test_training_workflow_can_use_precomputed_data(self) -> None:
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            precomputed_dir = _run_tiny_precompute(work_dir)
            output_dir = work_dir / "run"
            config = TreeDiffusionTrainingConfig(
                train_data=None,
                precomputed_data_dir=str(precomputed_dir),
                use_precomputed=True,
                output_dir=str(output_dir),
                train_limit=2,
                val_limit=2,
                val_fraction=0.0,
                seed=123,
                device="cpu",
                num_epochs=1,
                batch_size=2,
                num_workers=0,
                max_input_length=256,
                max_target_length=64,
                max_positions=128,
                d_model=32,
                n_heads=4,
                d_ff=64,
                n_encoder_layers=1,
                n_decoder_layers=1,
                dropout=0.0,
                log_every=1,
                val_every=1,
                checkpoint_every=1,
                val_batches=1,
                diagnostic_batches=1,
            )

            summary = train_tree_diffusion_policy(config)

            self.assertEqual(summary["final_step"], 1)
            self.assertTrue((output_dir / "metrics.jsonl").exists())
            self.assertTrue((output_dir / "checkpoint_step_latest.pt").exists())
            self.assertTrue((output_dir / "lightning" / "last.ckpt").exists())
            rows = [
                json.loads(line)
                for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            losses = [float(row["loss"]) for row in rows if "loss" in row]
            self.assertTrue(losses)
            self.assertTrue(all(math.isfinite(loss) for loss in losses))


def _run_tiny_precompute(work_dir: Path) -> Path:
    return run_tiny_precompute(work_dir)


def _small_model(tokenizer: TreeDiffusionTokenizer):
    return small_policy_model(tokenizer, max_input_length=256, max_target_length=64)


def _write_tiny_trajectory_precomputed(work_dir: Path) -> Path:
    import pandas as pd

    tokenizer = TreeDiffusionTokenizer(max_positions=128)
    output_dir = work_dir / "precomputed"
    val_dir = output_dir / "val"
    val_dir.mkdir(parents=True)
    (output_dir / "tokenizer_metadata.json").write_text(
        json.dumps(
            {
                "vocab_size": tokenizer.vocab_size,
                "pad_id": tokenizer.pad_id,
                "bos_id": tokenizer.bos_id,
                "eos_id": tokenizer.eos_id,
                "unk_id": tokenizer.unk_id,
                "max_positions": tokenizer.max_positions,
                "numeric_log_min": tokenizer.numeric_log_min,
                "numeric_log_max": tokenizer.numeric_log_max,
            }
        ),
        encoding="utf-8",
    )

    input_tokens = [
        "<F>",
        "mul",
        "INT+",
        "3",
        "pow",
        "x",
        "INT+",
        "2",
        "</F>",
        "<CUR>",
        "pow",
        "x",
        "INT+",
        "5",
        "</CUR>",
        "<DER>",
        "<NO_DER>",
        "</DER>",
        "<RES>",
        "<NO_RES>",
        "</RES>",
        "<NUM>",
        "<NO_NUM>",
        "</NUM>",
        "<EDIT>",
    ]
    target_tokens = ["<POS_2>", "INT+", "3", "<eos>"]
    input_ids = tokenizer.encode_tokens(input_tokens, pad_to_length=64)
    target_ids = tokenizer.encode_tokens(target_tokens, pad_to_length=16)
    labels = [(-100 if token_id == tokenizer.pad_id else token_id) for token_id in target_ids]
    trajectory = {
        "mode": "forward_and_repair",
        "example_index_for_pair": 0,
        "pair_index": 0,
        "forward": {
            "complete": True,
            "start_prefix": "pow x INT+ 3",
            "end_prefix": "pow x INT+ 5",
            "num_mutations": 1,
            "used_random_init": False,
            "steps": [
                {
                    "step_index": 0,
                    "mutation_kind": "local_const_edit",
                    "selected_node_id": 2,
                    "selected_token_start": 2,
                    "selected_token_end": 4,
                    "original_subtree_prefix": "INT+ 3",
                    "replacement_subtree_prefix": "INT+ 5",
                    "before_prefix": "pow x INT+ 3",
                    "after_prefix": "pow x INT+ 5",
                }
            ],
        },
        "repair": {
            "max_steps": 64,
            "reached_target": True,
            "start_prefix": "pow x INT+ 5",
            "steps": [
                {
                    "step_index": 0,
                    "mutation_kind": "local_const_edit",
                    "reason": "direct_mismatch_target",
                    "selected_node_id": 2,
                    "selected_node_span": [2, 4],
                    "distance_before": 1,
                    "distance_after": 0,
                    "original_subtree_prefix": "INT+ 5",
                    "replacement_subtree_prefix": "INT+ 3",
                    "before_prefix": "pow x INT+ 5",
                    "after_prefix": "pow x INT+ 3",
                }
            ],
        },
    }
    row = {
        "split": "val",
        "global_example_index": 0,
        "pair_index": 0,
        "source": "unit-test",
        "example_index_for_pair": 0,
        "rng_seed": 123,
        "target_integrand_prefix": "mul INT+ 3 pow x INT+ 2",
        "target_antiderivative_prefix": "pow x INT+ 3",
        "current_antiderivative_prefix": "pow x INT+ 5",
        "current_derivative_prefix": None,
        "symbolic_residual_prefix": None,
        "input_tokens_json": json.dumps(input_tokens),
        "target_tokens_json": json.dumps(target_tokens),
        "input_ids_json": json.dumps(input_ids),
        "target_ids_json": json.dumps(target_ids),
        "labels_json": json.dumps(labels),
        "input_length": len(input_tokens),
        "target_length": len(target_tokens),
        "selected_node_id": 2,
        "replacement_subtree_prefix": "INT+ 3",
        "resulting_tree_prefix": "pow x INT+ 3",
        "num_mutations": 1,
        "used_random_init": False,
        "sampled_s": None,
        "distance_before": 1,
        "distance_after": 0,
        "label_validation_ok": True,
        "label_strict_improvement": True,
        "observation_status": "ok",
        "warnings_json": "[]",
        "trajectory_json": json.dumps(trajectory),
    }
    pd.DataFrame([row]).to_parquet(val_dir / "shard_00000.parquet", index=False)
    return output_dir


if __name__ == "__main__":
    unittest.main()
