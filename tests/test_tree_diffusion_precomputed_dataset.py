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
            self.assertTrue((output_dir / "checkpoint_last.pt").exists())
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


if __name__ == "__main__":
    unittest.main()
