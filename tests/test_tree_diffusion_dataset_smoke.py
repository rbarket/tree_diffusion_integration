from __future__ import annotations

import itertools
import unittest
from pathlib import Path

from src.tree_diffusion.dataset import (
    load_integration_pairs_from_parquet,
    make_tree_diffusion_dataloader,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_prefix_filtered.parquet"
SAMPLE_SIZE = 25


class TreeDiffusionDatasetSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_PATH.exists():
            raise unittest.SkipTest(f"Dataset artifact not found: {DATASET_PATH}")

        cls.pairs = load_integration_pairs_from_parquet(
            DATASET_PATH,
            limit=SAMPLE_SIZE,
        )
        if len(cls.pairs) < SAMPLE_SIZE:
            raise unittest.SkipTest(
                f"Only found {len(cls.pairs)} dataset pairs in {DATASET_PATH}."
            )

    def test_dataloader_batches_from_real_dataset_pairs(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=512)
        loader = make_tree_diffusion_dataloader(
            self.pairs,
            tokenizer=tokenizer,
            batch_size=4,
            num_workers=0,
            sigma_small=2,
            smax=3,
            rho=0.2,
            residual_mode="both",
            simplify_symbolic_residual=False,
            max_input_length=512,
            max_target_length=128,
            base_seed=123,
        )

        for batch in itertools.islice(iter(loader), 3):
            self.assertEqual(batch["input_ids"].shape, (4, 512))
            self.assertEqual(batch["target_ids"].shape, (4, 128))
            self.assertEqual(batch["labels"].shape, (4, 128))
            self.assertTrue((batch["input_attention_mask"].sum(dim=1) > 0).all())
            self.assertTrue((batch["target_attention_mask"].sum(dim=1) > 0).all())
            self.assertEqual(len(batch["input_tokens"]), 4)
            self.assertEqual(len(batch["target_tokens"]), 4)

            for input_tokens, target_tokens in zip(batch["input_tokens"], batch["target_tokens"]):
                self.assertEqual(input_tokens[-1], "<EDIT>")
                self.assertTrue(target_tokens[0].startswith("<POS_"))
                self.assertEqual(target_tokens[-1], "<eos>")

            self.assertEqual(
                tokenizer.decode_ids(batch["input_ids"][0].tolist(), strip_pad=True),
                batch["input_tokens"][0],
            )
            self.assertEqual(
                tokenizer.decode_ids(batch["target_ids"][0].tolist(), strip_pad=True),
                batch["target_tokens"][0],
            )

    def test_short_multi_worker_smoke(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=512)
        loader = make_tree_diffusion_dataloader(
            self.pairs[:8],
            tokenizer=tokenizer,
            batch_size=2,
            num_workers=2,
            sigma_small=2,
            smax=2,
            rho=0.2,
            residual_mode="both",
            simplify_symbolic_residual=False,
            max_input_length=512,
            max_target_length=128,
            base_seed=456,
        )

        batch = next(iter(loader))

        self.assertEqual(batch["input_ids"].shape, (2, 512))
        self.assertEqual(batch["target_ids"].shape, (2, 128))
        self.assertTrue((batch["input_attention_mask"].sum(dim=1) > 0).all())

if __name__ == "__main__":
    unittest.main()
