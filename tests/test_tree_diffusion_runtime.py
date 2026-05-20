from __future__ import annotations

import unittest

import torch

from src.tree_diffusion.runtime import (
    batch_size,
    metadata_item,
    required_metadata,
    required_tensor,
    tensor_row,
    tokenizer_from_checkpoint,
)


class TreeDiffusionRuntimeTests(unittest.TestCase):
    def test_tensor_and_metadata_helpers_support_batched_rows(self) -> None:
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "current_prefix": ["x", "add x INT+ 1"],
            "score": torch.tensor([1.5, 2.5]),
        }

        self.assertEqual(batch_size(batch), 2)
        self.assertEqual(batch_size(batch["input_ids"]), 2)
        self.assertTrue(torch.equal(tensor_row(batch["input_ids"], 1), torch.tensor([4, 5, 6])))
        self.assertTrue(torch.equal(required_tensor(batch, "input_ids"), batch["input_ids"]))
        self.assertEqual(required_metadata(batch, "current_prefix", 1), "add x INT+ 1")
        self.assertEqual(metadata_item(batch, "score", 0), 1.5)
        self.assertEqual(metadata_item(batch, "missing", 0, default="fallback"), "fallback")

    def test_required_metadata_rejects_missing_none_and_short_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required metadata"):
            required_metadata({}, "current_prefix", 0)
        with self.assertRaisesRegex(ValueError, "contains None"):
            required_metadata({"current_prefix": [None]}, "current_prefix", 0)
        with self.assertRaisesRegex(ValueError, "shorter than the batch"):
            required_metadata({"current_prefix": []}, "current_prefix", 0)

    def test_tokenizer_from_checkpoint_uses_serialized_metadata(self) -> None:
        tokenizer = tokenizer_from_checkpoint(
            {
                "tokenizer": {
                    "max_positions": 99,
                    "numeric_log_min": -7,
                    "numeric_log_max": 8,
                }
            }
        )

        self.assertIsNotNone(tokenizer)
        assert tokenizer is not None
        self.assertEqual(tokenizer.max_positions, 99)
        self.assertEqual(tokenizer.numeric_log_min, -7)
        self.assertEqual(tokenizer.numeric_log_max, 8)


if __name__ == "__main__":
    unittest.main()
