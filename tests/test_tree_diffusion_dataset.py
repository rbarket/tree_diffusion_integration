from __future__ import annotations

import itertools
import unittest

import torch

from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion.dataset import (
    IntegrationPair,
    TreeDiffusionBatchCollator,
    TreeDiffusionIterableDataset,
    make_tree_diffusion_dataloader,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


class TreeDiffusionDatasetTests(unittest.TestCase):
    def test_validation(self) -> None:
        pairs = _pairs()
        invalid_dataset_kwargs = (
            {"pairs": []},
            {"pairs": pairs, "rho": -0.1},
            {"pairs": pairs, "rho": 1.1},
            {"pairs": pairs, "sigma_small": 0},
            {"pairs": pairs, "smax": 0},
            {"pairs": pairs, "max_input_length": 0},
            {"pairs": pairs, "max_target_length": 0},
            {"pairs": pairs, "max_attempts": 0},
        )

        for kwargs in invalid_dataset_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    TreeDiffusionIterableDataset(**kwargs)

        with self.assertRaises(ValueError):
            make_tree_diffusion_dataloader(pairs, batch_size=0)

    def test_iterable_dataset_yields_one_valid_item(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        dataset = _dataset(tokenizer=tokenizer)
        item = next(iter(dataset))

        for key in (
            "input_ids",
            "input_attention_mask",
            "target_ids",
            "target_attention_mask",
            "labels",
            "num_mutations",
            "used_random_init",
            "input_length",
            "target_length",
        ):
            self.assertIn(key, item)

        self.assertEqual(item["input_ids"].shape, (256,))
        self.assertEqual(item["target_ids"].shape, (64,))
        self.assertEqual(item["input_ids"].dtype, torch.long)
        self.assertEqual(item["target_ids"].dtype, torch.long)
        self.assertEqual(item["labels"].dtype, torch.long)

        expected_input_mask = (item["input_ids"] != tokenizer.pad_id).long()
        expected_target_mask = (item["target_ids"] != tokenizer.pad_id).long()
        self.assertTrue(torch.equal(item["input_attention_mask"], expected_input_mask))
        self.assertTrue(torch.equal(item["target_attention_mask"], expected_target_mask))

        expected_labels = item["target_ids"].clone()
        expected_labels[item["target_ids"] == tokenizer.pad_id] = -100
        self.assertTrue(torch.equal(item["labels"], expected_labels))

    def test_token_structure_survives_metadata(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        item = next(iter(_dataset(tokenizer=tokenizer, include_metadata=True)))

        self.assertEqual(item["input_tokens"][-1], "<EDIT>")
        self.assertTrue(item["target_tokens"][0].startswith("<POS_"))
        self.assertEqual(item["target_tokens"][-1], "<eos>")
        self.assertEqual(
            tokenizer.decode_ids(item["input_ids"].tolist(), strip_pad=True),
            item["input_tokens"],
        )
        self.assertEqual(
            tokenizer.decode_ids(item["target_ids"].tolist(), strip_pad=True),
            item["target_tokens"],
        )

    def test_collator_stacks_tensors_correctly(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        iterator = iter(_dataset(tokenizer=tokenizer, include_metadata=True))
        items = list(itertools.islice(iterator, 4))

        batch = TreeDiffusionBatchCollator(tokenizer)(items)

        self.assertEqual(batch["input_ids"].shape, (4, 256))
        self.assertEqual(batch["target_ids"].shape, (4, 64))
        self.assertEqual(batch["labels"].shape, (4, 64))
        self.assertEqual(batch["input_attention_mask"].shape, (4, 256))
        self.assertEqual(batch["target_attention_mask"].shape, (4, 64))
        self.assertEqual(len(batch["input_tokens"]), 4)
        self.assertEqual(len(batch["target_tokens"]), 4)

        expected_labels = batch["target_ids"].clone()
        expected_labels[batch["target_ids"] == tokenizer.pad_id] = -100
        self.assertTrue(torch.equal(batch["labels"], expected_labels))

    def test_make_tree_diffusion_dataloader_returns_valid_batch(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        loader = make_tree_diffusion_dataloader(
            _pairs(),
            tokenizer=tokenizer,
            batch_size=3,
            num_workers=0,
            sigma_small=2,
            smax=2,
            rho=0.0,
            max_input_length=256,
            max_target_length=64,
            base_seed=123,
            shuffle_pairs=False,
        )

        batch = next(iter(loader))

        self.assertEqual(batch["input_ids"].shape, (3, 256))
        self.assertEqual(batch["target_ids"].shape, (3, 64))
        self.assertEqual(batch["labels"].shape, (3, 64))
        self.assertEqual(len(batch["input_tokens"]), 3)

    def test_deterministic_with_same_seed(self) -> None:
        first = list(itertools.islice(iter(_dataset(base_seed=321, shuffle_pairs=False)), 5))
        second = list(itertools.islice(iter(_dataset(base_seed=321, shuffle_pairs=False)), 5))

        self.assertEqual(
            [item["input_tokens"] for item in first],
            [item["input_tokens"] for item in second],
        )
        self.assertEqual(
            [item["target_tokens"] for item in first],
            [item["target_tokens"] for item in second],
        )
        self.assertEqual(
            [item["current_prefix"] for item in first],
            [item["current_prefix"] for item in second],
        )
        self.assertEqual(
            [int(item["pair_index"]) for item in first],
            [int(item["pair_index"]) for item in second],
        )

    def test_different_seed_differs_with_high_probability(self) -> None:
        first = list(itertools.islice(iter(_dataset(base_seed=1, shuffle_pairs=True)), 10))
        second = list(itertools.islice(iter(_dataset(base_seed=2, shuffle_pairs=True)), 10))

        first_signature = [(item["current_prefix"], item["target_tokens"]) for item in first]
        second_signature = [(item["current_prefix"], item["target_tokens"]) for item in second]

        self.assertNotEqual(first_signature, second_signature)

    def test_shuffle_false_cycles_in_order(self) -> None:
        items = list(itertools.islice(iter(_dataset(shuffle_pairs=False, smax=1, rho=0.0)), 6))

        self.assertEqual([int(item["pair_index"]) for item in items], [0, 1, 2, 0, 1, 2])

    def test_rho_behavior(self) -> None:
        mutation_items = list(itertools.islice(iter(_dataset(rho=0.0, max_random_size=4)), 10))
        random_items = list(itertools.islice(iter(_dataset(rho=1.0, max_random_size=4)), 10))

        self.assertTrue(all(not bool(item["used_random_init"]) for item in mutation_items))
        self.assertTrue(all(bool(item["used_random_init"]) for item in random_items))

    def test_residual_modes(self) -> None:
        for residual_mode in ("none", "symbolic", "numeric", "both"):
            with self.subTest(residual_mode=residual_mode):
                item = next(iter(_dataset(residual_mode=residual_mode)))
                residual_tokens = _section(item["input_tokens"], "<RES>", "</RES>")
                numeric_tokens = _section(item["input_tokens"], "<NUM>", "</NUM>")
                if residual_mode == "none":
                    self.assertEqual(residual_tokens, ["<NO_RES>"])
                    self.assertEqual(numeric_tokens, ["<NO_NUM>"])
                elif residual_mode == "symbolic":
                    self.assertNotEqual(residual_tokens, ["<NO_RES>"])
                    self.assertEqual(numeric_tokens, ["<NO_NUM>"])
                elif residual_mode == "numeric":
                    self.assertEqual(residual_tokens, ["<NO_RES>"])
                    self.assertNotEqual(numeric_tokens, ["<NO_NUM>"])
                else:
                    self.assertNotEqual(residual_tokens, ["<NO_RES>"])
                    self.assertNotEqual(numeric_tokens, ["<NO_NUM>"])

    def test_collator_rejects_empty_list(self) -> None:
        with self.assertRaises(ValueError):
            TreeDiffusionBatchCollator(TreeDiffusionTokenizer(max_positions=128))([])

    def test_sequence_too_long_propagates_clearly(self) -> None:
        dataset = _dataset(max_input_length=2, max_target_length=2, max_attempts=2)

        with self.assertRaisesRegex(RuntimeError, "Failed to generate a tree-diffusion dataset item"):
            next(iter(dataset))


def _pairs() -> list[IntegrationPair]:
    return [
        IntegrationPair(
            target_integrand=parse_prefix_string("pow x INT+ 2"),
            target_antiderivative=parse_prefix_string("div pow x INT+ 3 INT+ 3"),
            source="unit",
            index=0,
        ),
        IntegrationPair(
            target_integrand=parse_prefix_string("cos x"),
            target_antiderivative=parse_prefix_string("sin x"),
            source="unit",
            index=1,
        ),
        IntegrationPair(
            target_integrand=parse_prefix_string("exp x"),
            target_antiderivative=parse_prefix_string("exp x"),
            source="unit",
            index=2,
        ),
    ]


def _dataset(
    *,
    tokenizer: TreeDiffusionTokenizer | None = None,
    include_metadata: bool = True,
    base_seed: int = 123,
    shuffle_pairs: bool = True,
    sigma_small: int = 2,
    smax: int = 2,
    rho: float = 0.0,
    residual_mode: str = "both",
    max_input_length: int = 256,
    max_target_length: int = 64,
    max_attempts: int = 32,
    max_random_size: int | None = None,
) -> TreeDiffusionIterableDataset:
    return TreeDiffusionIterableDataset(
        _pairs(),
        tokenizer=tokenizer or TreeDiffusionTokenizer(max_positions=128),
        sigma_small=sigma_small,
        smax=smax,
        rho=rho,
        residual_mode=residual_mode,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        base_seed=base_seed,
        shuffle_pairs=shuffle_pairs,
        max_attempts=max_attempts,
        max_random_size=max_random_size,
        include_metadata=include_metadata,
    )


def _section(tokens: list[str], start_token: str, end_token: str) -> list[str]:
    start = tokens.index(start_token) + 1
    end = tokens.index(end_token, start)
    return tokens[start:end]


if __name__ == "__main__":
    unittest.main()
