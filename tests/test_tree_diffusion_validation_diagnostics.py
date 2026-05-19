from __future__ import annotations

import importlib.util
import math
from types import SimpleNamespace
import unittest

import torch
from torch.utils.data import DataLoader

from src.tree_diffusion._common import diagnostic_metrics
from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion.dataset import IntegrationPair, make_tree_diffusion_dataloader
from src.tree_diffusion.edit_path import structural_distance
from src.tree_diffusion.model import TreeDiffusionModelConfig, TreeDiffusionPolicyModel
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.validation import run_one_step_edit_diagnostics


class TreeDiffusionValidationDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_do_not_crash_on_untrained_model(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        loader = _real_loader(tokenizer)
        model = _small_model(tokenizer)

        summary = run_one_step_edit_diagnostics(
            model,
            loader,
            tokenizer=tokenizer,
            device="cpu",
            num_batches=1,
        )

        self.assertGreater(summary.examples, 0)
        for value in (
            summary.valid_position_rate,
            summary.parseable_replacement_rate,
            summary.applicable_edit_rate,
            summary.structural_improvement_rate,
            summary.exact_target_rate,
        ):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        if summary.mean_structural_distance_before is not None:
            self.assertTrue(math.isfinite(summary.mean_structural_distance_before))

    def test_diagnostics_handle_invalid_predictions(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        loader = _real_loader(tokenizer)
        model = _DummyTokenModel(tokenizer, ["<POS_0>", "add", "x", "<eos>"])

        summary = run_one_step_edit_diagnostics(
            model,  # type: ignore[arg-type]
            loader,
            tokenizer=tokenizer,
            device="cpu",
            num_batches=1,
        )

        self.assertEqual(summary.valid_position_rate, 1.0)
        self.assertEqual(summary.parseable_replacement_rate, 0.0)
        self.assertEqual(summary.applicable_edit_rate, 0.0)

    def test_diagnostics_detect_valid_known_edit(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        batch = _synthetic_known_edit_batch(tokenizer)
        loader = DataLoader([batch], batch_size=None)
        model = _DummyTokenModel(tokenizer, ["<POS_2>", "INT+", "3", "<eos>"])

        summary = run_one_step_edit_diagnostics(
            model,  # type: ignore[arg-type]
            loader,
            tokenizer=tokenizer,
            device="cpu",
            num_batches=1,
        )

        self.assertEqual(summary.examples, 1)
        self.assertEqual(summary.valid_position_rate, 1.0)
        self.assertEqual(summary.parseable_replacement_rate, 1.0)
        self.assertEqual(summary.applicable_edit_rate, 1.0)
        self.assertEqual(summary.structural_improvement_rate, 1.0)
        self.assertEqual(summary.exact_target_rate, 1.0)

    def test_diagnostics_expose_top_k_candidate_metrics(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        batch = _synthetic_known_edit_batch(tokenizer)
        loader = DataLoader([batch], batch_size=None)
        model = _FixedLogitModel(
            tokenizer,
            [
                {"<POS_2>": 10.0},
                {"pow": 10.0, "INT+": 9.0},
                {"3": 10.0},
                {"<eos>": 10.0},
            ],
            max_target_length=8,
        )

        summary = run_one_step_edit_diagnostics(
            model,  # type: ignore[arg-type]
            loader,
            tokenizer=tokenizer,
            device="cpu",
            num_batches=1,
            candidate_k=2,
            use_first_applicable_candidate=True,
        )
        metrics = diagnostic_metrics(summary)

        self.assertEqual(summary.any_decoded_ok_rate, 1.0)
        self.assertEqual(summary.any_applicable_edit_rate, 1.0)
        self.assertEqual(summary.first_applicable_rank_mean, 2.0)
        self.assertEqual(metrics["any_decoded_ok_rate"], 1.0)
        self.assertEqual(metrics["any_applicable_edit_rate"], 1.0)
        self.assertEqual(metrics["first_applicable_rank_mean"], 2.0)

    def test_lightning_callback_import_still_succeeds(self) -> None:
        if importlib.util.find_spec("lightning") is None:
            self.skipTest("lightning is not installed in this environment")

        from src.training.lightning import tree_diffusion_callbacks

        self.assertTrue(hasattr(tree_diffusion_callbacks, "TreeDiffusionTrainingCallback"))

    def test_public_structural_distance_wrapper(self) -> None:
        x = parse_prefix_string("x")
        five = parse_prefix_string("INT+ 5")
        three = parse_prefix_string("INT+ 3")

        self.assertEqual(structural_distance(x, x), 0)
        self.assertGreaterEqual(structural_distance(five, three), 1)


class _DummyTokenModel(torch.nn.Module):
    def __init__(self, tokenizer: TreeDiffusionTokenizer, tokens: list[str]) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.config = SimpleNamespace(max_target_length=max(len(tokens), 1))
        self._decode_step = 0

    def encode(
        self,
        input_ids: torch.Tensor,
        input_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del input_attention_mask
        memory = torch.zeros(
            (input_ids.size(0), input_ids.size(1), 1),
            device=input_ids.device,
            dtype=torch.float32,
        )
        padding = torch.zeros(input_ids.shape, device=input_ids.device, dtype=torch.bool)
        return memory, padding

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        *,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        target_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del memory, memory_padding_mask, target_attention_mask
        self._decode_step = decoder_input_ids.size(1) - 1
        return torch.zeros(
            (decoder_input_ids.size(0), decoder_input_ids.size(1), 1),
            device=decoder_input_ids.device,
            dtype=torch.float32,
        )

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = hidden.new_full((hidden.size(0), hidden.size(1), self.tokenizer.vocab_size), -1000.0)
        token = self.tokens[self._decode_step] if self._decode_step < len(self.tokens) else "<eos>"
        logits[:, -1, self.tokenizer.token_to_id[token]] = 10.0
        return logits


class _FixedLogitModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer: TreeDiffusionTokenizer,
        step_scores: list[dict[str, float]],
        *,
        max_target_length: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.step_scores = step_scores
        self.config = SimpleNamespace(max_target_length=max_target_length)
        self._decode_step = 0

    def encode(
        self,
        input_ids: torch.Tensor,
        input_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del input_attention_mask
        memory = torch.zeros(
            (input_ids.size(0), input_ids.size(1), 1),
            device=input_ids.device,
            dtype=torch.float32,
        )
        padding = torch.zeros(input_ids.shape, device=input_ids.device, dtype=torch.bool)
        return memory, padding

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        *,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        target_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del memory, memory_padding_mask, target_attention_mask
        self._decode_step = decoder_input_ids.size(1) - 1
        return torch.zeros(
            (decoder_input_ids.size(0), decoder_input_ids.size(1), 1),
            device=decoder_input_ids.device,
            dtype=torch.float32,
        )

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = hidden.new_full((hidden.size(0), hidden.size(1), self.tokenizer.vocab_size), -1000.0)
        step_scores = (
            self.step_scores[self._decode_step]
            if self._decode_step < len(self.step_scores)
            else {"<eos>": 0.0}
        )
        for token, score in step_scores.items():
            logits[:, -1, self.tokenizer.token_to_id[token]] = float(score)
        return logits


def _real_loader(tokenizer: TreeDiffusionTokenizer) -> DataLoader:
    pairs = [
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
    ]
    return make_tree_diffusion_dataloader(
        pairs,
        tokenizer=tokenizer,
        batch_size=2,
        num_workers=0,
        sigma_small=2,
        smax=2,
        rho=0.2,
        max_input_length=128,
        max_target_length=32,
        base_seed=123,
        include_metadata=True,
    )


def _small_model(tokenizer: TreeDiffusionTokenizer) -> TreeDiffusionPolicyModel:
    return TreeDiffusionPolicyModel(
        TreeDiffusionModelConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_id,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            max_input_length=128,
            max_target_length=32,
            d_model=32,
            n_heads=4,
            d_ff=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            dropout=0.0,
        )
    )


def _synthetic_known_edit_batch(tokenizer: TreeDiffusionTokenizer) -> dict:
    target_tokens = ["<POS_2>", "INT+", "3", "<eos>"]
    input_ids = torch.tensor([tokenizer.encode_tokens(["<EDIT>"], pad_to_length=8)], dtype=torch.long)
    target_ids = torch.tensor([tokenizer.encode_tokens(target_tokens, pad_to_length=8)], dtype=torch.long)
    labels = target_ids.clone()
    labels[target_ids.eq(tokenizer.pad_id)] = -100
    return {
        "input_ids": input_ids,
        "input_attention_mask": input_ids.ne(tokenizer.pad_id).long(),
        "target_ids": target_ids,
        "target_attention_mask": target_ids.ne(tokenizer.pad_id).long(),
        "labels": labels,
        "current_prefix": ["pow x INT+ 5"],
        "target_antiderivative_prefix": ["pow x INT+ 3"],
        "target_integrand_prefix": ["mul INT+ 3 pow x INT+ 2"],
    }


if __name__ == "__main__":
    unittest.main()
