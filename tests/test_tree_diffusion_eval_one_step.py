from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion.dataset import IntegrationPair, make_tree_diffusion_dataloader
from src.tree_diffusion.eval_one_step import (
    evaluate_one_step_edits,
    main as eval_one_step_main,
    numeric_residual_score,
)
from src.tree_diffusion.model import TreeDiffusionModelConfig, TreeDiffusionPolicyModel
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.validation import run_one_step_edit_diagnostics


class TreeDiffusionOneStepEvaluationTests(unittest.TestCase):
    def test_evaluate_one_step_edits_runs_on_tiny_untrained_model(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        loader = _real_loader(tokenizer)
        model = _small_model(tokenizer)

        summary = evaluate_one_step_edits(
            model,
            loader,
            tokenizer=tokenizer,
            device="cpu",
            num_batches=1,
            max_decode_length=4,
            compute_numeric_residual=False,
        )

        self.assertGreater(summary.examples, 0)
        for value in (
            summary.decoded_ok_rate,
            summary.valid_position_rate,
            summary.parseable_replacement_rate,
            summary.applicable_edit_rate,
            summary.structural_improvement_rate,
            summary.nonincreasing_structural_rate,
            summary.exact_target_rate,
        ):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_invalid_dummy_model_produces_zero_applicable_rate(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        loader = DataLoader([_synthetic_known_edit_batch(tokenizer)], batch_size=None)
        model = _FixedLogitModel(
            tokenizer,
            [{"x": 10.0}, {"<eos>": 10.0}],
            max_target_length=4,
        )

        summary = evaluate_one_step_edits(
            model,  # type: ignore[arg-type]
            loader,
            tokenizer=tokenizer,
            device="cpu",
            num_batches=1,
            constrain_position=False,
            max_decode_length=2,
            compute_numeric_residual=False,
        )

        self.assertEqual(summary.examples, 1)
        self.assertEqual(summary.applicable_edit_rate, 0.0)
        self.assertEqual(summary.status_counts.get("invalid_position_token"), 1)

    def test_correct_dummy_model_produces_exact_target(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        loader = DataLoader([_synthetic_known_edit_batch(tokenizer)], batch_size=None)
        model = _correct_edit_model(tokenizer)

        summary = evaluate_one_step_edits(
            model,  # type: ignore[arg-type]
            loader,
            tokenizer=tokenizer,
            device="cpu",
            num_batches=1,
            max_decode_length=4,
            compute_numeric_residual=False,
        )

        self.assertEqual(summary.examples, 1)
        self.assertEqual(summary.decoded_ok_rate, 1.0)
        self.assertEqual(summary.valid_position_rate, 1.0)
        self.assertEqual(summary.parseable_replacement_rate, 1.0)
        self.assertEqual(summary.applicable_edit_rate, 1.0)
        self.assertEqual(summary.structural_improvement_rate, 1.0)
        self.assertEqual(summary.nonincreasing_structural_rate, 1.0)
        self.assertEqual(summary.exact_target_rate, 1.0)
        self.assertEqual(summary.status_counts.get("ok"), 1)

    def test_numeric_residual_score_detects_improving_edit(self) -> None:
        target_integrand = parse_prefix_string("mul INT+ 3 pow x INT+ 2")
        current = parse_prefix_string("pow x INT+ 5")
        edited = parse_prefix_string("pow x INT+ 3")

        before = numeric_residual_score(current, target_integrand, probe_points=(1.0, 2.0))
        after = numeric_residual_score(edited, target_integrand, probe_points=(1.0, 2.0))

        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        assert before is not None and after is not None
        self.assertGreater(before, after)
        self.assertEqual(after, 0.0)

    def test_numeric_residual_score_ignores_nonfinite_probes(self) -> None:
        score = numeric_residual_score(
            parse_prefix_string("x"),
            parse_prefix_string("div INT+ 1 x"),
            probe_points=(0.0, 2.0),
        )

        self.assertIsNotNone(score)
        assert score is not None
        self.assertAlmostEqual(score, 0.25)

    def test_numeric_residual_score_returns_none_when_no_finite_probes_exist(self) -> None:
        score = numeric_residual_score(
            parse_prefix_string("x"),
            parse_prefix_string("div INT+ 1 x"),
            probe_points=(0.0,),
        )

        self.assertIsNone(score)

    def test_apply_failed_is_counted_without_mutating_decode_status(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        loader = DataLoader([_synthetic_known_edit_batch(tokenizer)], batch_size=None)
        model = _correct_edit_model(tokenizer)

        with patch(
            "src.tree_diffusion.decoding.apply_subtree_replacement_by_position",
            side_effect=KeyError("forced"),
        ):
            summary = evaluate_one_step_edits(
                model,  # type: ignore[arg-type]
                loader,
                tokenizer=tokenizer,
                device="cpu",
                num_batches=1,
                max_decode_length=4,
                compute_numeric_residual=False,
            )

        self.assertEqual(summary.decoded_ok_rate, 1.0)
        self.assertEqual(summary.applicable_edit_rate, 0.0)
        self.assertEqual(summary.status_counts.get("apply_failed"), 1)

    def test_cli_without_checkpoint_requires_explicit_random_init_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "allow-random-init-model"):
            eval_one_step_main(["--data", "unused.parquet"])

    def test_validation_diagnostics_preserve_existing_fields(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        loader = DataLoader([_synthetic_known_edit_batch(tokenizer)], batch_size=None)
        model = _correct_edit_model(tokenizer)

        summary = run_one_step_edit_diagnostics(
            model,  # type: ignore[arg-type]
            loader,
            tokenizer=tokenizer,
            device="cpu",
            num_batches=1,
        )

        for field_name in (
            "examples",
            "valid_position_rate",
            "parseable_replacement_rate",
            "applicable_edit_rate",
            "structural_improvement_rate",
            "numeric_residual_improvement_rate",
            "exact_target_rate",
            "mean_structural_distance_before",
            "mean_structural_distance_after",
        ):
            self.assertTrue(hasattr(summary, field_name), field_name)
        self.assertEqual(summary.examples, 1)
        self.assertEqual(summary.exact_target_rate, 1.0)


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


def _correct_edit_model(tokenizer: TreeDiffusionTokenizer) -> _FixedLogitModel:
    return _FixedLogitModel(
        tokenizer,
        [{"<POS_2>": 10.0}, {"INT+": 10.0}, {"3": 10.0}, {"<eos>": 10.0}],
        max_target_length=8,
    )


def _synthetic_known_edit_batch(tokenizer: TreeDiffusionTokenizer) -> dict:
    input_ids = torch.tensor([tokenizer.encode_tokens(["<EDIT>"], pad_to_length=8)], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "input_attention_mask": input_ids.ne(tokenizer.pad_id).long(),
        "target_ids": torch.tensor(
            [tokenizer.encode_tokens(["<POS_2>", "INT+", "3", "<eos>"], pad_to_length=8)],
            dtype=torch.long,
        ),
        "target_attention_mask": input_ids.ne(tokenizer.pad_id).long(),
        "labels": input_ids.clone(),
        "current_prefix": ["pow x INT+ 5"],
        "target_antiderivative_prefix": ["pow x INT+ 3"],
        "target_integrand_prefix": ["mul INT+ 3 pow x INT+ 2"],
    }


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


if __name__ == "__main__":
    unittest.main()
