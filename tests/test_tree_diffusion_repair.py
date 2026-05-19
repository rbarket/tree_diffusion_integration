from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion.repair import (
    RepairScoringConfig,
    derivative_matches_target,
    greedy_repair,
    greedy_repair_from_seeds,
    score_repair_candidate,
    tree_size,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


class TreeDiffusionRepairTests(unittest.TestCase):
    def test_score_repair_candidate_prioritizes_numeric_then_size_then_policy(self) -> None:
        config = RepairScoringConfig(lambda_size=1e-3, lambda_policy=1e-2)

        self.assertLess(
            score_repair_candidate(
                numeric_residual=1.0,
                tree_size_value=100,
                policy_logprob=-10.0,
                config=config,
            ),
            score_repair_candidate(
                numeric_residual=2.0,
                tree_size_value=1,
                policy_logprob=0.0,
                config=config,
            ),
        )
        self.assertLess(
            score_repair_candidate(
                numeric_residual=1.0,
                tree_size_value=1,
                policy_logprob=0.0,
                config=config,
            ),
            score_repair_candidate(
                numeric_residual=1.0,
                tree_size_value=2,
                policy_logprob=0.0,
                config=config,
            ),
        )
        self.assertLess(
            score_repair_candidate(
                numeric_residual=1.0,
                tree_size_value=1,
                policy_logprob=-1.0,
                config=config,
            ),
            score_repair_candidate(
                numeric_residual=1.0,
                tree_size_value=1,
                policy_logprob=-2.0,
                config=config,
            ),
        )

    def test_tree_size_counts_ast_nodes(self) -> None:
        self.assertEqual(tree_size(parse_prefix_string("pow x INT+ 5")), 3)

    def test_derivative_matches_target(self) -> None:
        self.assertTrue(
            derivative_matches_target(
                parse_prefix_string("div pow x INT+ 3 INT+ 3"),
                parse_prefix_string("pow x INT+ 2"),
            )
        )
        self.assertFalse(
            derivative_matches_target(
                parse_prefix_string("pow x INT+ 5"),
                parse_prefix_string("pow x INT+ 2"),
            )
        )

    def test_greedy_repair_succeeds_in_one_step(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _correct_exponent_model(tokenizer)

        result = greedy_repair(
            model,  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            max_steps=3,
            candidate_k=1,
            max_decode_length=4,
            target_antiderivative=parse_prefix_string("pow x INT+ 3"),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "exact_symbolic_match")
        self.assertEqual(result.steps_taken, 1)
        self.assertEqual(result.final_prefix, "pow x INT+ 3")
        self.assertEqual(result.steps[0].candidate_rank, 1)
        self.assertLess(
            result.steps[0].numeric_residual_after,
            result.steps[0].numeric_residual_before,
        )

    def test_greedy_repair_skips_invalid_top_candidate(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _invalid_then_correct_candidate_model(tokenizer)

        result = greedy_repair(
            model,  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            max_steps=3,
            candidate_k=2,
            max_decode_length=4,
            target_antiderivative=parse_prefix_string("pow x INT+ 3"),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.final_prefix, "pow x INT+ 3")
        self.assertEqual(result.steps[0].candidate_rank, 2)
        self.assertEqual(result.steps[0].decoded_status, "ok")

    def test_greedy_repair_stops_on_repeated_state(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _same_exponent_model(tokenizer)

        result = greedy_repair(
            model,  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            max_steps=3,
            candidate_k=1,
            max_decode_length=4,
        )

        self.assertFalse(result.success)
        self.assertTrue(result.repeated_state)
        self.assertEqual(result.stop_reason, "repeated_state")
        self.assertEqual(result.steps_taken, 0)
        self.assertEqual(result.steps[0].decoded_status, "ok")

    def test_greedy_repair_stops_on_no_applicable_candidate(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _invalid_replacement_model(tokenizer)

        result = greedy_repair(
            model,  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            max_steps=3,
            candidate_k=1,
            max_decode_length=4,
        )

        self.assertFalse(result.success)
        self.assertTrue(result.no_candidate)
        self.assertEqual(result.stop_reason, "no_applicable_candidate")
        self.assertEqual(result.steps[0].decoded_status, "replacement_parse_failed")

    def test_greedy_repair_returns_success_for_already_correct_initial(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _correct_exponent_model(tokenizer)

        result = greedy_repair(
            model,  # type: ignore[arg-type]
            parse_prefix_string("pow x INT+ 2"),
            parse_prefix_string("div pow x INT+ 3 INT+ 3"),
            tokenizer=tokenizer,
            device="cpu",
            max_steps=3,
            candidate_k=1,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "exact_symbolic_match")
        self.assertEqual(result.steps_taken, 0)
        self.assertEqual(result.steps, [])

    def test_greedy_repair_max_steps_zero_only_runs_initial_checks(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _correct_exponent_model(tokenizer)

        result = greedy_repair(
            model,  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            max_steps=0,
            candidate_k=1,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.stop_reason, "max_steps")
        self.assertEqual(result.steps_taken, 0)

    def test_greedy_repair_from_seeds_returns_first_success(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _correct_exponent_model(tokenizer)

        result = greedy_repair_from_seeds(
            model,  # type: ignore[arg-type]
            _target_integrand(),
            [parse_prefix_string("x"), parse_prefix_string("pow x INT+ 5")],
            tokenizer=tokenizer,
            device="cpu",
            max_steps=2,
            candidate_k=1,
            max_decode_length=4,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.initial_prefix, "pow x INT+ 5")
        self.assertEqual(result.final_prefix, "pow x INT+ 3")

    def test_greedy_repair_from_seeds_returns_lowest_final_residual(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _invalid_replacement_model(tokenizer)

        result = greedy_repair_from_seeds(
            model,  # type: ignore[arg-type]
            parse_prefix_string("INT+ 2"),
            [parse_prefix_string("pow x INT+ 2"), parse_prefix_string("x")],
            tokenizer=tokenizer,
            device="cpu",
            max_steps=1,
            candidate_k=1,
            max_decode_length=4,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.initial_prefix, "x")


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
        self.config = SimpleNamespace(
            max_input_length=256,
            max_target_length=max_target_length,
        )
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


def _correct_exponent_model(tokenizer: TreeDiffusionTokenizer) -> _FixedLogitModel:
    return _FixedLogitModel(
        tokenizer,
        [{"<POS_2>": 10.0}, {"INT+": 10.0}, {"3": 10.0}, {"<eos>": 10.0}],
        max_target_length=8,
    )


def _invalid_then_correct_candidate_model(tokenizer: TreeDiffusionTokenizer) -> _FixedLogitModel:
    return _FixedLogitModel(
        tokenizer,
        [
            {"<POS_2>": 10.0},
            {"pow": 10.0, "INT+": 9.0},
            {"3": 10.0},
            {"<eos>": 10.0},
        ],
        max_target_length=8,
    )


def _same_exponent_model(tokenizer: TreeDiffusionTokenizer) -> _FixedLogitModel:
    return _FixedLogitModel(
        tokenizer,
        [{"<POS_2>": 10.0}, {"INT+": 10.0}, {"5": 10.0}, {"<eos>": 10.0}],
        max_target_length=8,
    )


def _invalid_replacement_model(tokenizer: TreeDiffusionTokenizer) -> _FixedLogitModel:
    return _FixedLogitModel(
        tokenizer,
        [{"<POS_2>": 10.0, "<POS_0>": 9.0}, {"pow": 10.0}, {"x": 10.0}, {"<eos>": 10.0}],
        max_target_length=8,
    )


def _target_integrand():
    return parse_prefix_string("mul INT+ 3 pow x INT+ 2")


if __name__ == "__main__":
    unittest.main()
