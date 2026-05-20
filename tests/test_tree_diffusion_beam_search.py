from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion.beam_search import (
    BeamSearchScoringConfig,
    BeamSearchStopConfig,
    beam_search_repair,
    beam_search_repair_from_seeds,
    score_beam_state,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from tests.test_tree_diffusion_repair import (
    _correct_exponent_model,
    _invalid_replacement_model,
    _rank1_worse_rank2_better_model,
    _same_exponent_model,
    _target_integrand,
    _worse_exponent_model,
)


class TreeDiffusionBeamSearchTests(unittest.TestCase):
    def test_score_beam_state_orders_by_residual_size_steps_policy(self) -> None:
        config = BeamSearchScoringConfig(
            lambda_residual=1.0,
            lambda_size=1e-3,
            lambda_steps=1e-3,
            lambda_policy=1e-2,
            use_log_residual=False,
        )

        self.assertLess(
            score_beam_state(
                numeric_residual=1.0,
                tree_size_value=100,
                steps=10,
                cumulative_policy_logprob=0.0,
                config=config,
            ),
            score_beam_state(
                numeric_residual=2.0,
                tree_size_value=1,
                steps=1,
                cumulative_policy_logprob=-1.0,
                config=config,
            ),
        )
        self.assertLess(
            score_beam_state(
                numeric_residual=1.0,
                tree_size_value=1,
                steps=1,
                cumulative_policy_logprob=0.0,
                config=config,
            ),
            score_beam_state(
                numeric_residual=1.0,
                tree_size_value=2,
                steps=1,
                cumulative_policy_logprob=0.0,
                config=config,
            ),
        )
        self.assertLess(
            score_beam_state(
                numeric_residual=1.0,
                tree_size_value=1,
                steps=1,
                cumulative_policy_logprob=0.0,
                config=config,
            ),
            score_beam_state(
                numeric_residual=1.0,
                tree_size_value=1,
                steps=2,
                cumulative_policy_logprob=0.0,
                config=config,
            ),
        )
        self.assertLess(
            score_beam_state(
                numeric_residual=1.0,
                tree_size_value=1,
                steps=1,
                cumulative_policy_logprob=-1.0,
                config=config,
            ),
            score_beam_state(
                numeric_residual=1.0,
                tree_size_value=1,
                steps=1,
                cumulative_policy_logprob=-2.0,
                config=config,
            ),
        )

    def test_score_beam_state_log_residual_changes_score(self) -> None:
        raw = score_beam_state(
            numeric_residual=9.0,
            tree_size_value=1,
            steps=0,
            cumulative_policy_logprob=0.0,
            config=BeamSearchScoringConfig(use_log_residual=False),
        )
        logged = score_beam_state(
            numeric_residual=9.0,
            tree_size_value=1,
            steps=0,
            cumulative_policy_logprob=0.0,
            config=BeamSearchScoringConfig(use_log_residual=True),
        )
        self.assertLess(logged, raw)

    def test_beam_search_succeeds_in_one_step(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        result = beam_search_repair(
            _correct_exponent_model(tokenizer),  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            beam_size=2,
            candidate_k=1,
            max_decode_length=4,
            target_antiderivative=parse_prefix_string("pow x INT+ 3"),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "exact_symbolic_match")
        self.assertEqual(result.best_prefix, "pow x INT+ 3")
        self.assertEqual(result.steps_taken, 1)
        self.assertEqual(result.path[0].candidate_rank, 1)

    def test_beam_search_can_choose_lower_rank_better_residual(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        result = beam_search_repair(
            _rank1_worse_rank2_better_model(tokenizer),  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            beam_size=1,
            candidate_k=2,
            max_decode_length=4,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.best_prefix, "pow x INT+ 3")
        self.assertEqual(result.path[0].candidate_rank, 2)

    def test_beam_search_succeeds_where_rank1_greedy_would_follow_bad_candidate(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        result = beam_search_repair(
            _rank1_worse_rank2_better_model(tokenizer),  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            beam_size=2,
            candidate_k=2,
            max_decode_length=4,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.best_prefix, "pow x INT+ 3")

    def test_beam_search_skips_repeated_states(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        result = beam_search_repair(
            _same_exponent_model(tokenizer),  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            beam_size=2,
            candidate_k=1,
            max_decode_length=4,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.stop_reason, "beam_empty")
        self.assertGreaterEqual(result.repeated_candidates, 1)

    def test_beam_search_returns_best_so_far_after_later_worsening(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _DepthEditModel(
            tokenizer,
            [
                [{"<POS_2>": 10.0}, {"INT+": 10.0}, {"4": 10.0}, {"<eos>": 10.0}],
                [{"<POS_2>": 10.0}, {"INT+": 10.0}, {"6": 10.0}, {"<eos>": 10.0}],
            ],
        )

        result = beam_search_repair(
            model,  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            beam_size=1,
            candidate_k=1,
            max_decode_length=4,
            stopping=BeamSearchStopConfig(max_steps=2, numeric_patience=None),
        )

        self.assertEqual(result.stop_reason, "max_steps")
        self.assertEqual(result.best_prefix, "pow x INT+ 4")
        self.assertEqual(result.steps_taken, 1)
        self.assertEqual(len(result.per_depth_best_numeric_residual), 3)

    def test_beam_search_numeric_patience_stop(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        result = beam_search_repair(
            _worse_exponent_model(tokenizer),  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            beam_size=1,
            candidate_k=1,
            max_decode_length=4,
            stopping=BeamSearchStopConfig(max_steps=3, numeric_patience=1),
        )

        self.assertEqual(result.stop_reason, "numeric_patience")

    def test_beam_search_structural_patience_stop(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        result = beam_search_repair(
            _worse_exponent_model(tokenizer),  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            beam_size=1,
            candidate_k=1,
            max_decode_length=4,
            stopping=BeamSearchStopConfig(
                max_steps=3,
                numeric_patience=None,
                structural_patience=1,
            ),
            target_antiderivative=parse_prefix_string("pow x INT+ 3"),
        )

        self.assertEqual(result.stop_reason, "structural_patience")

    def test_beam_search_max_steps_beam_empty_exact_and_numeric_tol_stops(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        max_steps = beam_search_repair(
            _rank1_worse_rank2_better_model(tokenizer),  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            beam_size=1,
            candidate_k=1,
            max_decode_length=4,
            stopping=BeamSearchStopConfig(max_steps=1, numeric_patience=None),
        )
        beam_empty = beam_search_repair(
            _invalid_replacement_model(tokenizer),  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            beam_size=1,
            candidate_k=1,
            max_decode_length=4,
        )
        exact = beam_search_repair(
            _correct_exponent_model(tokenizer),  # type: ignore[arg-type]
            parse_prefix_string("pow x INT+ 2"),
            parse_prefix_string("div pow x INT+ 3 INT+ 3"),
            tokenizer=tokenizer,
            device="cpu",
        )
        numeric_tol = beam_search_repair(
            _correct_exponent_model(tokenizer),  # type: ignore[arg-type]
            _target_integrand(),
            parse_prefix_string("pow x INT+ 5"),
            tokenizer=tokenizer,
            device="cpu",
            stopping=BeamSearchStopConfig(max_steps=3, numeric_tol=1e30),
        )

        self.assertEqual(max_steps.stop_reason, "max_steps")
        self.assertEqual(beam_empty.stop_reason, "beam_empty")
        self.assertEqual(exact.stop_reason, "exact_symbolic_match")
        self.assertEqual(numeric_tol.stop_reason, "numeric_tol")

    def test_invalid_beam_size_or_candidate_k_raises(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        with self.assertRaises(ValueError):
            beam_search_repair(
                _correct_exponent_model(tokenizer),  # type: ignore[arg-type]
                _target_integrand(),
                parse_prefix_string("pow x INT+ 5"),
                tokenizer=tokenizer,
                device="cpu",
                beam_size=0,
            )
        with self.assertRaises(ValueError):
            beam_search_repair(
                _correct_exponent_model(tokenizer),  # type: ignore[arg-type]
                _target_integrand(),
                parse_prefix_string("pow x INT+ 5"),
                tokenizer=tokenizer,
                device="cpu",
                candidate_k=0,
            )

    def test_beam_search_from_seeds_returns_success_then_lowest_best_residual(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        success = beam_search_repair_from_seeds(
            _correct_exponent_model(tokenizer),  # type: ignore[arg-type]
            _target_integrand(),
            [parse_prefix_string("x"), parse_prefix_string("pow x INT+ 5")],
            tokenizer=tokenizer,
            device="cpu",
            beam_size=1,
            candidate_k=1,
            max_decode_length=4,
        )
        lowest = beam_search_repair_from_seeds(
            _invalid_replacement_model(tokenizer),  # type: ignore[arg-type]
            parse_prefix_string("INT+ 2"),
            [parse_prefix_string("pow x INT+ 2"), parse_prefix_string("x")],
            tokenizer=tokenizer,
            device="cpu",
            beam_size=1,
            candidate_k=1,
            max_decode_length=4,
        )

        self.assertTrue(success.success)
        self.assertEqual(success.initial_prefix, "pow x INT+ 5")
        self.assertFalse(lowest.success)
        self.assertEqual(lowest.initial_prefix, "x")


class _DepthEditModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer: TreeDiffusionTokenizer,
        repairs: list[list[dict[str, float]]],
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.repairs = repairs
        self.config = SimpleNamespace(max_input_length=256, max_target_length=8)
        self._encode_calls = 0
        self._repair_index = 0
        self._decode_step = 0

    def encode(
        self,
        input_ids: torch.Tensor,
        input_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del input_attention_mask
        self._repair_index = min(self._encode_calls, len(self.repairs) - 1)
        self._encode_calls += 1
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
        step_scores = self.repairs[self._repair_index][self._decode_step]
        for token, score in step_scores.items():
            logits[:, -1, self.tokenizer.token_to_id[token]] = float(score)
        return logits


if __name__ == "__main__":
    unittest.main()
