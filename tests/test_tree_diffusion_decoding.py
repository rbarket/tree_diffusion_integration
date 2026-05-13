from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.decoding import (
    DecodedEdit,
    apply_decoded_edit,
    decode_edit_tokens,
    greedy_decode_edit_tokens,
    predict_greedy_edit,
    valid_position_token_ids,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


class TreeDiffusionDecodingTests(unittest.TestCase):
    def test_decode_valid_tokens_into_edit(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        current = parse_prefix_string("pow x INT+ 5")

        decoded = decode_edit_tokens(
            ["<POS_2>", "INT+", "3", "<eos>"],
            tokenizer=tokenizer,
            current_tree=current,
        )

        self.assertEqual(decoded.status, "ok")
        self.assertEqual(decoded.selected_node_id, 2)
        self.assertEqual(decoded.generated_tokens, ["<POS_2>", "INT+", "3", "<eos>"])
        self.assertEqual(decoded.normalized_tokens, ["<POS_2>", "INT+", "3"])
        self.assertEqual(decoded.replacement_tokens, ["INT+", "3"])
        self.assertEqual(serialize_prefix_string(decoded.replacement_subtree), "INT+ 3")

    def test_decode_ignores_leading_bos_and_trailing_pad(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        current = parse_prefix_string("pow x INT+ 5")

        decoded = decode_edit_tokens(
            ["<bos>", "<POS_2>", "INT+", "3", "<eos>", "<pad>", "<pad>"],
            tokenizer=tokenizer,
            current_tree=current,
        )

        self.assertEqual(decoded.status, "ok")
        self.assertEqual(
            decoded.generated_tokens,
            ["<bos>", "<POS_2>", "INT+", "3", "<eos>", "<pad>", "<pad>"],
        )
        self.assertEqual(decoded.normalized_tokens, ["<POS_2>", "INT+", "3"])

    def test_decode_invalid_and_empty_positions(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        current = parse_prefix_string("x")

        empty = decode_edit_tokens(["<eos>", "<pad>"], tokenizer=tokenizer, current_tree=current)
        invalid = decode_edit_tokens(["x", "<eos>"], tokenizer=tokenizer, current_tree=current)

        self.assertEqual(empty.status, "empty")
        self.assertEqual(invalid.status, "invalid_position_token")

    def test_decode_position_out_of_range(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        current = parse_prefix_string("x")

        decoded = decode_edit_tokens(
            ["<POS_1>", "INT+", "3", "<eos>"],
            tokenizer=tokenizer,
            current_tree=current,
        )

        self.assertEqual(decoded.status, "position_out_of_range")
        self.assertEqual(decoded.selected_node_id, 1)

    def test_decode_missing_replacement(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        current = parse_prefix_string("x")

        decoded = decode_edit_tokens(["<POS_0>", "<eos>"], tokenizer=tokenizer, current_tree=current)

        self.assertEqual(decoded.status, "missing_replacement")

    def test_decode_replacement_parse_failure(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        current = parse_prefix_string("x")

        decoded = decode_edit_tokens(
            ["<POS_0>", "add", "x", "<eos>"],
            tokenizer=tokenizer,
            current_tree=current,
        )

        self.assertEqual(decoded.status, "replacement_parse_failed")

    def test_decode_replacement_parse_failure_on_leftover_tokens(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        current = parse_prefix_string("x")

        decoded = decode_edit_tokens(
            ["<POS_0>", "INT+", "3", "x", "<eos>"],
            tokenizer=tokenizer,
            current_tree=current,
        )

        self.assertEqual(decoded.status, "replacement_parse_failed")

    def test_apply_decoded_edit_changes_subtree(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        current = parse_prefix_string("pow x INT+ 5")
        decoded = decode_edit_tokens(
            ["<POS_2>", "INT+", "3", "<eos>"],
            tokenizer=tokenizer,
            current_tree=current,
        )

        edited = apply_decoded_edit(current, decoded)

        self.assertEqual(serialize_prefix_string(edited), "pow x INT+ 3")

    def test_apply_invalid_decoded_edit_raises(self) -> None:
        decoded = DecodedEdit(
            selected_node_id=None,
            replacement_subtree=None,
            generated_tokens=[],
            normalized_tokens=[],
            replacement_tokens=[],
            status="empty",
        )

        with self.assertRaises(ValueError):
            apply_decoded_edit(parse_prefix_string("x"), decoded)

    def test_valid_position_token_ids_returns_preorder_ids(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        current = parse_prefix_string("pow x INT+ 5")

        ids = valid_position_token_ids(current, tokenizer)

        self.assertEqual(
            ids,
            [tokenizer.position_id(0), tokenizer.position_id(1), tokenizer.position_id(2)],
        )

    def test_valid_position_token_ids_raises_when_tree_exceeds_tokenizer_positions(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=2)
        current = parse_prefix_string("pow x INT+ 5")

        with self.assertRaisesRegex(ValueError, "max_positions"):
            valid_position_token_ids(current, tokenizer)

    def test_position_constrained_greedy_masks_invalid_first_token(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        current = parse_prefix_string("pow x INT+ 5")
        model = _FixedLogitModel(
            tokenizer,
            [{"x": 10.0, "<POS_2>": 5.0}, {"<eos>": 10.0}],
            max_target_length=4,
        )

        tokens, logprob = greedy_decode_edit_tokens(
            model,  # type: ignore[arg-type]
            _input_ids(tokenizer),
            tokenizer=tokenizer,
            current_tree=current,
            max_length=2,
            constrain_position=True,
        )

        self.assertEqual(tokens, ["<POS_2>", "<eos>"])
        self.assertIsNotNone(logprob)

    def test_unconstrained_greedy_preserves_raw_argmax(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        current = parse_prefix_string("pow x INT+ 5")
        model = _FixedLogitModel(
            tokenizer,
            [{"x": 10.0, "<POS_2>": 5.0}],
            max_target_length=4,
        )

        tokens, _ = greedy_decode_edit_tokens(
            model,  # type: ignore[arg-type]
            _input_ids(tokenizer),
            tokenizer=tokenizer,
            current_tree=current,
            max_length=1,
            constrain_position=False,
        )

        self.assertEqual(tokens, ["x"])

    def test_greedy_decode_stops_at_eos(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        current = parse_prefix_string("x")
        model = _FixedLogitModel(
            tokenizer,
            [{"<POS_0>": 10.0}, {"<eos>": 10.0}, {"x": 10.0}],
            max_target_length=8,
        )

        tokens, _ = greedy_decode_edit_tokens(
            model,  # type: ignore[arg-type]
            _input_ids(tokenizer),
            tokenizer=tokenizer,
            current_tree=current,
            max_length=8,
        )

        self.assertEqual(tokens, ["<POS_0>", "<eos>"])

    def test_predict_greedy_edit_returns_decoded_edit(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        current = parse_prefix_string("pow x INT+ 5")
        model = _FixedLogitModel(
            tokenizer,
            [{"<POS_2>": 10.0}, {"INT+": 10.0}, {"3": 10.0}, {"<eos>": 10.0}],
            max_target_length=8,
        )

        decoded = predict_greedy_edit(
            model,  # type: ignore[arg-type]
            _input_ids(tokenizer),
            tokenizer=tokenizer,
            current_tree=current,
            max_length=4,
        )

        self.assertEqual(decoded.status, "ok")
        self.assertEqual(decoded.selected_node_id, 2)
        self.assertEqual(decoded.replacement_tokens, ["INT+", "3"])
        self.assertIsNotNone(decoded.logprob)


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


def _input_ids(tokenizer: TreeDiffusionTokenizer) -> torch.Tensor:
    return torch.tensor(tokenizer.encode_tokens(["<EDIT>"], pad_to_length=4), dtype=torch.long)


if __name__ == "__main__":
    unittest.main()
