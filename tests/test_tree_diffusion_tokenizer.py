from __future__ import annotations

from fractions import Fraction
import unittest

from src.mathlang.ast import Const, Var
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_tokens
from src.tree_diffusion.edit_path import first_edit_toward_target
from src.tree_diffusion.observation import Observation, build_observation
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer, numeric_bucket_token


class TreeDiffusionTokenizerTests(unittest.TestCase):
    def test_vocab_contains_grammar_and_control_tokens(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=16)

        expected_tokens = {
            "add",
            "mul",
            "pow",
            "div",
            "x",
            "INT+",
            "INT-",
            "<F>",
            "</F>",
            "<CUR>",
            "</CUR>",
            "<DER>",
            "</DER>",
            "<RES>",
            "</RES>",
            "<NUM>",
            "</NUM>",
            "<EDIT>",
            "<POS_0>",
            "<POS_15>",
        }
        expected_tokens.update(str(digit) for digit in range(10))

        for token in expected_tokens:
            with self.subTest(token=token):
                self.assertIn(token, tokenizer.token_to_id)

    def test_position_token_roundtrip(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=16)

        self.assertEqual(tokenizer.position_token(3), "<POS_3>")
        self.assertEqual(tokenizer.token_to_position("<POS_3>"), 3)
        self.assertEqual(tokenizer.token_to_position(tokenizer.position_id(3)), 3)

        with self.assertRaises(ValueError):
            tokenizer.position_token(-1)
        with self.assertRaises(ValueError):
            tokenizer.position_token(16)

    def test_expression_encode_decode_roundtrip(self) -> None:
        tokenizer = TreeDiffusionTokenizer()
        expr = parse_prefix_string("add pow x INT+ 2 sin x")
        tokens = serialize_prefix_tokens(expr)

        self.assertEqual(tokenizer.serialize_expr(expr), tokens)
        self.assertEqual(tokenizer.decode_ids(tokenizer.encode_tokens(tokens)), tokens)

    def test_observation_serialization_with_full_residual_mode(self) -> None:
        tokenizer = TreeDiffusionTokenizer()
        target = parse_prefix_string("pow x INT+ 2")
        current = parse_prefix_string("div pow x INT+ 3 INT+ 3")
        observation = build_observation(target, current, residual_mode="both")

        tokens = tokenizer.serialize_observation(observation)

        self.assertLess(tokens.index("<F>"), tokens.index("</F>"))
        self.assertLess(tokens.index("</F>"), tokens.index("<CUR>"))
        self.assertLess(tokens.index("</CUR>"), tokens.index("<DER>"))
        self.assertLess(tokens.index("</DER>"), tokens.index("<RES>"))
        self.assertLess(tokens.index("</RES>"), tokens.index("<NUM>"))
        self.assertNotIn("<EDIT>", tokens)

        self.assertEqual(_section(tokens, "<F>", "</F>"), serialize_prefix_tokens(target))
        self.assertEqual(_section(tokens, "<CUR>", "</CUR>"), serialize_prefix_tokens(current))
        self.assertEqual(_section(tokens, "<DER>", "</DER>"), ["pow", "x", "INT+", "2"])
        self.assertEqual(_section(tokens, "<RES>", "</RES>"), ["INT+", "0"])

        numeric_tokens = _section(tokens, "<NUM>", "</NUM>")
        self.assertIn("<NUM_MEAN_ABS>", numeric_tokens)
        self.assertIn("<NUM_MSE>", numeric_tokens)
        self.assertIn("<NUM_MAX_ABS>", numeric_tokens)
        self.assertIn("<NUM_ZERO>", numeric_tokens)

        self.assertEqual(tokenizer.decode_ids(tokenizer.encode_tokens(tokens)), tokens)

    def test_observation_serialization_with_missing_fields(self) -> None:
        tokenizer = TreeDiffusionTokenizer()
        observation = Observation(
            target_integrand=Var(name="x"),
            current_antiderivative=parse_prefix_string("sin x"),
            current_derivative=None,
            symbolic_residual=None,
            numeric_probes=None,
            residual_mode="both",
            status="partial",
            warnings=(),
        )

        tokens = tokenizer.serialize_observation(observation)

        self.assertEqual(_section(tokens, "<DER>", "</DER>"), ["<NO_DER>"])
        self.assertEqual(_section(tokens, "<RES>", "</RES>"), ["<NO_RES>"])
        self.assertEqual(_section(tokens, "<NUM>", "</NUM>"), ["<NO_NUM>"])
        self.assertEqual(tokenizer.decode_ids(tokenizer.encode_tokens(tokens)), tokens)

    def test_numeric_bucket_behavior(self) -> None:
        tokenizer = TreeDiffusionTokenizer()
        clipped = TreeDiffusionTokenizer(numeric_log_min=-3, numeric_log_max=3)

        self.assertEqual(numeric_bucket_token(None), "<NUM_NAN>")
        self.assertEqual(tokenizer.numeric_bucket_token(float("inf")), "<NUM_NAN>")
        self.assertEqual(tokenizer.numeric_bucket_token(0.0), "<NUM_ZERO>")
        self.assertEqual(tokenizer.numeric_bucket_token(1e-13), "<NUM_ZERO>")
        self.assertEqual(tokenizer.numeric_bucket_token(1.0), "<NUM_POS_LOG_0>")
        self.assertEqual(tokenizer.numeric_bucket_token(10.0), "<NUM_POS_LOG_1>")
        self.assertEqual(tokenizer.numeric_bucket_token(-0.1), "<NUM_NEG_LOG_-1>")
        self.assertEqual(clipped.numeric_bucket_token(1e99), "<NUM_POS_LOG_3>")
        self.assertEqual(clipped.numeric_bucket_token(1e-4), "<NUM_POS_LOG_-3>")

    def test_edit_target_serialization(self) -> None:
        tokenizer = TreeDiffusionTokenizer()
        current = parse_prefix_string("pow x INT+ 5")
        target = parse_prefix_string("pow x INT+ 3")

        edit = first_edit_toward_target(current, target, sigma_small=2)

        self.assertIsNotNone(edit)
        assert edit is not None
        tokens = tokenizer.serialize_edit_target(edit)

        self.assertTrue(tokens[0].startswith("<POS_"))
        self.assertEqual(tokenizer.token_to_position(tokens[0]), edit.selected_node_id)
        self.assertEqual(tokens[1:-1], serialize_prefix_tokens(edit.replacement_subtree))
        self.assertEqual(tokens[-1], "<eos>")

    def test_training_pair_serialization(self) -> None:
        tokenizer = TreeDiffusionTokenizer()
        current = parse_prefix_string("pow x INT+ 5")
        gold_antiderivative = parse_prefix_string("pow x INT+ 3")
        target_integrand = parse_prefix_string("mul INT+ 3 pow x INT+ 2")
        observation = build_observation(target_integrand, current, residual_mode="both")
        edit = first_edit_toward_target(current, gold_antiderivative, sigma_small=2)

        self.assertIsNotNone(edit)
        assert edit is not None
        input_tokens, target_tokens = tokenizer.serialize_training_pair(observation, edit)

        self.assertEqual(input_tokens[-1], "<EDIT>")
        self.assertTrue(target_tokens[0].startswith("<POS_"))
        self.assertEqual(target_tokens[-1], "<eos>")
        self.assertFalse(
            _contains_subsequence(input_tokens, serialize_prefix_tokens(gold_antiderivative))
        )
        self.assertEqual(
            target_tokens,
            [
                tokenizer.position_token(edit.selected_node_id),
                *serialize_prefix_tokens(edit.replacement_subtree),
                "<eos>",
            ],
        )

    def test_padding_and_unknown_behavior(self) -> None:
        tokenizer = TreeDiffusionTokenizer()

        encoded = tokenizer.encode_tokens(["x"], pad_to_length=4)

        self.assertEqual(len(encoded), 4)
        self.assertEqual(encoded[1:], [tokenizer.pad_id, tokenizer.pad_id, tokenizer.pad_id])
        self.assertEqual(tokenizer.decode_ids(encoded, strip_pad=True), ["x"])

        with self.assertRaises(ValueError):
            tokenizer.encode_tokens(["<DOES_NOT_EXIST>"])
        self.assertEqual(
            tokenizer.encode_tokens(["<DOES_NOT_EXIST>"], allow_unk=True),
            [tokenizer.unk_id],
        )

    def test_max_sequence_length_errors(self) -> None:
        tokenizer = TreeDiffusionTokenizer()

        with self.assertRaises(ValueError):
            tokenizer.encode_tokens(["x", "x"], pad_to_length=1)

    def test_numeric_probe_count_above_reserved_labels_raises(self) -> None:
        tokenizer = TreeDiffusionTokenizer()
        observation = build_observation(
            parse_prefix_string("INT+ 0"),
            Const(value=Fraction(0, 1)),
            residual_mode="numeric",
            probe_points=tuple(float(index + 1) for index in range(33)),
        )

        with self.assertRaises(ValueError):
            tokenizer.serialize_observation(observation)


def _section(tokens: list[str], start_token: str, end_token: str) -> list[str]:
    start = tokens.index(start_token) + 1
    end = tokens.index(end_token, start)
    return tokens[start:end]


def _contains_subsequence(tokens: list[str], subsequence: list[str]) -> bool:
    if not subsequence:
        return True
    if len(subsequence) > len(tokens):
        return False
    return any(
        tokens[index : index + len(subsequence)] == subsequence
        for index in range(len(tokens))
    )


if __name__ == "__main__":
    unittest.main()
