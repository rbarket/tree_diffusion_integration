from __future__ import annotations

import unittest

import torch

from src.mathlang.parser import parse_prefix_string
from src.tree_diffusion.dataset import IntegrationPair, make_tree_diffusion_dataloader
from src.tree_diffusion.model import (
    TreeDiffusionModelConfig,
    TreeDiffusionPolicyModel,
    build_tree_diffusion_policy_model,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


class TreeDiffusionPolicyModelTests(unittest.TestCase):
    def test_config_validation(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        base = _config_kwargs(tokenizer)

        invalid_overrides = (
            {"vocab_size": 0},
            {"pad_token_id": -1},
            {"bos_token_id": tokenizer.vocab_size},
            {"eos_token_id": tokenizer.vocab_size},
            {"d_model": 63, "n_heads": 4},
            {"max_input_length": 0},
            {"max_target_length": 0},
            {"dropout": -0.1},
            {"dropout": 1.0},
        )

        for overrides in invalid_overrides:
            kwargs = dict(base)
            kwargs.update(overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    TreeDiffusionModelConfig(**kwargs)

    def test_shift_right(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer)
        target_ids = torch.tensor(
            [[tokenizer.position_id(3), tokenizer.token_to_id["x"], tokenizer.eos_id, tokenizer.pad_id]],
            dtype=torch.long,
        )

        shifted = model.shift_right(target_ids)

        self.assertEqual(shifted.shape, target_ids.shape)
        self.assertEqual(shifted.dtype, target_ids.dtype)
        self.assertEqual(shifted.device, target_ids.device)
        self.assertEqual(int(shifted[0, 0]), tokenizer.bos_id)
        self.assertTrue(torch.equal(shifted[0, 1:], target_ids[0, :-1]))

        with self.assertRaises(TypeError):
            model.shift_right(target_ids.float())
        with self.assertRaises(ValueError):
            model.shift_right(target_ids.unsqueeze(0))

    def test_forward_shape_with_synthetic_batch(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer)
        batch = _synthetic_batch(tokenizer, max_input_length=128, max_target_length=32)

        output = model(
            input_ids=batch["input_ids"],
            input_attention_mask=batch["input_attention_mask"],
            target_ids=batch["target_ids"],
            target_attention_mask=batch["target_attention_mask"],
            labels=batch["labels"],
        )

        self.assertEqual(output.logits.shape, (2, 32, tokenizer.vocab_size))
        self.assertIsNotNone(output.loss)
        assert output.loss is not None
        self.assertEqual(output.loss.ndim, 0)
        self.assertTrue(torch.isfinite(output.loss).item())
        self.assertIsNotNone(output.position_accuracy)
        assert output.position_accuracy is not None
        self.assertEqual(output.position_accuracy.ndim, 0)
        self.assertIsNotNone(output.token_accuracy)
        assert output.token_accuracy is not None
        self.assertEqual(output.token_accuracy.ndim, 0)

    def test_forward_without_labels(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer)
        batch = _synthetic_batch(tokenizer, max_input_length=128, max_target_length=32)

        output = model(
            input_ids=batch["input_ids"],
            input_attention_mask=batch["input_attention_mask"],
            target_ids=batch["target_ids"],
            target_attention_mask=batch["target_attention_mask"],
        )

        self.assertIsNotNone(output.loss)
        assert output.loss is not None
        self.assertTrue(torch.isfinite(output.loss).item())

    def test_labels_ignore_pad_positions(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer)
        model.eval()
        batch = _synthetic_batch(tokenizer, max_input_length=128, max_target_length=32)

        with torch.no_grad():
            output1 = model(
                input_ids=batch["input_ids"],
                input_attention_mask=batch["input_attention_mask"],
                target_ids=batch["target_ids"],
                target_attention_mask=batch["target_attention_mask"],
                labels=batch["labels"],
            )

            changed_targets = batch["target_ids"].clone()
            changed_targets[batch["labels"].eq(-100)] = tokenizer.token_to_id["x"]
            output2 = model(
                input_ids=batch["input_ids"],
                input_attention_mask=batch["input_attention_mask"],
                target_ids=changed_targets,
                target_attention_mask=batch["target_attention_mask"],
                labels=batch["labels"],
            )

        self.assertTrue(torch.equal(batch["labels"].eq(-100), batch["target_ids"].eq(tokenizer.pad_id)))
        assert output1.loss is not None
        assert output2.loss is not None
        self.assertTrue(torch.allclose(output1.loss, output2.loss, atol=1e-6, rtol=0.0))

    def test_decoder_is_causal_no_future_leakage(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer, max_target_length=5)
        model.eval()
        batch = _synthetic_batch(tokenizer, max_input_length=128, max_target_length=5)
        input_ids = batch["input_ids"][:1].repeat(2, 1)
        input_attention_mask = batch["input_attention_mask"][:1].repeat(2, 1)
        target_ids = torch.tensor(
            [
                tokenizer.encode_tokens(
                    [tokenizer.position_token(1), "x", tokenizer.eos_token],
                    pad_to_length=5,
                ),
                tokenizer.encode_tokens(
                    [tokenizer.position_token(1), "x", "sin", "x", tokenizer.eos_token],
                    pad_to_length=5,
                ),
            ],
            dtype=torch.long,
        )
        target_attention_mask = target_ids.ne(tokenizer.pad_id).long()

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                input_attention_mask=input_attention_mask,
                target_ids=target_ids,
                target_attention_mask=target_attention_mask,
            )

        self.assertTrue(torch.allclose(output.logits[0, :3], output.logits[1, :3], atol=1e-6, rtol=0.0))

    def test_backprop_works(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer, d_model=32, d_ff=64)
        batch = _real_batch(tokenizer, batch_size=2, max_input_length=128, max_target_length=32)

        output = model(
            input_ids=batch["input_ids"],
            input_attention_mask=batch["input_attention_mask"],
            target_ids=batch["target_ids"],
            target_attention_mask=batch["target_attention_mask"],
            labels=batch["labels"],
        )
        assert output.loss is not None
        output.loss.backward()

        found_nonzero = False
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            found_nonzero = found_nonzero or bool(parameter.grad.abs().sum().item() > 0.0)
        self.assertTrue(found_nonzero)

    def test_deterministic_initialization(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        batch = _synthetic_batch(tokenizer, max_input_length=128, max_target_length=32)

        torch.manual_seed(123)
        model_a = _small_model(tokenizer)
        torch.manual_seed(123)
        model_b = _small_model(tokenizer)
        model_a.eval()
        model_b.eval()

        with torch.no_grad():
            logits_a = model_a(
                input_ids=batch["input_ids"],
                input_attention_mask=batch["input_attention_mask"],
                target_ids=batch["target_ids"],
                target_attention_mask=batch["target_attention_mask"],
                labels=batch["labels"],
            ).logits
            logits_b = model_b(
                input_ids=batch["input_ids"],
                input_attention_mask=batch["input_attention_mask"],
                target_ids=batch["target_ids"],
                target_attention_mask=batch["target_attention_mask"],
                labels=batch["labels"],
            ).logits

        self.assertTrue(torch.allclose(logits_a, logits_b))

    def test_weight_tying(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        tied = _small_model(tokenizer, tie_embeddings=True)
        untied = _small_model(tokenizer, tie_embeddings=False)

        self.assertIs(tied.lm_head.weight, tied.token_embedding.weight)
        self.assertIsNot(untied.lm_head.weight, untied.token_embedding.weight)
        self.assertNotEqual(
            untied.lm_head.weight.data_ptr(),
            untied.token_embedding.weight.data_ptr(),
        )

    def test_dataloader_integration(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer, d_model=32, d_ff=64)
        batch = _real_batch(tokenizer, batch_size=3, max_input_length=128, max_target_length=32)

        output = model(
            input_ids=batch["input_ids"],
            input_attention_mask=batch["input_attention_mask"],
            target_ids=batch["target_ids"],
            target_attention_mask=batch["target_attention_mask"],
            labels=batch["labels"],
        )

        self.assertEqual(output.logits.shape, (3, 32, tokenizer.vocab_size))
        decoded_target = tokenizer.decode_ids(batch["target_ids"][0].tolist(), strip_pad=True)
        self.assertTrue(decoded_target[0].startswith("<POS_"))
        self.assertEqual(decoded_target[-1], tokenizer.eos_token)

    def test_tiny_overfit_smoke(self) -> None:
        torch.manual_seed(123)
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(
            tokenizer,
            max_input_length=64,
            max_target_length=8,
            d_model=32,
            d_ff=64,
        )
        batch = _synthetic_batch(tokenizer, max_input_length=64, max_target_length=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        initial = model(
            input_ids=batch["input_ids"],
            input_attention_mask=batch["input_attention_mask"],
            target_ids=batch["target_ids"],
            target_attention_mask=batch["target_attention_mask"],
            labels=batch["labels"],
        ).loss
        assert initial is not None
        initial_value = float(initial.detach())

        final = initial
        for _ in range(25):
            optimizer.zero_grad()
            output = model(
                input_ids=batch["input_ids"],
                input_attention_mask=batch["input_attention_mask"],
                target_ids=batch["target_ids"],
                target_attention_mask=batch["target_attention_mask"],
                labels=batch["labels"],
            )
            assert output.loss is not None
            output.loss.backward()
            optimizer.step()
            final = output.loss.detach()

        self.assertLess(float(final), initial_value)

    def test_greedy_decode_smoke(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer, max_target_length=8)
        batch = _synthetic_batch(tokenizer, max_input_length=128, max_target_length=8)

        generated = model.greedy_decode(
            batch["input_ids"],
            input_attention_mask=batch["input_attention_mask"],
            max_length=6,
        )

        self.assertEqual(generated.dtype, torch.long)
        self.assertEqual(generated.ndim, 2)
        self.assertEqual(generated.shape[0], 2)
        self.assertGreaterEqual(generated.shape[1], 1)
        self.assertLessEqual(generated.shape[1], 6)
        self.assertTrue(generated.ge(0).all().item())
        self.assertTrue(generated.lt(tokenizer.vocab_size).all().item())

    def test_build_tree_diffusion_policy_model(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = build_tree_diffusion_policy_model(
            tokenizer,
            max_input_length=64,
            max_target_length=16,
            d_model=32,
            n_heads=4,
            d_ff=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            dropout=0.0,
        )

        self.assertEqual(model.config.vocab_size, tokenizer.vocab_size)
        self.assertEqual(model.config.pad_token_id, tokenizer.pad_id)
        self.assertEqual(model.config.bos_token_id, tokenizer.bos_id)
        self.assertEqual(model.config.eos_token_id, tokenizer.eos_id)

    def test_max_length_errors(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer, max_input_length=4, max_target_length=4)
        token_x = tokenizer.token_to_id["x"]

        with self.assertRaises(ValueError):
            model(
                input_ids=torch.full((1, 5), token_x, dtype=torch.long),
                target_ids=torch.full((1, 4), token_x, dtype=torch.long),
            )

        with self.assertRaises(ValueError):
            model(
                input_ids=torch.full((1, 4), token_x, dtype=torch.long),
                target_ids=torch.full((1, 5), token_x, dtype=torch.long),
            )

    def test_dtype_and_shape_errors(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer)
        batch = _synthetic_batch(tokenizer, max_input_length=128, max_target_length=32)

        with self.assertRaises(TypeError):
            model(
                input_ids=batch["input_ids"].float(),
                target_ids=batch["target_ids"],
            )
        with self.assertRaises(TypeError):
            model(
                input_ids=batch["input_ids"],
                target_ids=batch["target_ids"].float(),
            )
        with self.assertRaises(ValueError):
            model(
                input_ids=batch["input_ids"],
                input_attention_mask=batch["input_attention_mask"][:, :-1],
                target_ids=batch["target_ids"],
            )
        with self.assertRaises(ValueError):
            model(
                input_ids=batch["input_ids"],
                target_ids=batch["target_ids"],
                target_attention_mask=batch["target_attention_mask"][:, :-1],
            )
        with self.assertRaises(ValueError):
            model(
                input_ids=batch["input_ids"],
                target_ids=batch["target_ids"],
                labels=batch["labels"][:, :-1],
            )


def _config_kwargs(
    tokenizer: TreeDiffusionTokenizer,
    *,
    max_input_length: int = 128,
    max_target_length: int = 32,
    d_model: int = 64,
    n_heads: int = 4,
    d_ff: int = 128,
    n_encoder_layers: int = 1,
    n_decoder_layers: int = 1,
    dropout: float = 0.0,
    tie_embeddings: bool = True,
) -> dict[str, object]:
    return {
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_id,
        "bos_token_id": tokenizer.bos_id,
        "eos_token_id": tokenizer.eos_id,
        "max_input_length": max_input_length,
        "max_target_length": max_target_length,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "n_encoder_layers": n_encoder_layers,
        "n_decoder_layers": n_decoder_layers,
        "dropout": dropout,
        "tie_embeddings": tie_embeddings,
    }


def _small_model(
    tokenizer: TreeDiffusionTokenizer,
    **overrides: object,
) -> TreeDiffusionPolicyModel:
    kwargs = _config_kwargs(tokenizer)
    kwargs.update(overrides)
    return TreeDiffusionPolicyModel(TreeDiffusionModelConfig(**kwargs))


def _synthetic_batch(
    tokenizer: TreeDiffusionTokenizer,
    *,
    max_input_length: int,
    max_target_length: int,
) -> dict[str, torch.Tensor]:
    input_tokens = [
        [
            "<F>",
            "pow",
            "x",
            "INT+",
            "2",
            "</F>",
            "<CUR>",
            "pow",
            "x",
            "INT+",
            "4",
            "</CUR>",
            "<DER>",
            "mul",
            "INT+",
            "4",
            "pow",
            "x",
            "INT+",
            "3",
            "</DER>",
            "<RES>",
            "add",
            "pow",
            "x",
            "INT+",
            "2",
            "INT-",
            "1",
            "</RES>",
            "<NUM>",
            "<NO_NUM>",
            "</NUM>",
            "<EDIT>",
        ],
        [
            "<F>",
            "cos",
            "x",
            "</F>",
            "<CUR>",
            "x",
            "</CUR>",
            "<DER>",
            "INT+",
            "1",
            "</DER>",
            "<RES>",
            "add",
            "cos",
            "x",
            "INT-",
            "1",
            "</RES>",
            "<NUM>",
            "<NO_NUM>",
            "</NUM>",
            "<EDIT>",
        ],
    ]
    target_tokens = [
        [tokenizer.position_token(0), "x", tokenizer.eos_token],
        [tokenizer.position_token(1), "sin", "x", tokenizer.eos_token],
    ]

    input_ids = torch.tensor(
        [tokenizer.encode_tokens(tokens, pad_to_length=max_input_length) for tokens in input_tokens],
        dtype=torch.long,
    )
    target_ids = torch.tensor(
        [tokenizer.encode_tokens(tokens, pad_to_length=max_target_length) for tokens in target_tokens],
        dtype=torch.long,
    )
    input_attention_mask = input_ids.ne(tokenizer.pad_id).long()
    target_attention_mask = target_ids.ne(tokenizer.pad_id).long()
    labels = target_ids.clone()
    labels[target_ids.eq(tokenizer.pad_id)] = -100
    return {
        "input_ids": input_ids,
        "input_attention_mask": input_attention_mask,
        "target_ids": target_ids,
        "target_attention_mask": target_attention_mask,
        "labels": labels,
    }


def _real_batch(
    tokenizer: TreeDiffusionTokenizer,
    *,
    batch_size: int,
    max_input_length: int,
    max_target_length: int,
) -> dict[str, torch.Tensor]:
    loader = make_tree_diffusion_dataloader(
        _pairs(),
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=0,
        sigma_small=2,
        smax=2,
        rho=0.0,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        base_seed=123,
        shuffle_pairs=False,
    )
    return next(iter(loader))


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


if __name__ == "__main__":
    unittest.main()
