from __future__ import annotations

import unittest

import torch

from src.tree_diffusion.dataset import make_tree_diffusion_dataloader
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.train_step import (
    compute_gradient_norm,
    inspect_batch_predictions,
    overfit_fixed_batch,
    tree_diffusion_eval_step,
    tree_diffusion_train_step,
    validate_tree_diffusion_batch,
)
from tests.tree_diffusion_test_utils import sample_integration_pairs, small_policy_model


class TreeDiffusionTrainStepTests(unittest.TestCase):
    def test_validate_tree_diffusion_batch_accepts_real_batch(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        batch = _real_batch(tokenizer)

        validate_tree_diffusion_batch(
            batch,
            pad_token_id=tokenizer.pad_id,
            require_metadata=True,
        )

    def test_validate_tree_diffusion_batch_catches_bad_masks(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        batch = _real_batch(tokenizer)

        bad_input = _clone_batch(batch)
        bad_input["input_attention_mask"][0, 0] = 1 - bad_input["input_attention_mask"][0, 0]
        with self.assertRaisesRegex(ValueError, "input_attention_mask"):
            validate_tree_diffusion_batch(bad_input, pad_token_id=tokenizer.pad_id)

        bad_target = _clone_batch(batch)
        bad_target["target_attention_mask"][0, 0] = 1 - bad_target["target_attention_mask"][0, 0]
        with self.assertRaisesRegex(ValueError, "target_attention_mask"):
            validate_tree_diffusion_batch(bad_target, pad_token_id=tokenizer.pad_id)

    def test_validate_tree_diffusion_batch_catches_bad_labels(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        batch = _real_batch(tokenizer)

        bad_pad = _clone_batch(batch)
        pad_positions = bad_pad["target_ids"].eq(tokenizer.pad_id).nonzero()
        self.assertGreater(pad_positions.size(0), 0)
        row, col = pad_positions[0].tolist()
        bad_pad["labels"][row, col] = tokenizer.pad_id
        with self.assertRaisesRegex(ValueError, "pad"):
            validate_tree_diffusion_batch(bad_pad, pad_token_id=tokenizer.pad_id)

        bad_nonpad = _clone_batch(batch)
        nonpad_positions = bad_nonpad["target_ids"].ne(tokenizer.pad_id).nonzero()
        row, col = nonpad_positions[0].tolist()
        bad_nonpad["labels"][row, col] = (int(bad_nonpad["target_ids"][row, col]) + 1) % tokenizer.vocab_size
        with self.assertRaisesRegex(ValueError, "not pad"):
            validate_tree_diffusion_batch(bad_nonpad, pad_token_id=tokenizer.pad_id)

    def test_train_step_updates_parameters(self) -> None:
        torch.manual_seed(123)
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        batch = _real_batch(tokenizer)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        output = tree_diffusion_train_step(model, batch, optimizer, tokenizer=tokenizer)

        self.assertTrue(torch.isfinite(torch.tensor(output.loss)).item())
        self.assertIsNotNone(output.grad_norm)
        assert output.grad_norm is not None
        self.assertGreater(output.grad_norm, 0.0)
        changed = any(
            not torch.equal(parameter.detach(), previous)
            for parameter, previous in zip(model.parameters(), before)
        )
        self.assertTrue(changed)

    def test_eval_step_does_not_update_parameters(self) -> None:
        torch.manual_seed(123)
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer)
        batch = _real_batch(tokenizer)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        output = tree_diffusion_eval_step(model, batch, tokenizer=tokenizer)

        self.assertTrue(torch.isfinite(torch.tensor(output.loss)).item())
        self.assertIsNone(output.grad_norm)
        for parameter, previous in zip(model.parameters(), before):
            self.assertTrue(torch.equal(parameter.detach(), previous))

    def test_gradient_norm_detects_invalid_gradients(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        parameter.grad = torch.tensor([float("nan"), 1.0])

        with self.assertRaises(RuntimeError):
            compute_gradient_norm([parameter])

    def test_fixed_batch_overfit_loss_decreases(self) -> None:
        torch.manual_seed(123)
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer, d_model=32, d_ff=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        batch = _real_batch(tokenizer)
        initial = tree_diffusion_eval_step(model, batch, tokenizer=tokenizer).loss

        history = overfit_fixed_batch(
            model,
            batch,
            optimizer,
            tokenizer=tokenizer,
            steps=20,
            grad_clip_norm=1.0,
        )

        self.assertEqual(len(history), 20)
        self.assertLess(history[-1].loss, initial)

    def test_inspect_batch_predictions_returns_valid_decoded_tokens(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer, d_model=32, d_ff=64)
        batch = _real_batch(tokenizer)

        records = inspect_batch_predictions(model, batch, tokenizer, num_examples=2)

        self.assertGreater(len(records), 0)
        self.assertLessEqual(len(records), 2)
        for record in records:
            self.assertIn("predicted_tokens", record)
            self.assertIsInstance(record["predicted_tokens"], list)
            for token in record["predicted_tokens"]:
                self.assertIn(token, tokenizer.token_to_id)
            self.assertIn("target_tokens", record)
            self.assertTrue(record["target_tokens"][0].startswith("<POS_"))

    def test_device_movement_preserves_metadata(self) -> None:
        torch.manual_seed(123)
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        batch = _real_batch(tokenizer)
        input_tokens_before = list(batch["input_tokens"])
        target_tokens_before = list(batch["target_tokens"])

        tree_diffusion_train_step(
            model,
            batch,
            optimizer,
            tokenizer=tokenizer,
            device="cpu",
        )

        self.assertEqual(batch["input_tokens"], input_tokens_before)
        self.assertEqual(batch["target_tokens"], target_tokens_before)

    def test_train_eval_outputs_include_diagnostics(self) -> None:
        torch.manual_seed(123)
        tokenizer = TreeDiffusionTokenizer(max_positions=128)
        model = _small_model(tokenizer)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        batch = _real_batch(tokenizer)

        output = tree_diffusion_train_step(model, batch, optimizer, tokenizer=tokenizer)

        self.assertIsNotNone(output.input_length_mean)
        self.assertIsNotNone(output.target_length_mean)
        self.assertIsNotNone(output.random_init_fraction)
        self.assertIsNotNone(output.num_mutations_mean)


def _pairs() -> list[IntegrationPair]:
    return sample_integration_pairs()


def _real_batch(tokenizer: TreeDiffusionTokenizer) -> dict:
    loader = make_tree_diffusion_dataloader(
        _pairs(),
        tokenizer=tokenizer,
        batch_size=2,
        num_workers=0,
        sigma_small=2,
        smax=2,
        rho=0.2,
        max_input_length=128,
        max_target_length=32,
        base_seed=123,
        shuffle_pairs=False,
        include_metadata=True,
    )
    return next(iter(loader))


def _small_model(
    tokenizer: TreeDiffusionTokenizer,
    *,
    d_model: int = 64,
    d_ff: int = 128,
):
    return small_policy_model(tokenizer, d_model=d_model, d_ff=d_ff)


def _clone_batch(batch: dict) -> dict:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


if __name__ == "__main__":
    unittest.main()
