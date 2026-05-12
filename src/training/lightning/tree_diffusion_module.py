from __future__ import annotations

import time
from typing import Any, Mapping

import lightning.pytorch as pl
import torch

from src.tree_diffusion.model import TreeDiffusionModelConfig, TreeDiffusionPolicyModel
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.train_step import (
    TrainStepOutput,
    compute_gradient_norm,
    train_step_output_from_forward,
    tree_diffusion_forward_loss,
)
from src.training.lightning.tree_diffusion_wandb import build_tree_diffusion_wandb_tracker


class TreeDiffusionLightningModule(pl.LightningModule):
    automatic_optimization = False

    def __init__(
        self,
        *,
        model: TreeDiffusionPolicyModel,
        model_cfg: TreeDiffusionModelConfig,
        cfg: Any,
        tokenizer: TreeDiffusionTokenizer,
        output_dir: str,
        validation_held_out: bool,
        target_step: int,
        legacy_step: int = 0,
        best_val_loss: float | None = None,
        legacy_optimizer_state: Mapping[str, Any] | None = None,
        wandb_run_id: str | None = None,
        wandb_resume: str | None = None,
        wandb_run_name: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.model_cfg = model_cfg
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.output_dir = str(output_dir)
        self.validation_held_out = bool(validation_held_out)
        self.target_step = int(target_step)
        self.legacy_step = int(legacy_step)
        self.best_val_loss = best_val_loss
        self.final_train_output: TrainStepOutput | None = None
        self.last_checkpoint: str | None = None
        self.best_checkpoint: str | None = None
        self.wandb_run_id = wandb_run_id
        self.wandb_resume = wandb_resume
        self.wandb_run_name = wandb_run_name
        self._legacy_optimizer_state = dict(legacy_optimizer_state) if legacy_optimizer_state is not None else None
        self._data_wait_seconds: float = 0.0
        self._last_train_metrics: dict[str, float] = {}
        self._last_train_output: TrainStepOutput | None = None
        self._last_train_step_seconds: float = 0.0
        self._last_lr: float = float(cfg.lr)
        self.latest_val_metrics: dict[str, float] = {}
        self.latest_diagnostic_metrics: dict[str, float] = {}
        self.latest_eval_step: int | None = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
            betas=tuple(float(value) for value in self.cfg.betas),
        )

    def on_fit_start(self) -> None:
        if self._legacy_optimizer_state is None:
            return
        optimizer = self._raw_optimizer()
        try:
            optimizer.load_state_dict(self._legacy_optimizer_state)
        except Exception:
            print("Resume note: failed to load legacy optimizer state into Lightning optimizer.")
        self._legacy_optimizer_state = None

    def set_data_wait_seconds(self, value: float) -> None:
        self._data_wait_seconds = float(value)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        train_step_start = time.time()
        self.model.train()
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)
        forward_output = tree_diffusion_forward_loss(
            self.model,
            batch,
            tokenizer=self.tokenizer,
            device=None,
            validate_batch=True,
        )
        self.manual_backward(forward_output.loss)
        if self.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip_norm))
        grad_norm = compute_gradient_norm(self.model.parameters())
        if grad_norm <= 0.0:
            raise RuntimeError("No finite nonzero gradients were produced.")
        optimizer.step()

        output = train_step_output_from_forward(forward_output, grad_norm=grad_norm)
        self.legacy_step += 1
        self.final_train_output = output
        self._last_train_output = output
        self._last_train_step_seconds = time.time() - train_step_start
        self._last_lr = self.current_lr()
        self._last_train_metrics = self.training_metrics_for_output(
            output,
            lr=self._last_lr,
            data_wait_seconds=self._data_wait_seconds,
            train_step_seconds=self._last_train_step_seconds,
        )
        self.log("train/loss", float(output.loss), on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return forward_output.loss.detach()

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        del batch, batch_idx
        return None

    def current_lr(self) -> float:
        return float(self._raw_optimizer().param_groups[0]["lr"])

    def last_train_tracking_metrics(self) -> dict[str, float]:
        return dict(self._last_train_metrics)

    def last_train_output(self) -> TrainStepOutput | None:
        return self._last_train_output

    def last_train_step_seconds(self) -> float:
        return float(self._last_train_step_seconds)

    def data_wait_seconds(self) -> float:
        return float(self._data_wait_seconds)

    def record_active_wandb_run(self, run: Any) -> None:
        run_id = getattr(run, "id", None)
        run_name = getattr(run, "name", None)
        if run_id is not None:
            self.wandb_run_id = str(run_id)
        if run_name is not None:
            self.wandb_run_name = str(run_name)

    def record_eval_tracking_metrics(
        self,
        *,
        step: int,
        val_metrics: Mapping[str, Any],
        diagnostic_metrics: Mapping[str, Any],
    ) -> None:
        self.latest_eval_step = int(step)
        self.latest_val_metrics = {
            str(name): float(value)
            for name, value in val_metrics.items()
            if value is not None
        }
        self.latest_diagnostic_metrics = {
            str(name): float(value)
            for name, value in diagnostic_metrics.items()
            if value is not None
        }

    def build_wandb_tracker(self) -> Any:
        return build_tree_diffusion_wandb_tracker(
            self.cfg,
            self.model_cfg,
            run_id=self.wandb_run_id,
            resume=self.wandb_resume,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["legacy_step"] = int(self.legacy_step)
        checkpoint["target_step"] = int(self.target_step)
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["last_checkpoint"] = self.last_checkpoint
        checkpoint["best_checkpoint"] = self.best_checkpoint
        checkpoint["validation_held_out"] = self.validation_held_out
        checkpoint["tokenizer"] = {
            "vocab_size": self.tokenizer.vocab_size,
            "max_positions": self.tokenizer.max_positions,
            "pad_id": self.tokenizer.pad_id,
            "bos_id": self.tokenizer.bos_id,
            "eos_id": self.tokenizer.eos_id,
        }
        checkpoint["model_cfg"] = vars(self.model_cfg)
        checkpoint["wandb_run_id"] = self.wandb_run_id
        checkpoint["wandb_run_name"] = self.wandb_run_name
        checkpoint["latest_val_metrics"] = dict(self.latest_val_metrics)
        checkpoint["latest_diagnostic_metrics"] = dict(self.latest_diagnostic_metrics)
        checkpoint["latest_eval_step"] = self.latest_eval_step

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        self.legacy_step = int(checkpoint.get("legacy_step", self.legacy_step))
        self.target_step = int(checkpoint.get("target_step", self.target_step))
        raw_best = checkpoint.get("best_val_loss", self.best_val_loss)
        self.best_val_loss = None if raw_best is None else float(raw_best)
        last_checkpoint = checkpoint.get("last_checkpoint")
        best_checkpoint = checkpoint.get("best_checkpoint")
        self.last_checkpoint = None if last_checkpoint is None else str(last_checkpoint)
        self.best_checkpoint = None if best_checkpoint is None else str(best_checkpoint)
        self.validation_held_out = bool(checkpoint.get("validation_held_out", self.validation_held_out))
        wandb_run_id = checkpoint.get("wandb_run_id")
        wandb_run_name = checkpoint.get("wandb_run_name")
        if wandb_run_id is not None:
            self.wandb_run_id = str(wandb_run_id)
        if wandb_run_name is not None:
            self.wandb_run_name = str(wandb_run_name)
        latest_val = checkpoint.get("latest_val_metrics")
        latest_diagnostic = checkpoint.get("latest_diagnostic_metrics")
        latest_eval_step = checkpoint.get("latest_eval_step")
        if isinstance(latest_val, dict):
            self.latest_val_metrics = {
                str(name): float(value)
                for name, value in latest_val.items()
                if isinstance(value, (float, int))
            }
        if isinstance(latest_diagnostic, dict):
            self.latest_diagnostic_metrics = {
                str(name): float(value)
                for name, value in latest_diagnostic.items()
                if isinstance(value, (float, int))
            }
        if latest_eval_step is not None:
            self.latest_eval_step = int(latest_eval_step)

    def _raw_optimizer(self) -> torch.optim.Optimizer:
        optimizer = self.optimizers(use_pl_optimizer=False)
        if isinstance(optimizer, (list, tuple)):
            return optimizer[0]
        return optimizer

    @staticmethod
    def training_metrics_for_output(
        output: TrainStepOutput,
        *,
        lr: float,
        data_wait_seconds: float,
        train_step_seconds: float,
    ) -> dict[str, float]:
        metrics = {
            "loss": output.loss,
            "position_accuracy": output.position_accuracy,
            "token_accuracy": output.token_accuracy,
            "grad_norm": output.grad_norm,
            "input_length_mean": output.input_length_mean,
            "target_length_mean": output.target_length_mean,
            "random_init_fraction": output.random_init_fraction,
            "num_mutations_mean": output.num_mutations_mean,
            "data_wait_seconds": data_wait_seconds,
            "train_step_seconds": train_step_seconds,
            "lr": lr,
        }
        return {
            str(name): float(value)
            for name, value in metrics.items()
            if value is not None
        }


__all__ = ["TreeDiffusionLightningModule"]
