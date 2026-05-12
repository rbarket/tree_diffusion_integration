from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, TQDMProgressBar

from src.tree_diffusion._common import append_jsonl, diagnostic_metrics
from src.tree_diffusion.validation import run_one_step_edit_diagnostics
from src.training.lightning.tree_diffusion_module import TreeDiffusionLightningModule


EvaluatePolicyFn = Callable[..., dict[str, float]]
SaveLegacyCheckpointFn = Callable[..., None]


class TreeDiffusionTrainingCallback(Callback):
    def __init__(
        self,
        *,
        output_dir: str,
        evaluate_policy: EvaluatePolicyFn,
        save_legacy_checkpoint: SaveLegacyCheckpointFn,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.evaluate_policy = evaluate_policy
        self.save_legacy_checkpoint = save_legacy_checkpoint
        self._start_time = time.time()
        self._last_batch_boundary = time.time()
        self._last_validated_step: int | None = None
        self._best_lightning_ckpt: str | None = None
        self._initial_legacy_step = 0

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "lightning").mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()
        self._last_batch_boundary = time.time()
        if isinstance(pl_module, TreeDiffusionLightningModule):
            self._initial_legacy_step = int(pl_module.legacy_step)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, batch, batch_idx
        if not isinstance(pl_module, TreeDiffusionLightningModule):
            return
        now = time.time()
        pl_module.set_data_wait_seconds(now - self._last_batch_boundary)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        if not isinstance(pl_module, TreeDiffusionLightningModule):
            raise TypeError("TreeDiffusionTrainingCallback expects TreeDiffusionLightningModule.")
        if not _trainer_is_global_zero(trainer):
            self._last_batch_boundary = time.time()
            return

        step = int(pl_module.legacy_step)
        train_output = pl_module.last_train_output()
        if train_output is None:
            return

        if self._should_log_train(pl_module, step):
            self._record_train_metrics(pl_module, step=step)

        if self._should_validate(pl_module, step):
            self._run_validation_and_diagnostics(trainer, pl_module, step=step)
            self._last_validated_step = step

        if step % int(pl_module.cfg.checkpoint_every) == 0:
            self._save_step_checkpoint(trainer, pl_module, step=step)

        self._last_batch_boundary = time.time()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not isinstance(pl_module, TreeDiffusionLightningModule):
            return
        if not _trainer_is_global_zero(trainer):
            return
        step = int(pl_module.legacy_step)
        if step > 0 and self._last_validated_step != step:
            self._run_validation_and_diagnostics(trainer, pl_module, step=step)
            self._last_validated_step = step
        if bool(pl_module.cfg.save_last):
            self._save_last_checkpoint(trainer, pl_module, step=step)

    def _should_log_train(self, pl_module: TreeDiffusionLightningModule, step: int) -> bool:
        return (
            step % int(pl_module.cfg.log_every) == 0
            or step == self._initial_legacy_step + 1
            or step == int(pl_module.target_step)
        )

    def _should_validate(self, pl_module: TreeDiffusionLightningModule, step: int) -> bool:
        return step % int(pl_module.cfg.val_every) == 0 or step == int(pl_module.target_step)

    def _record_train_metrics(self, pl_module: TreeDiffusionLightningModule, *, step: int) -> None:
        output = pl_module.last_train_output()
        if output is None:
            return
        row_metrics = {
            name: value
            for name, value in pl_module.last_train_tracking_metrics().items()
            if name != "lr"
        }
        append_jsonl(
            self.metrics_path,
            _metrics_row(
                split="train",
                step=step,
                elapsed_seconds=time.time() - self._start_time,
                lr=pl_module.current_lr(),
                metrics=row_metrics,
            ),
        )
        _print_train_progress(
            step,
            output,
            pl_module.current_lr(),
            data_wait_seconds=pl_module.data_wait_seconds(),
            train_step_seconds=pl_module.last_train_step_seconds(),
        )

    def _run_validation_and_diagnostics(
        self,
        trainer: pl.Trainer,
        pl_module: TreeDiffusionLightningModule,
        *,
        step: int,
    ) -> None:
        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("Trainer datamodule is required for tree-diffusion validation.")
        val_loader = datamodule.val_dataloader()
        print(
            f"[step {step}] validation_start "
            f"num_batches={pl_module.cfg.val_batches} elapsed={time.time() - self._start_time:.1f}s",
            flush=True,
        )
        val_metrics = self.evaluate_policy(
            pl_module.model,
            val_loader,
            tokenizer=pl_module.tokenizer,
            device=pl_module.device,
            num_batches=int(pl_module.cfg.val_batches),
        )
        val_metrics["validation_held_out"] = float(pl_module.validation_held_out)
        append_jsonl(
            self.metrics_path,
            _metrics_row(
                split="val",
                step=step,
                elapsed_seconds=time.time() - self._start_time,
                lr=pl_module.current_lr(),
                metrics=val_metrics,
            ),
        )
        pl_module.log_dict(
            {f"val/{name}": float(value) for name, value in val_metrics.items() if value is not None},
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
        )
        print(
            f"[step {step}] val_loss={val_metrics['loss']:.4f} "
            f"pos_acc={val_metrics['position_accuracy']:.4f} "
            f"tok_acc={val_metrics['token_accuracy']:.4f}",
            flush=True,
        )

        print(
            f"[step {step}] diagnostics_start "
            f"num_batches={pl_module.cfg.diagnostic_batches} elapsed={time.time() - self._start_time:.1f}s",
            flush=True,
        )
        diagnostics = run_one_step_edit_diagnostics(
            pl_module.model,
            val_loader,
            tokenizer=pl_module.tokenizer,
            device=pl_module.device,
            num_batches=int(pl_module.cfg.diagnostic_batches),
        )
        diagnostic_row = diagnostic_metrics(diagnostics)
        pl_module.record_eval_tracking_metrics(
            step=step,
            val_metrics=val_metrics,
            diagnostic_metrics=diagnostic_row,
        )
        append_jsonl(
            self.metrics_path,
            _metrics_row(
                split="diagnostic",
                step=step,
                elapsed_seconds=time.time() - self._start_time,
                lr=pl_module.current_lr(),
                metrics=diagnostic_row,
            ),
        )
        pl_module.log_dict(
            {
                f"diagnostic/{name}": float(value)
                for name, value in diagnostic_row.items()
                if value is not None
            },
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
        )
        print(
            f"[step {step}] diagnostics "
            f"valid_pos={diagnostics.valid_position_rate:.4f} "
            f"parseable={diagnostics.parseable_replacement_rate:.4f} "
            f"applicable={diagnostics.applicable_edit_rate:.4f} "
            f"struct_improve={diagnostics.structural_improvement_rate:.4f} "
            f"exact={diagnostics.exact_target_rate:.4f}",
            flush=True,
        )

        val_loss = float(val_metrics["loss"])
        if bool(pl_module.cfg.save_best) and (
            pl_module.best_val_loss is None or val_loss < float(pl_module.best_val_loss)
        ):
            pl_module.best_val_loss = val_loss
            lightning_ckpt = self._save_lightning_checkpoint(
                trainer,
                filename=f"best-loss-step_{step}.ckpt",
            )
            if (
                self._best_lightning_ckpt is not None
                and self._best_lightning_ckpt != lightning_ckpt
                and os.path.exists(self._best_lightning_ckpt)
            ):
                os.remove(self._best_lightning_ckpt)
            self._best_lightning_ckpt = lightning_ckpt
            best_path = self.output_dir / "checkpoint_best.pt"
            self._save_legacy(
                best_path,
                pl_module,
                step=step,
                extra={"validation_held_out": pl_module.validation_held_out, "val_metrics": val_metrics},
                lightning_resume_ckpt=lightning_ckpt,
            )
            pl_module.best_checkpoint = str(best_path)
            print(
                f"[step {step}] checkpoint_best_saved path={best_path} "
                f"best_val_loss={val_loss:.4f}",
                flush=True,
            )
        pl_module.model.train()

    def _save_step_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: TreeDiffusionLightningModule,
        *,
        step: int,
    ) -> None:
        lightning_ckpt = self._save_lightning_checkpoint(trainer, filename="last.ckpt")
        step_path = self.output_dir / f"checkpoint_step_{step}.pt"
        self._save_legacy(
            step_path,
            pl_module,
            step=step,
            extra={"validation_held_out": pl_module.validation_held_out},
            lightning_resume_ckpt=lightning_ckpt,
        )
        pl_module.last_checkpoint = str(step_path)
        print(f"[step {step}] checkpoint_saved path={step_path}", flush=True)

    def _save_last_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: TreeDiffusionLightningModule,
        *,
        step: int,
    ) -> None:
        lightning_ckpt = self._save_lightning_checkpoint(trainer, filename="last.ckpt")
        last_path = self.output_dir / "checkpoint_last.pt"
        self._save_legacy(
            last_path,
            pl_module,
            step=step,
            extra={"validation_held_out": pl_module.validation_held_out},
            lightning_resume_ckpt=lightning_ckpt,
        )
        pl_module.last_checkpoint = str(last_path)
        print(f"checkpoint_last_saved path={last_path}", flush=True)

    def _save_lightning_checkpoint(self, trainer: pl.Trainer, *, filename: str) -> str:
        path = self.output_dir / "lightning" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(path))
        return str(path)

    def _save_legacy(
        self,
        path: Path,
        pl_module: TreeDiffusionLightningModule,
        *,
        step: int,
        extra: Mapping[str, Any],
        lightning_resume_ckpt: str,
    ) -> None:
        self.save_legacy_checkpoint(
            path,
            model=pl_module.model,
            optimizer=pl_module._raw_optimizer(),
            config=pl_module.cfg,
            step=step,
            best_val_loss=pl_module.best_val_loss,
            tokenizer=pl_module.tokenizer,
            extra=extra,
            lightning_resume_ckpt=lightning_resume_ckpt,
            wandb_run_id=pl_module.wandb_run_id,
            wandb_run_name=pl_module.wandb_run_name,
        )


class TreeDiffusionWandbCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._tracker: Any = None
        self._last_eval_step_logged: int | None = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if stage != "fit":
            return
        if not _trainer_is_global_zero(trainer):
            return
        if not isinstance(pl_module, TreeDiffusionLightningModule):
            raise TypeError("TreeDiffusionWandbCallback expects TreeDiffusionLightningModule.")
        if self._tracker is None:
            self._tracker = pl_module.build_wandb_tracker()
            run = getattr(self._tracker, "run", None)
            if run is not None:
                pl_module.record_active_wandb_run(run)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        if not _trainer_is_global_zero(trainer):
            return
        if self._tracker is None or not isinstance(pl_module, TreeDiffusionLightningModule):
            return
        metrics = {
            f"train/{name}": value
            for name, value in pl_module.last_train_tracking_metrics().items()
        }
        self._tracker.track_many(metrics, step=int(pl_module.legacy_step))
        eval_step = pl_module.latest_eval_step
        if eval_step is None or eval_step == self._last_eval_step_logged:
            return
        self._tracker.track_prefixed_metrics(
            pl_module.latest_val_metrics,
            prefix="val",
            step=int(eval_step),
        )
        self._tracker.track_prefixed_metrics(
            pl_module.latest_diagnostic_metrics,
            prefix="diagnostic",
            step=int(eval_step),
        )
        self._last_eval_step_logged = int(eval_step)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if not _trainer_is_global_zero(trainer):
            return
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        del exception
        self.on_fit_end(trainer, pl_module)


def build_tree_diffusion_callbacks(
    *,
    output_dir: str,
    evaluate_policy: EvaluatePolicyFn,
    save_legacy_checkpoint: SaveLegacyCheckpointFn,
    enable_progress_bar: bool,
) -> list[Callback]:
    callbacks: list[Callback] = [
        TreeDiffusionTrainingCallback(
            output_dir=output_dir,
            evaluate_policy=evaluate_policy,
            save_legacy_checkpoint=save_legacy_checkpoint,
        ),
        TreeDiffusionWandbCallback(),
    ]
    if enable_progress_bar:
        callbacks.append(TQDMProgressBar())
    return callbacks


def _metrics_row(
    *,
    split: str,
    step: int,
    elapsed_seconds: float,
    lr: float,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "split": split,
        "step": int(step),
        "elapsed_seconds": float(elapsed_seconds),
        "lr": float(lr),
        **dict(metrics),
    }


def _print_train_progress(
    step: int,
    output: Any,
    lr: float,
    *,
    data_wait_seconds: float,
    train_step_seconds: float,
) -> None:
    print(
        f"[step {step}] train_loss={output.loss:.4f} "
        f"pos_acc={_format_optional(output.position_accuracy)} "
        f"tok_acc={_format_optional(output.token_accuracy)} "
        f"grad_norm={_format_optional(output.grad_norm)} lr={lr:.6g} "
        f"data_wait={data_wait_seconds:.3f}s train_step={train_step_seconds:.3f}s",
        flush=True,
    )


def _format_optional(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:.4f}"


def _trainer_is_global_zero(trainer: pl.Trainer) -> bool:
    return bool(getattr(trainer, "is_global_zero", True))


__all__ = [
    "TreeDiffusionTrainingCallback",
    "TreeDiffusionWandbCallback",
    "build_tree_diffusion_callbacks",
]
