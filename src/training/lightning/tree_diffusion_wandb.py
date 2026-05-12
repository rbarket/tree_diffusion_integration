from __future__ import annotations

import importlib
import os
import sys
from dataclasses import asdict
from typing import Any, Mapping


GLOBAL_STEP_METRIC = "trainer/global_step"


TREE_DIFFUSION_METRIC_DESCRIPTIONS: dict[str, str] = {
    "train/loss": "Tree-diffusion edit policy cross-entropy on training batches.",
    "train/position_accuracy": "Accuracy of the predicted edit-position token on training batches.",
    "train/token_accuracy": "Token accuracy over supervised target tokens on training batches.",
    "train/grad_norm": "Global gradient norm after optional clipping.",
    "train/input_length_mean": "Mean encoded input length for the training batch.",
    "train/target_length_mean": "Mean encoded target length for the training batch.",
    "train/random_init_fraction": "Fraction of training examples generated from random initialization.",
    "train/num_mutations_mean": "Mean number of mutations used to generate training examples.",
    "train/data_wait_seconds": "Seconds spent waiting for the next training batch.",
    "train/train_step_seconds": "Seconds spent in the training step.",
    "train/lr": "Current optimizer learning rate.",
    "val/loss": "Tree-diffusion edit policy cross-entropy on validation batches.",
    "val/position_accuracy": "Accuracy of the predicted edit-position token on validation batches.",
    "val/token_accuracy": "Token accuracy over supervised target tokens on validation batches.",
    "val/input_length_mean": "Mean encoded input length for validation batches.",
    "val/target_length_mean": "Mean encoded target length for validation batches.",
    "val/random_init_fraction": "Fraction of validation examples generated from random initialization.",
    "val/num_mutations_mean": "Mean number of mutations used to generate validation examples.",
    "val/validation_held_out": "Whether validation uses held-out examples or split data.",
    "diagnostic/examples": "Number of examples inspected by one-step edit diagnostics.",
    "diagnostic/valid_position_rate": "Fraction of diagnostic predictions with a valid edit position.",
    "diagnostic/parseable_replacement_rate": "Fraction of diagnostic predictions with parseable replacements.",
    "diagnostic/applicable_edit_rate": "Fraction of diagnostic predictions that can be applied.",
    "diagnostic/structural_improvement_rate": "Fraction of diagnostic edits improving structural distance.",
    "diagnostic/numeric_residual_improvement_rate": "Fraction of diagnostic edits improving numeric residual.",
    "diagnostic/exact_target_rate": "Fraction of diagnostic edits exactly reaching the target.",
    "diagnostic/mean_structural_distance_before": "Mean structural distance before diagnostic edits.",
    "diagnostic/mean_structural_distance_after": "Mean structural distance after diagnostic edits.",
}


class TreeDiffusionWandbTracker:
    def __init__(
        self,
        *,
        run: Any | None,
        metric_descriptions: Mapping[str, str],
    ) -> None:
        self.run = run
        self.metric_descriptions = dict(metric_descriptions)

    def track_prefixed_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        prefix: str,
        step: int,
    ) -> None:
        payload = {
            f"{prefix}/{name}": float(value)
            for name, value in metrics.items()
            if value is not None and f"{prefix}/{name}" in self.metric_descriptions
        }
        self.track_many(payload, step=step)

    def track_many(self, metrics: Mapping[str, Any], *, step: int) -> None:
        if self.run is None:
            return
        payload = {
            str(name): float(value)
            for name, value in metrics.items()
            if value is not None and str(name) in self.metric_descriptions
        }
        if not payload:
            return
        payload[GLOBAL_STEP_METRIC] = int(step)
        self.run.log(payload)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()


def build_tree_diffusion_wandb_tracker(
    cfg: Any,
    model_cfg: Any,
    *,
    run_id: str | None = None,
    resume: str | None = None,
) -> TreeDiffusionWandbTracker:
    if not bool(getattr(cfg, "enable_wandb", False)):
        print("W&B tracking disabled.")
        return TreeDiffusionWandbTracker(run=None, metric_descriptions=TREE_DIFFUSION_METRIC_DESCRIPTIONS)

    try:
        wandb = _import_wandb_sdk()
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed but enable_wandb=true. Install wandb or set enable_wandb=false."
        ) from exc

    hparams = {
        "train_cfg": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(vars(cfg)),
        "model_cfg": asdict(model_cfg) if hasattr(model_cfg, "__dataclass_fields__") else dict(vars(model_cfg)),
    }
    init_kwargs = {
        "project": getattr(cfg, "wandb_project"),
        "name": getattr(cfg, "wandb_run_name"),
        "id": run_id if run_id is not None else getattr(cfg, "wandb_run_id", None),
        "resume": resume if resume is not None else getattr(cfg, "wandb_resume", None),
        "entity": getattr(cfg, "wandb_entity", None),
        "dir": getattr(cfg, "wandb_dir", None),
        "mode": getattr(cfg, "wandb_mode", None),
        "config": {
            "hparams": hparams,
            "metric_descriptions": dict(TREE_DIFFUSION_METRIC_DESCRIPTIONS),
        },
    }
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "W&B initialization failed. Run `wandb login`, set WANDB_MODE=offline, "
            "or disable tracking with enable_wandb=false."
        ) from exc

    run.define_metric(GLOBAL_STEP_METRIC)
    run.define_metric("*", step_metric=GLOBAL_STEP_METRIC)
    if hasattr(run, "config"):
        metric_metadata = {
            name: {"description": description}
            for name, description in TREE_DIFFUSION_METRIC_DESCRIPTIONS.items()
        }
        run.config.update({"metric_metadata": metric_metadata}, allow_val_change=True)
    print(
        "W&B tracking active:",
        f"project={getattr(cfg, 'wandb_project')}",
        f"run_name={getattr(run, 'name', None)}",
        f"run_id={getattr(run, 'id', None)}",
    )
    return TreeDiffusionWandbTracker(run=run, metric_descriptions=TREE_DIFFUSION_METRIC_DESCRIPTIONS)


def _import_wandb_sdk() -> Any:
    try:
        wandb_module = importlib.import_module("wandb")
    except ModuleNotFoundError:
        raise
    if hasattr(wandb_module, "init"):
        return wandb_module

    workspace_root = os.path.abspath(os.getcwd())
    shadow_dir = os.path.join(workspace_root, "wandb")
    original_sys_path = list(sys.path)
    sys.modules.pop("wandb", None)
    filtered_sys_path = [
        path
        for path in original_sys_path
        if os.path.abspath(os.path.join(path or workspace_root, "wandb")) != shadow_dir
    ]
    try:
        sys.path = filtered_sys_path
        wandb_module = importlib.import_module("wandb")
    finally:
        sys.path = original_sys_path

    if not hasattr(wandb_module, "init"):
        raise ModuleNotFoundError(
            "Imported module 'wandb' does not expose the W&B SDK. "
            "A local directory named 'wandb' is shadowing the installed package."
        )
    return wandb_module


__all__ = [
    "GLOBAL_STEP_METRIC",
    "TREE_DIFFUSION_METRIC_DESCRIPTIONS",
    "TreeDiffusionWandbTracker",
    "build_tree_diffusion_wandb_tracker",
]
