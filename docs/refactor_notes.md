# Refactor Notes

## Modules Created Or Split

- `src/tree_diffusion/runtime.py`: public inference-time checkpoint, tokenizer, dataloader, tensor, and metadata helpers. Old private `eval_one_step.py` helper names remain as thin wrappers.
- `src/tree_diffusion/eval_metrics.py`: shared numeric metric, grouping, metadata coercion, and `RepairGroupSummary` helpers used by greedy and beam evaluation.
- `src/tree_diffusion/search_common.py`: shared repair/beam tree size, derivative matching, structural distance, numeric comparison, observation encoding, and timeout helpers.
- `src/tree_diffusion/evaluation_common.py`: shared batch row, repair input, mutation trace, JSONL, config, and residual-worker helpers for evaluation runners.
- `src/tree_diffusion/experiments/resumable.py`: shared resumable part-file, manifest, config merge, and progress utilities for greedy and beam repair evaluation.
- `src/tree_diffusion/precompute_config.py` and `src/tree_diffusion/precompute_records.py`: extracted precompute configuration validation and precomputed record/trajectory conversion.
- `src/tree_diffusion/precompute_runner.py` and `src/tree_diffusion/precompute_resume.py`: compatibility front doors for precompute runner and resume helpers while remaining orchestration code stays in `precompute_dataset.py`.
- `src/training/tree_diffusion_config.py`: extracted Lightning training configuration loading and validation.
- `src/training/tree_diffusion_builders.py` and `src/training/tree_diffusion_runner.py`: compatibility front doors for training builders and runner entrypoints.
- `src/tree_diffusion/legacy/audit_training_pipeline.py`: legacy preflight audit implementation moved out of the main production module namespace.

## Compatibility Shims

- `src/tree_diffusion/precompute_dataset.py` remains the public CLI and re-exports the precompute config/record APIs.
- `src/training/workflows/tree_diffusion.py` remains the public Lightning training CLI and import surface.
- `src/tree_diffusion/audit_training_pipeline.py` and `tree_diffusion/audit_training_pipeline.py` continue to expose `main` for old preflight commands.
- Root `tree_diffusion/` and `training/` wrappers still support existing `python -m ...` commands, with narrower explicit exports where practical.

## Behavior Preservation

- Checkpoint loading, tokenizer metadata loading, online/precomputed dataloaders, decoding, repair scoring, beam scoring, stopping criteria, precompute schema, Lightning objective, and JSON summary keys were preserved.
- Shared helpers were moved or wrapped without changing formulas or public CLI arguments.
- Obsolete preflight code was quarantined rather than deleted because tests still exercise it as a lightweight pipeline audit.

## Intentional Non-Changes

- No MDLM hybrid algorithm changes, value network, model architecture, training loss, decoding constraints, beam-search algorithm, or precompute schema changes were introduced.
- Long-running smoke tests were not deleted; existing coverage remains active.
- Worker-heavy precompute orchestration remains in `precompute_dataset.py` behind compatibility modules to avoid risky circular import churn in this refactor.
