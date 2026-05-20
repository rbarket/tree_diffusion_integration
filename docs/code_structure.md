# Code Structure

## Core Math And Data

- `src/mathlang/`: prefix AST parsing, serialization, canonicalization, grammar, and SymPy conversions.
- `src/tree_diffusion/dataset.py`: online integration-pair loading and tree-diffusion dataloader construction.
- `src/tree_diffusion/precomputed_dataset.py`: precomputed dataset loading and tokenizer metadata.

## Tree Diffusion Model

- `src/tree_diffusion/tokenizer.py`: tree-diffusion token vocabulary and numeric bucket tokens.
- `src/tree_diffusion/model.py`: policy model config and model implementation.
- `src/tree_diffusion/train_step.py`: train/eval step helpers and batch diagnostics.
- `src/training/lightning/`: Lightning data module, module, callbacks, and wandb integration.
- `src/training/tree_diffusion_config.py`: training configuration and validation.
- `src/training/workflows/tree_diffusion.py`: public Lightning training CLI.

## Precompute

- `src/tree_diffusion/precompute_config.py`: precompute config loading and validation.
- `src/tree_diffusion/precompute_records.py`: precomputed record and trajectory conversion.
- `src/tree_diffusion/precompute_dataset.py`: public precompute CLI and worker orchestration.
- `src/tree_diffusion/precompute_runner.py` and `src/tree_diffusion/precompute_resume.py`: compatibility entrypoints for runner/resume helpers.

## Search And Evaluation

- `src/tree_diffusion/decoding.py`: edit-token decoding and candidate application.
- `src/tree_diffusion/search_common.py`: shared repair/beam numeric, structural, observation, and timeout helpers.
- `src/tree_diffusion/repair.py`: greedy repair implementation.
- `src/tree_diffusion/beam_search.py`: beam repair implementation.
- `src/tree_diffusion/runtime.py`: inference model/tokenizer loading and evaluation dataloader helpers.
- `src/tree_diffusion/eval_metrics.py`: shared repair metric and grouping helpers.
- `src/tree_diffusion/evaluation_common.py`: shared evaluation batch/metadata/residual-worker helpers.
- `src/tree_diffusion/eval_one_step.py`, `evaluate_repair.py`, and `evaluate_beam_search.py`: public evaluation CLIs.

## Experiments And Legacy

- `src/tree_diffusion/experiments/`: one-step inference, resumable greedy/beam repair evaluation, policy validation, and hybrid MDLM repair experiments.
- `src/tree_diffusion/experiments/resumable.py`: shared part-file and resume utilities.
- `src/tree_diffusion/legacy/`: compatibility-only legacy utilities, including the old training preflight audit.
- Root `tree_diffusion/` and `training/` packages: compatibility wrappers for `python -m` entrypoints.
