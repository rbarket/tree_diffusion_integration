# Tree Diffusion for Symbolic Integration

This repository implements a tree-diffusion edit policy for symbolic integration. The model edits a valid antiderivative AST, differentiates the current candidate, compares that derivative against the target integrand, and repeats repair until the derivative residual is small or the search budget is exhausted.

The implementation adapts the syntax-tree diffusion framework from Kapur, Jenner, and Russell to antiderivative generation. The important domain mapping is:

| Tree-diffusion paper | Symbolic-integration implementation |
| --- | --- |
| program syntax tree `z_t` | current antiderivative AST `I_t` |
| target output `x_0` | target integrand `f` |
| current program output `x_t` | current derivative `g_t = dI_t/dx` |
| image/output difference | symbolic and numeric residual `r_t = canon(g_t - f)` |
| reverse edit target | first useful AST edit toward the canonical target antiderivative |

## Current implementation scope

Implemented components:

- prefix parser/serializer and typed math ASTs in `src/mathlang/`
- conservative canonicalization with separate handling for antiderivatives versus integrands/residuals
- grammar-valid AST mutation and random valid AST sampling
- reverse-path edit labels from corrupted current trees to canonical target antiderivatives
- observation construction with derivative, residual, and numeric probe features
- deterministic tokenizer with math, field, numeric-bucket, and position tokens
- precomputed training/validation example generation
- encoder-decoder Transformer edit policy
- Lightning training workflow
- one-step edit diagnostics
- greedy repair evaluation
- beam-search repair evaluation
- resumable validation runners and an optional MDLM-seeded hybrid repair experiment

Final benchmark numbers are not committed here. This repository is the implementation and evaluation harness.

## Repository layout

```text
config/
  precompute/      JSON configs for fixed precomputed examples
  train/           JSON configs for Lightning policy training
  experiments/     small policy-validation experiment configs
src/
  mathlang/        AST, grammar, parser, serializer, canonicalization, SymPy conversion
  tree_diffusion/  mutation, observations, labels, tokenizer, model, precompute, repair, eval
  training/        Lightning modules, callbacks, config, training workflow
  data/            legacy prefix preprocessing helpers
  eval/            legacy symbolic-evaluation helpers
tree_diffusion/    compatibility wrappers for `python -m tree_diffusion...` entrypoints
training/          compatibility wrappers for `python -m training...` entrypoints
docs/              implementation notes and focused workflow docs
tests/             unit, smoke, and workflow tests
data/              ignored generated/input data; only `.gitkeep` files are committed
runs/              ignored training outputs/checkpoints; only `.gitkeep` files are committed
artifacts/         ignored evaluation artifacts; only `.gitkeep` files are committed
```

Most implementation imports currently use the `src.*` package path. The root-level `tree_diffusion/` and `training/` packages are compatibility wrappers for command-line entrypoints.

## Setup

Python 3.10+ is expected.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The project dependencies are declared in `pyproject.toml`. GPU training additionally requires a PyTorch build compatible with the machine CUDA stack.

## Data expectations

Most workflows expect a processed parquet file at:

```text
data/processed/train_prefix_filtered.parquet
```

The default column names are:

```text
integrand_prefix
integral_prefix
```

Each row is one supervised integration pair `(f, I*)` in space-separated prefix tokens. Example prefix fragments look like:

```text
div pow x INT+ 3 INT+ 3
sin x
ln x
```

Generated data, checkpoints, parquet shards, JSONL files, and run artifacts are ignored by git. Copy the processed parquet into `data/processed/` before running precompute, training, or validation. If starting from raw infix data, `src/data/preprocess_prefix_dataset.py` contains a legacy helper, but it expects the legacy vocabulary file used by the earlier token-level integration repo.

## Core workflows

### 1. Precompute tree-diffusion examples

Precomputation generates fixed supervised edit examples and validation shards. This is preferred for repeatable training and validation because observation construction uses SymPy and can be slow online.

Small smoke precompute:

```bash
python -m tree_diffusion.precompute_dataset \
  --config config/precompute/tree_diffusion_precompute_light.json \
  --overwrite
```

Larger training precompute:

```bash
python -m tree_diffusion.precompute_dataset \
  --config config/precompute/tree_diffusion_precompute_train.json \
  --resume
```

Useful overrides:

```bash
python -m tree_diffusion.precompute_dataset \
  --config config/precompute/tree_diffusion_precompute_train.json \
  --input-data data/processed/train_prefix_filtered.parquet \
  --output-dir data/precomputed/tree_diffusion_v1 \
  --train-limit 1000000 \
  --val-limit 10000 \
  --num-workers 32 \
  --resume
```

Important outputs:

```text
data/precomputed/tree_diffusion_v1/
  metadata.json
  tokenizer_metadata.json
  audit_summary.json
  train/shard_*.parquet
  train/audit_summary.json
  val/shard_*.parquet
  val/audit_summary.json
```

Use `--overwrite` for a clean rebuild. Use `--resume` to continue a compatible interrupted run. Do not combine `--overwrite` and `--resume`.

### 2. Train the policy

Online training directly samples examples from the processed parquet:

```bash
python -m training.workflows.tree_diffusion \
  --config config/train/tree_diffusion.json
```

Precomputed training uses fixed shards:

```bash
python -m training.workflows.tree_diffusion \
  --config config/train/tree_diffusion.json \
  --use-precomputed \
  --precomputed-data-dir data/precomputed/tree_diffusion_v1 \
  --output-dir runs/tree_diffusion_precomputed
```

Small/tiny training run for debugging:

```bash
python -m training.workflows.tree_diffusion \
  --config config/train/tree_diffusion.json \
  --train-data data/processed/train_prefix_filtered.parquet \
  --output-dir runs/tree_diffusion_smoke \
  --num-epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --enable-wandb false
```

Resume from a saved checkpoint:

```bash
python -m training.workflows.tree_diffusion \
  --config config/train/tree_diffusion.json \
  --use-precomputed \
  --precomputed-data-dir data/precomputed/tree_diffusion_v1 \
  --resume-from runs/tree_diffusion_precomputed/checkpoint_step_latest.pt \
  --output-dir runs/tree_diffusion_precomputed
```

Common outputs:

```text
runs/<run_name>/
  metrics.jsonl
  checkpoint_step_latest.pt
  checkpoint_best.pt
  lightning/last.ckpt
  lightning/best.ckpt
```

Both legacy `.pt` checkpoints and Lightning `.ckpt` checkpoints are supported by the inference loaders.

### 3. One-step edit evaluation

One-step evaluation checks whether the model can decode one valid, applicable edit from held-out currents. It is a diagnostic for edit-position prediction, replacement parsing, structural improvement, and optional numeric residual improvement.

```bash
python -m src.tree_diffusion.eval_one_step \
  --checkpoint runs/tree_diffusion_precomputed/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/tree_diffusion_v1 \
  --precomputed-split val \
  --num-batches 5 \
  --batch-size 32 \
  --candidate-k 8 \
  --use-first-applicable-candidate \
  --output artifacts/test_summaries/one_step_eval.json \
  --device auto
```

To inspect raw position-token behavior, disable the first-token position mask:

```bash
python -m src.tree_diffusion.eval_one_step \
  --checkpoint runs/tree_diffusion_precomputed/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/tree_diffusion_v1 \
  --no-constrain-position
```

### 4. Greedy repair on validation data

Greedy repair repeatedly proposes top-k edits, applies valid candidates, and chooses the next state using residual-based scoring.

```bash
python -m tree_diffusion.evaluate_repair \
  --checkpoint runs/tree_diffusion_precomputed/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/tree_diffusion_v1 \
  --precomputed-split val \
  --num-pairs 512 \
  --batch-size 32 \
  --num-batches 16 \
  --candidate-k 8 \
  --max-steps 10 \
  --patience 2 \
  --output artifacts/test_summaries/greedy_repair_summary.json \
  --dump-examples artifacts/test_summaries/greedy_repair_examples.jsonl \
  --num-dump-examples 50 \
  --device auto
```

Key summary fields include success rate, exact symbolic match rate, numeric success rate, residual-improvement rate, stop reasons, and stratified metrics by corruption source and mutation count.

### 5. Beam-search repair on validation data

Beam search keeps multiple valid AST states and scores candidates using numeric residual, tree size, step count, and policy log probability.

```bash
python -m tree_diffusion.evaluate_beam_search \
  --checkpoint runs/tree_diffusion_precomputed/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/tree_diffusion_v1 \
  --precomputed-split val \
  --num-pairs 512 \
  --batch-size 16 \
  --num-batches 32 \
  --beam-size 8 \
  --candidate-k 8 \
  --max-steps 10 \
  --numeric-patience 5 \
  --residual-workers 16 \
  --output artifacts/test_summaries/beam_repair_summary.json \
  --dump-examples artifacts/test_summaries/beam_repair_examples.jsonl \
  --num-dump-examples 50 \
  --device auto
```

For longer validation runs, prefer the resumable runners under `src/tree_diffusion/experiments/repair_eval_resumable.py` and `src/tree_diffusion/experiments/beam_eval_resumable.py`.

### 6. Optional MDLM-seeded hybrid repair

The hybrid experiment parses MDLM candidate antiderivatives into the tree grammar and runs beam repair from every parseable seed.

```bash
python -m tree_diffusion.experiments.hybrid_mdlm_repair \
  --predictions artifacts/hybrid/mdlm_tree/mdlm_predictions.jsonl \
  --tree-checkpoint runs/tree_diffusion_precomputed/checkpoint_best.pt \
  --output artifacts/hybrid/mdlm_tree/hybrid_repair_summary.json \
  --examples-out artifacts/hybrid/mdlm_tree/hybrid_repair_examples.jsonl \
  --beam-size 8 \
  --candidate-k 8 \
  --max-steps 10 \
  --numeric-patience 5 \
  --seed-selection all_parseable \
  --residual-workers 16 \
  --device auto
```

See `docs/hybrid_mdlm_tree_repair.md` for the expected prediction JSONL format and hybrid metrics.

## Tests and quality checks

Run tests through Python so the repository root is on `sys.path`:

```bash
python -m pytest -q
```

Useful focused checks:

```bash
python -m pytest -q tests/test_ast_roundtrip.py tests/test_canonicalization.py tests/test_mutation.py
python -m pytest -q tests/test_tree_diffusion_training_examples.py tests/test_tree_diffusion_decoding.py
python -m pytest -q tests/test_tree_diffusion_repair.py tests/test_tree_diffusion_beam_search.py
```

Training and workflow tests require the dependencies in `pyproject.toml`, including Lightning. Some smoke tests create temporary parquet data and checkpoints; committed datasets and checkpoints are not required for those tests.

## Implementation details to preserve

- All training and inference states should remain valid ASTs.
- Antiderivative canonicalization may strip top-level additive constants; integrands, derivatives, and residuals must not.
- Position tokens are preorder node ids for the current canonical tree only. They are re-indexed after every edit.
- Reverse edit paths are supervised labels only. They must not be included in inference-time observations.
- Replacement decoding is currently validated after generation; only the first edit-position token is constrained by default.
- Beam search intentionally uses derivative residual scoring instead of a learned value network.
- Random noising defaults are `sigma_small=2`, `smax=5`, and `rho=0.2` unless a config overrides them.

## Documentation index

- `docs/code_structure.md`: module map
- `docs/refactor_notes.md`: recent refactor compatibility notes
- `docs/tree_diffusion_decoding_and_one_step_eval.md`: edit decoding and one-step metrics
- `docs/hybrid_mdlm_tree_repair.md`: MDLM-seeded repair workflow
- `notebooks/mutation_walkthrough.ipynb`: mutation examples and sanity checks

## Handoff checklist

Before transferring ownership, make sure the team has:

- the processed parquet dataset or clear instructions for regenerating it
- the exact precompute config used for any published run
- `metadata.json`, `audit_summary.json`, and tokenizer metadata for every precomputed dataset
- the training config, checkpoint, and metrics JSONL for every model to be reused
- one-step, greedy, and beam validation summaries for the chosen checkpoint
- a short note identifying whether validation used online or precomputed currents
- the dependency installation command used on the training machine
- the random seeds and hardware details for important runs
- known failure modes, especially SymPy timeouts, unsupported prefix tokens, and invalid replacement decoding
