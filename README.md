# Tree Diffusion for Symbolic Integration

## Project Purpose

This repository implements tree diffusion for symbolic integration. The editable state is a valid antiderivative AST `I_t`, not a flat token string. At each step, the policy observes the target integrand `f`, the current candidate antiderivative `I_t`, the current derivative `g_t = d/dx I_t`, symbolic and numeric residuals such as `r_t = canon(g_t - f)`, and numeric probe features.

Training labels are reverse edit targets that move a current AST toward the canonical gold antiderivative. Inference supports one-step edit diagnostics, greedy multi-step repair, beam repair over valid AST states, and the hybrid MDLM seed repair path when MDLM prediction JSONL files are available.

## Repository Map

- `src/mathlang/`: AST definitions, parser, serializer, grammar helpers, SymPy conversion, and canonicalization.
- `src/tree_diffusion/`: mutation, edit paths, observations, tokenizer, model, decoding, precompute, repair/search, and evaluation.
- `training/` and `src/training/`: training workflow entrypoints and Lightning integration.
- `tree_diffusion/`: compatibility wrappers for public `python -m tree_diffusion...` commands.
- `config/precompute/`: dataset generation and validation configs.
- `config/train/`: model/training configs accepted by `training.workflows.tree_diffusion`.
- `config/eval/`: one-step, greedy repair, and beam repair eval configs.
- `config/audit/`: preflight/data audit configs.
- `tests/`: unit, smoke, and workflow tests.
- `data/processed/`: place input parquet here.
- `data/precomputed/`: generated precompute outputs; do not commit.
- `runs/`: generated checkpoints and training logs; do not commit.
- `artifacts/`: generated eval summaries/examples; do not commit.

## Environment Setup

If your checkout includes `uv.lock`, prefer:

```bash
uv sync
source .venv/bin/activate
```

Pip fallback:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Then run:

```bash
python -m pytest -q
```

Full training and eval require the training stack in `pyproject.toml`, including PyTorch, Lightning, pandas, pyarrow, SymPy, and optionally wandb.

## Data Layout

Put the processed dataset at:

```text
data/processed/train_prefix_filtered.parquet
```

Required columns:

```text
integrand_prefix
integral_prefix
```

`integrand_prefix` is the target integrand `f` in space-separated prefix notation. `integral_prefix` is the gold antiderivative `I*` in the same notation. The parquet file is not committed to Git; retrieve it from the team artifact store or handoff location.

Schema check:

```bash
python - <<'PY'
import pandas as pd
path = "data/processed/train_prefix_filtered.parquet"
df = pd.read_parquet(path)
print(df.shape)
print(df[["integrand_prefix", "integral_prefix"]].head())
PY
```

## Canonical Fresh-Clone Workflow

### 1. Precompute Smoke Examples

```bash
python -m tree_diffusion.precompute_dataset \
  --config config/precompute/smoke.json \
  --input-data data/processed/train_prefix_filtered.parquet \
  --output-dir data/precomputed/smoke \
  --overwrite
```

Expected outputs:

```text
data/precomputed/smoke/
  metadata.json
  tokenizer_metadata.json
  audit_summary.json
  train/
    shard_*.parquet
    audit_summary.json
  val/
    shard_*.parquet
    audit_summary.json
```

Inspect the precompute audit:

```bash
cat data/precomputed/smoke/audit_summary.json
```

### 2. Train from Precomputed Examples

```bash
python -m training.workflows.tree_diffusion \
  --config config/train/precomputed_smoke.json
```

Expected checkpoint outputs:

```text
runs/tree_diffusion_smoke/
  checkpoint_best.pt
  checkpoint_step_latest.pt
  lightning/
    best.ckpt
    last.ckpt
```

Use this checkpoint for eval:

```text
runs/tree_diffusion_smoke/checkpoint_best.pt
```

### 3. One-Step Validation Eval

```bash
python -m tree_diffusion.experiments.one_step_inference_eval \
  --config config/eval/one_step_smoke.json
```

Equivalent explicit form:

```bash
python -m tree_diffusion.experiments.one_step_inference_eval \
  --checkpoint runs/tree_diffusion_smoke/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/smoke \
  --output-dir artifacts/eval/one_step_smoke \
  --num-batches 5 \
  --batch-size 32 \
  --device auto
```

### 4. Greedy Repair Validation Eval

```bash
python -m tree_diffusion.evaluate_repair \
  --config config/eval/greedy_smoke.json
```

Equivalent explicit form:

```bash
python -m tree_diffusion.evaluate_repair \
  --checkpoint runs/tree_diffusion_smoke/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/smoke \
  --precomputed-split val \
  --output artifacts/eval/greedy_repair_smoke.json \
  --dump-examples artifacts/eval/greedy_repair_smoke_examples.jsonl \
  --num-batches 5 \
  --batch-size 32 \
  --candidate-k 8 \
  --max-steps 10 \
  --device auto
```

### 5. Beam Repair Validation Eval

```bash
python -m tree_diffusion.evaluate_beam_search \
  --config config/eval/beam_smoke.json
```

Equivalent explicit form:

```bash
python -m tree_diffusion.evaluate_beam_search \
  --checkpoint runs/tree_diffusion_smoke/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/smoke \
  --precomputed-split val \
  --output artifacts/eval/beam_repair_smoke.json \
  --dump-examples artifacts/eval/beam_repair_smoke_examples.jsonl \
  --num-batches 5 \
  --batch-size 32 \
  --beam-size 8 \
  --candidate-k 8 \
  --max-steps 10 \
  --numeric-patience 5 \
  --device auto
```

### 6. Fresh-Clone Smoke Script

For a fast end-to-end precompute and help-check smoke run:

```bash
scripts/smoke_handoff.sh
```

The script creates a tiny parquet at `data/processed/handoff_smoke.parquet` if needed, runs precompute into `data/precomputed/handoff_smoke`, checks expected files, and prints the next training/eval commands.

## Config Guide

- `config/precompute/`: data generation, label validation, observation settings, and shard settings.
- `config/train/`: supervised policy training settings. `precomputed_smoke.json` and `precomputed_full.json` are the preferred handoff templates for fixed precomputed examples.
- `config/eval/`: one-step, greedy repair, and beam repair defaults.
- `config/audit/`: lightweight preflight/data audit configs.

Common fields:

- `input_data`: processed parquet for precompute.
- `output_dir`: generated output directory for precompute, training, or directory-based eval.
- `examples_per_pair_train`, `examples_per_pair_val`: number of sampled current states per pair.
- `sigma_small`, `smax`, `rho`: mutation/noise schedule settings.
- `residual_mode`: observation residual mode, usually `both`.
- `validate_labels`: precompute label validation flag.
- `batch_size`, `num_epochs`: training/eval batch and schedule controls.
- `use_precomputed`, `precomputed_data_dir`: train/eval from fixed precomputed shards.
- `checkpoint`: model checkpoint for eval.
- `num_batches`: eval batch budget.
- `candidate_k`: number of edit candidates decoded per state.
- `beam_size`: number of states retained by beam repair.
- `max_steps`: repair depth budget.
- `device`: `auto`, `cpu`, or a torch device string such as `cuda`.

CLI arguments override JSON config values. Keep method-defining fields in configs for reproducibility, and record any CLI overrides with final artifacts.

## Output File Map

- `metadata.json`: tokenizer/config/precompute metadata and example counts.
- `tokenizer_metadata.json`: tokenizer settings needed to load precomputed data.
- `audit_summary.json`: precompute validity, label, observation, and warning summary.
- `shard_*.parquet`: encoded precomputed examples.
- `checkpoint_best.pt`: default checkpoint to use for eval.
- `checkpoint_step_latest.pt`: latest non-Lightning checkpoint.
- `lightning/*.ckpt`: Lightning resume/checkpoint files.
- Eval summary JSON files: aggregate one-step, greedy, beam, or hybrid metrics.
- Eval examples JSONL files: per-example predictions, repair paths, failures, and diagnostics.

## Tests

Full suite:

```bash
python -m pytest -q
```

Focused fast subset:

```bash
python -m pytest -q \
  tests/test_ast_roundtrip.py \
  tests/test_canonicalization.py \
  tests/test_mutation.py \
  tests/test_observation.py \
  tests/test_tree_diffusion_decoding.py \
  tests/test_tree_diffusion_repair.py \
  tests/test_tree_diffusion_beam_search.py \
  tests/test_tree_diffusion_precompute_dataset.py \
  tests/test_tree_diffusion_evaluate_repair.py \
  tests/test_tree_diffusion_evaluate_beam_search.py \
  tests/test_tree_diffusion_beam_eval_resumable.py \
  tests/test_tree_diffusion_repair_eval_resumable.py
```

## Known Limitations

- Replacement-subtree generation is validated after decoding; it may not be fully grammar-constrained during generation.
- SymPy observation construction can fail or timeout for some expressions; partial observations use missing-field tokens and warnings.
- Structural distance metrics use the gold antiderivative and are validation-only, not real inference signals.
- Beam search currently uses numeric residual scoring rather than a learned value network.
- Position tokens are preorder ids for the current tree and are not stable across edits.
- Generated data/checkpoints are large and must live outside Git.

## Artifact Handoff Checklist

Outgoing owner should provide:

- `data/processed/train_prefix_filtered.parquet`
- final `data/precomputed/<name>/`
- final `runs/<name>/checkpoint_best.pt`
- final `runs/<name>/checkpoint_step_latest.pt`
- final eval JSON summaries
- final eval JSONL examples
- exact git commit hash
- exact precompute/train/eval configs
- random seeds
- hardware/runtime notes
- expected headline metrics
- known failure modes
- artifact storage location
- checksums:

```bash
sha256sum data/processed/train_prefix_filtered.parquet
find data/precomputed/<name> -type f -print0 | sort -z | xargs -0 sha256sum > precomputed.sha256
sha256sum runs/<name>/checkpoint_best.pt runs/<name>/checkpoint_step_latest.pt
```
