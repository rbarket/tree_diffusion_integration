# Project handoff

## Known-good environment

- Python version: 3.10+; current validation was run with Python 3.12.
- Dependency manager: use `uv sync` if `uv.lock` is present, otherwise `python -m pip install -e .`.
- Install command:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

- GPU/CPU notes: full training benefits from CUDA. Smoke precompute, help checks, and focused tests run on CPU.

## Required external data

- Expected dataset path inside repo: `data/processed/train_prefix_filtered.parquet`
- Required parquet columns: `integrand_prefix`, `integral_prefix`
- Retrieve the real dataset from: TODO: fill shared artifact store / handoff link.
- Checksum: TODO: fill `sha256sum data/processed/train_prefix_filtered.parquet`.

## Known-good commands

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

Precompute:

```bash
python -m tree_diffusion.precompute_dataset \
  --config config/precompute/smoke.json \
  --input-data data/processed/train_prefix_filtered.parquet \
  --output-dir data/precomputed/smoke \
  --overwrite
```

Train:

```bash
python -m training.workflows.tree_diffusion \
  --config config/train/precomputed_smoke.json
```

One-step eval:

```bash
python -m tree_diffusion.experiments.one_step_inference_eval \
  --config config/eval/one_step_smoke.json
```

Greedy eval:

```bash
python -m tree_diffusion.evaluate_repair \
  --config config/eval/greedy_smoke.json
```

Beam eval:

```bash
python -m tree_diffusion.evaluate_beam_search \
  --config config/eval/beam_smoke.json
```

Tests:

```bash
python -m pytest -q
```

## Final artifacts to preserve

- `data/precomputed/<final_name>/`
- `runs/<final_run>/checkpoint_best.pt`
- `runs/<final_run>/checkpoint_step_latest.pt`
- `artifacts/eval/<final_eval>.json`
- `artifacts/eval/<final_eval_examples>.jsonl`

## Configs used for final run

- Precompute: TODO: fill exact config path and CLI overrides.
- Train: TODO: fill exact config path and CLI overrides.
- One-step eval: TODO: fill exact config path and CLI overrides.
- Greedy eval: TODO: fill exact config path and CLI overrides.
- Beam eval: TODO: fill exact config path and CLI overrides.

## Expected metrics

| Metric | Value |
| --- | --- |
| one-step parseable rate | TODO: fill from final run |
| one-step applicable rate | TODO: fill from final run |
| one-step structural improvement rate | TODO: fill from final run |
| greedy exact derivative match | TODO: fill from final run |
| greedy numeric success | TODO: fill from final run |
| beam exact derivative match | TODO: fill from final run |
| beam numeric success | TODO: fill from final run |

## Known limitations

- Replacement-subtree decoding is validated after generation and is not fully grammar-constrained.
- SymPy observation construction can fail or timeout; missing fields are represented with sentinel tokens/warnings.
- Structural distance uses the gold antiderivative and is validation-only.
- Beam search uses numeric residual scoring, not a learned value network.
- Position tokens are preorder ids for the current tree and change after edits.
- Generated datasets, checkpoints, eval JSONL files, and run directories must stay outside Git.

## Open TODOs

- Improve constrained replacement-subtree decoding.
- Make eval configs uniform across every runner.
- Add a learned value model only if future work needs it.
- Confirm final artifact storage location.
- Ensure all stale local paths are removed before final handoff.
