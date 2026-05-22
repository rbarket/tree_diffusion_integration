#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SMOKE_DATA="${SMOKE_DATA:-data/processed/handoff_smoke.parquet}"
SMOKE_PRECOMPUTED="${SMOKE_PRECOMPUTED:-data/precomputed/handoff_smoke}"

if [[ ! -f "$SMOKE_DATA" ]]; then
  mkdir -p "$(dirname "$SMOKE_DATA")"
  python - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("data/processed/handoff_smoke.parquet")
rows = [
    {"integrand_prefix": "INT+ 1", "integral_prefix": "x"},
    {"integrand_prefix": "mul INT+ 2 x", "integral_prefix": "pow x INT+ 2"},
    {"integrand_prefix": "cos x", "integral_prefix": "sin x"},
    {"integrand_prefix": "pow x INT+ 2", "integral_prefix": "div pow x INT+ 3 INT+ 3"},
]
pd.DataFrame(rows).to_parquet(path, index=False)
print(f"wrote {path}")
PY
else
  echo "using existing smoke dataset: $SMOKE_DATA"
fi

rm -rf "$SMOKE_PRECOMPUTED"
python -m tree_diffusion.precompute_dataset \
  --config config/precompute/smoke.json \
  --input-data "$SMOKE_DATA" \
  --output-dir "$SMOKE_PRECOMPUTED" \
  --train-limit 3 \
  --val-limit 1 \
  --examples-per-pair-train 1 \
  --examples-per-pair-val 1 \
  --shard-size 4 \
  --overwrite

test -f "$SMOKE_PRECOMPUTED/metadata.json"
test -f "$SMOKE_PRECOMPUTED/audit_summary.json"
test -n "$(find "$SMOKE_PRECOMPUTED/train" -name 'shard_*.parquet' -print -quit)"
test -n "$(find "$SMOKE_PRECOMPUTED/val" -name 'shard_*.parquet' -print -quit)"

python -m tree_diffusion.precompute_dataset --help >/dev/null
python -m training.workflows.tree_diffusion --help >/dev/null
python -m tree_diffusion.evaluate_repair --help >/dev/null
python -m tree_diffusion.evaluate_beam_search --help >/dev/null
python -m tree_diffusion.experiments.one_step_inference_eval --help >/dev/null

cat "$SMOKE_PRECOMPUTED/audit_summary.json"
cat <<EOF

Smoke precompute succeeded.

Next commands:
  python -m training.workflows.tree_diffusion --config config/train/precomputed_smoke.json
  python -m tree_diffusion.experiments.one_step_inference_eval --config config/eval/one_step_smoke.json
  python -m tree_diffusion.evaluate_repair --config config/eval/greedy_smoke.json
  python -m tree_diffusion.evaluate_beam_search --config config/eval/beam_smoke.json
EOF
