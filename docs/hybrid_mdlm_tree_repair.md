# Hybrid MDLM Tree Repair

The hybrid pipeline uses MDLM predictions as initial antiderivative seeds for tree-diffusion beam repair. MDLM produces candidate antiderivative strings; this repo parses the candidates into the tree AST grammar and only repairs candidates that parse.

## Prediction JSONL Schema

Each line must be a JSON object with:

- `integrand_prefix`: target integrand in space-separated prefix notation.
- `pred_prefix`: one MDLM antiderivative prediction attempt in prefix notation.
- `attempt_index`: zero-based attempt index for this example.

Optional grouping fields:

- `row_index`: preferred stable example id when available.
- `pair_index`: fallback stable example id.
- `target_antiderivative_prefix`: optional gold antiderivative for diagnostics and grouping.

Rows are grouped by `row_index` when present, then `pair_index`, then by `(integrand_prefix, target_antiderivative_prefix)`.

## Flow

1. Generate up to K MDLM candidate predictions per integrand in the MDLM repo.
2. Export one JSONL row per prediction attempt.
3. Load those attempts in this tree-diffusion repo.
4. Parse every MDLM attempt into the tree AST grammar.
5. Run tree-diffusion beam repair from parseable MDLM seeds.
6. Report failures by stage.

The primary seed-selection mode is `all_parseable`, which passes every parseable MDLM candidate for an example into beam repair in attempt order.

## CLI

```bash
python -m tree_diffusion.experiments.hybrid_mdlm_repair \
  --predictions artifacts/hybrid/mdlm_tree/mdlm_predictions.jsonl \
  --tree-checkpoint runs/<tree_diffusion_run>/checkpoint_best.pt \
  --output artifacts/hybrid/mdlm_tree/hybrid_repair_summary.json \
  --examples-out artifacts/hybrid/mdlm_tree/hybrid_repair_examples.jsonl \
  --beam-size 8 \
  --candidate-k 8 \
  --max-steps 10 \
  --numeric-patience 5 \
  --seed-selection all_parseable \
  --examples-parts-dir artifacts/hybrid/mdlm_tree/hybrid_repair_parts \
  --part-size 500 \
  --progress-every 25 \
  --residual-workers 16 \
  --device auto
```

Seed selection modes:

- `all_parseable`: repair from every parseable MDLM candidate.
- `first_parseable`: repair only from the lowest-attempt-index parseable candidate.
- `best_numeric_seed`: repair only from the parseable candidate with the lowest initial numeric residual.

Fallback seeds are optional via `--use-fallback-seeds`; they are appended only after parseable MDLM seeds and are reported separately.

## Outputs

Summary JSON includes aggregate parseability, repair, and hybrid success metrics. Example JSONL records include MDLM attempt prefixes, parse errors, parseable attempt indices, exactness flags, repair flags, residuals, and beam diagnostics.

When `--examples-parts-dir` is used, sharded example outputs look like:

```text
hybrid_repair_parts/
  part_000000.jsonl
  part_000000.summary.json
  part_000001.jsonl
  part_000001.summary.json
  manifest.json
```

## Failure Stages

- `mdlm_no_parseable_seed`: all MDLM attempts failed to parse; tree diffusion is not run.
- `integrand_parse_failed`: target integrand could not be parsed.
- `tree_repair_failed`: at least one MDLM seed parsed, but beam repair did not solve the example.

## Limitations

This stage does not repair unparseable MDLM strings directly. It only runs tree-diffusion repair from MDLM candidates that parse into the tree AST grammar. Do not claim final hybrid results in docs unless the summary JSON from the final run is present and has been checked.
