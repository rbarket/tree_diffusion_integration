# Hybrid MDLM Tree Repair

The hybrid pipeline uses MDLM predictions as initial antiderivative seeds for tree-diffusion beam repair.

## Flow

1. Generate up to K MDLM candidate predictions per integrand in `Diffusion_Integration`.
2. Export one MDLM prediction row per attempt.
3. Load those attempts in this tree-diffusion repo.
4. Group attempts by the original example.
5. Parse every MDLM attempt into the tree AST grammar.
6. Run tree-diffusion beam repair from all parseable MDLM seeds.
7. Report failures by stage.

The primary seed-selection mode is `all_parseable`. It passes every parseable MDLM candidate for an example into beam repair, in attempt order. It does not stop at the first prediction or the first parseable prediction.

## Failure Stages

If no MDLM candidate parses, the example is counted as:

```text
failure_stage = "mdlm_no_parseable_seed"
```

Tree diffusion is not run for that example. This is an MDLM seed-generation failure, not a tree-repair failure.

If at least one MDLM candidate parses and beam repair does not solve the example, the example is counted as:

```text
failure_stage = "tree_repair_failed"
```

This separation keeps MDLM parseability failures distinct from tree-diffusion repair failures.

## Metrics

Important summary metrics include:

- `mdlm_no_parseable_seed_rate`: fraction of examples where all K MDLM attempts failed to parse.
- `tree_repair_failure_rate_over_parseable`: tree repair failure rate among examples with at least one parseable MDLM seed.
- `hybrid_success_rate_over_all`: end-to-end success including MDLM parse failures.
- `hybrid_success_rate_over_parseable`: repair effectiveness once a valid MDLM seed exists.

The examples JSONL also records every MDLM attempt prefix, parse error, parseable attempt index, exactness flags, repair flags, residuals, and beam diagnostics.

## CLI

```bash
python -m tree_diffusion.experiments.hybrid_mdlm_repair \
  --predictions artifacts/hybrid/mdlm_tree/mdlm_predictions.jsonl \
  --tree-checkpoint runs/tree_diffusion_full_precomputed_20260513T133248Z/lightning/best.ckpt \
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

Fallback seeds are optional via `--use-fallback-seeds`; they are appended only after parseable MDLM seeds and are reported separately. They are not used when no MDLM candidate parses.

Progress is printed to stderr by default. Use `--progress-every 25` to mirror the beam validation cadence, or `--quiet` to suppress progress. Use `--residual-workers 16` to mirror the beam validation CPU worker pool for residual scoring inside beam repair. Use `--examples-parts-dir` with `--part-size 500` to write sharded example JSONL files:

```text
hybrid_repair_parts/
  part_000000.jsonl
  part_000000.summary.json
  part_000001.jsonl
  part_000001.summary.json
  manifest.json
```

## Limitations

This stage does not repair unparseable MDLM strings directly. It only runs tree-diffusion repair from MDLM candidates that parse into the tree AST grammar.
