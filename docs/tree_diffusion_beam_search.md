# Tree-Diffusion Beam Search Repair

Beam search repair is the next inference step after greedy multi-step repair. Greedy repair follows one selected edit at a time; beam search keeps several valid antiderivative AST states at each depth and returns the best state seen anywhere in the search, not just the final beam head.

The loop is:

```text
beam states
  -> build observation(target integrand, candidate antiderivative)
  -> decode top-k edit candidates
  -> apply valid edits
  -> score resulting AST states
  -> keep top beam_size states
```

## Scoring

Beam state scores are lower-is-better:

```text
lambda_residual * residual_term
+ lambda_size * tree_size
+ lambda_steps * num_steps
- lambda_policy * cumulative_policy_logprob
```

By default, `residual_term = log1p(numeric_residual)`. Missing residuals receive a large finite penalty. Numeric residual dominates; tree size, step count, and policy logprob are tie-breakers.

Default config:

- `beam_size=8`
- `candidate_k=8`
- `max_steps=10`
- `numeric_patience=5`
- `structural_patience=None`
- `lambda_residual=1.0`
- `lambda_size=1e-3`
- `lambda_steps=1e-3`
- `lambda_policy=1e-2`
- `use_log_residual=True`

## Stopping

Hard success stops:

- `exact_symbolic_match`
- `numeric_tol`

Budget or failure stops:

- `max_steps`
- `beam_empty`
- `max_expanded_states`
- `timeout`

Patience stops:

- `numeric_patience`: global best numeric residual has not improved for this many depths.
- `structural_patience`: optional supervised diagnostic that uses the gold target antiderivative. It should only be used for validation/debugging, not real inference.

Numeric patience is based on the best state seen across the whole beam, not whether every depth improves.

## Validation Commands

Greedy rank-1:

```bash
uv run python -m tree_diffusion.evaluate_repair \
  --checkpoint runs/<tree_diffusion_run>/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/<name> \
  --precomputed-split val \
  --output artifacts/repair_eval/greedy_rank1.json \
  --batch-size 32 \
  --num-batches 50 \
  --max-steps 10 \
  --candidate-k 8 \
  --selection-strategy rank1 \
  --device auto
```

Greedy residual-scored:

```bash
uv run python -m tree_diffusion.evaluate_repair \
  --checkpoint runs/<tree_diffusion_run>/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/<name> \
  --precomputed-split val \
  --output artifacts/repair_eval/greedy_residual_scored.json \
  --batch-size 32 \
  --num-batches 50 \
  --max-steps 10 \
  --candidate-k 8 \
  --selection-strategy residual_scored \
  --device auto
```

Beam search:

```bash
uv run python -m tree_diffusion.evaluate_beam_search \
  --checkpoint runs/<tree_diffusion_run>/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/<name> \
  --precomputed-split val \
  --output artifacts/beam_eval/beam8_k8.json \
  --batch-size 32 \
  --num-batches 50 \
  --beam-size 8 \
  --candidate-k 8 \
  --max-steps 10 \
  --numeric-patience 5 \
  --structural-patience none \
  --device auto
```

Recommended first comparisons:

- `beam_size=1`, `candidate_k=8` as a sanity run.
- `beam_size=4`, `candidate_k=8`.
- `beam_size=8`, `candidate_k=8`.
- `beam_size=16`, `candidate_k=8`.
- `beam_size=8`, `candidate_k=16`.

Compare exact match rate, numeric success rate, mean best numeric residual, best residual improvement rate, random-init success, `s=1` through `s=5` success, and expanded states.

## Hybrid Repair

`beam_search_repair_from_seeds(...)` accepts a sequence of parsed AST seeds. The MDLM hybrid runner can parse MDLM prediction attempts into expressions and pass them in as initial seeds before tree-diffusion repair.

## Limits

- No learned value network yet.
- Replacement generation is still not fully grammar-constrained.
- Structural patience uses gold target antiderivatives and is validation/debug only.
- Hybrid MDLM seed repair exists, but external MDLM prediction generation is not implemented in this repo.
