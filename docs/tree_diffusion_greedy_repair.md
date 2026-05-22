# Greedy Tree-Diffusion Repair

Greedy repair evaluates the trained edit policy in multi-step inference mode. It starts from a corrupted current antiderivative, repeatedly rebuilds the observation for the current state, proposes top-k edit candidates, applies valid edits, scores resulting trees by derivative residual, and keeps the best local move.

This is the bridge between one-step edit evaluation and future beam search. It is still local greedy search: it keeps one current tree, does not branch over AST states, and does not use the gold target antiderivative for choosing edits.

## Algorithm

For each repair step:

1. Build `build_observation(target_integrand, current_antiderivative)`.
2. Serialize the model input as `serialize_observation(...) + ["<EDIT>"]`.
3. Decode top-k edit candidates with optional first-token position constraint.
4. Skip candidates whose decoded status is not `ok`, candidates that fail application, and candidates that revisit a previous canonical tree.
5. Choose the next tree with one of two local strategies:

   - `rank1`: use the first applicable non-repeated policy candidate.
   - `residual_scored`: score all applicable non-repeated candidates and use the lowest score.

   Residual-scored selection uses:

   ```text
   numeric_residual + 1e-3 * tree_size - 1e-2 * policy_logprob
   ```

6. Track the best numeric residual seen so far across the repair trajectory.
7. Stop on exact symbolic derivative match, numeric tolerance, max steps, no applicable candidate, repeated state, decode/observation failure, or repeated lack of numeric improvement.

The target antiderivative is optional and diagnostic only. When supplied, it is used for structural distance fields in traces and evaluation summaries, never for candidate selection.

## Defaults

- `max_steps=10`
- `candidate_k=8`
- `numeric_tol=1e-10`
- `constrain_position=True`
- `lambda_size=1e-3`
- `lambda_policy=1e-2`
- `require_numeric_improvement=False`
- `patience=2`
- `selection_strategy="residual_scored"`

## Evaluation CLI

Recommended first run on precomputed validation examples:

```bash
python -m tree_diffusion.evaluate_repair \
  --checkpoint runs/<tree_diffusion_run>/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/<name> \
  --precomputed-split val \
  --output artifacts/repair_eval/greedy_repair.json \
  --batch-size 32 \
  --num-batches 50 \
  --max-steps 10 \
  --candidate-k 8 \
  --patience 2 \
  --selection-strategy residual_scored \
  --residual-workers 8 \
  --device auto
```

`--residual-workers` is optional and defaults to `0`. When it is greater than
zero, model decoding stays in the main process/GPU, while candidate numeric
residual and symbolic derivative checks are scored in a CPU process pool. This
is useful because greedy repair is often CPU-bound on SymPy/residual scoring.

With qualitative traces:

```bash
python -m tree_diffusion.evaluate_repair \
  --checkpoint runs/<tree_diffusion_run>/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/<name> \
  --output artifacts/repair_eval/greedy_repair.json \
  --dump-examples artifacts/repair_eval/greedy_repair_examples.jsonl \
  --num-dump-examples 50
```

## Metrics

The repair evaluator reports a compact set of metrics meant to answer whether greedy repair is working and whether a broader search is needed.

Overall success:

- `exact_symbolic_match_rate`: derivative simplifies exactly to the target integrand.
- `numeric_success_rate`: final numeric residual is at/below tolerance.
- `numeric_residual_improvement_rate`: final residual is lower than the initial residual among finite-residual examples.
- `success_rate`: exact symbolic match or numeric success.
- Stop reason rates and `stop_reason_counts`: diagnose local search failures.

Corruption source stratification:

- `by_used_random_init.local_corruption`: examples produced by local mutations near the target.
- `by_used_random_init.random_init`: examples that started from a random valid tree.
- `by_num_mutations.s=0` through `s=5`, plus `s>5` or `unknown` if present.

Best-so-far:

- `mean_best_numeric_residual`: lowest numeric residual reached anywhere in the repair trace.
- `best_numeric_residual_improvement_rate`: fraction where the best trace state improves over the initial state.
- `best_numeric_success_rate`: fraction where any trace state reaches numeric tolerance.

Per-step curves:

- `per_step.numeric_residual_mean` / `numeric_residual_median`: residual after each repair step, with `step_0` as the initial tree.
- `per_step.active_examples`: examples with a finite numeric residual at each step.
- `per_step.exact_match_rate`: fraction of all examples that have reached exact derivative match by that step.

Candidate rank:

- `mean_chosen_candidate_rank`: average policy rank of selected edits.
- `rank1_chosen_rate`: fraction of applied repair edits that used the top-ranked policy candidate.

## Interpretation

Greedy repair is promising when success and numeric improvement rates are nonzero, structural distance often decreases on corrupted-current examples, and failures are mostly local-search limits rather than decode failures. Compare `--selection-strategy rank1` against `--selection-strategy residual_scored` with the same `candidate_k`, usually 8, to see whether numeric residual scoring adds value over simply following the policy order. The most important deltas are `success_rate`, `exact_symbolic_match_rate`, `numeric_success_rate`, `mean_final_numeric_residual`, `mean_best_numeric_residual`, `mean_chosen_candidate_rank`, and `rank1_chosen_rate`.

These metrics are intentionally small: they tell us whether repair works near the target versus far from it, whether residual-scored top-k beats rank-1, and whether greedy repair improves over multiple steps or gets stuck. A high `no_numeric_improvement_rate` means the policy proposes valid edits but local residual scoring cannot find a useful next state. A high `no_candidate_rate` means decoding/applicability is still the bottleneck.

## Known Limits

- This is greedy repair, not beam search.
- Replacement generation is not fully grammar-constrained.
- Top-k retry skips invalid candidates but does not prevent invalid generation.
- Local candidate selection can get stuck or choose a short-term residual improvement that blocks later progress.
- Exact symbolic checks use SymPy simplification and can be expensive on hard expressions.
- This first evaluation starts from corrupted current trees, not from scratch integration seeds.
- There is no automatic rank-1 versus residual-scored comparison runner yet; run the CLI twice with different `--selection-strategy` values.
- There is no full mutation-kind trajectory analysis yet.
