# Tree-Diffusion One-Step Inference Evaluation

## Purpose

This evaluation checks whether a trained tree-diffusion policy can autoregressively propose useful one-step edits on held-out examples. It is an inference-mode diagnostic: the model must generate edit tokens, decode them, and apply the resulting edit without teacher forcing.

This is different from teacher-forced validation metrics:

- `val/position_accuracy` and `val/token_accuracy` measure next-token prediction under gold previous tokens.
- Greedy decoded edit metrics measure the actual autoregressive policy output.
- Top-k first-applicable metrics measure whether useful lower-ranked edit candidates exist when the top candidate is invalid or unhelpful.

## Modes

The runner compares:

- `unconstrained_greedy`: raw greedy decoding with no first-token position mask.
- `position_constrained_greedy`: greedy decoding with the first token masked to valid current-tree positions.
- `position_constrained_topk_4`
- `position_constrained_topk_8`
- `position_constrained_topk_16`

Top-k modes use first-applicable candidate selection. They do not implement beam search over AST states and they do not repair invalid replacement strings.

## Metrics

Decode validity:

- `valid_position_rate`: first decoded position refers to a current-tree node.
- `parseable_replacement_rate`: replacement tokens parse as one complete expression.
- `decoded_ok_rate`: position and replacement decode into an edit.
- `applicable_edit_rate`: decoded or selected edit can be applied.
- `status_counts`: decode and application failure counts.

Structural progress:

- `structural_improvement_rate`: edited tree is closer to the supervised target.
- `nonincreasing_structural_rate`: edited tree is no farther from the target.
- `exact_target_rate`: edited tree exactly matches the canonical target.
- `mean_structural_distance_before` and `mean_structural_distance_after`: average supervised edit distance before and after.

Numeric progress:

- `numeric_residual_improvement_rate`: derivative residual improves on finite probe points.
- `mean_numeric_residual_before` and `mean_numeric_residual_after`: average numeric residual when computable.

Top-k metrics:

- `any_decoded_ok_rate`: at least one candidate decodes successfully.
- `any_applicable_edit_rate`: at least one candidate applies.
- `any_structural_improvement_rate`: at least one applicable candidate improves structural distance.
- `first_applicable_rank_mean`: average 1-based rank of the first applicable candidate.

## Interpretation

Top-k is useful if it improves `applicable_edit_rate` and `structural_improvement_rate` over `position_constrained_greedy`. A low `first_applicable_rank_mean` means useful candidates are near the top of the policy distribution.

A reasonable go/no-go signal for greedy repair is:

- nonzero or high `applicable_edit_rate`,
- nonzero `structural_improvement_rate`,
- nonzero `numeric_residual_improvement_rate` when numeric scoring is enabled,
- top-k first-applicable improves over constrained greedy,
- qualitative failures are mostly valid-but-unhelpful edits rather than invalid replacements.

If invalid replacements dominate, grammar-constrained replacement decoding is likely more valuable than search. If valid edits are common but often unhelpful, repair/search can start, but ranking and scoring will matter.

## Commands

Using precomputed validation data:

```bash
python -m tree_diffusion.experiments.one_step_inference_eval \
  --checkpoint runs/<tree_diffusion_run>/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/<name> \
  --output-dir artifacts/eval/one_step_<name> \
  --batch-size 32 \
  --num-batches 50 \
  --device auto \
  --compute-numeric-residual
```

Using online held-out data:

```bash
python -m tree_diffusion.experiments.one_step_inference_eval \
  --checkpoint runs/<tree_diffusion_run>/checkpoint_best.pt \
  --data data/processed/train_prefix_filtered.parquet \
  --output-dir artifacts/eval/one_step_online \
  --num-pairs 512 \
  --batch-size 32 \
  --num-batches 50 \
  --device auto \
  --compute-numeric-residual
```

Fast smoke run:

```bash
python -m tree_diffusion.experiments.one_step_inference_eval \
  --checkpoint runs/<tree_diffusion_run>/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/<name> \
  --output-dir artifacts/eval/one_step_smoke \
  --batch-size 2 \
  --num-batches 1 \
  --top-k-values 2 \
  --num-dump-examples 2 \
  --device cpu \
  --no-compute-numeric-residual
```

Outputs:

- one JSON file per mode,
- `one_step_eval_summary.json`,
- `examples.jsonl` when `--num-dump-examples > 0`.
