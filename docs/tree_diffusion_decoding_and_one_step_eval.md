# Tree-Diffusion Decoding And One-Step Evaluation

## Predicted Edit Format

At inference time the policy emits one edit target:

```text
<POS_i> replacement_subtree <eos>
```

`<POS_i>` selects a preorder node id in the current antiderivative AST. The remaining tokens are parsed as exactly one complete prefix expression and used as the replacement subtree.

Example:

```text
current:      pow x INT+ 5
model output: <POS_2> INT+ 3 <eos>
result:       pow x INT+ 3
```

## Current Constraint

Only the first generated token is constrained. During greedy decoding, the first-token logits can be masked so that only `<POS_i>` tokens for nodes that exist in the current tree are allowed.

This matters because a position token is meaningful only relative to the current AST. With `--constrain-position`, `valid_position_rate` is expected to be high by construction. Use `--no-constrain-position` to inspect raw model behavior.

## Not Yet Constrained

Replacement subtree generation is still unconstrained. The decoder emits tokens greedily, then `decode_edit_tokens(...)` parses the replacement afterward.

Current limitations:

- No beam search.
- No multi-step repair.
- No full grammar-constrained replacement decoding.
- No precompute, objective, architecture, or checkpoint format changes.

## Metrics

Cross-entropy token metrics measure teacher-forced token prediction during training or validation. They do not prove that an autoregressive model output can be applied as an edit.

One-step diagnostics measure the inference path:

- `decoded_ok_rate`: decoded output is a syntactically valid edit.
- `valid_position_rate`: first decoded position exists in the current tree.
- `parseable_replacement_rate`: replacement tokens parse as exactly one prefix expression.
- `applicable_edit_rate`: decoded edit can be applied to the current tree.
- `structural_improvement_rate`: edited tree is closer to the supervised target antiderivative.
- `nonincreasing_structural_rate`: edited tree is no farther from the supervised target.
- `exact_target_rate`: edited tree exactly matches the canonical target antiderivative.
- `numeric_residual_improvement_rate`: derivative residual against the target integrand improves numerically.

Structural distance is a supervised held-out diagnostic because it uses the target antiderivative. Numeric residual is closer to future search-time scoring because it only needs the target integrand.

`numeric_residual_score(...)` computes `mean_squared_abs_residual` over finite probe points only. If derivative or probe evaluation fails, or no finite probes exist, the score is `None`.

## Manual CLI

Evaluate a checkpoint on precomputed examples:

```bash
python -m src.tree_diffusion.eval_one_step \
  --checkpoint runs/tree_diffusion/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/tree_diffusion_v1 \
  --num-batches 5 \
  --batch-size 32 \
  --device auto
```

Evaluate raw position behavior:

```bash
python -m src.tree_diffusion.eval_one_step \
  --checkpoint runs/tree_diffusion/checkpoint_best.pt \
  --precomputed-data-dir data/precomputed/tree_diffusion_v1 \
  --no-constrain-position
```

Smoke-test with a random model only when explicit:

```bash
python -m src.tree_diffusion.eval_one_step \
  --allow-random-init-model \
  --data data/processed/train_prefix_filtered.parquet \
  --num-pairs 8 \
  --num-batches 1
```

Unit tests use dummy or tiny random models. Real policy quality requires a trained checkpoint.
