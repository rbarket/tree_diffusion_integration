from __future__ import annotations

from dataclasses import dataclass
import json
import random
from typing import Any

from src.mathlang.canonicalize import canonicalize
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.dataset import IntegrationPair
from src.tree_diffusion.edit_path import EditTarget, first_edit_toward_target, structural_distance
from src.tree_diffusion.label_validation import validate_edit_label_progress
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer
from src.tree_diffusion.training_examples import TreeDiffusionTrainingExample


REPAIR_TRAJECTORY_MAX_STEPS = 64


@dataclass(frozen=True)
class PrecomputedTreeDiffusionExampleRecord:
    split: str
    global_example_index: int
    pair_index: int | None
    source: str | None
    example_index_for_pair: int
    rng_seed: int

    target_integrand_prefix: str
    target_antiderivative_prefix: str
    current_antiderivative_prefix: str
    current_derivative_prefix: str | None
    symbolic_residual_prefix: str | None

    input_tokens_json: str
    target_tokens_json: str
    input_ids_json: str
    target_ids_json: str
    labels_json: str
    input_length: int
    target_length: int

    selected_node_id: int
    replacement_subtree_prefix: str
    resulting_tree_prefix: str | None

    num_mutations: int
    used_random_init: bool
    sampled_s: int | None

    distance_before: int | None
    distance_after: int | None
    label_validation_ok: bool
    label_strict_improvement: bool

    observation_status: str | None
    warnings_json: str
    trajectory_json: str | None = None


def precomputed_record_from_training_example(
    example: TreeDiffusionTrainingExample,
    *,
    split: str,
    global_example_index: int,
    pair: IntegrationPair,
    example_index_for_pair: int,
    rng_seed: int,
    tokenizer: TreeDiffusionTokenizer,
    validate_labels: bool = True,
    require_strict_label_improvement: bool = False,
    trajectory_mode: str = "none",
    repair_trajectory_rng_seed: int | None = None,
    sigma_small: int = 2,
) -> PrecomputedTreeDiffusionExampleRecord:
    input_ids = example.input_ids
    target_ids = example.target_ids
    if input_ids is None:
        input_ids = tokenizer.encode_tokens(example.input_tokens)
    if target_ids is None:
        target_ids = tokenizer.encode_tokens(example.target_tokens)
    labels = [token_id if token_id != tokenizer.pad_id else -100 for token_id in target_ids]

    distance_before: int | None = None
    distance_after: int | None = None
    label_validation_ok = True
    label_strict_improvement = False
    if validate_labels:
        validation = validate_edit_label_progress(
            example.current_antiderivative,
            example.target_antiderivative,
            example.edit_target,
            require_strict_improvement=require_strict_label_improvement,
        )
        distance_before = validation.distance_before
        distance_after = validation.distance_after
        label_validation_ok = validation.ok
        label_strict_improvement = validation.strict_improvement
        if not validation.ok:
            raise ValueError(
                "label_validation_failed:"
                f"{validation.error or 'unknown'}; split={split}; "
                f"global_example_index={global_example_index}; pair_index={pair.index}; "
                f"rng_seed={rng_seed}"
            )
    else:
        try:
            distance_before = structural_distance(
                example.current_antiderivative,
                example.target_antiderivative,
            )
            distance_after = structural_distance(
                example.edit_target.resulting_tree,
                example.target_antiderivative,
            )
            label_strict_improvement = distance_after < distance_before
            label_validation_ok = distance_after <= distance_before
        except Exception:
            distance_before = None
            distance_after = None
            label_validation_ok = False
            label_strict_improvement = False

    trajectory_json = None
    if split == "val" and trajectory_mode == "forward_and_repair":
        trajectory_json = json.dumps(
            _build_forward_and_repair_trajectory(
                example,
                pair=pair,
                example_index_for_pair=example_index_for_pair,
                rng_seed=rng_seed,
                repair_rng_seed=repair_trajectory_rng_seed
                if repair_trajectory_rng_seed is not None
                else rng_seed + 31_000_031,
                sigma_small=sigma_small,
            ),
            sort_keys=True,
        )

    return PrecomputedTreeDiffusionExampleRecord(
        split=split,
        global_example_index=global_example_index,
        pair_index=pair.index,
        source=pair.source,
        example_index_for_pair=example_index_for_pair,
        rng_seed=rng_seed,
        target_integrand_prefix=serialize_prefix_string(example.target_integrand),
        target_antiderivative_prefix=serialize_prefix_string(example.target_antiderivative),
        current_antiderivative_prefix=serialize_prefix_string(example.current_antiderivative),
        current_derivative_prefix=(
            None
            if example.observation.current_derivative is None
            else serialize_prefix_string(example.observation.current_derivative)
        ),
        symbolic_residual_prefix=(
            None
            if example.observation.symbolic_residual is None
            else serialize_prefix_string(example.observation.symbolic_residual)
        ),
        input_tokens_json=json.dumps(example.input_tokens),
        target_tokens_json=json.dumps(example.target_tokens),
        input_ids_json=json.dumps(list(input_ids)),
        target_ids_json=json.dumps(list(target_ids)),
        labels_json=json.dumps(labels),
        input_length=len(example.input_tokens),
        target_length=len(example.target_tokens),
        selected_node_id=example.edit_target.selected_node_id,
        replacement_subtree_prefix=serialize_prefix_string(example.edit_target.replacement_subtree),
        resulting_tree_prefix=serialize_prefix_string(example.edit_target.resulting_tree),
        num_mutations=example.num_mutations,
        used_random_init=example.used_random_init,
        sampled_s=None,
        distance_before=distance_before,
        distance_after=distance_after,
        label_validation_ok=label_validation_ok,
        label_strict_improvement=label_strict_improvement,
        observation_status=example.observation.status,
        warnings_json=json.dumps(list(example.warnings)),
        trajectory_json=trajectory_json,
    )


def _build_forward_and_repair_trajectory(
    example: TreeDiffusionTrainingExample,
    *,
    pair: IntegrationPair,
    example_index_for_pair: int,
    rng_seed: int,
    repair_rng_seed: int,
    sigma_small: int,
) -> dict[str, Any]:
    target_prefix = serialize_prefix_string(example.target_antiderivative)
    current_prefix = serialize_prefix_string(example.current_antiderivative)
    forward_steps: list[dict[str, Any]] = []
    before = example.target_antiderivative
    for step_index, mutation in enumerate(example.forward_mutations):
        after = mutation.mutated_expr
        forward_steps.append(
            {
                "step_index": step_index,
                "before_prefix": serialize_prefix_string(before),
                "after_prefix": serialize_prefix_string(after),
                "selected_node_id": mutation.selected_node_id,
                "selected_family": mutation.selected_family,
                "mutation_kind": mutation.mutation_kind,
                "original_subtree_prefix": serialize_prefix_string(mutation.original_subtree),
                "replacement_subtree_prefix": serialize_prefix_string(mutation.replacement_subtree),
                "selected_token_start": mutation.selected_token_start,
                "selected_token_end": mutation.selected_token_end,
            }
        )
        before = after

    forward_complete = (
        not example.used_random_init
        and len(forward_steps) == example.num_mutations
        and (not forward_steps or forward_steps[-1]["after_prefix"] == current_prefix)
    )
    repair_steps, repair_reached_target, repair_truncated = _build_repair_trajectory_steps(
        example,
        sigma_small=sigma_small,
        repair_rng_seed=repair_rng_seed,
    )
    return {
        "version": 1,
        "mode": "forward_and_repair",
        "rng_seed": rng_seed,
        "repair_rng_seed": repair_rng_seed,
        "pair_index": pair.index,
        "source": pair.source,
        "example_index_for_pair": example_index_for_pair,
        "forward": {
            "start_prefix": target_prefix,
            "end_prefix": current_prefix,
            "used_random_init": example.used_random_init,
            "num_mutations": example.num_mutations,
            "complete": forward_complete,
            "steps": forward_steps,
        },
        "repair": {
            "start_prefix": current_prefix,
            "target_prefix": target_prefix,
            "max_steps": REPAIR_TRAJECTORY_MAX_STEPS,
            "reached_target": repair_reached_target,
            "truncated": repair_truncated,
            "steps": repair_steps,
        },
    }


def _build_repair_trajectory_steps(
    example: TreeDiffusionTrainingExample,
    *,
    sigma_small: int,
    repair_rng_seed: int,
) -> tuple[list[dict[str, Any]], bool, bool]:
    target = canonicalize(example.target_antiderivative)
    current = canonicalize(example.current_antiderivative)
    if current == target:
        return [], True, False

    rng = random.Random(repair_rng_seed)
    edit: EditTarget | None = example.edit_target
    steps: list[dict[str, Any]] = []
    for step_index in range(REPAIR_TRAJECTORY_MAX_STEPS):
        if edit is None:
            return steps, current == target, False
        steps.append(_repair_step_record(step_index, current, target, edit))
        current = canonicalize(edit.resulting_tree)
        if current == target:
            return steps, True, False
        edit = first_edit_toward_target(current, target, sigma_small=sigma_small, rng=rng)

    return steps, current == target, current != target


def _repair_step_record(
    step_index: int,
    before_tree: Any,
    target_tree: Any,
    edit: EditTarget,
) -> dict[str, Any]:
    return {
        "step_index": step_index,
        "before_prefix": serialize_prefix_string(before_tree),
        "after_prefix": serialize_prefix_string(edit.resulting_tree),
        "selected_node_id": edit.selected_node_id,
        "selected_node_span": list(edit.selected_node_span),
        "mutation_kind": edit.mutation_kind,
        "reason": edit.reason,
        "original_subtree_prefix": serialize_prefix_string(edit.original_subtree),
        "replacement_subtree_prefix": serialize_prefix_string(edit.replacement_subtree),
        "distance_before": structural_distance(before_tree, target_tree),
        "distance_after": structural_distance(edit.resulting_tree, target_tree),
    }


__all__ = [
    "PrecomputedTreeDiffusionExampleRecord",
    "REPAIR_TRAJECTORY_MAX_STEPS",
    "_build_forward_and_repair_trajectory",
    "_build_repair_trajectory_steps",
    "_repair_step_record",
    "precomputed_record_from_training_example",
]
