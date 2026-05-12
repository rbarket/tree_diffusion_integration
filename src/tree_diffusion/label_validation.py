from __future__ import annotations

from dataclasses import dataclass

from src.mathlang.ast import Expr
from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_tokens
from src.mathlang.serializer import serialize_prefix_tokens
from src.tree_diffusion.edit_path import EditTarget, structural_distance
from src.tree_diffusion.mutation import replace_subtree_by_node_id
from src.tree_diffusion.positions import index_tree_positions


@dataclass(frozen=True)
class EditLabelValidationResult:
    ok: bool
    valid_position: bool
    replacement_serializes: bool
    replacement_reparses: bool
    applicable: bool
    nonincreasing_distance: bool
    strict_improvement: bool
    distance_before: int | None
    distance_after: int | None
    error: str | None = None


def validate_edit_label_progress(
    current_tree: Expr,
    target_tree: Expr,
    edit_target: EditTarget,
    *,
    require_strict_improvement: bool = False,
    canonicalize_inputs: bool = True,
) -> EditLabelValidationResult:
    current = canonicalize(current_tree) if canonicalize_inputs else current_tree
    target = canonicalize(target_tree) if canonicalize_inputs else target_tree

    valid_position = False
    replacement_serializes = False
    replacement_reparses = False
    applicable = False
    nonincreasing_distance = False
    strict_improvement = False
    distance_before: int | None = None
    distance_after: int | None = None

    try:
        index = index_tree_positions(current)
        valid_position = edit_target.selected_node_id in index.node_id_to_node
    except Exception as exc:
        return _result(
            valid_position=False,
            replacement_serializes=False,
            replacement_reparses=False,
            applicable=False,
            nonincreasing_distance=False,
            strict_improvement=False,
            distance_before=None,
            distance_after=None,
            require_strict_improvement=require_strict_improvement,
            error=f"position_index_failed:{type(exc).__name__}",
        )

    if not valid_position:
        return _result(
            valid_position=False,
            replacement_serializes=False,
            replacement_reparses=False,
            applicable=False,
            nonincreasing_distance=False,
            strict_improvement=False,
            distance_before=None,
            distance_after=None,
            require_strict_improvement=require_strict_improvement,
            error=f"unknown_selected_node_id:{edit_target.selected_node_id}",
        )

    try:
        replacement_tokens = serialize_prefix_tokens(edit_target.replacement_subtree)
        replacement_serializes = True
    except Exception as exc:
        return _result(
            valid_position=valid_position,
            replacement_serializes=False,
            replacement_reparses=False,
            applicable=False,
            nonincreasing_distance=False,
            strict_improvement=False,
            distance_before=None,
            distance_after=None,
            require_strict_improvement=require_strict_improvement,
            error=f"replacement_serialize_failed:{type(exc).__name__}",
        )

    try:
        reparsed = parse_prefix_tokens(replacement_tokens)
        replacement_reparses = canonicalize(reparsed) == canonicalize(edit_target.replacement_subtree)
    except Exception as exc:
        return _result(
            valid_position=valid_position,
            replacement_serializes=replacement_serializes,
            replacement_reparses=False,
            applicable=False,
            nonincreasing_distance=False,
            strict_improvement=False,
            distance_before=None,
            distance_after=None,
            require_strict_improvement=require_strict_improvement,
            error=f"replacement_reparse_failed:{type(exc).__name__}",
        )

    if not replacement_reparses:
        return _result(
            valid_position=valid_position,
            replacement_serializes=replacement_serializes,
            replacement_reparses=False,
            applicable=False,
            nonincreasing_distance=False,
            strict_improvement=False,
            distance_before=None,
            distance_after=None,
            require_strict_improvement=require_strict_improvement,
            error="replacement_reparse_mismatch",
        )

    try:
        applied = apply_subtree_replacement_by_position(
            current,
            edit_target.selected_node_id,
            edit_target.replacement_subtree,
        )
        applied = canonicalize(applied)
        applicable = True
    except Exception as exc:
        return _result(
            valid_position=valid_position,
            replacement_serializes=replacement_serializes,
            replacement_reparses=replacement_reparses,
            applicable=False,
            nonincreasing_distance=False,
            strict_improvement=False,
            distance_before=None,
            distance_after=None,
            require_strict_improvement=require_strict_improvement,
            error=f"edit_apply_failed:{type(exc).__name__}",
        )

    expected_result = canonicalize(edit_target.resulting_tree)
    if applied != expected_result:
        return _result(
            valid_position=valid_position,
            replacement_serializes=replacement_serializes,
            replacement_reparses=replacement_reparses,
            applicable=applicable,
            nonincreasing_distance=False,
            strict_improvement=False,
            distance_before=None,
            distance_after=None,
            require_strict_improvement=require_strict_improvement,
            error="resulting_tree_mismatch",
        )

    distance_before = structural_distance(current, target)
    distance_after = structural_distance(applied, target)
    nonincreasing_distance = distance_after <= distance_before
    strict_improvement = distance_after < distance_before
    error = None
    if not nonincreasing_distance:
        error = f"distance_increased:{distance_before}->{distance_after}"
    elif require_strict_improvement and not strict_improvement:
        error = f"distance_not_strictly_improved:{distance_before}->{distance_after}"

    return _result(
        valid_position=valid_position,
        replacement_serializes=replacement_serializes,
        replacement_reparses=replacement_reparses,
        applicable=applicable,
        nonincreasing_distance=nonincreasing_distance,
        strict_improvement=strict_improvement,
        distance_before=distance_before,
        distance_after=distance_after,
        require_strict_improvement=require_strict_improvement,
        error=error,
    )


def apply_subtree_replacement_by_position(
    tree: Expr,
    selected_node_id: int,
    replacement_subtree: Expr,
) -> Expr:
    return replace_subtree_by_node_id(tree, selected_node_id, replacement_subtree)


def _result(
    *,
    valid_position: bool,
    replacement_serializes: bool,
    replacement_reparses: bool,
    applicable: bool,
    nonincreasing_distance: bool,
    strict_improvement: bool,
    distance_before: int | None,
    distance_after: int | None,
    require_strict_improvement: bool,
    error: str | None,
) -> EditLabelValidationResult:
    ok = (
        valid_position
        and replacement_serializes
        and replacement_reparses
        and applicable
        and nonincreasing_distance
        and (strict_improvement or not require_strict_improvement)
    )
    return EditLabelValidationResult(
        ok=ok,
        valid_position=valid_position,
        replacement_serializes=replacement_serializes,
        replacement_reparses=replacement_reparses,
        applicable=applicable,
        nonincreasing_distance=nonincreasing_distance,
        strict_improvement=strict_improvement,
        distance_before=distance_before,
        distance_after=distance_after,
        error=error,
    )


__all__ = [
    "EditLabelValidationResult",
    "apply_subtree_replacement_by_position",
    "validate_edit_label_progress",
]
