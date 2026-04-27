from __future__ import annotations

import random
from dataclasses import dataclass

from src.mathlang.ast import BinaryOp, Const, Expr, UnaryOp, Var
from src.mathlang.canonicalize import canonicalize
from src.tree_diffusion.mutation import (
    LOCAL_CONST_EDIT,
    LOCAL_SAME_ARITY_REPLACEMENT,
    SAMPLED_SMALL_SUBTREE_REPLACEMENT,
    replace_subtree_by_node_id,
)
from src.tree_diffusion.mutation_grammar import (
    can_locally_replace,
    can_sampled_subtree_replace,
    subtree_size as _subtree_size,
)
from src.tree_diffusion.positions import PositionIndex, index_tree_positions


@dataclass(frozen=True)
class FirstMismatch:
    path: tuple[int, ...]
    current_node_id: int
    current_subtree: Expr
    target_subtree: Expr


@dataclass(frozen=True)
class EditTarget:
    selected_node_id: int
    selected_node_span: tuple[int, int]
    original_subtree: Expr
    replacement_subtree: Expr
    mutation_kind: str
    reason: str
    resulting_tree: Expr


def first_edit_toward_target(
    current_tree: Expr,
    target_tree: Expr,
    sigma_small: int,
    rng: random.Random | None = None,
) -> EditTarget | None:
    if sigma_small < 0:
        raise ValueError("sigma_small must be non-negative.")

    canonical_current = canonicalize(current_tree)
    canonical_target = canonicalize(target_tree)
    if canonical_current == canonical_target:
        return None

    index = index_tree_positions(canonical_current)
    path_to_node_id = _path_to_node_ids(canonical_current)
    mismatch = _find_first_mismatch(canonical_current, canonical_target, path_to_node_id)
    if mismatch is None:
        return None

    current_distance = _structural_distance(canonical_current, canonical_target)

    direct = _direct_exact_edit(
        canonical_current,
        canonical_target,
        index,
        mismatch.current_node_id,
        mismatch.target_subtree,
        sigma_small,
        reason="direct_mismatch_target",
        current_distance=current_distance,
        require_distance_reduction=False,
    )
    if direct is not None:
        return direct

    for path in _ancestor_paths(mismatch.path, include_self=False):
        node_id = path_to_node_id[path]
        target_subtree = _subtree_at_path(canonical_target, path)
        edit = _direct_exact_edit(
            canonical_current,
            canonical_target,
            index,
            node_id,
            target_subtree,
            sigma_small,
            reason="direct_ancestor_target",
            current_distance=current_distance,
            require_distance_reduction=True,
        )
        if edit is not None:
            return edit

    candidate_edits: list[EditTarget] = []
    candidate_edits.extend(
        _local_root_operator_edits(
            canonical_current,
            canonical_target,
            index,
            path_to_node_id,
            mismatch.path,
            sigma_small,
            current_distance,
        )
    )
    candidate_edits.extend(
        _direct_child_edits(
            canonical_current,
            canonical_target,
            index,
            path_to_node_id,
            mismatch.path,
            sigma_small,
            current_distance,
        )
    )
    candidate_edits.extend(
        _target_family_intermediate_edits(
            canonical_current,
            canonical_target,
            index,
            path_to_node_id,
            mismatch.path,
            sigma_small,
            current_distance,
        )
    )

    if not candidate_edits:
        return None

    rng = rng or random.Random(0)
    best_distance = min(_structural_distance(edit.resulting_tree, canonical_target) for edit in candidate_edits)
    best_edits = [
        edit
        for edit in candidate_edits
        if _structural_distance(edit.resulting_tree, canonical_target) == best_distance
    ]
    return rng.choice(best_edits)


def compute_edit_path(
    current_tree: Expr,
    target_tree: Expr,
    sigma_small: int,
    rng: random.Random | None = None,
    *,
    max_steps: int = 64,
) -> list[EditTarget]:
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative.")

    canonical_target = canonicalize(target_tree)
    current = canonicalize(current_tree)
    path: list[EditTarget] = []

    for _ in range(max_steps):
        if current == canonical_target:
            return path
        edit = first_edit_toward_target(current, canonical_target, sigma_small, rng=rng)
        if edit is None:
            return path
        path.append(edit)
        current = edit.resulting_tree

    return path


def trees_equal(a: Expr, b: Expr) -> bool:
    return canonicalize(a) == canonicalize(b)


def find_first_mismatch(current_tree: Expr, target_tree: Expr) -> FirstMismatch | None:
    canonical_current = canonicalize(current_tree)
    canonical_target = canonicalize(target_tree)
    if canonical_current == canonical_target:
        return None
    return _find_first_mismatch(
        canonical_current,
        canonical_target,
        _path_to_node_ids(canonical_current),
    )


def subtree_size(expr: Expr) -> int:
    return _subtree_size(expr)


def is_small_enough(expr: Expr, sigma_small: int) -> bool:
    if sigma_small < 0:
        raise ValueError("sigma_small must be non-negative.")
    return subtree_size(expr) <= sigma_small


def structural_distance(a: Expr, b: Expr) -> int:
    return _structural_distance(canonicalize(a), canonicalize(b))


def _find_first_mismatch(
    current: Expr,
    target: Expr,
    path_to_node_id: dict[tuple[int, ...], int],
    path: tuple[int, ...] = (),
) -> FirstMismatch | None:
    if current == target:
        return None

    if not _same_constructor_label_and_arity(current, target):
        return FirstMismatch(
            path=path,
            current_node_id=path_to_node_id[path],
            current_subtree=current,
            target_subtree=target,
        )

    for child_index, (current_child, target_child) in enumerate(zip(current.children(), target.children())):
        mismatch = _find_first_mismatch(
            current_child,
            target_child,
            path_to_node_id,
            path + (child_index,),
        )
        if mismatch is not None:
            return mismatch

    return FirstMismatch(
        path=path,
        current_node_id=path_to_node_id[path],
        current_subtree=current,
        target_subtree=target,
    )


def _same_constructor_label_and_arity(a: Expr, b: Expr) -> bool:
    if type(a) is not type(b):
        return False

    if isinstance(a, Const) and isinstance(b, Const):
        return a == b
    if isinstance(a, Var) and isinstance(b, Var):
        return a == b
    if isinstance(a, UnaryOp) and isinstance(b, UnaryOp):
        return a.op == b.op
    if isinstance(a, BinaryOp) and isinstance(b, BinaryOp):
        return a.op == b.op

    return False


def _direct_exact_edit(
    canonical_current: Expr,
    canonical_target: Expr,
    index: PositionIndex,
    node_id: int,
    target_subtree: Expr,
    sigma_small: int,
    *,
    reason: str,
    current_distance: int,
    require_distance_reduction: bool,
) -> EditTarget | None:
    if not is_small_enough(target_subtree, sigma_small):
        return None
    return _make_candidate_edit(
        canonical_current,
        canonical_target,
        index,
        node_id,
        target_subtree,
        sigma_small,
        reason=reason,
        current_distance=current_distance,
        require_distance_reduction=require_distance_reduction,
    )


def _make_candidate_edit(
    canonical_current: Expr,
    canonical_target: Expr,
    index: PositionIndex,
    node_id: int,
    replacement: Expr,
    sigma_small: int,
    *,
    reason: str,
    current_distance: int,
    require_distance_reduction: bool,
) -> EditTarget | None:
    original = index.node_id_to_node[node_id]
    mutation_kind = _classify_mutation_kind(original, replacement, sigma_small)
    if mutation_kind is None:
        return None

    resulting_tree = canonicalize(replace_subtree_by_node_id(canonical_current, node_id, replacement))
    if resulting_tree == canonical_current:
        return None

    if require_distance_reduction and _structural_distance(resulting_tree, canonical_target) >= current_distance:
        return None

    selected_span = index.node_id_to_span[node_id]
    return EditTarget(
        selected_node_id=node_id,
        selected_node_span=selected_span,
        original_subtree=original,
        replacement_subtree=replacement,
        mutation_kind=mutation_kind,
        reason=reason,
        resulting_tree=resulting_tree,
    )


def _classify_mutation_kind(original: Expr, replacement: Expr, sigma_small: int) -> str | None:
    if replacement == original:
        return None

    if can_locally_replace(original, replacement):
        if isinstance(original, Const) and isinstance(replacement, Const) and _same_const_leaf_kind(original, replacement):
            return LOCAL_CONST_EDIT
        return LOCAL_SAME_ARITY_REPLACEMENT

    if is_small_enough(replacement, sigma_small) and can_sampled_subtree_replace(original, replacement):
        return SAMPLED_SMALL_SUBTREE_REPLACEMENT

    return None


def _local_root_operator_edits(
    canonical_current: Expr,
    canonical_target: Expr,
    index: PositionIndex,
    path_to_node_id: dict[tuple[int, ...], int],
    mismatch_path: tuple[int, ...],
    sigma_small: int,
    current_distance: int,
) -> list[EditTarget]:
    edits: list[EditTarget] = []
    for path in _ancestor_paths(mismatch_path, include_self=True):
        current_subtree = _subtree_at_path(canonical_current, path)
        target_subtree = _subtree_at_path(canonical_target, path)
        replacement = _root_operator_replacement(current_subtree, target_subtree)
        if replacement is None:
            continue
        edit = _make_candidate_edit(
            canonical_current,
            canonical_target,
            index,
            path_to_node_id[path],
            replacement,
            sigma_small,
            reason="local_root_operator",
            current_distance=current_distance,
            require_distance_reduction=True,
        )
        if edit is not None:
            edits.append(edit)
    return edits


def _direct_child_edits(
    canonical_current: Expr,
    canonical_target: Expr,
    index: PositionIndex,
    path_to_node_id: dict[tuple[int, ...], int],
    mismatch_path: tuple[int, ...],
    sigma_small: int,
    current_distance: int,
) -> list[EditTarget]:
    edits: list[EditTarget] = []
    for path in _ancestor_paths(mismatch_path, include_self=True):
        current_subtree = _subtree_at_path(canonical_current, path)
        target_subtree = _subtree_at_path(canonical_target, path)
        if not _same_container_shape(current_subtree, target_subtree):
            continue
        for child_index, (current_child, target_child) in enumerate(
            zip(current_subtree.children(), target_subtree.children())
        ):
            if current_child == target_child:
                continue
            child_path = path + (child_index,)
            if child_path not in path_to_node_id:
                continue
            edit = _direct_exact_edit(
                canonical_current,
                canonical_target,
                index,
                path_to_node_id[child_path],
                target_child,
                sigma_small,
                reason="direct_child_target",
                current_distance=current_distance,
                require_distance_reduction=True,
            )
            if edit is not None:
                edits.append(edit)
    return edits


def _target_family_intermediate_edits(
    canonical_current: Expr,
    canonical_target: Expr,
    index: PositionIndex,
    path_to_node_id: dict[tuple[int, ...], int],
    mismatch_path: tuple[int, ...],
    sigma_small: int,
    current_distance: int,
) -> list[EditTarget]:
    edits: list[EditTarget] = []
    for path in _ancestor_paths(mismatch_path, include_self=True):
        target_subtree = _subtree_at_path(canonical_target, path)
        replacement = _shrink_target_like(target_subtree, sigma_small)
        if replacement == target_subtree:
            continue
        edit = _make_candidate_edit(
            canonical_current,
            canonical_target,
            index,
            path_to_node_id[path],
            replacement,
            sigma_small,
            reason="target_family_intermediate",
            current_distance=current_distance,
            require_distance_reduction=True,
        )
        if edit is not None:
            edits.append(edit)
    return edits


def _root_operator_replacement(current: Expr, target: Expr) -> Expr | None:
    if isinstance(current, UnaryOp) and isinstance(target, UnaryOp):
        return UnaryOp(op=target.op, operand=current.operand)

    if isinstance(current, BinaryOp) and isinstance(target, BinaryOp):
        return BinaryOp(op=target.op, left=current.left, right=current.right)

    return None


def _same_container_shape(current: Expr, target: Expr) -> bool:
    if type(current) is not type(target):
        return False
    if isinstance(current, UnaryOp) and isinstance(target, UnaryOp):
        return current.op == target.op
    if isinstance(current, BinaryOp) and isinstance(target, BinaryOp):
        return current.op == target.op
    return False


def _shrink_target_like(target: Expr, budget: int) -> Expr:
    if isinstance(target, (Const, Var)):
        return target

    if budget <= 0:
        return _leaf_projection(target)

    if isinstance(target, UnaryOp):
        return UnaryOp(op=target.op, operand=_shrink_target_like(target.operand, budget - 1))

    if isinstance(target, BinaryOp):
        left, right = _shrink_children(target.children(), budget - 1)
        return BinaryOp(op=target.op, left=left, right=right)

    raise TypeError(f"Unsupported expression type: {type(target).__name__}")


def _shrink_children(children: tuple[Expr, ...], budget: int) -> tuple[Expr, ...]:
    remaining = budget
    shrunk: list[Expr] = []
    for child in children:
        child_size = subtree_size(child)
        if child_size <= remaining:
            shrunk.append(child)
            remaining -= child_size
        else:
            shrunk.append(_shrink_target_like(child, remaining))
            remaining = 0
    return tuple(shrunk)


def _leaf_projection(expr: Expr) -> Expr:
    if isinstance(expr, (Const, Var)):
        return expr

    for child in expr.children():
        leaf = _leaf_projection(child)
        if isinstance(leaf, Var):
            return leaf

    return _leaf_projection(expr.children()[0])


def _same_const_leaf_kind(left: Const, right: Const) -> bool:
    return (left.is_numeric and right.is_numeric) or (left.is_named and right.is_named)


def _ancestor_paths(path: tuple[int, ...], *, include_self: bool) -> list[tuple[int, ...]]:
    start = len(path) if include_self else len(path) - 1
    return [path[:length] for length in range(start, -1, -1)]


def _path_to_node_ids(expr: Expr) -> dict[tuple[int, ...], int]:
    path_to_node_id: dict[tuple[int, ...], int] = {}

    def walk(node: Expr, path: tuple[int, ...]) -> None:
        path_to_node_id[path] = len(path_to_node_id)
        for child_index, child in enumerate(node.children()):
            walk(child, path + (child_index,))

    walk(expr, ())
    return path_to_node_id


def _subtree_at_path(expr: Expr, path: tuple[int, ...]) -> Expr:
    current = expr
    for child_index in path:
        current = current.children()[child_index]
    return current


def _structural_distance(a: Expr, b: Expr) -> int:
    if a == b:
        return 0

    if not _same_distance_root(a, b):
        return 1 + _node_count(a) + _node_count(b)

    children_a = a.children()
    children_b = b.children()
    distance = 0
    for child_a, child_b in zip(children_a, children_b):
        distance += _structural_distance(child_a, child_b)

    if len(children_a) > len(children_b):
        distance += sum(_node_count(child) for child in children_a[len(children_b):])
    elif len(children_b) > len(children_a):
        distance += sum(_node_count(child) for child in children_b[len(children_a):])

    if isinstance(a, (Const, Var)):
        distance += 1
    elif not _same_operator_label(a, b):
        distance += 1

    return distance


def _same_distance_root(a: Expr, b: Expr) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, (Const, Var)):
        return True
    if isinstance(a, UnaryOp) and isinstance(b, UnaryOp):
        return True
    if isinstance(a, BinaryOp) and isinstance(b, BinaryOp):
        return True
    return False


def _same_operator_label(a: Expr, b: Expr) -> bool:
    if isinstance(a, UnaryOp) and isinstance(b, UnaryOp):
        return a.op == b.op
    if isinstance(a, BinaryOp) and isinstance(b, BinaryOp):
        return a.op == b.op
    return True


def _node_count(expr: Expr) -> int:
    return 1 + sum(_node_count(child) for child in expr.children())


__all__ = [
    "EditTarget",
    "FirstMismatch",
    "compute_edit_path",
    "find_first_mismatch",
    "first_edit_toward_target",
    "is_small_enough",
    "structural_distance",
    "subtree_size",
    "trees_equal",
]
