from __future__ import annotations

from dataclasses import dataclass

from src.mathlang.ast import BinaryOp, Const, Expr, NaryOp, UnaryOp, Var
from src.mathlang.serializer import serialize_prefix_tokens
from src.tree_diffusion.mutation_grammar import has_local_replacement, production_family, subtree_size


@dataclass(frozen=True)
class NodePosition:
    node_id: int
    preorder_index: int
    parent_id: int | None
    depth: int
    production_family: str
    op: str | None
    token_start: int
    token_end: int
    subtree_size: int
    is_mutable: bool


@dataclass(frozen=True)
class PositionIndex:
    positions: tuple[NodePosition, ...]
    node_id_to_node: dict[int, Expr]
    node_id_to_span: dict[int, tuple[int, int]]
    serialized_tokens: tuple[str, ...]


def index_tree_positions(expr: Expr, sigma_small: int | None = None) -> PositionIndex:
    serialized_tokens = tuple(serialize_prefix_tokens(expr))
    positions: list[NodePosition] = []
    node_id_to_node: dict[int, Expr] = {}
    node_id_to_span: dict[int, tuple[int, int]] = {}
    length_cache: dict[Expr, int] = {}

    def token_length(node: Expr) -> int:
        if node in length_cache:
            return length_cache[node]

        if isinstance(node, (Const, Var)):
            length = len(serialize_prefix_tokens(node))
        elif isinstance(node, UnaryOp):
            length = 1 + token_length(node.operand)
        elif isinstance(node, BinaryOp):
            length = 1 + token_length(node.left) + token_length(node.right)
        elif isinstance(node, NaryOp):
            length = _nary_token_length(node.operands, token_length)
        else:
            raise TypeError(f"Unsupported expression type: {type(node).__name__}")

        length_cache[node] = length
        return length

    def walk(node: Expr, *, parent_id: int | None, depth: int, cursor: int) -> None:
        node_id = len(positions)
        family = production_family(node)
        span = (cursor, cursor + token_length(node))
        mutable = has_local_replacement(node)
        if sigma_small is not None:
            mutable = mutable and subtree_size(node) <= sigma_small

        position = NodePosition(
            node_id=node_id,
            preorder_index=node_id,
            parent_id=parent_id,
            depth=depth,
            production_family=family,
            op=_node_op(node),
            token_start=span[0],
            token_end=span[1],
            subtree_size=subtree_size(node),
            is_mutable=mutable,
        )
        positions.append(position)
        node_id_to_node[node_id] = node
        node_id_to_span[node_id] = span

        if isinstance(node, (Const, Var)):
            return

        if isinstance(node, UnaryOp):
            walk(node.operand, parent_id=node_id, depth=depth + 1, cursor=cursor + 1)
            return

        if isinstance(node, BinaryOp):
            left_cursor = cursor + 1
            walk(node.left, parent_id=node_id, depth=depth + 1, cursor=left_cursor)
            right_cursor = left_cursor + token_length(node.left)
            walk(node.right, parent_id=node_id, depth=depth + 1, cursor=right_cursor)
            return

        if isinstance(node, NaryOp):
            _walk_nary_operands(
                node.operands,
                parent_id=node_id,
                depth=depth + 1,
                cursor=cursor,
                visit=walk,
                token_length=token_length,
            )
            return

        raise TypeError(f"Unsupported expression type: {type(node).__name__}")

    walk(expr, parent_id=None, depth=0, cursor=0)
    return PositionIndex(
        positions=tuple(positions),
        node_id_to_node=node_id_to_node,
        node_id_to_span=node_id_to_span,
        serialized_tokens=serialized_tokens,
    )


def _node_op(node: Expr) -> str | None:
    if isinstance(node, Const):
        return node.symbol if node.is_named else "const"
    if isinstance(node, Var):
        return node.name
    if isinstance(node, (UnaryOp, BinaryOp, NaryOp)):
        return node.op
    return None


def _nary_token_length(operands: tuple[Expr, ...], token_length) -> int:
    if len(operands) == 2:
        return 1 + token_length(operands[0]) + token_length(operands[1])
    return 1 + token_length(operands[0]) + _nary_token_length(operands[1:], token_length)


def _walk_nary_operands(
    operands: tuple[Expr, ...],
    *,
    parent_id: int,
    depth: int,
    cursor: int,
    visit,
    token_length,
) -> None:
    first_cursor = cursor + 1
    visit(operands[0], parent_id=parent_id, depth=depth, cursor=first_cursor)
    if len(operands) == 2:
        second_cursor = first_cursor + token_length(operands[0])
        visit(operands[1], parent_id=parent_id, depth=depth, cursor=second_cursor)
        return

    rest_cursor = first_cursor + token_length(operands[0])
    _walk_nary_operands(
        operands[1:],
        parent_id=parent_id,
        depth=depth,
        cursor=rest_cursor,
        visit=visit,
        token_length=token_length,
    )
