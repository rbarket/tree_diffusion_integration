from __future__ import annotations

from fractions import Fraction

from src.mathlang.ast import BinaryOp, Const, Expr, NaryOp, UnaryOp, Var
from src.mathlang.grammar import NAMED_CONSTANT_TOKENS, SIGNED_INT_TOKENS, VARIABLE_TOKENS, is_binary_operator, is_unary_function, normalize_token


class PrefixParseError(ValueError):
    pass


class UnexpectedEndOfTokens(PrefixParseError):
    pass


class UnknownTokenError(PrefixParseError):
    pass


class TrailingTokensError(PrefixParseError):
    pass


class MalformedIntegerError(PrefixParseError):
    pass


class MalformedExpressionError(PrefixParseError):
    pass


def parse_prefix_string(s: str) -> Expr:
    tokens = s.split()
    if not tokens:
        raise PrefixParseError("Empty prefix expression.")
    return parse_prefix_tokens(tokens)


def parse_prefix_tokens(tokens: list[str]) -> Expr:
    expr, next_index = _parse_expr(tokens, 0)
    if next_index != len(tokens):
        tail = tokens[next_index:next_index + 8]
        raise TrailingTokensError(
            f"Unconsumed trailing tokens starting at index {next_index}: {tail}"
        )
    return expr


def _parse_expr(tokens: list[str], index: int) -> tuple[Expr, int]:
    if index >= len(tokens):
        raise UnexpectedEndOfTokens("Unexpected end of prefix tokens.")

    raw_token = tokens[index]
    token = normalize_token(raw_token)

    if token in NAMED_CONSTANT_TOKENS:
        return Const(symbol=token, token_start=index, token_end=index + 1), index + 1

    if token in VARIABLE_TOKENS:
        return Var(name=token, token_start=index, token_end=index + 1), index + 1

    if token in SIGNED_INT_TOKENS:
        return _parse_signed_int(tokens, index)

    if raw_token.isdigit():
        raise MalformedIntegerError(
            f"Unexpected bare digit token '{raw_token}' at index {index}; expected INT+ or INT- first."
        )

    if is_unary_function(token):
        operand, next_index = _parse_expr(tokens, index + 1)
        return UnaryOp(op=token, operand=operand, token_start=index, token_end=next_index), next_index

    if token in {"add", "mul"}:
        left, next_index = _parse_expr(tokens, index + 1)
        right, end_index = _parse_expr(tokens, next_index)
        operands = _collect_operands(token, left) + _collect_operands(token, right)
        return NaryOp(op=token, operands=tuple(operands), token_start=index, token_end=end_index), end_index

    if is_binary_operator(token):
        left, next_index = _parse_expr(tokens, index + 1)
        right, end_index = _parse_expr(tokens, next_index)
        if token == "div" and isinstance(left, Const) and isinstance(right, Const) and left.is_numeric and right.is_numeric:
            if right.value == 0:
                raise MalformedExpressionError(
                    f"Constant division by zero at token span [{index}, {end_index})."
                )
            return Const(value=left.value / right.value, token_start=index, token_end=end_index), end_index
        return BinaryOp(op=token, left=left, right=right, token_start=index, token_end=end_index), end_index

    raise UnknownTokenError(f"Unknown token '{raw_token}' at index {index}.")


def _parse_signed_int(tokens: list[str], index: int) -> tuple[Const, int]:
    sign_token = tokens[index]
    digits: list[str] = []
    next_index = index + 1
    while next_index < len(tokens) and tokens[next_index].isdigit():
        digits.append(tokens[next_index])
        next_index += 1

    if not digits:
        raise MalformedIntegerError(
            f"Malformed integer at index {index}: expected one or more digit tokens after {sign_token}."
        )

    value = int("".join(digits))
    if sign_token == "INT-":
        value = -value

    return Const(value=Fraction(value, 1), token_start=index, token_end=next_index), next_index


def _collect_operands(op: str, expr: Expr) -> list[Expr]:
    if isinstance(expr, NaryOp) and expr.op == op:
        return list(expr.operands)
    return [expr]
