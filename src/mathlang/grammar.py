from __future__ import annotations

from dataclasses import dataclass


TOKEN_ALIASES = {
    "PI": "Pi",
    "pi": "Pi",
}

VARIABLE_TOKENS = frozenset({"x"})
NAMED_CONSTANT_TOKENS = frozenset({"E", "I", "Pi"})
DIGIT_TOKENS = frozenset(str(i) for i in range(10))
SIGNED_INT_TOKENS = frozenset({"INT+", "INT-"})


@dataclass(frozen=True)
class OperatorSpec:
    token: str
    arity_kind: str
    arity: int
    commutative: bool
    associative: bool
    category: str


OPERATOR_SPECS = {
    "add": OperatorSpec("add", "binary", 2, True, True, "operator"),
    "mul": OperatorSpec("mul", "binary", 2, True, True, "operator"),
    "pow": OperatorSpec("pow", "binary", 2, False, False, "operator"),
    "div": OperatorSpec("div", "binary", 2, False, False, "operator"),
    "ln": OperatorSpec("ln", "unary", 1, False, False, "function"),
    "exp": OperatorSpec("exp", "unary", 1, False, False, "function"),
    "sqrt": OperatorSpec("sqrt", "unary", 1, False, False, "function"),
    "abs": OperatorSpec("abs", "unary", 1, False, False, "function"),
    "sin": OperatorSpec("sin", "unary", 1, False, False, "function"),
    "cos": OperatorSpec("cos", "unary", 1, False, False, "function"),
    "tan": OperatorSpec("tan", "unary", 1, False, False, "function"),
    "cot": OperatorSpec("cot", "unary", 1, False, False, "function"),
    "sinh": OperatorSpec("sinh", "unary", 1, False, False, "function"),
    "cosh": OperatorSpec("cosh", "unary", 1, False, False, "function"),
    "tanh": OperatorSpec("tanh", "unary", 1, False, False, "function"),
    "coth": OperatorSpec("coth", "unary", 1, False, False, "function"),
    "asin": OperatorSpec("asin", "unary", 1, False, False, "function"),
    "acos": OperatorSpec("acos", "unary", 1, False, False, "function"),
    "atan": OperatorSpec("atan", "unary", 1, False, False, "function"),
    "acot": OperatorSpec("acot", "unary", 1, False, False, "function"),
    "asinh": OperatorSpec("asinh", "unary", 1, False, False, "function"),
    "acosh": OperatorSpec("acosh", "unary", 1, False, False, "function"),
    "atanh": OperatorSpec("atanh", "unary", 1, False, False, "function"),
}


ALL_VALID_TOKENS = frozenset(OPERATOR_SPECS) | VARIABLE_TOKENS | NAMED_CONSTANT_TOKENS | DIGIT_TOKENS | SIGNED_INT_TOKENS


def normalize_token(token: str) -> str:
    return TOKEN_ALIASES.get(token, token)


def is_operator(token: str) -> bool:
    return normalize_token(token) in OPERATOR_SPECS


def arity(token: str) -> int:
    normalized = normalize_token(token)
    if normalized not in OPERATOR_SPECS:
        raise KeyError(f"Unknown operator: {token}")
    return OPERATOR_SPECS[normalized].arity


def is_commutative(token: str) -> bool:
    normalized = normalize_token(token)
    return normalized in OPERATOR_SPECS and OPERATOR_SPECS[normalized].commutative


def is_associative(token: str) -> bool:
    normalized = normalize_token(token)
    return normalized in OPERATOR_SPECS and OPERATOR_SPECS[normalized].associative


def is_unary_function(token: str) -> bool:
    normalized = normalize_token(token)
    return normalized in OPERATOR_SPECS and OPERATOR_SPECS[normalized].arity_kind == "unary"


def is_binary_operator(token: str) -> bool:
    normalized = normalize_token(token)
    return normalized in OPERATOR_SPECS and OPERATOR_SPECS[normalized].arity_kind == "binary"


def valid_token(token: str) -> bool:
    return normalize_token(token) in ALL_VALID_TOKENS
