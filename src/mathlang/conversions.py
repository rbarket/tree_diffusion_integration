from __future__ import annotations

from typing import List, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from src.mathlang.ast import BinaryOp, Const, Expr, UnaryOp, Var
from src.mathlang.parser import parse_prefix_tokens

# ---------- Unary-function registries ----------
_PREFIX_UNARY_TO_SYMPY = {
    "abs": sp.Abs,
    "acos": sp.acos,
    "acosh": sp.acosh,
    "acot": sp.acot,
    "asin": sp.asin,
    "asinh": sp.asinh,
    "atan": sp.atan,
    "atanh": sp.atanh,
    "cos": sp.cos,
    "cosh": sp.cosh,
    "cot": sp.cot,
    "coth": sp.coth,
    "csc": sp.csc,
    "csch": sp.csch,
    "exp": sp.exp,
    "ln": sp.log,
    "sec": sp.sec,
    "sech": sp.sech,
    "sign": sp.sign,
    "sin": sp.sin,
    "sinh": sp.sinh,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
    "tanh": sp.tanh,
}

_EXTRA_LOCAL_DICT_ENTRIES = {
    "ln": sp.log,
    "log": sp.log,
    "sec": sp.sec,
    "csc": sp.csc,
    "sech": sp.sech,
    "csch": sp.csch,
    "asec": sp.asec,
    "acsc": sp.acsc,
    "acoth": sp.acoth,
    "asech": sp.asech,
    "acsch": sp.acsch,
}

_SYMPY_UNARY_TO_PREFIX = {
    sympy_func: token for token, sympy_func in _PREFIX_UNARY_TO_SYMPY.items()
}
_SYMPY_UNARY_TO_PREFIX.update(
    {
        sp.sec: "sec",
        sp.csc: "csc",
        sp.asec: "asec",
        sp.acsc: "acsc",
        sp.sech: "sech",
        sp.csch: "csch",
        sp.acoth: "acoth",
        sp.asech: "asech",
        sp.acsch: "acsch",
    }
)

# ---------- SymPy parsing config ----------
_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,  # allow "^" as power
)

_LOCAL_DICT = {
    # variables
    "x": sp.Symbol("x", real=True),
    "y": sp.Symbol("y", real=True),
    "z": sp.Symbol("z", real=True),
    "t": sp.Symbol("t", real=True),
    # constants
    "E": sp.E,
    "Pi": sp.pi,
    "pi": sp.pi,
    # common function names
    **{token: func for token, func in _PREFIX_UNARY_TO_SYMPY.items() if token != "ln"},
    **_EXTRA_LOCAL_DICT_ENTRIES,
}


def _normalize_infix(s: str) -> str:
    s = s.strip()

    # normalize log alias
    s = s.replace("ln(", "log(")
    repl = {
        "arcsin(": "asin(",
        "arccos(": "acos(",
        "arctan(": "atan(",
        "arccot(": "acot(",
        "arcsec(": "asec(",
        "arccsc(": "acsc(",

        "arcsinh(": "asinh(",
        "arccosh(": "acosh(",
        "arctanh(": "atanh(",
        "arccoth(": "acoth(",
        "arcsech(": "asech(",
        "arccsch(": "acsch(",
    }
    for k, v in repl.items():
        s = s.replace(k, v)

    return s



def infix_to_sympy(s: str) -> sp.Expr:
    s = _normalize_infix(s)
    return parse_expr(s, local_dict=_LOCAL_DICT, transformations=_TRANSFORMS, evaluate=True)


# ---------- SymPy -> prefix ----------
def _write_int_base10(n: int) -> List[str]:
    sign = "INT-" if n < 0 else "INT+"
    digits = list(str(abs(n)))
    return [sign] + digits


def sympy_to_prefix(expr: sp.Expr) -> List[str]:
    # symbols
    if isinstance(expr, sp.Symbol):
        return [str(expr)]

    # integers / rationals
    if isinstance(expr, sp.Integer):
        return _write_int_base10(int(expr))
    if isinstance(expr, sp.Rational):
        return ["div"] + _write_int_base10(int(expr.p)) + _write_int_base10(int(expr.q))

    # constants
    if expr == sp.E:
        return ["E"]
    if expr == sp.pi:
        return ["Pi"]
    if expr == sp.I:
        return ["I"]

    # unary functions
    if expr.func in _SYMPY_UNARY_TO_PREFIX:
        return [_SYMPY_UNARY_TO_PREFIX[expr.func]] + sympy_to_prefix(expr.args[0])

    # power
    if expr.func == sp.Pow:
        base, exp = expr.args
        if isinstance(exp, sp.Rational) and exp.p == 1 and exp.q == 2:
            return ["sqrt"] + sympy_to_prefix(base)
        return ["pow"] + sympy_to_prefix(base) + sympy_to_prefix(exp)

    # add / mul: emit right-nested binary prefix like Lample/Charton
    if expr.func == sp.Add:
        args = list(expr.args)
        if len(args) == 1:
            return sympy_to_prefix(args[0])
        out: List[str] = []
        for i, a in enumerate(args):
            if i == 0 or i < len(args) - 1:
                out.append("add")
            out += sympy_to_prefix(a)
        return out

    if expr.func == sp.Mul:
        args = list(expr.args)
        if len(args) == 1:
            return sympy_to_prefix(args[0])
        out: List[str] = []
        for i, a in enumerate(args):
            if i == 0 or i < len(args) - 1:
                out.append("mul")
            out += sympy_to_prefix(a)
        return out

    raise ValueError(f"Unsupported SymPy expression: {expr} (func={expr.func})")


def infix_to_prefix_tokens(s: str) -> List[str]:
    return sympy_to_prefix(infix_to_sympy(s))


# ---------- prefix -> infix (minimal, for later verification/debug) ----------
_UNARY_TOKENS = {
    "exp", "ln", "abs",
    "sign",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
    "sqrt", 'INT+','INT-'
}
_BINARY_TOKENS = {"add", "mul", "pow", "div"}
_ATOM_TOKEN_ALIASES = {
    "Pi": "pi",
    "PI": "pi",
}


def _read_digit_run(tokens: List[str], i: int) -> Tuple[str, int]:
    """
    Read one or more base-10 digit tokens starting at tokens[i].
    Returns (digits_string, next_index).
    """
    digits = []
    while i < len(tokens) and tokens[i].isdigit():
        digits.append(tokens[i])
        i += 1
    if not digits:
        raise ValueError("Malformed integer: expected at least one digit token.")
    return "".join(digits), i


def _read_signed_int(tokens: List[str], i: int) -> Tuple[str, int]:
    """
    Parse INT+ / INT- as a unary operator whose child is a digit-run leaf.
    Grammar: INT+ <digit-run>  |  INT- <digit-run>

    Returns (signed_integer_string, next_index).
    """
    sign = tokens[i]
    if sign not in ("INT+", "INT-"):
        raise ValueError("Expected INT+ or INT-")
    digits, j = _read_digit_run(tokens, i + 1)
    n = int(digits)
    if sign == "INT-":
        n = -n
    return str(n), j



def prefix_tokens_to_infix(tokens: List[str]) -> str:
    """
    Deterministic parse of prefix tokens into an infix string.
    """
    def parse(i: int) -> Tuple[str, int]:
        if i >= len(tokens):
            raise ValueError("Unexpected end of prefix tokens.")
        t = tokens[i]
        t = _ATOM_TOKEN_ALIASES.get(t, t)

        # integer (INT+ / INT- are unary; child is a digit-run leaf)
        if t in ("INT+", "INT-"):
            return _read_signed_int(tokens, i)

        # digit-run leaf (only appears as child of INT+/INT- in well-formed sequences)
        if t.isdigit():
            digits, j = _read_digit_run(tokens, i)
            return digits, j


        # atoms
        if t in ("x", "y", "z", "t", "E", "pi", "I"):
            return t, i + 1

        # unary
        if t in _UNARY_TOKENS:
            arg, j = parse(i + 1)
            if t == "ln":
                return f"log({arg})", j
            if t == "sqrt":
                return f"sqrt({arg})", j
            if t == "abs":
                return f"Abs({arg})", j
            return f"{t}({arg})", j

        # binary
        if t in _BINARY_TOKENS:
            a, j = parse(i + 1)
            b, k = parse(j)
            if t == "add":
                return f"({a})+({b})", k
            if t == "mul":
                return f"({a})*({b})", k
            if t == "div":
                return f"({a})/({b})", k
            if t == "pow":
                return f"({a})**({b})", k

        raise ValueError(f"Unknown token in prefix parse: {t}")

    s, j = parse(0)
    if j != len(tokens):
        raise ValueError(f"Unconsumed tokens remaining at index {j}: {tokens[j:j+10]}")
    return s


def prefix_tokens_to_sympy(tokens: List[str]) -> sp.Expr:
    """
    Convert prefix tokens -> AST -> SymPy.
    """
    return ast_to_sympy(parse_prefix_tokens(list(tokens)))


def sympy_to_ast(expr: sp.Expr) -> Expr:
    return parse_prefix_tokens(sympy_to_prefix(expr))


def ast_to_sympy(expr: Expr) -> sp.Expr:
    if isinstance(expr, Const):
        if expr.is_named:
            mapping = {
                "E": sp.E,
                "I": sp.I,
                "Pi": sp.pi,
            }
            return mapping[expr.symbol]
        if expr.value.denominator == 1:
            return sp.Integer(expr.value.numerator)
        return sp.Rational(expr.value.numerator, expr.value.denominator)

    if isinstance(expr, Var):
        return sp.Symbol(expr.name, real=True)

    if isinstance(expr, UnaryOp):
        return _PREFIX_UNARY_TO_SYMPY[expr.op](ast_to_sympy(expr.operand))

    if isinstance(expr, BinaryOp):
        if expr.op == "add":
            return sp.Add(ast_to_sympy(expr.left), ast_to_sympy(expr.right))
        if expr.op == "mul":
            return sp.Mul(ast_to_sympy(expr.left), ast_to_sympy(expr.right))
        if expr.op == "pow":
            return sp.Pow(ast_to_sympy(expr.left), ast_to_sympy(expr.right))
        if expr.op == "div":
            return ast_to_sympy(expr.left) / ast_to_sympy(expr.right)
        raise ValueError(f"Unsupported binary operator: {expr.op}")

    raise TypeError(f"Unsupported AST node: {type(expr).__name__}")
