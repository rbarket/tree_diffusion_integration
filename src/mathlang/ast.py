from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction


NAMED_CONSTANTS = frozenset({"E", "I", "Pi"})


@dataclass(frozen=True, kw_only=True)
class Expr:
    token_start: int | None = field(default=None, compare=False)
    token_end: int | None = field(default=None, compare=False)

    def children(self) -> tuple[Expr, ...]:
        return ()

    @property
    def node_type(self) -> str:
        return type(self).__name__

    @property
    def production_label(self) -> str:
        return self.node_type

    def contains_var(self, variable: str = "x") -> bool:
        return any(child.contains_var(variable) for child in self.children())


@dataclass(frozen=True)
class Const(Expr):
    value: Fraction | None = None
    symbol: str | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.symbol is None):
            raise ValueError("Const must have exactly one of 'value' or 'symbol'.")

        if self.value is not None and not isinstance(self.value, Fraction):
            object.__setattr__(self, "value", Fraction(self.value))

        if self.symbol is not None and self.symbol not in NAMED_CONSTANTS:
            raise ValueError(f"Unsupported constant symbol: {self.symbol}")

    @property
    def production_label(self) -> str:
        return self.symbol or "const"

    @property
    def is_numeric(self) -> bool:
        return self.value is not None

    @property
    def is_named(self) -> bool:
        return self.symbol is not None


@dataclass(frozen=True)
class Var(Expr):
    name: str

    @property
    def production_label(self) -> str:
        return self.name

    def contains_var(self, variable: str = "x") -> bool:
        return self.name == variable


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str
    operand: Expr

    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)

    @property
    def production_label(self) -> str:
        return self.op


@dataclass(frozen=True)
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr

    def children(self) -> tuple[Expr, ...]:
        return (self.left, self.right)

    @property
    def production_label(self) -> str:
        return self.op
