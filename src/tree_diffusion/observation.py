from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from src.mathlang.ast import Expr
from src.mathlang.canonicalize import canonicalize
from src.mathlang.conversions import ast_to_sympy, sympy_to_ast
from src.mathlang.serializer import serialize_prefix_tokens


DEFAULT_PROBE_POINTS: tuple[float, ...] = (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0)
RESIDUAL_MODES = frozenset({"none", "symbolic", "numeric", "both"})
_NO_STRIP_VARIABLE = "__observation_no_strip__"


@dataclass(frozen=True)
class NumericProbeFeatures:
    probe_points: tuple[float, ...]
    residual_values: tuple[float | None, ...]
    finite_mask: tuple[bool, ...]
    mean_abs_residual: float | None
    mean_squared_residual: float | None
    max_abs_residual: float | None
    fraction_finite: float


@dataclass(frozen=True)
class Observation:
    target_integrand: Expr
    current_antiderivative: Expr
    current_derivative: Expr | None
    symbolic_residual: Expr | None
    numeric_probes: NumericProbeFeatures | None
    residual_mode: str
    status: str
    warnings: tuple[str, ...]


def compute_current_derivative(
    current_antiderivative: Expr,
    var: str = "x",
    simplify_derivative: bool = False,
) -> Expr:
    derivative_sym = _compute_current_derivative_sympy(
        current_antiderivative,
        var=var,
        simplify_derivative=simplify_derivative,
    )
    return _canonicalize_expr(sympy_to_ast(derivative_sym))


def compute_symbolic_residual(
    current_derivative: Expr,
    target_integrand: Expr,
    var: str = "x",
    simplify_residual: bool = True,
) -> Expr:
    del var
    residual_sym = _compute_symbolic_residual_sympy(
        ast_to_sympy(current_derivative),
        ast_to_sympy(target_integrand),
        simplify_residual=simplify_residual,
    )
    return _canonicalize_expr(sympy_to_ast(residual_sym))


def compute_numeric_probes(
    current_derivative: Expr,
    target_integrand: Expr,
    probe_points: Sequence[float] | None = None,
    var: str = "x",
) -> NumericProbeFeatures:
    current_derivative_sym = ast_to_sympy(current_derivative)
    target_integrand_sym = ast_to_sympy(target_integrand)
    return _compute_numeric_probes_from_sympy(
        current_derivative_sym - target_integrand_sym,
        probe_points=probe_points,
        var=var,
    )


def build_observation(
    target_integrand: Expr,
    current_antiderivative: Expr,
    *,
    residual_mode: str = "both",
    var: str = "x",
    probe_points: Sequence[float] | None = None,
    simplify_derivative: bool = False,
    simplify_symbolic_residual: bool = True,
    max_derivative_tokens: int | None = None,
    max_residual_tokens: int | None = None,
) -> Observation:
    _validate_residual_mode(residual_mode)

    warnings: list[str] = []
    canonical_target = _canonicalize_expr(target_integrand)
    canonical_current = _canonicalize_expr(current_antiderivative)

    current_derivative_sym: sp.Expr | None = None
    current_derivative: Expr | None = None
    symbolic_residual: Expr | None = None
    numeric_probes: NumericProbeFeatures | None = None
    target_integrand_sym: sp.Expr | None = None

    try:
        current_derivative_sym = _compute_current_derivative_sympy(
            canonical_current,
            var=var,
            simplify_derivative=simplify_derivative,
        )
    except Exception as exc:
        warnings.append(f"derivative_failed:{type(exc).__name__}")
        return Observation(
            target_integrand=canonical_target,
            current_antiderivative=canonical_current,
            current_derivative=None,
            symbolic_residual=None,
            numeric_probes=None,
            residual_mode=residual_mode,
            status="derivative_failed",
            warnings=tuple(warnings),
        )

    try:
        current_derivative = _canonicalize_expr(sympy_to_ast(current_derivative_sym))
    except Exception as exc:
        warnings.append(f"derivative_ast_conversion_failed:{type(exc).__name__}")
    else:
        current_derivative = _apply_token_cap(
            current_derivative,
            component_name="current_derivative",
            max_tokens=max_derivative_tokens,
            warnings=warnings,
        )

    if residual_mode in {"symbolic", "numeric", "both"}:
        try:
            target_integrand_sym = ast_to_sympy(canonical_target)
        except Exception as exc:
            warnings.append(f"target_integrand_sympy_failed:{type(exc).__name__}")

    if residual_mode in {"symbolic", "both"} and target_integrand_sym is not None:
        try:
            symbolic_residual_sym = _compute_symbolic_residual_sympy(
                current_derivative_sym,
                target_integrand_sym,
                simplify_residual=simplify_symbolic_residual,
            )
            symbolic_residual = _canonicalize_expr(sympy_to_ast(symbolic_residual_sym))
            symbolic_residual = _apply_token_cap(
                symbolic_residual,
                component_name="symbolic_residual",
                max_tokens=max_residual_tokens,
                warnings=warnings,
            )
        except Exception as exc:
            warnings.append(f"symbolic_residual_failed:{type(exc).__name__}")

    if residual_mode in {"numeric", "both"} and target_integrand_sym is not None:
        try:
            numeric_probes = _compute_numeric_probes_from_sympy(
                current_derivative_sym - target_integrand_sym,
                probe_points=probe_points,
                var=var,
            )
            nonfinite_count = len(numeric_probes.probe_points) - sum(numeric_probes.finite_mask)
            if nonfinite_count > 0:
                warnings.append(
                    f"numeric_probe_nonfinite:{nonfinite_count}/{len(numeric_probes.probe_points)}"
                )
        except Exception as exc:
            warnings.append(f"numeric_probe_failed:{type(exc).__name__}")

    status = _derive_status(
        current_derivative=current_derivative,
        symbolic_residual=symbolic_residual,
        numeric_probes=numeric_probes,
        residual_mode=residual_mode,
    )

    return Observation(
        target_integrand=canonical_target,
        current_antiderivative=canonical_current,
        current_derivative=current_derivative,
        symbolic_residual=symbolic_residual,
        numeric_probes=numeric_probes,
        residual_mode=residual_mode,
        status=status,
        warnings=tuple(warnings),
    )


def _compute_current_derivative_sympy(
    current_antiderivative: Expr,
    *,
    var: str,
    simplify_derivative: bool,
) -> sp.Expr:
    variable = sp.Symbol(var, real=True)
    derivative_sym = sp.diff(ast_to_sympy(current_antiderivative), variable)
    if simplify_derivative:
        derivative_sym = sp.simplify(derivative_sym)
    return derivative_sym


def _compute_symbolic_residual_sympy(
    current_derivative_sym: sp.Expr,
    target_integrand_sym: sp.Expr,
    *,
    simplify_residual: bool,
) -> sp.Expr:
    residual_sym = current_derivative_sym - target_integrand_sym
    if simplify_residual:
        residual_sym = sp.simplify(residual_sym)
    return residual_sym


def _compute_numeric_probes_from_sympy(
    residual_sym: sp.Expr,
    *,
    probe_points: Sequence[float] | None,
    var: str,
) -> NumericProbeFeatures:
    raw_points = DEFAULT_PROBE_POINTS if probe_points is None else probe_points
    points = tuple(float(point) for point in raw_points)
    variable = sp.Symbol(var, real=True)

    residual_values: list[float | None] = []
    finite_mask: list[bool] = []

    for point in points:
        try:
            evaluated = sp.N(residual_sym.subs(variable, point))
            if evaluated.is_real is not True or evaluated.is_finite is not True:
                raise ValueError("non-finite-or-non-real")
            residual_value = float(evaluated)
        except Exception:
            residual_values.append(None)
            finite_mask.append(False)
            continue

        residual_values.append(residual_value)
        finite_mask.append(True)

    finite_values = [value for value in residual_values if value is not None]
    if finite_values:
        abs_values = [abs(value) for value in finite_values]
        mean_abs_residual = sum(abs_values) / len(abs_values)
        mean_squared_residual = sum(value * value for value in finite_values) / len(finite_values)
        max_abs_residual = max(abs_values)
    else:
        mean_abs_residual = None
        mean_squared_residual = None
        max_abs_residual = None

    return NumericProbeFeatures(
        probe_points=points,
        residual_values=tuple(residual_values),
        finite_mask=tuple(finite_mask),
        mean_abs_residual=mean_abs_residual,
        mean_squared_residual=mean_squared_residual,
        max_abs_residual=max_abs_residual,
        fraction_finite=(len(finite_values) / len(points)) if points else 0.0,
    )


def _canonicalize_expr(expr: Expr) -> Expr:
    return canonicalize(expr, variable=_NO_STRIP_VARIABLE)


def _apply_token_cap(
    expr: Expr,
    *,
    component_name: str,
    max_tokens: int | None,
    warnings: list[str],
) -> Expr | None:
    if max_tokens is None:
        return expr

    token_count = len(serialize_prefix_tokens(expr))
    if token_count <= max_tokens:
        return expr

    warnings.append(f"{component_name}_token_cap_exceeded:{token_count}>{max_tokens}")
    return None


def _derive_status(
    *,
    current_derivative: Expr | None,
    symbolic_residual: Expr | None,
    numeric_probes: NumericProbeFeatures | None,
    residual_mode: str,
) -> str:
    if current_derivative is None:
        return "partial"

    if residual_mode in {"symbolic", "both"} and symbolic_residual is None:
        return "partial"

    if residual_mode in {"numeric", "both"}:
        if numeric_probes is None or numeric_probes.fraction_finite <= 0.0:
            return "partial"

    return "ok"


def _validate_residual_mode(residual_mode: str) -> None:
    if residual_mode not in RESIDUAL_MODES:
        raise ValueError(
            f"Unsupported residual_mode {residual_mode!r}; expected one of {sorted(RESIDUAL_MODES)}."
        )
