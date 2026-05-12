from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import signal
import threading
from typing import Sequence

import sympy as sp

from src.mathlang.ast import Expr
from src.mathlang.canonicalize import canonicalize
from src.mathlang.conversions import ast_to_sympy, sympy_to_ast
from src.mathlang.serializer import serialize_prefix_tokens


DEFAULT_PROBE_POINTS: tuple[float, ...] = (
    -3.0,
    -2.0,
    -1.0,
    -0.5,
    -0.25,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
)
RESIDUAL_MODES = frozenset({"none", "symbolic", "numeric", "both"})
_DEFAULT_COMPLEX_TOLERANCE = 1e-10


class ObservationTimeoutError(TimeoutError):
    """Raised when SymPy-backed observation construction exceeds its budget."""


@dataclass(frozen=True)
class NumericProbeFeatures:
    probe_points: tuple[float, ...]
    residual_real: tuple[float | None, ...]
    residual_imag: tuple[float | None, ...]
    residual_abs: tuple[float | None, ...]
    residual_abs_squared: tuple[float | None, ...]
    finite_mask: tuple[bool, ...]
    complex_mask: tuple[bool, ...]
    mean_abs_residual: float | None
    mean_squared_abs_residual: float | None
    max_abs_residual: float | None
    fraction_finite: float
    fraction_complex: float

    @property
    def residual_values(self) -> tuple[float | None, ...]:
        return self.residual_real

    @property
    def mean_squared_residual(self) -> float | None:
        return self.mean_squared_abs_residual


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
    residual_sym = current_derivative_sym - target_integrand_sym
    return _compute_numeric_probes_from_sympy(
        residual_sym,
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
    observation_timeout_seconds: float | None = None,
) -> Observation:
    _validate_residual_mode(residual_mode)
    _validate_timeout(observation_timeout_seconds)

    warnings: list[str] = []
    canonical_target = _canonicalize_expr(target_integrand)
    canonical_current = _canonicalize_expr(current_antiderivative)

    current_derivative_sym: sp.Expr | None = None
    current_derivative: Expr | None = None
    symbolic_residual: Expr | None = None
    numeric_probes: NumericProbeFeatures | None = None
    target_integrand_sym: sp.Expr | None = None

    try:
        with _observation_timeout(observation_timeout_seconds):
            current_derivative_sym = _compute_current_derivative_sympy(
                canonical_current,
                var=var,
                simplify_derivative=simplify_derivative,
            )
    except ObservationTimeoutError:
        warnings.append("derivative_timeout")
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
        with _observation_timeout(observation_timeout_seconds):
            current_derivative = _canonicalize_expr(sympy_to_ast(current_derivative_sym))
    except Exception as exc:
        if isinstance(exc, ObservationTimeoutError):
            warnings.append("derivative_ast_conversion_timeout")
        else:
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
            with _observation_timeout(observation_timeout_seconds):
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
        except ObservationTimeoutError:
            warnings.append("symbolic_residual_timeout")
        except Exception as exc:
            warnings.append(f"symbolic_residual_failed:{type(exc).__name__}")

    if residual_mode in {"numeric", "both"} and target_integrand_sym is not None:
        try:
            with _observation_timeout(observation_timeout_seconds):
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
            if numeric_probes.fraction_complex > 0.0:
                warnings.append(
                    f"numeric_probe_complex:{sum(numeric_probes.complex_mask)}/{sum(numeric_probes.finite_mask)}"
                )
        except ObservationTimeoutError:
            warnings.append("numeric_probe_timeout")
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

    residual_real: list[float | None] = []
    residual_imag: list[float | None] = []
    residual_abs: list[float | None] = []
    residual_abs_squared: list[float | None] = []
    finite_mask: list[bool] = []
    complex_mask: list[bool] = []

    for point in points:
        try:
            try:
                substitution_value: sp.Expr = sp.Rational(str(point))
            except Exception:
                substitution_value = sp.Float(point)
            evaluated = sp.N(residual_sym.subs(variable, substitution_value))
            residual_value = complex(evaluated)
            real_part = float(residual_value.real)
            imag_part = float(residual_value.imag)
            if not math.isfinite(real_part) or not math.isfinite(imag_part):
                raise ValueError("non-finite")
            abs_squared = (real_part * real_part) + (imag_part * imag_part)
            abs_value = math.sqrt(abs_squared)
        except ObservationTimeoutError:
            raise
        except Exception:
            residual_real.append(None)
            residual_imag.append(None)
            residual_abs.append(None)
            residual_abs_squared.append(None)
            finite_mask.append(False)
            complex_mask.append(False)
            continue

        residual_real.append(real_part)
        residual_imag.append(imag_part)
        residual_abs.append(abs_value)
        residual_abs_squared.append(abs_squared)
        finite_mask.append(True)
        complex_mask.append(abs(imag_part) > _DEFAULT_COMPLEX_TOLERANCE)

    finite_abs_values = [value for value in residual_abs if value is not None]
    finite_abs_squared_values = [value for value in residual_abs_squared if value is not None]
    finite_count = sum(finite_mask)
    complex_finite_count = sum(complex_mask)
    if finite_abs_values:
        mean_abs_residual = sum(finite_abs_values) / len(finite_abs_values)
        mean_squared_abs_residual = sum(finite_abs_squared_values) / len(finite_abs_squared_values)
        max_abs_residual = max(finite_abs_values)
    else:
        mean_abs_residual = None
        mean_squared_abs_residual = None
        max_abs_residual = None

    return NumericProbeFeatures(
        probe_points=points,
        residual_real=tuple(residual_real),
        residual_imag=tuple(residual_imag),
        residual_abs=tuple(residual_abs),
        residual_abs_squared=tuple(residual_abs_squared),
        finite_mask=tuple(finite_mask),
        complex_mask=tuple(complex_mask),
        mean_abs_residual=mean_abs_residual,
        mean_squared_abs_residual=mean_squared_abs_residual,
        max_abs_residual=max_abs_residual,
        fraction_finite=(finite_count / len(points)) if points else 0.0,
        fraction_complex=(complex_finite_count / finite_count) if finite_count else 0.0,
    )


def _canonicalize_expr(expr: Expr) -> Expr:
    return canonicalize(expr, strip_additive_constants=False)


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


def _validate_timeout(timeout_seconds: float | None) -> None:
    if timeout_seconds is not None and timeout_seconds <= 0.0:
        raise ValueError("observation_timeout_seconds must be > 0 when provided.")


@contextmanager
def _observation_timeout(timeout_seconds: float | None):
    if (
        timeout_seconds is None
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def _raise_timeout(signum, frame):
        del signum, frame
        raise ObservationTimeoutError(
            f"Observation construction exceeded {timeout_seconds:.3f}s."
        )

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
