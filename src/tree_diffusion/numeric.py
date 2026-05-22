from __future__ import annotations

import math
from typing import Any


def finite_numeric(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def is_finite_numeric(value: Any) -> bool:
    return finite_numeric(value) is not None


def meets_numeric_tol(value: Any, numeric_tol: float) -> bool:
    numeric = finite_numeric(value)
    return numeric is not None and numeric <= float(numeric_tol)


__all__ = [
    "finite_numeric",
    "is_finite_numeric",
    "meets_numeric_tol",
]
