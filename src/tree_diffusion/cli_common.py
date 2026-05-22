from __future__ import annotations


def optional_int_arg(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"none", "null"}:
        return None
    return int(lowered)


def optional_float_arg(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"none", "null"}:
        return None
    return float(lowered)


__all__ = [
    "optional_float_arg",
    "optional_int_arg",
]
