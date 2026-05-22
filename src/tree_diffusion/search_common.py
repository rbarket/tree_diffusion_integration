from __future__ import annotations

import math
import time
from typing import Any

import sympy as sp
import torch

from src.mathlang.ast import Expr
from src.mathlang.canonicalize import canonicalize
from src.mathlang.conversions import ast_to_sympy
from src.tree_diffusion.edit_path import structural_distance
from src.tree_diffusion.observation import (
    _observation_timeout,
    build_observation,
    compute_current_derivative,
)
from src.tree_diffusion.numeric import (
    finite_numeric,
    is_finite_numeric,
    meets_numeric_tol,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


def tree_size(expr: Expr) -> int:
    return 1 + sum(tree_size(child) for child in expr.children())


def derivative_matches_target(
    antiderivative: Expr,
    target_integrand: Expr,
    *,
    simplify: bool = True,
    timeout_seconds: float | None = None,
) -> bool:
    try:
        with _observation_timeout(timeout_seconds):
            derivative = compute_current_derivative(
                antiderivative,
                simplify_derivative=simplify,
            )
            if simplify:
                residual = ast_to_sympy(derivative) - ast_to_sympy(
                    canonicalize(target_integrand, strip_additive_constants=False)
                )
                return bool(sp.simplify(residual) == 0)
            return canonicalize(derivative, strip_additive_constants=False) == canonicalize(
                target_integrand,
                strip_additive_constants=False,
            )
    except Exception:
        return False


def structural_distance_or_none(current: Expr, target: Expr | None) -> int | None:
    if target is None:
        return None
    try:
        return int(structural_distance(current, target))
    except Exception:
        return None


def numeric_key(value: Any) -> float:
    numeric = finite_numeric(value)
    return math.inf if numeric is None else numeric


def numeric_better(a: float | None, b: float | None) -> bool:
    return numeric_key(a) < numeric_key(b)


def best_numeric_residual(
    current_best: float | None,
    candidate_value: float | None,
) -> float | None:
    candidate = finite_numeric(candidate_value)
    if candidate is None:
        return current_best
    current = finite_numeric(current_best)
    if current is None:
        return candidate
    return min(current, candidate)


def structural_better(candidate: int | None, current: int | None) -> bool:
    if candidate is None:
        return False
    if current is None:
        return True
    return int(candidate) < int(current)


def encode_repair_observation(
    target_integrand: Expr,
    current_antiderivative: Expr,
    *,
    tokenizer: TreeDiffusionTokenizer,
    residual_mode: str = "both",
    simplify_symbolic_residual: bool = True,
    observation_timeout_seconds: float | None = None,
    max_input_length: int | None = None,
) -> tuple[list[str], torch.LongTensor, torch.LongTensor]:
    observation = build_observation(
        target_integrand,
        current_antiderivative,
        residual_mode=residual_mode,
        simplify_symbolic_residual=simplify_symbolic_residual,
        observation_timeout_seconds=observation_timeout_seconds,
    )
    input_tokens = tokenizer.serialize_observation(observation) + ["<EDIT>"]
    input_ids = torch.tensor(
        tokenizer.encode_tokens(input_tokens, pad_to_length=max_input_length),
        dtype=torch.long,
    )
    input_attention_mask = input_ids.ne(tokenizer.pad_id).to(dtype=torch.long)
    return input_tokens, input_ids, input_attention_mask


def remaining_timeout(timeout_seconds: float | None, *, deadline: float | None) -> float | None:
    if deadline is None:
        return timeout_seconds
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise TimeoutError("Search timeout expired.")
    if timeout_seconds is None:
        return remaining
    return min(float(timeout_seconds), remaining)


def deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


__all__ = [
    "best_numeric_residual",
    "deadline_expired",
    "derivative_matches_target",
    "encode_repair_observation",
    "finite_numeric",
    "is_finite_numeric",
    "meets_numeric_tol",
    "numeric_better",
    "numeric_key",
    "remaining_timeout",
    "structural_better",
    "structural_distance_or_none",
    "tree_size",
]
