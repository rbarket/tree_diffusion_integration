from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Collection

from src.mathlang.ast import Expr
from src.mathlang.canonicalize import canonicalize
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.edit_path import EditTarget, first_edit_toward_target, structural_distance
from src.tree_diffusion.label_validation import validate_edit_label_progress
from src.tree_diffusion.mutation import mutate_once, sample_random_expr
from src.tree_diffusion.observation import Observation, build_observation
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


@dataclass(frozen=True)
class TreeDiffusionTrainingExample:
    target_integrand: Expr
    target_antiderivative: Expr
    current_antiderivative: Expr
    observation: Observation
    edit_target: EditTarget
    input_tokens: list[str]
    target_tokens: list[str]
    input_ids: list[int] | None = None
    target_ids: list[int] | None = None
    num_mutations: int = 0
    used_random_init: bool = False
    attempts: int = 1
    warnings: tuple[str, ...] = ()


def generate_current_candidate(
    target_antiderivative: Expr,
    *,
    rng: random.Random | None = None,
    sigma_small: int = 2,
    smax: int = 5,
    rho: float = 0.2,
    max_random_size: int | None = None,
    max_attempts: int = 32,
    allow_complex_constants: bool = False,
    allow_distributional_unary_ops: bool = False,
    excluded_random_tokens: Collection[str] | None = None,
) -> tuple[Expr, int, bool]:
    _validate_generation_args(
        sigma_small=sigma_small,
        smax=smax,
        rho=rho,
        max_attempts=max_attempts,
        observation_timeout_seconds=None,
    )
    if max_random_size is not None and max_random_size < 0:
        raise ValueError("max_random_size must be non-negative.")

    rng = rng or random.Random()
    target = canonicalize(target_antiderivative)
    random_size = max(sigma_small, 3) if max_random_size is None else max_random_size

    for _ in range(max_attempts):
        use_random_init = _sample_random_init(rng, rho)
        if use_random_init:
            current = sample_random_expr(
                rng=rng,
                max_size=random_size,
                allow_complex_constants=allow_complex_constants,
                allow_distributional_unary_ops=allow_distributional_unary_ops,
                excluded_random_tokens=excluded_random_tokens,
            )
            if canonicalize(current) != target:
                return current, 0, True
            continue

        requested_mutations = rng.randint(1, smax)
        current = target
        successful_mutations = 0
        for _ in range(requested_mutations):
            mutation = mutate_once(
                current,
                sigma_small=sigma_small,
                rng=rng,
                allow_complex_constants=allow_complex_constants,
                allow_distributional_unary_ops=allow_distributional_unary_ops,
                excluded_random_tokens=excluded_random_tokens,
            )
            if mutation is None:
                break
            current = mutation.mutated_expr
            successful_mutations += 1

        current = canonicalize(current)
        if successful_mutations > 0 and current != target:
            return current, successful_mutations, False

    raise RuntimeError(
        "Failed to generate a current candidate different from target "
        f"after {max_attempts} attempts: target={serialize_prefix_string(target)!r}, "
        f"sigma_small={sigma_small}, smax={smax}, rho={rho}, "
        f"max_random_size={random_size}."
    )


def generate_training_example(
    target_integrand: Expr,
    target_antiderivative: Expr,
    *,
    tokenizer: TreeDiffusionTokenizer | None = None,
    rng: random.Random | None = None,
    sigma_small: int = 2,
    smax: int = 5,
    rho: float = 0.2,
    residual_mode: str = "both",
    encode: bool = False,
    max_input_length: int | None = None,
    max_target_length: int | None = None,
    max_random_size: int | None = None,
    max_attempts: int = 32,
    observation_timeout_seconds: float | None = None,
    simplify_symbolic_residual: bool = True,
    allow_complex_constants: bool = False,
    allow_distributional_unary_ops: bool = False,
    excluded_random_tokens: Collection[str] | None = None,
    validate_label: bool = False,
    max_derivative_tokens: int | None = None,
    max_residual_tokens: int | None = None,
) -> TreeDiffusionTrainingExample:
    _validate_generation_args(
        sigma_small=sigma_small,
        smax=smax,
        rho=rho,
        max_attempts=max_attempts,
        observation_timeout_seconds=observation_timeout_seconds,
        max_derivative_tokens=max_derivative_tokens,
        max_residual_tokens=max_residual_tokens,
    )

    rng = rng or random.Random()
    tokenizer = tokenizer or TreeDiffusionTokenizer()
    canonical_target_integrand = canonicalize(target_integrand, strip_additive_constants=False)
    canonical_target_antiderivative = canonicalize(target_antiderivative)
    last_failure: str | None = None

    for attempt in range(1, max_attempts + 1):
        current_antiderivative, num_mutations, used_random_init = generate_current_candidate(
            canonical_target_antiderivative,
            rng=rng,
            sigma_small=sigma_small,
            smax=smax,
            rho=rho,
            max_random_size=max_random_size,
            max_attempts=max_attempts,
            allow_complex_constants=allow_complex_constants,
            allow_distributional_unary_ops=allow_distributional_unary_ops,
            excluded_random_tokens=excluded_random_tokens,
        )

        edit_target = first_edit_toward_target(
            current_antiderivative,
            canonical_target_antiderivative,
            sigma_small=sigma_small,
            rng=rng,
        )
        if edit_target is None:
            last_failure = "first_edit_toward_target returned None"
            continue

        before_distance = structural_distance(current_antiderivative, canonical_target_antiderivative)
        after_distance = structural_distance(
            edit_target.resulting_tree,
            canonical_target_antiderivative,
        )
        if after_distance > before_distance:
            last_failure = (
                "edit_target increased structural distance "
                f"from {before_distance} to {after_distance}"
            )
            continue

        if validate_label:
            validation = validate_edit_label_progress(
                current_antiderivative,
                canonical_target_antiderivative,
                edit_target,
            )
            if not validation.ok:
                last_failure = f"edit_label_validation_failed:{validation.error or 'unknown'}"
                continue

        observation = build_observation(
            canonical_target_integrand,
            current_antiderivative,
            residual_mode=residual_mode,
            simplify_symbolic_residual=simplify_symbolic_residual,
            max_derivative_tokens=max_derivative_tokens,
            max_residual_tokens=max_residual_tokens,
            observation_timeout_seconds=observation_timeout_seconds,
        )
        input_tokens, target_tokens = tokenizer.serialize_training_pair(observation, edit_target)

        input_ids: list[int] | None = None
        target_ids: list[int] | None = None
        if encode:
            try:
                input_ids = tokenizer.encode_tokens(input_tokens, pad_to_length=max_input_length)
            except ValueError as exc:
                raise ValueError(
                    f"Failed to encode input tokens of length {len(input_tokens)} "
                    f"with max_input_length={max_input_length}."
                ) from exc
            try:
                target_ids = tokenizer.encode_tokens(target_tokens, pad_to_length=max_target_length)
            except ValueError as exc:
                raise ValueError(
                    f"Failed to encode target tokens of length {len(target_tokens)} "
                    f"with max_target_length={max_target_length}."
                ) from exc

        return TreeDiffusionTrainingExample(
            target_integrand=canonical_target_integrand,
            target_antiderivative=canonical_target_antiderivative,
            current_antiderivative=current_antiderivative,
            observation=observation,
            edit_target=edit_target,
            input_tokens=input_tokens,
            target_tokens=target_tokens,
            input_ids=input_ids,
            target_ids=target_ids,
            num_mutations=num_mutations,
            used_random_init=used_random_init,
            attempts=attempt,
            warnings=observation.warnings,
        )

    diagnostic = last_failure or "no successful candidate/edit pair was produced"
    raise RuntimeError(
        "Failed to generate a supervised tree-diffusion training example "
        f"after {max_attempts} attempts: target_integrand="
        f"{serialize_prefix_string(canonical_target_integrand)!r}, "
        f"target_antiderivative={serialize_prefix_string(canonical_target_antiderivative)!r}, "
        f"sigma_small={sigma_small}, smax={smax}, rho={rho}, "
        f"residual_mode={residual_mode!r}; last_failure={diagnostic}."
    )


def _validate_generation_args(
    *,
    sigma_small: int,
    smax: int,
    rho: float,
    max_attempts: int,
    observation_timeout_seconds: float | None = None,
    max_derivative_tokens: int | None = None,
    max_residual_tokens: int | None = None,
) -> None:
    if sigma_small < 1:
        raise ValueError("sigma_small must be >= 1.")
    if smax < 1:
        raise ValueError("smax must be >= 1.")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must satisfy 0.0 <= rho <= 1.0.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")
    if observation_timeout_seconds is not None and observation_timeout_seconds <= 0.0:
        raise ValueError("observation_timeout_seconds must be > 0 when provided.")
    if max_derivative_tokens is not None and max_derivative_tokens < 1:
        raise ValueError("max_derivative_tokens must be >= 1 when provided.")
    if max_residual_tokens is not None and max_residual_tokens < 1:
        raise ValueError("max_residual_tokens must be >= 1 when provided.")


def _sample_random_init(rng: random.Random, rho: float) -> bool:
    if rho <= 0.0:
        return False
    if rho >= 1.0:
        return True
    return rng.random() < rho


__all__ = [
    "TreeDiffusionTrainingExample",
    "generate_current_candidate",
    "generate_training_example",
]
