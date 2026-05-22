"""Compatibility wrappers for tree-diffusion experiment runners."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SRC_EXPERIMENTS = "src.tree_diffusion.experiments"

__all__ = [
    "FinalEvalConfig",
    "HybridMdlmRepairSummary",
    "HybridRepairExampleResult",
    "MdlmSeedParseResult",
    "OneStepInferenceEvalMode",
    "PolicyExperimentConfig",
    "compare_policy_experiment_summaries",
    "combine_resumable_beam_repair_eval",
    "combine_resumable_repair_eval",
    "evaluate_hybrid_mdlm_repair",
    "load_policy_experiment_config",
    "parse_mdlm_seed",
    "parse_mdlm_seed_attempts",
    "run_one_step_inference_eval",
    "run_policy_experiment",
    "run_resumable_beam_repair_eval",
    "run_resumable_greedy_repair_eval",
    "summarize_hybrid_mdlm_repair_results",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_SRC_EXPERIMENTS), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
