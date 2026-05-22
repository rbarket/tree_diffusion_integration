"""Experiment runners for tree-diffusion policy validation."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "FinalEvalConfig": "src.tree_diffusion.experiments.policy_validation_experiment",
    "HybridMdlmRepairSummary": "src.tree_diffusion.experiments.hybrid_mdlm_repair",
    "HybridRepairExampleResult": "src.tree_diffusion.experiments.hybrid_mdlm_repair",
    "MdlmSeedParseResult": "src.tree_diffusion.experiments.hybrid_mdlm_repair",
    "OneStepInferenceEvalMode": "src.tree_diffusion.experiments.one_step_inference_eval",
    "PolicyExperimentConfig": "src.tree_diffusion.experiments.policy_validation_experiment",
    "compare_policy_experiment_summaries": "src.tree_diffusion.experiments.policy_validation_experiment",
    "combine_resumable_beam_repair_eval": "src.tree_diffusion.experiments.beam_eval_resumable",
    "combine_resumable_repair_eval": "src.tree_diffusion.experiments.repair_eval_resumable",
    "evaluate_hybrid_mdlm_repair": "src.tree_diffusion.experiments.hybrid_mdlm_repair",
    "load_policy_experiment_config": "src.tree_diffusion.experiments.policy_validation_experiment",
    "parse_mdlm_seed": "src.tree_diffusion.experiments.hybrid_mdlm_repair",
    "parse_mdlm_seed_attempts": "src.tree_diffusion.experiments.hybrid_mdlm_repair",
    "run_one_step_inference_eval": "src.tree_diffusion.experiments.one_step_inference_eval",
    "run_policy_experiment": "src.tree_diffusion.experiments.policy_validation_experiment",
    "run_resumable_beam_repair_eval": "src.tree_diffusion.experiments.beam_eval_resumable",
    "run_resumable_greedy_repair_eval": "src.tree_diffusion.experiments.repair_eval_resumable",
    "summarize_hybrid_mdlm_repair_results": "src.tree_diffusion.experiments.hybrid_mdlm_repair",
}

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
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
