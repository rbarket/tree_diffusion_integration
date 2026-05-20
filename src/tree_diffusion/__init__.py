"""Stable public tree-diffusion APIs for symbolic integration."""

from src.tree_diffusion.beam_search import (
    BeamSearchResult,
    BeamSearchScoringConfig,
    BeamSearchStopConfig,
    beam_search_repair,
    beam_search_repair_from_seeds,
)
from src.tree_diffusion.dataset import (
    IntegrationPair,
    TreeDiffusionBatchCollator,
    TreeDiffusionIterableDataset,
    load_integration_pairs_from_parquet,
    make_tree_diffusion_dataloader,
    pairs_from_prefix_rows,
)
from src.tree_diffusion.model import (
    TreeDiffusionModelConfig,
    TreeDiffusionModelOutput,
    TreeDiffusionPolicyModel,
    build_tree_diffusion_policy_model,
)
from src.tree_diffusion.precompute_dataset import (
    PrecomputedTreeDiffusionExampleRecord,
    TreeDiffusionPrecomputeConfig,
    load_precompute_config,
    precompute_split,
    precompute_tree_diffusion_dataset,
    precomputed_record_from_training_example,
    split_pairs_for_precompute,
)
from src.tree_diffusion.precomputed_dataset import (
    PrecomputedTreeDiffusionDataset,
    load_precomputed_tokenizer_metadata,
)
from src.tree_diffusion.repair import (
    RepairResult,
    RepairScoringConfig,
    RepairStep,
    derivative_matches_target,
    encode_repair_observation,
    greedy_repair,
    greedy_repair_from_seeds,
    score_repair_candidate,
    tree_size,
)
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer, numeric_bucket_token


__all__ = [
    "BeamSearchResult",
    "BeamSearchScoringConfig",
    "BeamSearchStopConfig",
    "IntegrationPair",
    "PrecomputedTreeDiffusionDataset",
    "PrecomputedTreeDiffusionExampleRecord",
    "RepairResult",
    "RepairScoringConfig",
    "RepairStep",
    "TreeDiffusionBatchCollator",
    "TreeDiffusionIterableDataset",
    "TreeDiffusionModelConfig",
    "TreeDiffusionModelOutput",
    "TreeDiffusionPolicyModel",
    "TreeDiffusionPrecomputeConfig",
    "TreeDiffusionTokenizer",
    "beam_search_repair",
    "beam_search_repair_from_seeds",
    "build_tree_diffusion_policy_model",
    "derivative_matches_target",
    "encode_repair_observation",
    "greedy_repair",
    "greedy_repair_from_seeds",
    "load_integration_pairs_from_parquet",
    "load_precompute_config",
    "load_precomputed_tokenizer_metadata",
    "make_tree_diffusion_dataloader",
    "numeric_bucket_token",
    "pairs_from_prefix_rows",
    "precompute_split",
    "precompute_tree_diffusion_dataset",
    "precomputed_record_from_training_example",
    "score_repair_candidate",
    "split_pairs_for_precompute",
    "tree_size",
]
