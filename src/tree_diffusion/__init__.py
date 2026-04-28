"""Tree diffusion infrastructure for symbolic integration."""

from src.tree_diffusion.edit_path import (
    EditTarget,
    FirstMismatch,
    compute_edit_path,
    find_first_mismatch,
    first_edit_toward_target,
    is_small_enough,
    structural_distance,
    subtree_size,
    trees_equal,
)
from src.tree_diffusion.mutation import (
    MutationResult,
    collect_candidate_nodes,
    local_replace_once,
    mutate_once,
    sample_valid_subtree,
)
from src.tree_diffusion.observation import (
    DEFAULT_PROBE_POINTS,
    NumericProbeFeatures,
    Observation,
    build_observation,
    compute_current_derivative,
    compute_numeric_probes,
    compute_symbolic_residual,
)
from src.tree_diffusion.positions import NodePosition, PositionIndex, index_tree_positions
