"""Tree diffusion infrastructure for symbolic integration."""

from src.tree_diffusion.dataset import (
    IntegrationPair,
    TreeDiffusionBatchCollator,
    TreeDiffusionIterableDataset,
    load_integration_pairs_from_parquet,
    make_tree_diffusion_dataloader,
    pairs_from_prefix_rows,
)
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
    sample_random_expr,
    sample_valid_subtree,
)
from src.tree_diffusion.model import (
    LearnedPositionalEmbedding,
    TreeDiffusionModelConfig,
    TreeDiffusionModelOutput,
    TreeDiffusionPolicyModel,
    build_tree_diffusion_policy_model,
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
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer, numeric_bucket_token
from src.tree_diffusion.train_step import (
    TrainStepOutput,
    compute_gradient_norm,
    inspect_batch_predictions,
    overfit_fixed_batch,
    tree_diffusion_eval_step,
    tree_diffusion_train_step,
    validate_tree_diffusion_batch,
)
from src.tree_diffusion.training_examples import (
    TreeDiffusionTrainingExample,
    generate_current_candidate,
    generate_training_example,
)
from src.tree_diffusion.validation import (
    OneStepEditDiagnosticSummary,
    run_one_step_edit_diagnostics,
)
