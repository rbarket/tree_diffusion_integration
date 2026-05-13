"""Tree diffusion infrastructure for symbolic integration."""

from src.tree_diffusion.dataset import (
    IntegrationPair,
    TreeDiffusionBatchCollator,
    TreeDiffusionIterableDataset,
    load_integration_pairs_from_parquet,
    make_tree_diffusion_dataloader,
    pairs_from_prefix_rows,
)
from src.tree_diffusion.decoding import (
    DecodedEdit,
    apply_decoded_edit,
    decode_edit_tokens,
    greedy_decode_edit_tokens,
    predict_greedy_edit,
    valid_position_token_ids,
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
from src.tree_diffusion.eval_one_step import (
    OneStepEditEvaluationSummary,
    evaluate_one_step_edits,
    numeric_residual_score,
)
from src.tree_diffusion.label_validation import (
    EditLabelValidationResult,
    apply_subtree_replacement_by_position,
    validate_edit_label_progress,
)
from src.tree_diffusion.mutation import (
    MutationResult,
    collect_candidate_nodes,
    is_obviously_zero,
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
