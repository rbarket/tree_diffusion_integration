# tree_diffusion_integration

This repository is for tree diffusion on symbolic integration.

The current implementation includes AST infrastructure, mutation, edit labels, observations, tokenization, precompute, Lightning training, decoding, one-step evaluation, greedy repair, beam repair, and resumable repair evaluation. The current data representation is prefix expressions. Symbolic correctness is evaluated by differentiating the predicted antiderivative and checking equivalence to the integrand.

## Canonical Workflows

1. Precompute tree-diffusion examples:
   `python -m tree_diffusion.precompute_dataset --config config/precompute/tree_diffusion_precompute_light.json`
2. Train a tree-diffusion policy with Lightning:
   `python -m training.workflows.tree_diffusion --config config/train/tree_diffusion.json`
3. Run one-step inference evaluation:
   `python -m tree_diffusion.experiments.one_step_inference_eval --help`
4. Run greedy repair evaluation:
   `python -m tree_diffusion.evaluate_repair --help`
5. Run beam repair evaluation:
   `python -m tree_diffusion.evaluate_beam_search --help`
6. Future work: MDLM-seeded hybrid repair remains documented separately in `docs/hybrid_mdlm_tree_repair.md`.

See `docs/code_structure.md` for the current module layout and `docs/refactor_notes.md` for refactor compatibility notes.

## Source Repos

- Symbolic integration source: `/workspace/rbarket/Diffusion_Integration`
- Tree diffusion reference only: `/workspace/rbarket/tree-diffusion`

The tree-diffusion repository is used only as a reference for design ideas.

## Copied From Previous Repo

Copied code files:

- `src/data/vocab.py`
- `src/data/prefix_filters.py`
- `src/data/preprocess_prefix_dataset.py`
- `src/eval/symbolic_eval.py`
- `src/mathlang/conversions.py`
- `src/utils/seeding.py`

Copied data artifacts:

- `data/processed/train_prefix_filtered.parquet`
- `data/processed/vocab.json`

Not present in the source repo at copy time:

- `data/raw/train_data.parquet`
- validation or test parquet files
