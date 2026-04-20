# tree_diffusion_integration

This repository is for tree diffusion on symbolic integration.

Phase 1 focuses on dataset preparation and symbolic evaluation compatibility. The current data representation is prefix expressions. Symbolic correctness is evaluated by differentiating the predicted antiderivative and checking equivalence to the integrand. AST infrastructure and tree-diffusion training will be added later.

## Source Repos

- Symbolic integration source: `/workspace/rbarket/Diffusion_Integration`
- Tree diffusion reference only: `/workspace/rbarket/tree-diffusion`

The tree-diffusion repository is being used only as a reference. Later stages may borrow design ideas around grammar or mutation, tokenizer design, constrained decoding, and search, but none of that is implemented here yet.

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
