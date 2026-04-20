# Canonical prefix-dataset preprocessing helpers.
from __future__ import annotations

import os
import pandas as pd

from src.data.prefix_filters import filter_prefix_pairs
from src.mathlang.conversions import infix_to_prefix_tokens
from src.data.vocab import Vocab


def preprocess_prefix_dataset(
    *,
    in_path: str = "data/raw/train_data.parquet",
    out_path: str = "data/processed/train_prefix_filtered.parquet",
    x_col: str = "integrand",
    y_col: str = "integral",
    vocab_path: str = "data/processed/vocab.json",
    max_len: int = 256,
    limit: int | None = None,
) -> None:
    vocab = Vocab.load(vocab_path)
    vocab_set = set(vocab.token2id.keys())

    df = pd.read_parquet(in_path)
    df = df[[x_col, y_col]].dropna()

    if limit is not None:
        df = df.iloc[: limit].copy()

    # Convert to prefix token strings (space-separated)
    def to_prefix_str(s: str) -> str | None:
        try:
            toks = infix_to_prefix_tokens(str(s))
        except Exception:
            return None
        return " ".join(toks)

    df_out = pd.DataFrame({
        "integrand_prefix": df[x_col].map(to_prefix_str),
        "integral_prefix": df[y_col].map(to_prefix_str),
    }).dropna()

    df_filtered = filter_prefix_pairs(
        df_out,
        x_col="integrand_prefix",
        y_col="integral_prefix",
        vocab_set=vocab_set,
        max_len=max_len,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_filtered.to_parquet(out_path, index=False)

    print(f"Loaded rows: {len(df)}")
    print(f"Parsed rows: {len(df_out)}")
    print(f"Kept rows:   {len(df_filtered)}")
    print(f"Dropped:     {len(df_out) - len(df_filtered)}")
    print(f"Wrote processed+filtered prefix dataset: {out_path}")
    print(df_filtered.head(2))
