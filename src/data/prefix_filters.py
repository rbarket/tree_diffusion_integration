from __future__ import annotations

import pandas as pd


def tokens_in_vocab(s: str, vocab_set: set[str]) -> bool:
    tokens = str(s).split()
    if not tokens:
        return False
    return all(token in vocab_set for token in tokens)


def tokens_max_len(s: str, max_len: int) -> bool:
    tokens = str(s).split()
    if not tokens:
        return False
    return len(tokens) <= max_len


def filter_prefix_pairs(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    vocab_set: set[str],
    max_len: int,
) -> pd.DataFrame:
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Missing columns. Expected '{x_col}' and '{y_col}'. Found: {list(df.columns)}")

    out = df[[x_col, y_col]].dropna().copy()
    out[x_col] = out[x_col].astype(str)
    out[y_col] = out[y_col].astype(str)

    x_ok = out[x_col].map(lambda value: tokens_in_vocab(value, vocab_set))
    y_ok = out[y_col].map(lambda value: tokens_in_vocab(value, vocab_set))
    x_len_ok = out[x_col].map(lambda value: tokens_max_len(value, max_len))
    y_len_ok = out[y_col].map(lambda value: tokens_max_len(value, max_len))

    return out[x_ok & y_ok & x_len_ok & y_len_ok].reset_index(drop=True)
