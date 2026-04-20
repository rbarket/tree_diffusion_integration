from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_prefix_filtered.parquet"
SAMPLE_SIZE = 500
PREVIEW_COUNT = 3
PROGRESS_EVERY = 100


class DatasetParseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_PATH.exists():
            raise unittest.SkipTest(f"Dataset artifact not found: {DATASET_PATH}")

        cls.sample = pd.read_parquet(
            DATASET_PATH,
            columns=["integrand_prefix", "integral_prefix"],
        ).head(SAMPLE_SIZE)

    def test_dataset_columns_exist(self) -> None:
        self.assertEqual(list(self.sample.columns), ["integrand_prefix", "integral_prefix"])
        self.assertGreater(len(self.sample), 0)
        self.assertEqual(len(self.sample), SAMPLE_SIZE)

    def test_parse_and_roundtrip_sample(self) -> None:
        for column in ("integrand_prefix", "integral_prefix"):
            print(f"\n[{column}] parse/roundtrip over {len(self.sample)} examples")
            for row_index, expression in enumerate(self.sample[column].astype(str)):
                with self.subTest(column=column, row_index=row_index):
                    try:
                        expr = parse_prefix_string(expression)
                        serialized = serialize_prefix_string(expr)
                        reparsed = parse_prefix_string(serialized)
                    except Exception as exc:
                        self.fail(
                            f"parse/roundtrip failed for {column} row {row_index}: "
                            f"expression={expression!r} error={type(exc).__name__}: {exc}"
                        )

                    if row_index < PREVIEW_COUNT:
                        print(f"\n[{column}] example {row_index}")
                        print(f"input: {expression}")
                        print(f"parsed: {expr!r}")
                        print(f"serialized: {serialized}")
                    elif (row_index + 1) % PROGRESS_EVERY == 0:
                        print(f"[{column}] processed {row_index + 1}/{len(self.sample)} examples")

                    self.assertEqual(
                        expr,
                        reparsed,
                        msg=(
                            f"roundtrip mismatch for {column} row {row_index}: "
                            f"input={expression!r} serialized={serialized!r}"
                        ),
                    )
            print(f"[{column}] parse/roundtrip completed: {len(self.sample)}/{len(self.sample)} passed")

    def test_canonicalize_integral_sample(self) -> None:
        print(f"\n[integral_prefix] canonicalize over {len(self.sample)} examples")
        for row_index, expression in enumerate(self.sample["integral_prefix"].astype(str)):
            with self.subTest(column="integral_prefix", row_index=row_index):
                try:
                    expr = parse_prefix_string(expression)
                    canonical_expr = canonicalize(expr)
                    serialized = serialize_prefix_string(canonical_expr)
                    reparsed = parse_prefix_string(serialized)
                except Exception as exc:
                    self.fail(
                        f"canonicalize failed for integral_prefix row {row_index}: "
                        f"expression={expression!r} error={type(exc).__name__}: {exc}"
                    )

                self.assertEqual(
                    canonical_expr,
                    reparsed,
                    msg=(
                        f"canonicalize roundtrip mismatch for integral_prefix row {row_index}: "
                        f"input={expression!r} serialized={serialized!r}"
                    ),
                )
                if (row_index + 1) % PROGRESS_EVERY == 0:
                    print(f"[integral_prefix] canonicalized {row_index + 1}/{len(self.sample)} examples")
        print(f"[integral_prefix] canonicalize completed: {len(self.sample)}/{len(self.sample)} passed")
