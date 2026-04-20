from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_prefix_filtered.parquet"


class DatasetParseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_PATH.exists():
            raise unittest.SkipTest(f"Dataset artifact not found: {DATASET_PATH}")

        cls.sample = pd.read_parquet(
            DATASET_PATH,
            columns=["integrand_prefix", "integral_prefix"],
        ).head(32)

    def test_dataset_columns_exist(self) -> None:
        self.assertEqual(list(self.sample.columns), ["integrand_prefix", "integral_prefix"])
        self.assertGreater(len(self.sample), 0)

    def test_parse_and_roundtrip_sample(self) -> None:
        for column in ("integrand_prefix", "integral_prefix"):
            for row_index, expression in enumerate(self.sample[column].astype(str)):
                expr = parse_prefix_string(expression)
                serialized = serialize_prefix_string(expr)
                reparsed = parse_prefix_string(serialized)
                if row_index < 3:
                    print(f"\n[{column}] example {row_index}")
                    print(f"input: {expression}")
                    print(f"parsed: {expr!r}")
                    print(f"serialized: {serialized}")
                self.assertEqual(expr, reparsed)

    def test_canonicalize_integral_sample(self) -> None:
        for expression in self.sample["integral_prefix"].astype(str):
            expr = parse_prefix_string(expression)
            canonical_expr = canonicalize(expr)
            reparsed = parse_prefix_string(serialize_prefix_string(canonical_expr))
            self.assertEqual(canonical_expr, reparsed)
