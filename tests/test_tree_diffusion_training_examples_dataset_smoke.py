from __future__ import annotations

import random
import unittest
from pathlib import Path

from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string, parse_prefix_tokens
from src.mathlang.serializer import serialize_prefix_tokens
from src.tree_diffusion.training_examples import generate_training_example
from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_prefix_filtered.parquet"
SAMPLE_SIZE = 25
READ_SIZE = 500
MAX_PREFIX_TOKENS = 40


class TreeDiffusionTrainingExamplesDatasetSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATASET_PATH.exists():
            raise unittest.SkipTest(f"Dataset artifact not found: {DATASET_PATH}")
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"Dataset dependency not available: {exc.name}") from exc

        raw_sample = pd.read_parquet(
            DATASET_PATH,
            columns=["integrand_prefix", "integral_prefix"],
        ).head(READ_SIZE)
        lightweight_rows = [
            row
            for row in raw_sample.itertuples(index=False)
            if len(str(row.integrand_prefix).split()) <= MAX_PREFIX_TOKENS
            and len(str(row.integral_prefix).split()) <= MAX_PREFIX_TOKENS
        ][:SAMPLE_SIZE]
        if len(lightweight_rows) < SAMPLE_SIZE:
            raise unittest.SkipTest(
                f"Only found {len(lightweight_rows)} lightweight dataset rows in first {READ_SIZE}."
            )
        cls.sample = lightweight_rows

    def test_generate_encoded_examples_on_dataset_sample(self) -> None:
        tokenizer = TreeDiffusionTokenizer(max_positions=512)
        diagnostics: list[str] = []
        successes = 0

        for row_index, row in enumerate(self.sample):
            integrand_prefix = str(row.integrand_prefix)
            integral_prefix = str(row.integral_prefix)

            try:
                target_integrand = parse_prefix_string(integrand_prefix)
                target_antiderivative = parse_prefix_string(integral_prefix)
                example = generate_training_example(
                    target_integrand,
                    target_antiderivative,
                    tokenizer=tokenizer,
                    rng=random.Random(50_000 + row_index),
                    sigma_small=2,
                    smax=3,
                    rho=0.2,
                    residual_mode="both",
                    encode=True,
                    max_input_length=512,
                    max_target_length=128,
                )

                self.assertIsNotNone(example.input_ids)
                self.assertIsNotNone(example.target_ids)
                self.assertEqual(example.input_tokens[-1], "<EDIT>")
                self.assertTrue(example.target_tokens[0].startswith("<POS_"))
                self.assertEqual(example.target_tokens[-1], "<eos>")
                self.assertIsNotNone(example.edit_target)

                reparsed_current = parse_prefix_tokens(
                    serialize_prefix_tokens(example.current_antiderivative)
                )
                self.assertEqual(
                    canonicalize(example.current_antiderivative),
                    canonicalize(reparsed_current),
                )
                successes += 1
            except Exception as exc:
                diagnostics.append(
                    f"row={row_index} integrand={integrand_prefix!r} "
                    f"integral={integral_prefix!r} error={type(exc).__name__}: {exc}"
                )

        self.assertEqual(
            diagnostics,
            [],
            msg=(
                f"Generated {successes}/{len(self.sample)} examples; failures:\n"
                + "\n".join(diagnostics)
            ),
        )


if __name__ == "__main__":
    unittest.main()
