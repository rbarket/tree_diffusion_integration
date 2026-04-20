from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.mathlang.canonicalize import canonicalize
from src.mathlang.grammar import OPERATOR_SPECS, SIGNED_INT_TOKENS
from src.mathlang.parser import PrefixParseError, parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit prefix dataset operators and AST parse coverage.")
    parser.add_argument(
        "--parquet-path",
        default="data/processed/train_prefix_filtered.parquet",
        help="Parquet file to audit.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for audit outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for faster debugging.",
    )
    args = parser.parse_args(argv)

    parquet_path = Path(args.parquet_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    prefix_columns = [column for column in df.columns if "prefix" in column]
    inventory = build_operator_inventory(df, prefix_columns)
    write_operator_inventory(output_dir, inventory)

    summary, failures = audit_dataset(df, prefix_columns)
    (output_dir / "parse_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "parse_failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure) + "\n")

    print(f"Dataset columns: {list(df.columns)}")
    print(f"Prefix columns: {prefix_columns}")
    print(f"Proposed grammar operators: {inventory['proposed_grammar']['operators']}")
    print(json.dumps(summary, indent=2))
    return 0


def build_operator_inventory(df: pd.DataFrame, prefix_columns: list[str]) -> dict:
    operator_counts: Counter[str] = Counter()
    named_constant_counts: Counter[str] = Counter()
    signed_int_counts: Counter[str] = Counter()
    digit_counts: Counter[str] = Counter()
    max_token_length: dict[str, int] = {}
    rare_examples: dict[str, str] = {}

    for column in prefix_columns:
        max_token_length[column] = 0
        for expression in df[column].astype(str):
            tokens = expression.split()
            max_token_length[column] = max(max_token_length[column], len(tokens))
            seen_in_expression = set()
            for token in tokens:
                if token in OPERATOR_SPECS:
                    operator_counts[token] += 1
                    if token not in seen_in_expression and token not in rare_examples:
                        rare_examples[token] = expression
                    seen_in_expression.add(token)
                elif token in {"E", "I", "Pi"}:
                    named_constant_counts[token] += 1
                elif token in SIGNED_INT_TOKENS:
                    signed_int_counts[token] += 1
                elif token.isdigit():
                    digit_counts[token] += 1

    rare_operator_examples = {}
    for token, _count in sorted(operator_counts.items(), key=lambda item: (item[1], item[0]))[:10]:
        rare_operator_examples[token] = rare_examples[token]

    return {
        "dataset_columns": list(df.columns),
        "prefix_columns": prefix_columns,
        "row_count": int(len(df)),
        "max_token_length": max_token_length,
        "operator_frequencies": dict(sorted(operator_counts.items())),
        "named_constant_frequencies": dict(sorted(named_constant_counts.items())),
        "signed_integer_marker_frequencies": dict(sorted(signed_int_counts.items())),
        "digit_frequencies": dict(sorted(digit_counts.items())),
        "rare_operator_examples": rare_operator_examples,
        "proposed_grammar": {
            "variables": ["x"],
            "named_constants": sorted(named_constant_counts),
            "operators": sorted(operator_counts),
        },
    }


def write_operator_inventory(output_dir: Path, inventory: dict) -> None:
    (output_dir / "operator_inventory.json").write_text(
        json.dumps(inventory, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"Dataset columns: {inventory['dataset_columns']}",
        f"Prefix columns: {inventory['prefix_columns']}",
        f"Row count: {inventory['row_count']}",
        f"Max token length: {inventory['max_token_length']}",
        f"Operators: {inventory['proposed_grammar']['operators']}",
        f"Named constants: {inventory['proposed_grammar']['named_constants']}",
        "Operator frequencies:",
    ]
    for token, count in inventory["operator_frequencies"].items():
        lines.append(f"  {token}: {count}")
    lines.append("Rare operator examples:")
    for token, example in inventory["rare_operator_examples"].items():
        lines.append(f"  {token}: {example}")

    (output_dir / "operator_inventory.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_dataset(df: pd.DataFrame, prefix_columns: list[str]) -> tuple[dict, list[dict]]:
    summary = {
        "row_count": int(len(df)),
        "dataset_columns": list(df.columns),
        "prefix_columns": prefix_columns,
        "per_column": {},
        "error_type_counts": {},
    }
    failures: list[dict] = []
    error_counts: Counter[str] = Counter()

    for column in prefix_columns:
        column_summary = {
            "total": 0,
            "parse_ok": 0,
            "roundtrip_ok": 0,
            "canonicalize_ok": 0,
            "failures": 0,
        }
        for row_index, expression in enumerate(df[column].astype(str)):
            column_summary["total"] += 1
            try:
                expr = parse_prefix_string(expression)
                column_summary["parse_ok"] += 1

                roundtrip = serialize_prefix_string(expr)
                reparsed = parse_prefix_string(roundtrip)
                if reparsed != expr:
                    raise PrefixParseError("Round-trip reparsed AST does not match original AST.")
                column_summary["roundtrip_ok"] += 1

                if column == "integral_prefix":
                    canonical_expr = canonicalize(expr)
                    reparsed_canonical = parse_prefix_string(serialize_prefix_string(canonical_expr))
                    if reparsed_canonical != canonical_expr:
                        raise PrefixParseError("Canonicalized AST does not survive serialize/reparse.")
                    column_summary["canonicalize_ok"] += 1
            except Exception as exc:
                column_summary["failures"] += 1
                error_type = type(exc).__name__
                error_counts[error_type] += 1
                failures.append(
                    {
                        "row_index": row_index,
                        "column": column,
                        "original_expression": expression,
                        "error_type": error_type,
                        "error_message": str(exc),
                    }
                )
        summary["per_column"][column] = column_summary

    summary["error_type_counts"] = dict(sorted(error_counts.items()))
    summary["failure_count"] = len(failures)
    return summary, failures


if __name__ == "__main__":
    raise SystemExit(main())
