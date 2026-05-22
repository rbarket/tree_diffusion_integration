from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.mathlang.canonicalize import canonicalize
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.beam_search import BeamSearchResult
from src.tree_diffusion.experiments import hybrid_mdlm_repair as hybrid


class _FakeModel:
    def to(self, device):
        del device
        return self

    def eval(self):
        return self


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prediction_row(
    *,
    row_index: int,
    attempt_index: int,
    pred_prefix: str,
    integrand_prefix: str = "pow x INT+ 2",
    target_antiderivative_prefix: str | None = "div pow x INT+ 3 INT+ 3",
) -> dict:
    return {
        "row_index": row_index,
        "attempt_index": attempt_index,
        "integrand_prefix": integrand_prefix,
        "target_antiderivative_prefix": target_antiderivative_prefix,
        "pred_prefix": pred_prefix,
    }


def _beam_result(
    *,
    initial_prefix: str = "x",
    success: bool = False,
    exact: bool = False,
    initial_numeric: float | None = 5.0,
    final_numeric: float | None = 5.0,
    best_numeric: float | None = 5.0,
    stop_reason: str = "beam_empty",
) -> BeamSearchResult:
    return BeamSearchResult(
        target_integrand_prefix="pow x INT+ 2",
        initial_prefix=initial_prefix,
        best_prefix=initial_prefix,
        final_beam_prefixes=[initial_prefix],
        success=success,
        stop_reason=stop_reason,
        steps_taken=1,
        expanded_states=2,
        generated_candidates=0,
        applicable_candidates=0,
        repeated_candidates=0,
        pruned_candidates=0,
        initial_numeric_residual=initial_numeric,
        best_numeric_residual=best_numeric,
        final_best_numeric_residual=final_numeric,
        best_structural_distance=None,
        exact_symbolic_match=exact,
        best_step_index=None,
        path=[],
        per_depth_best_numeric_residual=[initial_numeric, best_numeric],
        per_depth_best_structural_distance=[None, None],
        stop_diagnostics={"best_score": 1.0},
    )


def _patch_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hybrid,
        "_load_cli_model_and_tokenizer",
        lambda **kwargs: (object(), _FakeModel()),
    )


def _prefix(expr) -> str:
    return serialize_prefix_string(canonicalize(expr))


def test_parse_mdlm_seed_attempts_preserves_failed_and_parseable_attempts() -> None:
    attempts = [
        {"attempt_index": 0, "pred_prefix": "pow x"},
        {"attempt_index": 1, "pred_prefix": "div pow x INT+ 3 INT+ 3"},
    ]

    results = hybrid.parse_mdlm_seed_attempts(attempts)

    assert [result.attempt_index for result in results] == [0, 1]
    assert results[0].ok is False
    assert results[0].error is not None
    assert results[1].ok is True
    assert results[1].normalized_prefix == "div pow x INT+ 3 INT+ 3"


def test_first_invalid_second_parseable_attempts_still_run_repair(tmp_path, monkeypatch) -> None:
    _patch_loader(monkeypatch)
    predictions = tmp_path / "predictions.jsonl"
    examples_out = tmp_path / "examples.jsonl"
    _write_jsonl(
        predictions,
        [
            _prediction_row(row_index=7, attempt_index=0, pred_prefix="pow x"),
            _prediction_row(row_index=7, attempt_index=1, pred_prefix="div pow x INT+ 3 INT+ 3"),
        ],
    )
    monkeypatch.setattr(hybrid, "derivative_matches_target", lambda *args, **kwargs: False)
    monkeypatch.setattr(hybrid, "numeric_residual_score", lambda *args, **kwargs: 3.0)
    monkeypatch.setattr(
        hybrid,
        "beam_search_repair_from_seeds",
        lambda *args, **kwargs: _beam_result(success=False),
    )

    hybrid.evaluate_hybrid_mdlm_repair(
        predictions_path=predictions,
        tree_checkpoint="fake.ckpt",
        examples_out_path=examples_out,
        device="cpu",
    )

    row = _read_jsonl(examples_out)[0]
    assert row["mdlm_any_seed_parse_ok"] is True
    assert row["num_parseable_mdlm_seeds"] == 1
    assert row["parseable_attempt_indices"] == [1]
    assert row["repair_attempted"] is True
    assert row["failure_stage"] == "tree_repair_failed"
    assert row["mdlm_parse_errors"][0] is not None


def test_no_parseable_attempts_are_mdlm_seed_failures_not_tree_failures(tmp_path, monkeypatch) -> None:
    _patch_loader(monkeypatch)
    predictions = tmp_path / "predictions.jsonl"
    examples_out = tmp_path / "examples.jsonl"
    _write_jsonl(
        predictions,
        [
            _prediction_row(row_index=1, attempt_index=0, pred_prefix="pow x"),
            _prediction_row(row_index=1, attempt_index=1, pred_prefix="INT+ 3 x"),
            _prediction_row(row_index=1, attempt_index=2, pred_prefix="<mask> x"),
        ],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("beam repair should not run without parseable MDLM seeds")

    monkeypatch.setattr(hybrid, "beam_search_repair_from_seeds", fail_if_called)

    summary = hybrid.evaluate_hybrid_mdlm_repair(
        predictions_path=predictions,
        tree_checkpoint="fake.ckpt",
        examples_out_path=examples_out,
        device="cpu",
    )

    row = _read_jsonl(examples_out)[0]
    assert row["mdlm_no_parseable_seed"] is True
    assert row["repair_attempted"] is False
    assert row["failure_stage"] == "mdlm_no_parseable_seed"
    assert summary.failure_stage_counts == {"mdlm_no_parseable_seed": 1}
    assert summary.tree_repair_failure_rate_over_all == 0.0


def test_all_parseable_seed_selection_passes_every_parseable_seed(tmp_path, monkeypatch) -> None:
    _patch_loader(monkeypatch)
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            _prediction_row(row_index=2, attempt_index=0, pred_prefix="x"),
            _prediction_row(row_index=2, attempt_index=1, pred_prefix="pow x"),
            _prediction_row(row_index=2, attempt_index=2, pred_prefix="pow x INT+ 5"),
        ],
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(hybrid, "derivative_matches_target", lambda *args, **kwargs: False)
    monkeypatch.setattr(hybrid, "numeric_residual_score", lambda *args, **kwargs: 1.0)

    def fake_repair(model, target_integrand, seeds, **kwargs):
        del model, target_integrand
        captured["seeds"] = [_prefix(seed) for seed in seeds]
        captured["residual_executor"] = kwargs.get("residual_executor")
        return _beam_result(success=False)

    monkeypatch.setattr(hybrid, "beam_search_repair_from_seeds", fake_repair)

    hybrid.evaluate_hybrid_mdlm_repair(
        predictions_path=predictions,
        tree_checkpoint="fake.ckpt",
        seed_selection="all_parseable",
        device="cpu",
    )

    assert captured["seeds"] == ["x", "pow x INT+ 5"]
    assert captured["residual_executor"] is not None


def test_first_parseable_seed_selection_passes_only_first_parseable_seed(tmp_path, monkeypatch) -> None:
    _patch_loader(monkeypatch)
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            _prediction_row(row_index=3, attempt_index=0, pred_prefix="x"),
            _prediction_row(row_index=3, attempt_index=1, pred_prefix="pow x INT+ 5"),
        ],
    )
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(hybrid, "derivative_matches_target", lambda *args, **kwargs: False)
    monkeypatch.setattr(hybrid, "numeric_residual_score", lambda *args, **kwargs: 1.0)

    def fake_repair(model, target_integrand, seeds, **kwargs):
        del model, target_integrand, kwargs
        captured["seeds"] = [_prefix(seed) for seed in seeds]
        return _beam_result(success=False)

    monkeypatch.setattr(hybrid, "beam_search_repair_from_seeds", fake_repair)

    hybrid.evaluate_hybrid_mdlm_repair(
        predictions_path=predictions,
        tree_checkpoint="fake.ckpt",
        seed_selection="first_parseable",
        device="cpu",
    )

    assert captured["seeds"] == ["x"]


def test_best_numeric_seed_selection_chooses_lowest_residual_seed(tmp_path, monkeypatch) -> None:
    _patch_loader(monkeypatch)
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            _prediction_row(row_index=4, attempt_index=0, pred_prefix="x"),
            _prediction_row(row_index=4, attempt_index=1, pred_prefix="pow x INT+ 5"),
        ],
    )
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(hybrid, "derivative_matches_target", lambda *args, **kwargs: False)

    def fake_numeric(seed, target_integrand, **kwargs):
        del target_integrand, kwargs
        return {"x": 10.0, "pow x INT+ 5": 2.0}[_prefix(seed)]

    def fake_repair(model, target_integrand, seeds, **kwargs):
        del model, target_integrand, kwargs
        captured["seeds"] = [_prefix(seed) for seed in seeds]
        return _beam_result(success=False)

    monkeypatch.setattr(hybrid, "numeric_residual_score", fake_numeric)
    monkeypatch.setattr(hybrid, "beam_search_repair_from_seeds", fake_repair)

    hybrid.evaluate_hybrid_mdlm_repair(
        predictions_path=predictions,
        tree_checkpoint="fake.ckpt",
        seed_selection="best_numeric_seed",
        device="cpu",
    )

    assert captured["seeds"] == ["pow x INT+ 5"]


def test_initially_correct_parseable_seed_is_hybrid_success_without_gain(tmp_path, monkeypatch) -> None:
    _patch_loader(monkeypatch)
    predictions = tmp_path / "predictions.jsonl"
    examples_out = tmp_path / "examples.jsonl"
    _write_jsonl(
        predictions,
        [
            _prediction_row(
                row_index=5,
                attempt_index=0,
                integrand_prefix="mul INT+ 3 pow x INT+ 2",
                pred_prefix="pow x INT+ 3",
                target_antiderivative_prefix="pow x INT+ 3",
            ),
        ],
    )
    monkeypatch.setattr(
        hybrid,
        "derivative_matches_target",
        lambda seed, target_integrand, **kwargs: _prefix(seed) == "pow x INT+ 3",
    )
    monkeypatch.setattr(hybrid, "numeric_residual_score", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        hybrid,
        "beam_search_repair_from_seeds",
        lambda *args, **kwargs: _beam_result(
            initial_prefix="pow x INT+ 3",
            success=True,
            exact=True,
            initial_numeric=0.0,
            final_numeric=0.0,
            best_numeric=0.0,
            stop_reason="exact_symbolic_match",
        ),
    )

    hybrid.evaluate_hybrid_mdlm_repair(
        predictions_path=predictions,
        tree_checkpoint="fake.ckpt",
        examples_out_path=examples_out,
        device="cpu",
    )

    row = _read_jsonl(examples_out)[0]
    assert row["mdlm_any_seed_exact_symbolic_match"] is True
    assert row["hybrid_success"] is True
    assert row["repair_gain"] is False
    assert row["regression"] is False


def test_incorrect_parseable_seed_repaired_successfully_counts_as_gain(tmp_path, monkeypatch) -> None:
    _patch_loader(monkeypatch)
    predictions = tmp_path / "predictions.jsonl"
    examples_out = tmp_path / "examples.jsonl"
    _write_jsonl(predictions, [_prediction_row(row_index=6, attempt_index=0, pred_prefix="x")])
    monkeypatch.setattr(hybrid, "derivative_matches_target", lambda *args, **kwargs: False)
    monkeypatch.setattr(hybrid, "numeric_residual_score", lambda *args, **kwargs: 9.0)
    monkeypatch.setattr(
        hybrid,
        "beam_search_repair_from_seeds",
        lambda *args, **kwargs: _beam_result(
            success=True,
            exact=True,
            initial_numeric=9.0,
            final_numeric=0.0,
            best_numeric=0.0,
            stop_reason="exact_symbolic_match",
        ),
    )

    hybrid.evaluate_hybrid_mdlm_repair(
        predictions_path=predictions,
        tree_checkpoint="fake.ckpt",
        examples_out_path=examples_out,
        device="cpu",
    )

    row = _read_jsonl(examples_out)[0]
    assert row["tree_repair_success"] is True
    assert row["hybrid_success"] is True
    assert row["repair_gain"] is True
    assert row["failure_stage"] is None


def test_parseable_seed_with_failed_repair_is_tree_repair_failure(tmp_path, monkeypatch) -> None:
    _patch_loader(monkeypatch)
    predictions = tmp_path / "predictions.jsonl"
    examples_out = tmp_path / "examples.jsonl"
    _write_jsonl(predictions, [_prediction_row(row_index=8, attempt_index=0, pred_prefix="x")])
    monkeypatch.setattr(hybrid, "derivative_matches_target", lambda *args, **kwargs: False)
    monkeypatch.setattr(hybrid, "numeric_residual_score", lambda *args, **kwargs: 4.0)
    monkeypatch.setattr(
        hybrid,
        "beam_search_repair_from_seeds",
        lambda *args, **kwargs: _beam_result(success=False, exact=False),
    )

    summary = hybrid.evaluate_hybrid_mdlm_repair(
        predictions_path=predictions,
        tree_checkpoint="fake.ckpt",
        examples_out_path=examples_out,
        device="cpu",
    )

    row = _read_jsonl(examples_out)[0]
    assert row["failure_stage"] == "tree_repair_failed"
    assert row["tree_repair_success"] is False
    assert summary.tree_repair_failure_rate_over_parseable == 1.0


def test_summary_rates_distinguish_no_seed_from_tree_repair_failure(tmp_path, monkeypatch) -> None:
    _patch_loader(monkeypatch)
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            _prediction_row(row_index=10, attempt_index=0, pred_prefix="pow x"),
            _prediction_row(row_index=11, attempt_index=0, pred_prefix="x"),
            _prediction_row(row_index=12, attempt_index=0, pred_prefix="pow x INT+ 3"),
        ],
    )
    monkeypatch.setattr(hybrid, "derivative_matches_target", lambda *args, **kwargs: False)
    monkeypatch.setattr(hybrid, "numeric_residual_score", lambda *args, **kwargs: 5.0)

    def fake_repair(model, target_integrand, seeds, **kwargs):
        del model, target_integrand, kwargs
        seed_prefix = _prefix(seeds[0])
        return _beam_result(
            initial_prefix=seed_prefix,
            success=seed_prefix == "pow x INT+ 3",
            exact=seed_prefix == "pow x INT+ 3",
            final_numeric=0.0 if seed_prefix == "pow x INT+ 3" else 5.0,
            best_numeric=0.0 if seed_prefix == "pow x INT+ 3" else 5.0,
            stop_reason="exact_symbolic_match" if seed_prefix == "pow x INT+ 3" else "beam_empty",
        )

    monkeypatch.setattr(hybrid, "beam_search_repair_from_seeds", fake_repair)

    summary = hybrid.evaluate_hybrid_mdlm_repair(
        predictions_path=predictions,
        tree_checkpoint="fake.ckpt",
        device="cpu",
    )

    assert summary.examples == 3
    assert summary.mdlm_first_attempt_parseable_rate == pytest.approx(2 / 3)
    assert summary.mdlm_any_attempt_parseable_rate == pytest.approx(2 / 3)
    assert summary.mdlm_no_parseable_seed_rate == pytest.approx(1 / 3)
    assert summary.tree_repair_failure_rate_over_parseable == pytest.approx(1 / 2)
    assert summary.hybrid_success_rate_over_all == pytest.approx(1 / 3)
    assert summary.hybrid_success_rate_over_parseable == pytest.approx(1 / 2)
    assert summary.failure_stage_counts == {
        "mdlm_no_parseable_seed": 1,
        "tree_repair_failed": 1,
    }


def test_examples_jsonl_serializes_candidate_details(tmp_path, monkeypatch) -> None:
    _patch_loader(monkeypatch)
    predictions = tmp_path / "predictions.jsonl"
    examples_out = tmp_path / "examples.jsonl"
    _write_jsonl(
        predictions,
        [
            _prediction_row(row_index=13, attempt_index=0, pred_prefix="pow x"),
            _prediction_row(row_index=13, attempt_index=1, pred_prefix="x"),
        ],
    )
    monkeypatch.setattr(hybrid, "derivative_matches_target", lambda *args, **kwargs: False)
    monkeypatch.setattr(hybrid, "numeric_residual_score", lambda *args, **kwargs: 1.5)
    monkeypatch.setattr(
        hybrid,
        "beam_search_repair_from_seeds",
        lambda *args, **kwargs: _beam_result(success=False, final_numeric=1.5, best_numeric=1.5),
    )

    hybrid.evaluate_hybrid_mdlm_repair(
        predictions_path=predictions,
        tree_checkpoint="fake.ckpt",
        examples_out_path=examples_out,
        device="cpu",
    )

    row = _read_jsonl(examples_out)[0]
    for key in (
        "row_index",
        "pair_index",
        "target_integrand_prefix",
        "target_antiderivative_prefix",
        "num_mdlm_attempts",
        "num_parseable_mdlm_seeds",
        "parseable_attempt_indices",
        "first_parseable_attempt_index",
        "mdlm_attempt_prefixes",
        "mdlm_parse_errors",
        "mdlm_any_seed_parse_ok",
        "repair_attempted",
        "tree_repair_success",
        "hybrid_success",
        "failure_stage",
        "initial_best_mdlm_numeric_residual",
        "final_numeric_residual",
        "best_numeric_residual",
        "beam_stop_reason",
        "beam_steps_taken",
        "expanded_states",
    ):
        assert key in row
    assert row["mdlm_attempt_prefixes"] == ["pow x", "x"]
    assert row["mdlm_parse_errors"][0] is not None
    assert row["mdlm_parse_errors"][1] is None


def test_progress_and_sharded_example_parts(tmp_path, monkeypatch, capsys) -> None:
    _patch_loader(monkeypatch)
    predictions = tmp_path / "predictions.jsonl"
    parts_dir = tmp_path / "parts"
    _write_jsonl(
        predictions,
        [
            _prediction_row(row_index=20, attempt_index=0, pred_prefix="x"),
            _prediction_row(row_index=21, attempt_index=0, pred_prefix="x"),
            _prediction_row(row_index=22, attempt_index=0, pred_prefix="x"),
        ],
    )
    monkeypatch.setattr(hybrid, "derivative_matches_target", lambda *args, **kwargs: False)
    monkeypatch.setattr(hybrid, "numeric_residual_score", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        hybrid,
        "beam_search_repair_from_seeds",
        lambda *args, **kwargs: _beam_result(success=False),
    )

    hybrid.evaluate_hybrid_mdlm_repair(
        predictions_path=predictions,
        tree_checkpoint="fake.ckpt",
        examples_parts_dir=parts_dir,
        part_size=2,
        progress_every=1,
        device="cpu",
    )

    captured = capsys.readouterr()
    assert "hybrid_mdlm_repair_progress completed=1/3" in captured.err
    assert "hybrid_mdlm_repair_part_written part=000000 rows=2" in captured.err
    assert "hybrid_mdlm_repair_complete completed=3/3" in captured.err
    assert len(_read_jsonl(parts_dir / "part_000000.jsonl")) == 2
    assert len(_read_jsonl(parts_dir / "part_000001.jsonl")) == 1
    manifest = json.loads((parts_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["part_count"] == 2
    assert manifest["completed_examples"] == 3


def test_resume_continues_after_existing_example_parts(tmp_path, monkeypatch, capsys) -> None:
    _patch_loader(monkeypatch)
    predictions = tmp_path / "predictions.jsonl"
    parts_dir = tmp_path / "parts"
    _write_jsonl(
        predictions,
        [
            _prediction_row(row_index=30, attempt_index=0, pred_prefix="x"),
            _prediction_row(row_index=31, attempt_index=0, pred_prefix="x"),
            _prediction_row(row_index=32, attempt_index=0, pred_prefix="x"),
        ],
    )
    monkeypatch.setattr(hybrid, "derivative_matches_target", lambda *args, **kwargs: False)
    monkeypatch.setattr(hybrid, "numeric_residual_score", lambda *args, **kwargs: 1.0)
    evaluated: list[str] = []

    def fake_repair(model, target_integrand, seeds, **kwargs):
        del model, target_integrand, kwargs
        evaluated.append(_prefix(seeds[0]))
        return _beam_result(success=False)

    monkeypatch.setattr(hybrid, "beam_search_repair_from_seeds", fake_repair)

    first_summary = hybrid.evaluate_hybrid_mdlm_repair(
        predictions_path=predictions,
        tree_checkpoint="fake.ckpt",
        examples_parts_dir=parts_dir,
        limit=1,
        part_size=1,
        progress_every=1,
        device="cpu",
    )
    assert first_summary.examples == 1
    assert len(_read_jsonl(parts_dir / "part_000000.jsonl")) == 1

    evaluated.clear()
    resumed_summary = hybrid.evaluate_hybrid_mdlm_repair(
        predictions_path=predictions,
        tree_checkpoint="fake.ckpt",
        examples_parts_dir=parts_dir,
        part_size=1,
        progress_every=1,
        resume=True,
        device="cpu",
    )

    captured = capsys.readouterr()
    assert "resume=True completed=1 next_part=000001" in captured.err
    assert resumed_summary.examples == 3
    assert len(evaluated) == 2
    assert len(_read_jsonl(parts_dir / "part_000000.jsonl")) == 1
    assert len(_read_jsonl(parts_dir / "part_000001.jsonl")) == 1
    assert len(_read_jsonl(parts_dir / "part_000002.jsonl")) == 1
    assert _read_jsonl(parts_dir / "part_000001.jsonl")[0]["row_index"] == 31
    assert _read_jsonl(parts_dir / "part_000002.jsonl")[0]["row_index"] == 32
    manifest = json.loads((parts_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["part_count"] == 3
    assert manifest["completed_examples"] == 3
