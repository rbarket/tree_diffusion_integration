from __future__ import annotations

import unittest

from src.tree_diffusion.evaluate_repair import (
    RepairEvaluationRecord,
    repair_evaluation_summary_to_json,
    summarize_repair_results,
)
from src.tree_diffusion.repair import RepairResult, RepairStep


class TreeDiffusionRepairMetricsTests(unittest.TestCase):
    def test_group_summaries_by_used_random_init_and_num_mutations(self) -> None:
        records = [
            RepairEvaluationRecord(
                result=_result(initial=10.0, final=5.0, best=5.0, stop_reason="max_steps"),
                used_random_init=False,
                num_mutations=0,
            ),
            RepairEvaluationRecord(
                result=_result(initial=10.0, final=10.0, best=10.0, stop_reason="max_steps"),
                used_random_init=True,
                num_mutations=1,
            ),
            RepairEvaluationRecord(
                result=_result(initial=10.0, final=2.0, best=2.0, stop_reason="numeric_tol", success=True),
                used_random_init=False,
                num_mutations=5,
            ),
            RepairEvaluationRecord(
                result=_result(initial=10.0, final=9.0, best=9.0, stop_reason="max_steps"),
                used_random_init=None,
                num_mutations=None,
            ),
        ]

        summary = summarize_repair_results(
            records,
            numeric_tol=2.0,
            max_steps=2,
            candidate_k=8,
            selection_strategy="residual_scored",
        )

        self.assertEqual(summary.by_used_random_init["local_corruption"].examples, 2)
        self.assertEqual(summary.by_used_random_init["random_init"].examples, 1)
        self.assertEqual(summary.by_used_random_init["unknown"].examples, 1)
        self.assertEqual(summary.by_num_mutations["s=0"].examples, 1)
        self.assertEqual(summary.by_num_mutations["s=1"].examples, 1)
        self.assertEqual(summary.by_num_mutations["s=5"].examples, 1)
        self.assertEqual(summary.by_num_mutations["unknown"].examples, 1)
        self.assertEqual(
            sum(group.examples for group in summary.by_used_random_init.values()),
            summary.examples,
        )
        self.assertEqual(
            sum(group.examples for group in summary.by_num_mutations.values()),
            summary.examples,
        )

    def test_best_so_far_residual_metrics_track_best_even_if_final_worsens(self) -> None:
        result = _result(
            initial=10.0,
            final=8.0,
            best=5.0,
            best_step_index=1,
            steps=[
                _step(index=0, before=10.0, after=5.0, best=5.0, rank=1),
                _step(index=1, before=5.0, after=8.0, best=5.0, rank=1),
            ],
        )

        summary = summarize_repair_results(
            [RepairEvaluationRecord(result=result)],
            numeric_tol=1e-10,
            max_steps=2,
            candidate_k=8,
            selection_strategy="residual_scored",
        )

        self.assertEqual(result.final_numeric_residual, 8.0)
        self.assertEqual(result.best_numeric_residual, 5.0)
        self.assertEqual(result.best_step_index, 1)
        self.assertEqual(summary.mean_final_numeric_residual, 8.0)
        self.assertEqual(summary.mean_best_numeric_residual, 5.0)
        self.assertEqual(summary.best_numeric_residual_improvement_rate, 1.0)

    def test_per_step_residual_curves(self) -> None:
        first = _result(
            initial=10.0,
            final=8.0,
            best=5.0,
            steps=[
                _step(index=0, before=10.0, after=5.0, best=5.0, rank=1),
                _step(index=1, before=5.0, after=8.0, best=5.0, rank=1),
            ],
        )
        second = _result(
            initial=4.0,
            final=2.0,
            best=2.0,
            success=True,
            stop_reason="exact_symbolic_match",
            exact=True,
            steps=[
                _step(index=0, before=4.0, after=2.0, best=2.0, rank=1, exact=True),
            ],
        )

        summary = summarize_repair_results(
            [
                RepairEvaluationRecord(result=first),
                RepairEvaluationRecord(result=second),
            ],
            numeric_tol=1e-10,
            max_steps=2,
            candidate_k=8,
            selection_strategy="residual_scored",
        )

        self.assertEqual(summary.per_step_active_examples["step_0"], 2)
        self.assertEqual(summary.per_step_active_examples["step_1"], 2)
        self.assertEqual(summary.per_step_active_examples["step_2"], 1)
        self.assertEqual(summary.per_step_numeric_residual_mean["step_0"], 7.0)
        self.assertEqual(summary.per_step_numeric_residual_median["step_1"], 3.5)
        self.assertEqual(summary.per_step_numeric_residual_mean["step_2"], 8.0)
        self.assertEqual(summary.per_step_exact_match_rate["step_0"], 0.0)
        self.assertEqual(summary.per_step_exact_match_rate["step_1"], 0.5)
        self.assertEqual(summary.per_step_exact_match_rate["step_2"], 0.5)

    def test_candidate_rank_metrics(self) -> None:
        result = _result(
            initial=10.0,
            final=4.0,
            best=4.0,
            steps=[
                _step(index=0, before=10.0, after=8.0, best=8.0, rank=1),
                _step(index=1, before=8.0, after=6.0, best=6.0, rank=1),
                _step(index=2, before=6.0, after=4.0, best=4.0, rank=3),
            ],
        )

        summary = summarize_repair_results(
            [RepairEvaluationRecord(result=result)],
            numeric_tol=1e-10,
            max_steps=3,
            candidate_k=8,
            selection_strategy="residual_scored",
        )

        self.assertAlmostEqual(summary.mean_chosen_candidate_rank, 5.0 / 3.0)
        self.assertAlmostEqual(summary.rank1_chosen_rate, 2.0 / 3.0)

    def test_json_output_contains_core_sections(self) -> None:
        summary = summarize_repair_results(
            [
                RepairEvaluationRecord(
                    result=_result(initial=10.0, final=5.0, best=5.0),
                    used_random_init=False,
                    num_mutations=1,
                )
            ],
            numeric_tol=1e-10,
            max_steps=1,
            candidate_k=8,
            selection_strategy="rank1",
        )

        payload = repair_evaluation_summary_to_json(summary)

        for key in (
            "overall",
            "by_used_random_init",
            "by_num_mutations",
            "best_so_far",
            "per_step",
            "candidate_rank",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["selection_strategy"], "rank1")
        self.assertEqual(payload["candidate_k"], 8)


def _result(
    *,
    initial: float,
    final: float,
    best: float,
    stop_reason: str = "max_steps",
    success: bool = False,
    exact: bool = False,
    steps: list[RepairStep] | None = None,
    best_step_index: int | None = None,
) -> RepairResult:
    if steps is None:
        steps = []
    if best_step_index is None:
        best_step_index = 0 if best == initial else None
    return RepairResult(
        target_integrand_prefix="x",
        initial_prefix="initial",
        final_prefix="final",
        success=success,
        stop_reason=stop_reason,
        steps_taken=sum(1 for step in steps if step.chosen_prefix is not None),
        initial_numeric_residual=initial,
        final_numeric_residual=final,
        best_numeric_residual=best,
        best_prefix="best",
        best_step_index=best_step_index,
        exact_symbolic_match=exact,
        repeated_state=stop_reason == "repeated_state",
        no_candidate=stop_reason == "no_applicable_candidate",
        steps=steps,
    )


def _step(
    *,
    index: int,
    before: float,
    after: float,
    best: float,
    rank: int,
    exact: bool = False,
) -> RepairStep:
    return RepairStep(
        step_index=index,
        current_prefix=f"step_{index}",
        chosen_prefix=f"step_{index + 1}",
        decoded_status="ok",
        selected_node_id=0,
        replacement_tokens=["x"],
        replacement_subtree_prefix="x",
        candidate_rank=rank,
        policy_logprob=-float(rank),
        numeric_residual_before=before,
        numeric_residual_after=after,
        best_numeric_residual_so_far=best,
        score=after,
        exact_symbolic_match=exact,
    )


if __name__ == "__main__":
    unittest.main()
