from __future__ import annotations

import unittest

from chapter3_bser.experiments.phase1c_prrac.evaluation_metrics import (
    aggregate_checkpoint,
    mcnemar_exact_p_value,
    paired_checkpoint_comparison,
    recommend_checkpoint,
    router_class_metrics,
    wilson_interval,
)


def _row(scenario: str, *, found: bool, contact: bool, hold: bool, success: bool, collision: bool = False):
    return {
        "scenario_id": scenario,
        "found": found,
        "contact_episode": contact,
        "hold_episode": hold,
        "success": success,
        "collision_episode": collision,
        "post_found_collision_count": int(collision and found),
        "executor_invalid_count": 2 if found else 99,
        "executor_invalid_assignment_unreachable_count": 1 if found else 99,
        "executor_min_distance_to_target": 3.0 if found else 99.0,
        "executor_final_distance_to_target": 4.0 if found else 99.0,
        "executor_replan_count": 2 if found else 99,
        "executor_residual_ratio_post_found": 0.25 if found else None,
        "handoff_delay": 3 if found else None,
        "found_to_success_steps": 7 if success else None,
        "router_confusion_matrix": [[2, 0, 0], [0, 1, 0], [0, 0, 0]],
        "gate_mean": 0.5,
        "gate_p10": 0.2,
        "gate_p90": 0.8,
        "alignment_negative_rate": 0.1,
    }


class PRRACEvaluationMetricsTests(unittest.TestCase):
    def test_funnel_rates_and_conditional_denominators(self) -> None:
        rows = [
            _row("s0", found=False, contact=False, hold=False, success=False),
            _row("s1", found=True, contact=False, hold=False, success=False),
            _row("s2", found=True, contact=True, hold=False, success=False),
            _row("s3", found=True, contact=True, hold=True, success=True),
        ]
        summary = aggregate_checkpoint(rows, {})
        self.assertEqual(summary["found_rate"], 3 / 4)
        self.assertEqual(summary["contact_if_found_rate"], 2 / 3)
        self.assertEqual(summary["hold_if_contact_rate"], 1 / 2)
        self.assertEqual(summary["success_if_found_rate"], 1 / 3)
        self.assertEqual(summary["success_if_contact_rate"], 1 / 2)
        self.assertEqual(summary["mean_executor_invalid_count_if_found"], 2.0)
        self.assertEqual(summary["median_assignment_unreachable_if_found"], 1.0)
        self.assertEqual(summary["mean_found_to_success_steps_if_success"], 7.0)

    def test_wilson_interval(self) -> None:
        low, high = wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.2365930905, places=8)
        self.assertAlmostEqual(high, 0.7634069095, places=8)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_router_precision_recall_and_balanced_accuracy(self) -> None:
        metrics = router_class_metrics([[8, 2, 0], [1, 4, 0], [0, 1, 4]])
        self.assertAlmostEqual(metrics["router_recall_search"], 0.8)
        self.assertAlmostEqual(metrics["router_precision_search"], 8 / 9)
        self.assertAlmostEqual(metrics["router_recall_intercept"], 0.8)
        self.assertAlmostEqual(metrics["router_precision_intercept"], 4 / 7)
        self.assertAlmostEqual(metrics["router_recall_hold"], 0.8)
        self.assertAlmostEqual(metrics["router_balanced_accuracy"], 0.8)

    def test_selection_rule_is_lexicographic_and_earlier_breaks_tie(self) -> None:
        rows = [
            {
                "checkpoint": "late.pt",
                "checkpoint_episode": 20,
                "evaluation_mode": "full_prrac",
                "success_rate": 0.8,
                "success_if_found_rate": 0.9,
                "contact_if_found_rate": 0.9,
                "collision_episode_rate": 0.1,
                "mean_assignment_unreachable_if_found": 1.0,
            },
            {
                "checkpoint": "early.pt",
                "checkpoint_episode": 10,
                "evaluation_mode": "full_prrac",
                "success_rate": 0.8,
                "success_if_found_rate": 0.9,
                "contact_if_found_rate": 0.9,
                "collision_episode_rate": 0.1,
                "mean_assignment_unreachable_if_found": 1.0,
            },
        ]
        result = recommend_checkpoint(rows)
        self.assertEqual(result["recommended_checkpoint"], "early.pt")
        self.assertIsNone(result["performance_passed"])

    def test_paired_comparison_and_exact_mcnemar(self) -> None:
        base = [
            _row("s0", found=True, contact=True, hold=True, success=True),
            _row("s1", found=True, contact=True, hold=False, success=True),
            _row("s2", found=True, contact=False, hold=False, success=False),
            _row("s3", found=False, contact=False, hold=False, success=False),
        ]
        candidate = [
            _row("s0", found=True, contact=True, hold=True, success=True),
            _row("s1", found=True, contact=True, hold=False, success=False),
            _row("s2", found=True, contact=True, hold=True, success=True),
            _row("s3", found=False, contact=False, hold=False, success=False),
        ]
        result = paired_checkpoint_comparison(
            base,
            candidate,
            base_checkpoint="base.pt",
            candidate_checkpoint="candidate.pt",
            evaluation_mode="full_prrac",
        )
        self.assertEqual(result["both_success"], 1)
        self.assertEqual(result["base_only_success"], 1)
        self.assertEqual(result["candidate_only_success"], 1)
        self.assertEqual(result["neither_success"], 1)
        self.assertEqual(result["mcnemar_exact_p_value"], 1.0)
        self.assertEqual(mcnemar_exact_p_value(0, 4), 0.125)


if __name__ == "__main__":
    unittest.main()
