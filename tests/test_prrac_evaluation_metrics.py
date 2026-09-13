from __future__ import annotations

import unittest

from chapter3_bser.experiments.phase1c_prrac.evaluation_metrics import (
    aggregate_checkpoint,
    failure_stage,
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
    def test_strict_outcomes_share_validation_across_all_consumers(self):
        from core.env.task_protocol import STRICT, protocol_identity
        from chapter3_bser.experiments.phase1c_prrac.task_metrics import aggregate_task_outcomes
        from chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints import _failure_funnel, _read_csv, _write_csv
        from chapter3_bser.experiments.phase1c_prrac.search_continuity.aggregation import search_failure_funnel
        from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery.aggregation import search_collision_recovery_failure_funnel
        from pathlib import Path
        import tempfile
        def row(reason):
            return {**_row('case-42', found=True, contact=False, hold=False, success=reason == 'success'),
                **protocol_identity({'task_protocol': STRICT}), 'termination_reason': reason,
                'safe_success': reason == 'success', 'terminated': reason != 'running', 'truncated': False,
                'collision_episode': reason == 'obstacle_collision', 'checkpoint': 'test.pt',
                'evaluation_mode': 'full_prrac', 'first_collision_agent_ids': [0] if reason == 'obstacle_collision' else []}
        for reason, stage in [('success', 'SUCCESS'), ('obstacle_collision', 'OBSTACLE_COLLISION'), ('timeout', 'TIMEOUT'), ('running', 'INCOMPLETE')]:
            for csv_values in (False, True):
                with self.subTest(reason=reason, csv=csv_values):
                    value = row(reason)
                    if csv_values:
                        value = {k: str(v) if type(v) is bool else v for k, v in value.items()}
                    self.assertEqual(failure_stage(value), stage)
                    self.assertEqual(aggregate_task_outcomes([value])['evaluation_complete'], reason != 'running')
                    for funnel in (search_failure_funnel, search_collision_recovery_failure_funnel):
                        counts = {r['category']: r['count'] for r in funnel([value])}
                        self.assertEqual(counts[stage], 1)
                        self.assertEqual(sum(counts.values()), 1)
                    self.assertEqual(_failure_funnel([value])[0][stage.lower()], 1)
        for missing in (None, '', '  '):
            value = row('timeout'); value['termination_reason'] = missing
            self.assertEqual(failure_stage(value), 'INCOMPLETE')
            self.assertFalse(aggregate_task_outcomes([value])['evaluation_complete'])
            for funnel in (search_failure_funnel, search_collision_recovery_failure_funnel):
                self.assertEqual({r['category']: r['count'] for r in funnel([value])}['INCOMPLETE'], 1)
        value.pop('termination_reason')
        self.assertEqual(failure_stage(value), 'INCOMPLETE')
        for field in ('success', 'safe_success', 'found', 'terminated', 'truncated', 'collision_episode'):
            with self.subTest(missing_field=field):
                value = row('success'); value.pop(field)
                self.assertEqual(failure_stage(value), 'INCOMPLETE')
                summary = aggregate_checkpoint([value], {})
                self.assertEqual(summary['n_valid_episodes'], 0)
                self.assertIsNone(summary['success_rate'])
                self.assertIsNone(recommend_checkpoint([summary])['recommended_checkpoint'])
                with self.assertRaisesRegex(ValueError, 'complete'):
                    paired_checkpoint_comparison([value], [row('success')], base_checkpoint='a', candidate_checkpoint='b', evaluation_mode='full_prrac')
        cases = [('success', 'success', False), ('success', 'safe_success', False), ('success', 'found', False),
                 ('obstacle_collision', 'success', True), ('timeout', 'terminated', False),
                 ('timeout', 'collision_episode', True), ('success', 'truncated', True),
                 ('running', 'terminated', True), ('timeout', 'found', 'invalid')]
        for reason, key, value in cases:
            bad = row(reason); bad[key] = value
            for consumer in (failure_stage, lambda r: aggregate_task_outcomes([r]), lambda r: _failure_funnel([r]),
                             lambda r: search_failure_funnel([r]), lambda r: search_collision_recovery_failure_funnel([r])):
                with self.subTest(reason=reason, field=key, consumer=consumer):
                    with self.assertRaisesRegex(ValueError, 'case-42.*termination_reason'):
                        consumer(bad)
        bad = row('timeout'); bad['termination_reason'] = 'alien'
        with self.assertRaisesRegex(ValueError, 'case-42.*alien'):
            failure_stage(bad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'out.csv'
            _write_csv(path, [row('success'), {**row('timeout'), 'scenario_id': 'case-43'}])
            restored = _read_csv(path)
            self.assertEqual(aggregate_task_outcomes(restored)['success_rate'], .5)
            comparison = paired_checkpoint_comparison(restored, restored, base_checkpoint='a', candidate_checkpoint='b', evaluation_mode='full_prrac')
            self.assertEqual(comparison['both_success'], 1)
            self.assertEqual(comparison['neither_success'], 1)
            _write_csv(path, [bad])
            with self.assertRaisesRegex(ValueError, 'alien'):
                _read_csv(path)
        incomplete_summary = {**protocol_identity({'task_protocol': STRICT}), 'evaluation_mode': 'full_prrac', 'evaluation_complete': 'False'}
        self.assertIsNone(recommend_checkpoint([incomplete_summary])['recommended_checkpoint'])
        self.assertFalse(aggregate_task_outcomes([])['evaluation_complete'])
        self.assertEqual(failure_stage(_row('legacy', found=True, contact=False, hold=False, success=True)), 'SUCCESS')

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
