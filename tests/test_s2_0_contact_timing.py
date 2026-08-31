from __future__ import annotations

import unittest

from chapter3_bser.experiments.phase1c_prrac.evaluation_metrics import EvaluationTransitionDiagnostics


class ContactTimingTests(unittest.TestCase):
    def test_transition_times_are_not_cumulative_counts(self):
        d=EvaluationTransitionDiagnostics()
        d.observe(step=4,stage_before=0,stage_after=1,contact_step_count=0,full_hold_step_count=0,mission_complete=False)
        d.observe(step=9,stage_before=1,stage_after=1,contact_step_count=1,full_hold_step_count=0,mission_complete=False)
        d.observe(step=11,stage_before=1,stage_after=2,contact_step_count=3,full_hold_step_count=1,mission_complete=False)
        d.observe(step=14,stage_before=2,stage_after=2,contact_step_count=6,full_hold_step_count=4,mission_complete=True)
        row=d.summary(); self.assertEqual(row["first_contact_step"],9); self.assertEqual(row["first_full_hold_step"],11); self.assertEqual(row["success_step"],14)
        self.assertEqual(row["found_to_first_contact_steps"],5); self.assertEqual(row["first_contact_to_success_steps"],5)

    def test_missing_events_are_null(self):
        self.assertIsNone(EvaluationTransitionDiagnostics().summary()["first_contact_step"])


if __name__ == "__main__": unittest.main()
