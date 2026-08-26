import unittest

import numpy as np

from chapter3_bser.controllers.action_adapter import assignment_to_fixed_actions
from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.diagnostics.event_semantics import (
    Phase1B3ADiagnosticRecorder,
    classify_executor_invalid,
)
from chapter3_bser.online.config import load_phase1b2_config
from chapter3_bser.online.controller import OnlineBSERController
from tests.bser_online_test_utils import mission_context, state_at


class Phase1B3ABehaviorPreservationTest(unittest.TestCase):
    def test_diagnostic_calls_do_not_change_events_decision_allocation_or_actions(self):
        config = load_phase1b2_config()
        initial = state_at(0)
        current = state_at(1)
        baseline_controller = OnlineBSERController(config)
        observed_controller = OnlineBSERController(config)
        baseline_controller.initialize(initial, mission_context(initial))
        observed_controller.initialize(initial, mission_context(initial))
        baseline = baseline_controller.step(current, mission_context(current))
        _ = classify_executor_invalid(("RELATIVE_COST_INCREASE",))
        observed = observed_controller.step(current, mission_context(current))
        self.assertEqual(baseline.events, observed.events)
        self.assertEqual(baseline.decision_reason, observed.decision_reason)
        self.assertEqual(baseline.replanned, observed.replanned)
        self.assertEqual(baseline.allocation, observed.allocation)
        baseline_actions = assignment_to_fixed_actions(current, baseline.allocation)
        observed_actions = assignment_to_fixed_actions(current, observed.allocation)
        np.testing.assert_array_equal(baseline_actions, observed_actions)

    def test_path_tracker_diagnostics_are_idempotent_and_read_only(self):
        tracker = PathTracker(threshold=0.5)
        path = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        tracker.tracking_target(0, (0.0, 0.0, 0.0), path, (3.0, 0.0, 0.0))
        before = tracker.snapshot(0)
        first_error = tracker.cross_track_error(0, (0.4, 0.3, 0.0))
        first_length = tracker.remaining_path_length(0, (0.4, 0.3, 0.0))
        self.assertEqual(before, tracker.snapshot(0))
        self.assertEqual(first_error, tracker.cross_track_error(0, (0.4, 0.3, 0.0)))
        self.assertEqual(first_length, tracker.remaining_path_length(0, (0.4, 0.3, 0.0)))

    def test_recorder_uses_the_committed_public_target_as_lock_baseline(self):
        config = load_phase1b2_config()
        controller = OnlineBSERController(config)
        initial = state_at(0)
        initial_context = mission_context(initial)
        initial_allocation = controller.initialize(initial, initial_context).allocation
        recorder = Phase1B3ADiagnosticRecorder(2729, 2, config)
        recorder.initialize(initial, initial_allocation)

        found = state_at(1, target_found=True)
        found_context = mission_context(
            found, target_found=True, executor_knows_target=False
        )
        before_found = controller.current_allocation
        found_result = controller.step(found, found_context)
        recorder.record_step(
            state=found,
            context=found_context,
            before=before_found,
            result=found_result,
            path_tracker=None,
        )

        received = state_at(2, target_found=True)
        received_context = mission_context(
            received, target_found=True, executor_knows_target=True
        )
        before_received = controller.current_allocation
        received_result = controller.step(received, received_context)
        recorder.record_step(
            state=received,
            context=received_context,
            before=before_received,
            result=received_result,
            path_tracker=None,
        )
        summary = recorder.target_summary()
        self.assertEqual(summary["public_target_received_count"], 1)
        self.assertEqual(summary["public_target_lock_violation_count"], 0)
        self.assertEqual(summary["standby_source_after_public_handoff_count"], 0)


if __name__ == "__main__":
    unittest.main()
