"""Telemetry route mismatch regression; no simulator, training, or tracker mutation."""
import copy
import importlib.util
import pickle
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.experiments.safe_search_v1.telemetry import capture_runtime
from scripts.search_diagnostic_observer import capture_runtime as historical_capture


def fixture():
    tracker = PathTracker()
    positions, velocities, assignments = [], [], []
    for agent_id in range(4):
        position = (float(agent_id), 0.0, 0.0)
        path = (position, (float(agent_id) + 1, 0.0, 0.0), (float(agent_id) + 2, 1.0, 0.0))
        target = tracker.tracking_target(agent_id, position, path, path[-1])
        positions.append(position)
        velocities.append((0.1, 0.2, 0.0))
        assignments.append(SimpleNamespace(assignment_id=f"candidate_{agent_id}",
            assignment_kind="search" if agent_id < 3 else "executor_standby",
            final_waypoint=path[-1], planned_path=path, tracking_waypoint=target,
            hold_state=False, reachable=True))
    runtime = SimpleNamespace(
        state=SimpleNamespace(agents=[SimpleNamespace(agent_id=i, role="search" if i < 3 else "executor")
                                     for i in range(4)], map_revision=3,
                              occupancy=SimpleNamespace(known_mask=[True, False], occupied_mask=[False, False])),
        env=SimpleNamespace(get_agent_state=lambda: SimpleNamespace(positions=positions, velocities=velocities),
                            get_task_state=lambda: SimpleNamespace(step=7, target_found=False, executor_knows_target=False),
                            get_mapping_state=lambda: SimpleNamespace(map_revision=4)),
        guidance=SimpleNamespace(assignment_for=lambda i: assignments[i], decision_reason="NO_REPLAN_EVENT"),
        bridge=SimpleNamespace(path_tracker=tracker),
        controller=SimpleNamespace(current_allocation=SimpleNamespace(allocation_sha256="allocation"),
                                   replan_count=1, replan_steps=[0]),
        provider=SimpleNamespace(_last_full_refresh_step=0, last_snapshot_was_full_refresh=False))
    return runtime, tracker, assignments


def original_fields(captured):
    captured = copy.deepcopy(captured)
    for row in captured["agents"]:
        row.pop("path_diagnostics")
    return captured


class CaptureTests(unittest.TestCase):
    def test_active_aligned_fields_are_exactly_historical(self):
        runtime, tracker, assignments = fixture()
        observed = capture_runtime(runtime)
        self.assertEqual(original_fields(observed), historical_capture(runtime))
        self.assertTrue(all(row["path_diagnostics"]["geometry_status"] == "active_aligned"
                            for row in observed["agents"]))

    def test_hold_with_old_cursor_keeps_index_and_remaining_length_but_no_geometry(self):
        runtime, tracker, assignments = fixture()
        assignments[0].hold_state = True
        assignments[0].tracking_waypoint = (0.0, 0.0, 0.0)
        before = copy.deepcopy(tracker.__dict__)
        length = tracker.remaining_path_length(0, (0, 0, 0))
        row = capture_runtime(runtime)["agents"][0]
        self.assertEqual(row["path_diagnostics"]["geometry_status"], "inactive_hold")
        self.assertIsNone(row["active_segment_cross_track_error"])
        self.assertIsNone(row["next_turn_angle_deg"])
        self.assertEqual(row["path_index"], 1)
        self.assertEqual(row["remaining_path_length"], length)
        self.assertEqual(tracker.__dict__, before)

    def test_shorter_replaced_guidance_does_not_index_old_route_or_clamp_cursor(self):
        runtime, tracker, assignments = fixture()
        old_path = tuple((float(i), 0.0, 0.0) for i in range(9))
        tracker.tracking_target(0, old_path[0], old_path, old_path[-1])
        for point in old_path[1:8]:
            tracker.tracking_target(0, point, old_path, old_path[-1])
        self.assertEqual(tracker.snapshot(0).next_index, 8)
        assignments[0].planned_path = ((0.0, 0.0, 0.0),)
        with self.assertRaises(IndexError):
            historical_capture(runtime)
        retained = copy.deepcopy(tracker.__dict__)
        row = capture_runtime(runtime)["agents"][0]
        self.assertEqual(row["path_index"], 8)
        self.assertEqual(row["path_point_count"], 1)
        self.assertEqual(row["path_diagnostics"]["tracker_path_point_count"], 9)
        self.assertFalse(row["path_diagnostics"]["tracker_matches_guidance_route"])
        self.assertIsNone(row["active_segment_cross_track_error"])
        self.assertEqual(tracker.__dict__, retained)

    def test_equal_length_different_prefix_is_not_misattributed_by_matching_tail(self):
        runtime, tracker, assignments = fixture()
        path = assignments[0].planned_path
        assignments[0].planned_path = ((99.0, 99.0, 0.0), *path[1:])
        self.assertEqual(tracker.snapshot(0).remaining_path_points, path[1:])
        row = capture_runtime(runtime)["agents"][0]
        self.assertFalse(row["path_diagnostics"]["tracker_matches_guidance_route"])
        self.assertEqual(row["path_diagnostics"]["geometry_status"], "unavailable_guidance_tracker_route_mismatch")
        self.assertIsNone(row["next_turn_angle_deg"])

    def test_empty_path_and_unreachable_assignment_have_no_active_segment(self):
        for mode in ("empty", "unreachable"):
            runtime, tracker, assignments = fixture()
            if mode == "empty":
                assignments[0].planned_path = ()
            else:
                assignments[0].reachable = False
            row = capture_runtime(runtime)["agents"][0]
            self.assertEqual(row["path_diagnostics"]["geometry_status"],
                             "inactive_empty_guidance_path" if mode == "empty" else "inactive_unreachable")
            self.assertIsNone(row["active_segment_cross_track_error"])
            self.assertIsInstance(row["remaining_path_length"], float)

    def test_completed_index_equal_to_path_length_is_legal_but_inactive(self):
        runtime, tracker, assignments = fixture()
        assignment = assignments[0]
        for point in assignment.planned_path[1:]:
            assignment.tracking_waypoint = tracker.tracking_target(
                0, point, assignment.planned_path, assignment.final_waypoint)
        row = capture_runtime(runtime)["agents"][0]
        self.assertEqual(row["path_index"], len(assignment.planned_path))
        self.assertTrue(row["path_diagnostics"]["tracker_matches_guidance_route"])
        self.assertEqual(row["path_diagnostics"]["geometry_status"], "inactive_completed_route")
        self.assertIsNone(row["active_segment_cross_track_error"])

    def test_changed_final_or_installed_target_never_reports_aligned_geometry(self):
        for mode in ("final", "target"):
            runtime, tracker, assignments = fixture()
            setattr(assignments[0], "final_waypoint" if mode == "final" else "tracking_waypoint", (9.0, 9.0, 0.0))
            row = capture_runtime(runtime)["agents"][0]
            self.assertIsNone(row["active_segment_cross_track_error"])
            self.assertNotEqual(row["path_diagnostics"]["geometry_status"], "active_aligned")

    def test_missing_complete_route_identity_is_conservatively_unavailable(self):
        runtime, tracker, assignments = fixture()
        runtime.bridge.path_tracker = SimpleNamespace(snapshot=tracker.snapshot,
                                                       remaining_path_length=tracker.remaining_path_length)
        row = capture_runtime(runtime)["agents"][0]
        self.assertIsNone(row["path_diagnostics"]["tracker_matches_guidance_route"])
        self.assertEqual(row["path_diagnostics"]["geometry_status"], "unavailable_tracker_route_identity")
        self.assertIsNone(row["active_segment_cross_track_error"])

    def test_does_not_call_advance_reset_or_change_python_numpy_rng(self):
        runtime, tracker, assignments = fixture()
        retained = copy.deepcopy(tracker.__dict__)
        random_state = random.getstate()
        numpy_state = pickle.dumps(np.random.get_state())
        with (patch.object(tracker, "tracking_target", side_effect=AssertionError("telemetry advanced tracker")),
              patch.object(tracker, "reset", side_effect=AssertionError("telemetry reset tracker"))):
            capture_runtime(runtime)
            capture_runtime(runtime)
        self.assertEqual(tracker.__dict__, retained)
        self.assertEqual(random.getstate(), random_state)
        self.assertEqual(pickle.dumps(np.random.get_state()), numpy_state)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_torch_rng_and_global_settings_unchanged(self):
        import torch
        runtime, tracker, assignments = fixture()
        before = (torch.get_rng_state().clone(), torch.get_num_threads(), torch.get_default_dtype(), torch.is_grad_enabled())
        capture_runtime(runtime)
        self.assertTrue(torch.equal(torch.get_rng_state(), before[0]))
        self.assertEqual((torch.get_num_threads(), torch.get_default_dtype(), torch.is_grad_enabled()), before[1:])

    def test_optional_live_public_map_preserves_historical_fields(self):
        runtime, tracker, assignments = fixture()
        live = SimpleNamespace(map_revision=8, grid=SimpleNamespace(shape=(2, 1, 1), origin=(0, 0, 0), spacing=(1, 1, 1)),
            occupancy=SimpleNamespace(occupied_mask=[True, False], unknown_mask=[False, True]))
        self.assertEqual(original_fields(capture_runtime(runtime, live)), historical_capture(runtime, live))


if __name__ == "__main__":
    unittest.main()
