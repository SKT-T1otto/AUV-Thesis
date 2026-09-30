"""Mechanism regressions for pre-Found repairs; no full scenario evaluations."""
from collections import Counter
import copy
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, Mock

import numpy as np

from chapter3_bser.experiments.safe_search_v2.options import ARMS, SETTINGS
from chapter3_bser.experiments.safe_search_v2.motion import Vehicle, forecast_safe
from chapter3_bser.experiments.safe_search_v2.guidance import FoundSafetyBridge, SearchTracker
from chapter3_bser.experiments.safe_search_v2.recovery import RecoveryController
from chapter3_bser.experiments.safe_search_v2.runtime import make_runtime
from chapter3_bser.experiments.safe_search_v1.safe_guidance import SafetyBridge
from chapter3_bser.experiments.safe_search_v1.public_geometry import PublicGeometry
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from tests.test_safe_search_geometry import fixture
from tests.test_safe_search_guidance import allocation, context, owner
from tests.test_safe_search_runtime_review import config


def constants():
    return NS(use_residual_prior=True, dt=.2, prior_kv_xy=1.1, prior_kv_z=1.,
        prior_slow_radius_xy=2.4, prior_slow_radius_z=1.2, prior_strength_search=1.,
        agent_specs=[dict(v_xy_max=2.8, v_z_max=1.2, a_xy_max=1.3, a_z_max=.75,
                         drag_xy=.1, drag_z=.16, buoyancy_bias=0.) for _ in range(4)])


def found_owner(state, arm="R1"):
    result = owner(state)
    result.found_options = ARMS[arm]
    result.found_settings = SETTINGS
    result.found_counts = Counter()
    result.env = NS(unwrapped=constants())
    return result


class HoldAndBrakingTests(unittest.TestCase):
    def test_latched_hold_keeps_reference_across_drift_and_route_change(self):
        state = fixture((4,))
        state = replace(state, agents=(replace(state.agents[0], position=(.6,.5,1.)), *state.agents[1:]))
        runtime = found_owner(state)
        bridge = FoundSafetyBridge(runtime)
        assigned = allocation(state)
        first = bridge.compile_guidance(assigned, state, context(state))
        self.assertTrue(first.assignment_for(0).hold_state)
        moved = replace(state, step=1, agents=(replace(state.agents[0], position=(.7,.5,1.)), *state.agents[1:]))
        runtime.planning_views.latest = moved
        changed = replace(assigned, search_assignments=(replace(assigned.search_assignments[0], candidate_id="changed"), *assigned.search_assignments[1:]))
        second = bridge.compile_guidance(changed, moved, context(moved))
        self.assertEqual(first.assignment_for(0).hold_position, second.assignment_for(0).hold_position)
        self.assertEqual(bridge.anchors[0], (.6,.5,1.))
        self.assertAlmostEqual(bridge.last_found_safety[0]["anchor_error"], .1)

    def test_safe_route_releases_hold_anchor(self):
        state = fixture()
        runtime = found_owner(state)
        bridge = FoundSafetyBridge(runtime)
        bridge.anchors[0] = (.1,.1,1.)
        result = bridge.compile_guidance(allocation(state), state, context(state))
        self.assertFalse(result.assignment_for(0).hold_state)
        self.assertNotIn(0, bridge.anchors)

    def test_repeated_compile_is_idempotent_and_conflicting_assignment_rejected(self):
        state = fixture()
        runtime = found_owner(state)
        bridge = FoundSafetyBridge(runtime)
        assigned = allocation(state)
        one = bridge.compile_guidance(assigned, state, context(state))
        counts = copy.deepcopy(runtime.found_counts)
        two = bridge.compile_guidance(assigned, state, context(state))
        self.assertIs(one, two)
        self.assertEqual(counts, runtime.found_counts)
        with self.assertRaises(ValueError):
            bridge.compile_guidance(replace(assigned, trigger_reason="different"), state, context(state))

    def test_new_obstacle_invalidates_anchor_explicitly(self):
        state = fixture((4,))
        runtime = found_owner(state)
        bridge = FoundSafetyBridge(runtime)
        bridge.anchors[0] = (1.5,1.5,1.)
        np.testing.assert_array_equal(bridge._anchor(0, np.array([.5,.5,1.]), PublicGeometry(state)), [.5,.5,1.])
        self.assertEqual(runtime.found_counts["hold_anchor_invalidations"], 1)

    def test_search_turn_threshold_does_not_change_executor_or_found_behavior(self):
        tracker = SearchTracker(.25)
        path = ((.5,0.,1.), (2.,0.,1.))
        self.assertEqual(tracker.tracking_target(0, (0.,0.,1.), path, path[-1]), path[0])
        self.assertEqual(tracker.tracking_target(3, (0.,0.,1.), path, path[-1]), path[1])
        self.assertEqual(tracker.threshold, .75)
        tracker.search_enabled = False
        self.assertEqual(tracker.tracking_target(0, (0.,0.,1.), path, path[-1]), path[1])

    def test_found_disables_overlay_and_preserves_executor_guidance(self):
        state = fixture()
        runtime = found_owner(state, "R4")
        bridge = FoundSafetyBridge(runtime)
        assigned = allocation(state)
        actual = bridge.compile_guidance(assigned, state, context(state))
        expected = SafetyBridge(owner(state)).compile_guidance(assigned, state, context(state))
        self.assertEqual(actual.assignment_for(3), expected.assignment_for(3))
        self.assertEqual(actual.executor_assignment, expected.executor_assignment)
        found = replace(state, step=1, target_found=True)
        before = copy.deepcopy(bridge.path_tracker)
        before.search_enabled = False
        expected = RMADDPGGuidanceBridge(before).compile_guidance(assigned, found, context(found))
        with patch.object(runtime.planning_views, "live", side_effect=AssertionError("read after Found")):
            actual = bridge.compile_guidance(assigned, found, context(found))
        self.assertEqual(actual, expected)
        self.assertFalse(bridge.anchors)

    def test_fixed_reference_limits_drift_under_constant_public_disturbance(self):
        vehicle = Vehicle.from_constants(constants(), 0)
        anchor = np.array([10.,10.,1.])
        fixed, chasing = anchor.copy(), anchor.copy()
        vf = np.zeros(3)
        vc = np.zeros(3)
        for _ in range(150):
            fixed, vf = vehicle.step(fixed, vf, anchor, np.array([.18,0,0]))
            chasing, vc = vehicle.step(chasing, vc, chasing.copy(), np.array([.18,0,0]))
        self.assertLess(np.linalg.norm(fixed-anchor), .25)
        self.assertGreater(np.linalg.norm(chasing-anchor), 3.)

    def test_damped_hold_reduces_stopping_distance_without_zeroing_velocity(self):
        vehicle = Vehicle.from_constants(constants(), 0)
        anchor = np.array([10.,10.,1.])
        p, q = anchor.copy(), anchor.copy()
        v, w = np.array([1.5,0,0]), np.array([1.5,0,0])
        distances, original = [], []
        for _ in range(15):
            command = vehicle.hold_target(p, v, anchor, SETTINGS.hold_velocity_feedback)
            p, v = vehicle.step(p, v, command, np.zeros(3))
            q, w = vehicle.step(q, w, anchor, np.zeros(3))
            distances.append(p[0]-10)
            original.append(q[0]-10)
        self.assertGreater(distances[0], 0.)
        self.assertLess(max(distances), max(original))

    def test_velocity_encoding_respects_existing_speed_limits(self):
        vehicle = Vehicle.from_constants(constants(), 0)
        position = np.array([1.,1.,1.])
        for desired in ([.5,-.7,.3], [100.,100.,100.]):
            target = vehicle.target_for_velocity(position, desired)
            np.testing.assert_allclose(vehicle.desired_velocity(position, target), vehicle.clamp_velocity(desired), atol=1e-12)

    def test_prediction_rejects_swept_collision_despite_clear_command_segment(self):
        vehicle = Vehicle.from_constants(constants(), 0)
        geometry = PublicGeometry(fixture((4,)))
        position = np.array([.8,1.5,1.])
        self.assertTrue(geometry.segment_free(position, position))
        self.assertFalse(forecast_safe(vehicle, geometry, position, np.array([2.,0,0]), position, np.zeros(3), SETTINGS))
        self.assertTrue(forecast_safe(vehicle, geometry, np.array([.5,.5,1.]), np.zeros(3), np.array([.5,.5,1.]), np.zeros(3), SETTINGS))

    def test_observed_disturbance_needs_no_flow_or_truth_accessor(self):
        state = fixture()
        bridge = FoundSafetyBridge(found_owner(state))
        vehicle = bridge.vehicles[0]
        p, v, target, wind = np.array([1.,1.,1.]), np.zeros(3), np.array([1.,1.,1.]), np.array([.1,-.1,0])
        _, after = vehicle.step(p, v, target, wind)
        bridge.previous[0] = (0, p, v, target)
        np.testing.assert_allclose(bridge._disturbance(replace(state.agents[0], velocity=tuple(after)), vehicle, 1), wind)

    def test_reference_factory_is_exact_parent(self):
        for baseline, variant in (("B0_search_prior", "V5"), ("B1_bser_prior", "V4")):
            with patch("chapter3_bser.experiments.safe_search_v2.runtime.parent.make_runtime") as original:
                result = make_runtime({}, {}, baseline=baseline, arm="R0", seed=8, episode_id=3)
                self.assertIs(result, original.return_value)
                original.assert_called_once_with({}, {}, baseline=baseline, variant=variant, seed=8, episode_id=3)


class RecoveryTests(unittest.TestCase):
    def build(self):
        state = fixture()
        current = allocation(state)
        runtime = found_owner(state, "R4")
        runtime.safe_options["failure_policy"] = True
        runtime.state = state
        runtime.planning_views.refresh = Mock(side_effect=lambda: runtime.state)
        runtime.planning_views.search = lambda s: s
        inner = NS(config=config(), current_allocation=current, replan_count=0, replan_steps=[],
                   waypoints=NS(updates=Mock(return_value=())), _diagnostics=Mock(return_value=None))
        inner.detector = NS(detect=lambda *a, **kw: NS(stale_searcher_ids=(), events=()))
        inner.cache, inner.current_context = NS(current=state), None
        inner.step = lambda s, c: NS(replanned=False, allocation=inner.current_allocation, events=(),
                                    event_detection=None, diagnostics=None)
        calls = []
        def partial(s, old, *, affected_searcher_ids, executor_affected, trigger_reason):
            i = affected_searcher_ids[0]
            calls.append(i)
            changed = replace(old, search_assignments=tuple(replace(a, candidate_id="new"+str(i)) if a.agent_id == i else a for a in old.search_assignments))
            return changed, True, "PROPOSED"
        inner.allocator = NS(allocate_partial=partial)
        controller = RecoveryController(inner, runtime)
        controller._pending = lambda *args: {0,1,2}
        return controller, runtime, state, calls

    def test_invalid_agent_does_not_block_two_healthy_agents(self):
        controller, runtime, state, calls = self.build()
        before = controller.inner.current_allocation
        with patch("chapter3_bser.experiments.safe_search_v2.recovery.start_status", side_effect=lambda s,i: "invalid_start" if i == 0 else "ready"):
            result = controller.step(state, context(state))
        self.assertEqual(calls, [1,2])
        self.assertTrue(result.replanned)
        self.assertIs(result.allocation.search_assignments[0], before.search_assignments[0])
        self.assertEqual(result.allocation.executor_assignment, before.executor_assignment)
        self.assertEqual(runtime.safe_pending, {0})
        self.assertEqual(controller.inner.replan_count, 1)

    def test_forced_refresh_does_not_bypass_per_agent_retry_clock(self):
        controller, runtime, state, calls = self.build()
        with patch("chapter3_bser.experiments.safe_search_v2.recovery.start_status", return_value="invalid_start"):
            for step in (0,1,2,9,10):
                runtime.state = replace(state, step=step)
                controller.step(runtime.state, context(runtime.state))
        self.assertEqual(runtime.planning_views.refresh.call_count, 2)
        self.assertEqual(calls, [])
        self.assertEqual(runtime.found_counts["recovery_preflight:invalid_start"], 6)

    def test_found_has_no_search_recovery_queries(self):
        controller, runtime, state, calls = self.build()
        found = replace(state, target_found=True)
        controller._pending = Mock(side_effect=AssertionError("search after Found"))
        controller.step(found, context(found))
        self.assertEqual(calls, [])
        runtime.planning_views.refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
