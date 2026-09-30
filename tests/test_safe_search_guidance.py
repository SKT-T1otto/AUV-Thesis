"""Action-free safety guidance, route persistence and ablation isolation."""
from collections import Counter
import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest

from chapter3_bser.experiments.safe_search_v1.public_geometry import PublicGeometry, filter_planning_state
from chapter3_bser.experiments.safe_search_v1.safe_guidance import SafetyBridge
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from chapter3_bser.online.mission_context import OnlineMissionContext
from chapter3_bser.online.types import ExecutorAssignment, OnlineAllocation, SearchAssignment
from core.mapping.planning_state import planning_state_sha256
from tests.test_safe_search_geometry import fixture


class Views:
    def __init__(self, live, clearance=0.0):
        self.latest = live
        self.clearance = clearance
        self.live_calls = 0
        self.search_inputs = []

    def live(self):
        self.live_calls += 1
        return self.latest

    def search(self, state):
        self.search_inputs.append(state)
        return filter_planning_state(state, self.clearance)

    def refresh(self):
        raise AssertionError("V3 safety bridge must not refresh the planning snapshot")


def owner(state, live=None, enabled=True):
    return SimpleNamespace(state=state, planning_views=Views(state if live is None else live),
                           safe_options={"path_safety": enabled}, safe_counts=Counter(), safe_pending=set())


def allocation(state, path=None, goal=(2.5, 2.5, 1.0)):
    assignments = tuple(SearchAssignment(agent.agent_id, f"candidate_{agent.agent_id}",
                                         goal if agent.agent_id == 0 else agent.position,
                                         tuple(path) if agent.agent_id == 0 and path is not None else
                                         (agent.position, goal if agent.agent_id == 0 else agent.position),
                                         1.0)
                        for agent in state.agents[:3])
    executor = state.agents[3]
    return OnlineAllocation(assignments, ExecutorAssignment(3, executor.position, (executor.position,), 0.0,
                                                             "FIXED_STANDBY", True),
                            1.0, 0.5, 1.0, "INITIALIZE")


def context(state):
    return replace(OnlineMissionContext.from_planning_view(state), searcher_finished_flags=(False, True, True))


class SafeSearchGuidanceTests(unittest.TestCase):
    def test_corner_touch_is_replaced_without_changing_semantic_assignment(self):
        state = fixture((4,))
        agent = replace(state.agents[0], position=(0.5, 1.5, 1.0))
        state = replace(state, agents=(agent, *state.agents[1:]))
        assigned = allocation(state, goal=(1.5, 0.5, 1.0))
        runtime = owner(state)
        bridge = SafetyBridge(runtime)
        before = planning_state_sha256(state)
        output = bridge.compile_guidance(assigned, state, context(state))
        item = output.assignment_for(0)
        self.assertFalse(item.hold_state)
        self.assertTrue(item.reachable)
        self.assertNotEqual(item.planned_path, assigned.search_assignments[0].path)
        self.assertTrue(PublicGeometry(state).path_free(item.planned_path))
        self.assertEqual(item.final_waypoint, assigned.search_assignments[0].waypoint)
        self.assertEqual(item.assignment_id, assigned.search_assignments[0].candidate_id)
        self.assertEqual(output.allocation_hash, assigned.allocation_sha256)
        self.assertEqual(bridge.last_safety_status[0], "SAFE_REROUTE")
        self.assertEqual(runtime.safe_counts["safety_route_queries"], 1)
        self.assertEqual(planning_state_sha256(state), before)

    def test_failed_repair_holds_preserves_goal_and_does_not_zero_velocity(self):
        state = fixture((4,))
        agent = replace(state.agents[0], position=(0.6, 0.5, 1.0), velocity=(1.2, 0.4, 0.0))
        state = replace(state, agents=(agent, *state.agents[1:]))
        assigned = allocation(state)
        runtime = owner(state)
        bridge = SafetyBridge(runtime)
        output = bridge.compile_guidance(assigned, state, context(state))
        item = output.assignment_for(0)
        self.assertTrue(item.hold_state)
        self.assertFalse(item.reachable)
        self.assertEqual(item.hold_position, agent.position)
        self.assertEqual(item.tracking_waypoint, agent.position)
        self.assertEqual(item.final_waypoint, assigned.search_assignments[0].waypoint)
        self.assertEqual(item.assignment_id, assigned.search_assignments[0].candidate_id)
        self.assertEqual(state.agents[0].velocity, (1.2, 0.4, 0.0))
        self.assertEqual(bridge.last_safety_status[0], "UNVERIFIED_BRAKING")
        self.assertEqual(runtime.safe_pending, {0})
        self.assertFalse(hasattr(item, "action"))
        self.assertEqual(runtime.safe_counts["safety_route_failure:no_start_connector"], 1)

    def test_stale_snapshot_route_must_pass_latest_geometry_without_refresh(self):
        snapshot = fixture()
        live = fixture((4,))
        assigned = allocation(live)
        runtime = owner(snapshot, live=live)
        bridge = SafetyBridge(runtime)
        item = bridge.compile_guidance(assigned, live, context(live)).assignment_for(0)
        self.assertTrue(item.hold_state)
        self.assertIs(runtime.planning_views.search_inputs[0], snapshot)
        self.assertIs(runtime.state, snapshot)
        self.assertEqual(runtime.safe_counts["safety_route_failure:latest_public_geometry_rejected"], 1)

    def test_at_most_one_query_per_agent_and_step_even_on_repeated_compile(self):
        state = fixture((4,))
        agent = replace(state.agents[0], position=(0.6, 0.5, 1.0))
        state = replace(state, agents=(agent, *state.agents[1:]))
        runtime = owner(state)
        bridge = SafetyBridge(runtime)
        assigned = allocation(state)
        bridge.compile_guidance(assigned, state, context(state))
        bridge.compile_guidance(assigned, state, context(state))
        self.assertEqual(runtime.safe_counts["safety_route_queries"], 1)
        self.assertEqual(runtime.safe_counts["safety_braking_agent_steps"], 1)

    def test_override_keeps_progress_and_does_not_recheck_traversed_prefix(self):
        state = fixture((4,))
        agent = replace(state.agents[0], position=(0.5, 1.5, 1.0))
        state = replace(state, agents=(agent, *state.agents[1:]))
        assigned = allocation(state, goal=(1.5, 0.5, 1.0))
        runtime = owner(state)
        bridge = SafetyBridge(runtime)
        first = bridge.compile_guidance(assigned, state, context(state)).assignment_for(0)
        self.assertEqual(first.tracking_waypoint, (0.5, 0.5, 1.0))
        next_state = fixture((1, 4))
        next_agent = replace(next_state.agents[0], position=first.tracking_waypoint)
        next_state = replace(next_state, step=1, agents=(next_agent, *next_state.agents[1:]))
        runtime.planning_views.latest = next_state
        second = bridge.compile_guidance(assigned, next_state, context(next_state)).assignment_for(0)
        self.assertEqual(second.planned_path, first.planned_path)
        self.assertEqual(second.tracking_waypoint, (1.5, 0.5, 1.0))
        self.assertFalse(second.hold_state)
        self.assertEqual(runtime.safe_counts["safety_route_queries"], 1)
        self.assertEqual(bridge.path_tracker.snapshot(0).next_index, 2)

    def test_safe_route_tracking_matches_original_bridge_and_executor_unchanged(self):
        state = fixture()
        assigned = allocation(state, path=((0.5, 0.5, 1), (0.5, 1.5, 1), (2.5, 2.5, 1)))
        runtime = owner(state)
        actual = SafetyBridge(runtime).compile_guidance(assigned, state, context(state))
        expected = RMADDPGGuidanceBridge().compile_guidance(assigned, state, context(state))
        self.assertEqual(actual, expected)
        self.assertEqual(runtime.safe_counts["safety_route_queries"], 0)

    def test_found_and_disabled_safety_use_original_bridge_without_public_reads(self):
        for found, enabled in ((True, True), (False, False)):
            state = replace(fixture((4,)), target_found=found)
            runtime = owner(state, enabled=enabled)
            assigned = allocation(state)
            mission = context(state)
            actual = SafetyBridge(runtime).compile_guidance(assigned, state, mission)
            expected = RMADDPGGuidanceBridge().compile_guidance(assigned, state, mission)
            self.assertEqual(actual, expected)
            self.assertEqual(runtime.planning_views.live_calls, 0)
            self.assertEqual(runtime.safe_counts, {})

    def test_new_semantic_assignment_releases_old_override(self):
        state = fixture((4,))
        agent = replace(state.agents[0], position=(0.5, 1.5, 1.0))
        state = replace(state, agents=(agent, *state.agents[1:]))
        runtime = owner(state)
        bridge = SafetyBridge(runtime)
        old = allocation(state, goal=(1.5, 0.5, 1))
        bridge.compile_guidance(old, state, context(state))
        changed = allocation(state, goal=(0.5, 2.5, 1))
        changed = replace(changed, search_assignments=(replace(changed.search_assignments[0], candidate_id="NEW"),
                                                       *changed.search_assignments[1:]))
        item = bridge.compile_guidance(changed, state, context(state)).assignment_for(0)
        self.assertEqual(item.assignment_id, "NEW")
        self.assertEqual(item.final_waypoint, (0.5, 2.5, 1))
        self.assertEqual(item.planned_path, changed.search_assignments[0].path)

    def test_new_controller_route_to_same_semantic_goal_releases_old_override(self):
        state = fixture((4,))
        agent = replace(state.agents[0], position=(0.5, 1.5, 1.0))
        state = replace(state, agents=(agent, *state.agents[1:]))
        runtime = owner(state)
        bridge = SafetyBridge(runtime)
        original = allocation(state, goal=(1.5, 0.5, 1))
        initial = bridge.compile_guidance(original, state, context(state)).assignment_for(0)
        self.assertEqual(initial.tracking_waypoint, (0.5, 0.5, 1))
        next_state = replace(state, step=1)
        runtime.planning_views.latest = next_state
        controller_path = ((0.5, 1.5, 1), (0.5, 2.5, 1), (1.5, 2.5, 1),
                           (2.5, 2.5, 1), (2.5, 1.5, 1), (2.5, 0.5, 1), (1.5, 0.5, 1))
        new_assignment = replace(original.search_assignments[0], path=controller_path)
        updated = replace(original, search_assignments=(new_assignment, *original.search_assignments[1:]))
        actual = bridge.compile_guidance(updated, next_state, context(next_state)).assignment_for(0)
        self.assertEqual(actual.planned_path, controller_path)
        self.assertEqual(actual.tracking_waypoint, (0.5, 2.5, 1))
        self.assertEqual(actual.final_waypoint, original.search_assignments[0].waypoint)
        self.assertEqual(actual.assignment_id, original.search_assignments[0].candidate_id)
        self.assertEqual(updated.allocation_sha256, original.allocation_sha256)
        self.assertEqual(runtime.safe_counts["safety_route_queries"], 1)

    def test_found_after_overlay_returns_original_guidance_and_clears_stale_status(self):
        state = fixture((4,))
        agent = replace(state.agents[0], position=(0.5, 1.5, 1.0))
        state = replace(state, agents=(agent, *state.agents[1:]))
        runtime = owner(state)
        bridge = SafetyBridge(runtime)
        assigned = allocation(state, goal=(1.5, 0.5, 1))
        bridge.compile_guidance(assigned, state, context(state))
        expected_bridge = RMADDPGGuidanceBridge(copy.deepcopy(bridge.path_tracker))
        found = replace(state, step=1, target_found=True)
        reads_before = runtime.planning_views.live_calls
        counts_before = dict(runtime.safe_counts)
        expected = expected_bridge.compile_guidance(assigned, found, context(found))
        actual = bridge.compile_guidance(assigned, found, context(found))
        self.assertEqual(actual, expected)
        self.assertEqual(actual.assignment_for(0).planned_path, assigned.search_assignments[0].path)
        self.assertEqual(bridge.last_safety_status, {})
        self.assertEqual(runtime.planning_views.live_calls, reads_before)
        self.assertEqual(runtime.safe_counts, counts_before)

    def test_pending_clears_when_a_safe_route_is_available(self):
        state = fixture()
        runtime = owner(state)
        runtime.safe_pending.add(0)
        bridge = SafetyBridge(runtime)
        bridge.compile_guidance(allocation(state), state, context(state))
        self.assertEqual(runtime.safe_pending, set())


if __name__ == "__main__":
    unittest.main()
