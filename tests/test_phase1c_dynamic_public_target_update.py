from __future__ import annotations

from dataclasses import replace
import unittest

from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.online.config import load_phase1b2_config
from chapter3_bser.online.controller import OnlineBSERController
from chapter3_bser.online.mission_context import OnlineMissionContext
from core.mapping.planning_graph import EndpointConnectorSet, PlanningConnectorView
from tests.bser_online_test_utils import mission_context, shifted_belief, state_at


def _config():
    config = load_phase1b2_config()
    config["execution_runtime"] = {
        "dynamic_public_target_enabled": True,
        "public_target_update_distance": 0.75,
        "public_target_update_min_steps": 20,
        "defer_stale_endpoint_invalid": True,
        "refresh_on_executor_handoff": True,
        "refresh_on_public_target_shift": True,
    }
    return config


def _state_with_public_target(step: int, target):
    state = state_at(step, target_found=True)
    agents = list(state.agents)
    agents[state.executor_id] = replace(
        agents[state.executor_id], current_navigation_target=tuple(target)
    )
    endpoint = EndpointConnectorSet(
        f"public_target_{step}",
        "executor",
        tuple(target),
        (PlanningConnectorView(0, 0, 0.0, 0.0),),
        True,
    )
    return replace(
        state,
        agents=tuple(agents),
        planning_graph=replace(
            state.planning_graph,
            endpoint_connectors=state.planning_graph.endpoint_connectors
            + (endpoint,),
        ),
    )


def _context(state, target, *, knows=True):
    base = mission_context(
        state, target_found=True, executor_knows_target=knows
    )
    return replace(base, executor_navigation_target=tuple(target))


class _RejectUpdatedTargetAllocator(BSEROnlineAllocator):
    def replan_executor_to_public_target(
        self, state, current, public_target, *, trigger_reason="EXECUTOR_PUBLIC_TARGET_REPLAN"
    ):
        if trigger_reason == "EXECUTOR_PUBLIC_TARGET_UPDATED":
            return current, False, "ATOMIC_REJECT_TEST"
        return super().replan_executor_to_public_target(
            state,
            current,
            public_target,
            trigger_reason=trigger_reason,
        )


def _handoff(controller):
    initial = state_at(0)
    controller.initialize(initial, mission_context(initial))
    found = state_at(1, target_found=True)
    controller.step(
        found,
        mission_context(found, target_found=True, executor_knows_target=False),
    )
    target_a = (1.5, 1.5, 1.0)
    received = _state_with_public_target(2, target_a)
    result = controller.step(received, _context(received, target_a))
    return received, target_a, result


class Phase1CDynamicPublicTargetUpdateTests(unittest.TestCase):
    def test_dynamic_update_is_executor_only_and_transactional(self) -> None:
        controller = OnlineBSERController(_config())
        _, target_a, handoff = _handoff(controller)
        self.assertEqual(controller.execution_target, target_a)
        self.assertEqual(controller.execution_target_source, "PUBLIC_HANDOFF")
        frozen_search = handoff.allocation.search_assignments

        target_b = (0.5, 0.5, 1.0)
        shifted = _state_with_public_target(22, target_b)
        updated = controller.step(shifted, _context(shifted, target_b))
        self.assertIn(BSEREvent.EXECUTOR_PUBLIC_TARGET_UPDATED, updated.events)
        self.assertTrue(updated.replanned)
        self.assertEqual(updated.diagnostics.allocation_scope, "executor_public_target")
        self.assertEqual(updated.diagnostics.affected_agent_ids, (3,))
        self.assertEqual(updated.allocation.search_assignments, frozen_search)
        self.assertEqual(controller.execution_target, target_b)
        self.assertEqual(controller.public_target_update_count, 1)
        self.assertEqual(
            updated.decision_reason, "EXECUTOR_PUBLIC_TARGET_UPDATED"
        )

        belief = replace(
            shifted_belief(23),
            target_found=True,
            agents=shifted.agents,
            planning_graph=shifted.planning_graph,
        )
        after_belief = controller.step(belief, _context(belief, target_b))
        self.assertEqual(controller.execution_target, target_b)
        self.assertEqual(
            after_belief.allocation.executor_assignment.target_region, target_b
        )

    def test_rejected_replan_keeps_old_execution_target_and_count(self) -> None:
        controller = OnlineBSERController(
            _config(), allocator=_RejectUpdatedTargetAllocator()
        )
        _, target_a, handoff = _handoff(controller)
        frozen_search = handoff.allocation.search_assignments
        target_b = (0.5, 0.5, 1.0)
        shifted = _state_with_public_target(22, target_b)
        rejected = controller.step(shifted, _context(shifted, target_b))
        self.assertFalse(rejected.replanned)
        self.assertEqual(rejected.decision_reason, "ATOMIC_REJECT_TEST")
        self.assertEqual(controller.execution_target, target_a)
        self.assertEqual(controller.public_target_update_count, 0)
        self.assertEqual(rejected.allocation.search_assignments, frozen_search)


if __name__ == "__main__":
    unittest.main()
