from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from chapter3_bser.events.event_detector import EventDetector
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.online.config import load_phase1b2_config
from core.mapping.planning_graph import EndpointConnectorSet, PlanningConnectorView
from tests.bser_online_test_utils import state_at


def _endpoint(point, *, connectors, endpoint_id):
    return EndpointConnectorSet(
        endpoint_id,
        "executor",
        tuple(float(value) for value in point),
        tuple(connectors),
        True,
    )


def _state_with_executor(state, position, target, *, start_connectors=None):
    agents = list(state.agents)
    agents[state.executor_id] = replace(
        agents[state.executor_id],
        position=tuple(position),
        current_navigation_target=tuple(target),
    )
    endpoints = list(state.planning_graph.endpoint_connectors)
    if start_connectors is not None:
        endpoints.append(
            _endpoint(
                position,
                connectors=start_connectors,
                endpoint_id="current_executor",
            )
        )
    endpoints.append(
        _endpoint(
            target,
            connectors=(PlanningConnectorView(4, 0, 0.0, 0.0),),
            endpoint_id="current_public_target",
        )
    )
    return replace(
        state,
        agents=tuple(agents),
        planning_graph=replace(
            state.planning_graph, endpoint_connectors=tuple(endpoints)
        ),
    )


def _config():
    config = load_phase1b2_config()
    config["execution_runtime"] = {
        "defer_stale_endpoint_invalid": True,
        "dynamic_public_target_enabled": False,
    }
    return config


class Phase1CStaleEndpointDeferralTests(unittest.TestCase):
    def test_moved_executor_with_old_endpoint_is_deferred(self) -> None:
        previous = state_at(0)
        allocation = BSEROnlineAllocator().allocate(previous)
        target = (1.2, 1.2, 1.0)
        allocation = replace(
            allocation,
            executor_assignment=replace(
                allocation.executor_assignment,
                target_region=target,
                reachable=True,
            ),
        )
        current = _state_with_executor(
            state_at(1), (2.2, 2.2, 1.0), target, start_connectors=None
        )
        result = EventDetector(_config()).detect(
            previous, current, assignment=allocation
        )
        self.assertTrue(result.executor_validity_deferred)
        self.assertFalse(result.executor_validity_evaluated)
        self.assertFalse(result.executor_start_endpoint_current)
        self.assertTrue(result.executor_goal_endpoint_current)
        self.assertEqual(
            result.executor_invalid_reason, "STALE_ENDPOINT_SNAPSHOT_DEFERRED"
        )
        self.assertNotIn(BSEREvent.EXECUTOR_INVALID, result.events)
        self.assertTrue(result.executor_reachable)
        self.assertIsNone(result.executor_current_planning_cost)


if __name__ == "__main__":
    unittest.main()
