"""Small deterministic BSER fixture used only by CPU unit tests."""

from __future__ import annotations

import numpy as np

from chapter3_bser.candidate_generator import generate_candidates
from chapter3_bser.config import load_bser_config
from chapter3_bser.objective import build_objective_context
from core.mapping.planning_graph import EndpointConnectorSet, PlanningConnectorView, PlanningEdgeView, PlanningGraphView
from core.mapping.planning_state import GridGeometryView, OccupancyBeliefView, PlanningAgentView, PlanningStateView, TargetBeliefView


def locked(value, dtype=None):
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def synthetic_state():
    centers = locked([(x, y, 1.0) for x in (0.5, 1.5, 2.5) for y in (0.5, 1.5, 2.5)], np.float64)
    belief = locked([0.23, 0.13, 0.04, 0.10, 0.12, 0.06, 0.14, 0.11, 0.07], np.float64)
    occupied = locked([False] * 9, np.bool_)
    unknown = locked([False, True, True, False, False, True, False, False, True], np.bool_)
    free = locked(~unknown, np.bool_)
    known = locked(~unknown, np.bool_)
    occupancy_probability = locked(np.where(unknown, 0.5, 0.05), np.float64)
    roles = ("search_fast", "search_balanced", "search_precise", "executor")
    positions = ((0.5, 0.5, 1.0), (2.5, 0.5, 1.0), (0.5, 2.5, 1.0), (2.5, 2.5, 1.0))
    speeds = ((2.8, 1.2, 2.0), (2.2, 1.0, 2.35), (1.8, 0.9, 2.75), (1.5, 0.75, None))
    agents = tuple(PlanningAgentView(index, roles[index], positions[index], (0.0, 0.0, 0.0), speeds[index][0], speeds[index][1], speeds[index][2], (1.5, 1.5, 1.0)) for index in range(4))
    shape = (3, 3, 1)
    def adjacency(rate):
        rows = []
        for source, left in enumerate(centers):
            cell = np.unravel_index(source, shape)
            edges = []
            for destination, right in enumerate(centers):
                other = np.unravel_index(destination, shape)
                if source != destination and max(abs(cell[i] - other[i]) for i in range(3)) <= 1:
                    distance = float(np.linalg.norm(right - left))
                    edges.append(PlanningEdgeView(destination, distance * rate, distance * rate))
            rows.append(tuple(edges))
        return tuple(rows)
    endpoints = []
    for agent in agents:
        source = int(np.argmin(np.sum((centers - np.asarray(agent.position)) ** 2, axis=1)))
        target = int(np.argmin(np.sum((centers - np.asarray(agent.current_navigation_target)) ** 2, axis=1)))
        endpoints.append(EndpointConnectorSet(f"agent_{agent.agent_id}", agent.role, agent.position, (PlanningConnectorView(source, 0, 0.0, 0.0),)))
        endpoints.append(EndpointConnectorSet(f"navigation_target_{agent.agent_id}", agent.role, agent.current_navigation_target, (PlanningConnectorView(target, 0, 0.0, 0.0),)))
    graph = PlanningGraphView(
        shape, centers, locked([True] * 9, np.bool_), locked([0] * 9, np.int64),
        adjacency(1.0), adjacency(1.0 / 1.15), tuple(endpoints),
        1.0, 0.6 / 0.55, 1.0 / 1.15, 1.0, 1, "online_unknown", "synthetic_graph",
    )
    return PlanningStateView(
        0, False, False,
        GridGeometryView((3, 3, 1), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), centers),
        TargetBeliefView(belief, float(-np.sum(belief * np.log(belief))), float(np.max(belief)), int(np.argmax(belief)), 1, True),
        OccupancyBeliefView(occupancy_probability, known, free, occupied, unknown, 1, "online_unknown"),
        agents, 3, (0, 1, 2), 1, "online_unknown", graph,
    )


def synthetic_instance(k_search=2, k_standby=3):
    state = synthetic_state()
    config = load_bser_config()
    generated = generate_candidates(state, config, k_search=k_search, k_standby=k_standby)
    context = build_objective_context(state, generated.search_candidates, generated.standby_candidates, config)
    return state, config, generated, context
