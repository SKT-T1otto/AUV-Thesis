"""Bounded same-goal candidates on the existing immutable public graph."""
from collections import Counter
from dataclasses import dataclass
import hashlib

import numpy as np

from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.experiments.safe_search_v1.public_geometry import PublicGeometry
from chapter3_bser.experiments.safe_search_v2.motion import Vehicle
from core.mapping.travel_cost_service import TravelCostService


@dataclass(frozen=True, eq=False)
class Route:
    route_id: str
    source: str
    points: np.ndarray
    cells: np.ndarray
    planning_cost: float
    physical_time: float
    length: float


def route_pool(state, agent_id, goal, settings):
    """Keep the original query; alternative midpoints are existing grid nodes.

    This is a bounded via-point heuristic, not a complete K-shortest solver.
    Every route uses identical existing costs and public safety predicates.
    """
    agent = state.agents[agent_id]
    service = TravelCostService(state)
    geometry = PublicGeometry(state, settings['planner_clearance'])
    original = service.query(agent.position, goal, agent)
    counts = Counter(queries=1)
    if not original.reachable:
        return (), dict(counts, failure=original.failure_reason)
    if original.path_length < 1e-9:
        return (), dict(counts, failure='already_at_goal')
    routes, seen = [], set()

    def accept(points, cells, cost, physical, source):
        points = np.asarray(points, dtype=float)
        if (not np.allclose(points[0], agent.position, atol=1e-10, rtol=0)
                or not np.allclose(points[-1], goal, atol=1e-10, rtol=0)):
            raise ValueError('route endpoint mismatch')
        signature = tuple(tuple(p) for p in np.round(points, 10))
        if signature in seen:
            counts['duplicate'] += 1
            return
        seen.add(signature)
        if len(set(signature)) != len(signature):
            counts['loop'] += 1
            return
        if cost > original.planning_cost * settings['max_cost_ratio'] + 1e-9:
            counts['cost_budget'] += 1
            return
        if not geometry.path_free(points):
            counts['public_geometry_reject'] += 1
            return
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        rid = hashlib.sha256(repr(signature).encode()).hexdigest()[:12]
        locked = points.copy(); locked.setflags(write=False)
        routes.append(Route(rid, source, locked, np.asarray(cells), float(cost), float(physical), length))

    accept(original.path_points, original.path_cell_indices, original.planning_cost,
           original.physical_travel_time, 'unchanged_astar_query')
    start, goal = np.asarray(agent.position), np.asarray(goal)
    delta = goal - start
    xy = delta[:2]
    perpendicular = np.array([-xy[1], xy[0], 0.]) / max(float(np.linalg.norm(xy)), 1e-9)
    if np.linalg.norm(perpendicular) < .5:
        perpendicular = np.array([1., 0., 0.])
    spacing = np.asarray(state.grid.spacing)
    width = float(max(spacing[:2]))
    proposals = []
    for fraction in (.33, .67):
        for offset in (perpendicular * width, -perpendicular * width,
                       perpendicular * 2 * width, -perpendicular * 2 * width,
                       np.array([0., 0., spacing[2]]), np.array([0., 0., -spacing[2]])):
            proposals.append(start + fraction * delta + offset)
    centers = np.asarray(state.grid.cell_centers)
    valid = np.flatnonzero(state.planning_graph.valid_mask)
    via_seen = set()
    for point in proposals[:settings['via_limit']]:
        if not len(valid):
            break
        via = int(valid[np.argmin(np.sum((centers[valid] - point) ** 2, axis=1))])
        if via in via_seen:
            continue
        via_seen.add(via)
        a = service.query(start, centers[via], agent)
        b = service.query(centers[via], goal, agent)
        counts['queries'] += 2
        if not a.reachable or not b.reachable:
            counts['unreachable_via'] += 1
            continue
        accept(np.concatenate((a.path_points, b.path_points[1:])),
               np.r_[a.path_cell_indices, b.path_cell_indices[1:]],
               a.planning_cost + b.planning_cost,
               a.physical_travel_time + b.physical_travel_time, 'bounded_via_node')
    # Prefer alternatives with large geometric separation; no score-dependent pool.
    selected = routes[:1]
    remaining = routes[1:]
    while remaining and len(selected) < settings['routes_per_agent']:
        def diversity(route):
            return min(float(np.mean(np.min(np.linalg.norm(
                route.points[:, None] - old.points[None, :], axis=2), axis=1))) for old in selected)
        chosen = min(remaining, key=lambda r: (-diversity(r), r.planning_cost, r.route_id))
        remaining.remove(chosen)
        selected.append(chosen)
    counts['valid_generated'] = len(routes)
    counts['selected'] = len(selected)
    return tuple(selected), dict(counts)


def forecast(route, agent, vehicle_constants, horizon):
    """Fresh ordered tracker, observed velocity, zero residual and zero future flow.

    Uses the existing public Vehicle model. It is not a simulator continuation
    and does not model future safety replacements or high-level replans.
    """
    vehicle = Vehicle(**vehicle_constants)
    tracker = PathTracker(threshold=.75)
    p, v = np.asarray(agent.position), np.asarray(agent.velocity)
    positions = [p.copy()]
    for _ in range(horizon):
        target = tracker.tracking_target(agent.agent_id, p, route.points, route.points[-1])
        p, v = vehicle.step(p, v, np.asarray(target), np.zeros(3))
        positions.append(p.copy())
    return np.asarray(positions)
