"""BEDS-only public-map routing; shared planners and dynamics remain untouched."""
from dataclasses import dataclass

import numpy as np

from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.online.executor_standby import compute_standby_target
from core.mapping.travel_cost_service import TravelCostService


@dataclass(frozen=True)
class StandbyNavigationParameters:
    update_interval: int
    target_shift_threshold: float
    path_tracking_threshold: float
    safe_dist: float
    prior_slow_radius_xy: float
    prior_slow_radius_z: float
    hold_radius: float

    def __post_init__(self):
        for value in vars(self).values():
            if not np.isfinite(value) or value <= 0:
                raise ValueError("standby navigation parameters must be positive and finite")


def min_searcher_distance(point, state):
    agents = {a.agent_id: a for a in state.agents}
    return min(float(np.linalg.norm(np.asarray(point)-agents[i].position)) for i in state.searcher_ids)


def local_searcher_clearance(start, end, state, safe_dist):
    """Current-position point/segment separation; no future motion prediction."""
    start, end = np.asarray(start), np.asarray(end)
    agents = {a.agent_id: a for a in state.agents}
    return all(PathTracker._point_segment_distance(np.asarray(agents[i].position), start, end) >= safe_dist
               for i in state.searcher_ids)


def resolve_safe_route(state, raw_target, safe_dist, segment_clear, service_factory=TravelCostService):
    """Thin endpoint repair using existing public graph, Dijkstra and A*.

    The existing intercept proxy helper imposes strict progress and has no
    dynamic-agent filter. Standby instead orders all reachable valid centers
    by raw-target distance, then existing planning cost and stable cell index.
    """
    agent = {a.agent_id: a for a in state.agents}[state.executor_id]
    service = service_factory(state)
    if min_searcher_distance(raw_target, state) >= safe_dist:
        exact = service.query(agent.position, raw_target, agent)
        if exact.reachable and all(segment_clear(a, b) for a, b in zip(exact.path_points, exact.path_points[1:])):
            return tuple(raw_target), exact, "RAW_REACHABLE"
    single = service.single_source(agent.position, agent)
    centers = np.asarray(state.grid.cell_centers)
    eligible = np.flatnonzero(np.asarray(state.planning_graph.valid_mask)
                             & np.asarray(single.reachable_mask)
                             & np.isfinite(single.planning_cost_by_cell))
    ordered = sorted((int(i) for i in eligible), key=lambda i: (
        float(np.linalg.norm(centers[i]-raw_target)), float(single.planning_cost_by_cell[i]), i))
    for index in ordered:
        target = tuple(float(v) for v in centers[index])
        if min_searcher_distance(target, state) < safe_dist:
            continue
        query = service.query(agent.position, target, agent)
        if query.reachable and all(segment_clear(a, b) for a, b in zip(query.path_points, query.path_points[1:])):
            return target, query, "PUBLIC_ENDPOINT_REPAIR"
    return None, None, "NO_SAFE_PUBLIC_ROUTE"


class SafeStandbyNavigation:
    """Own only BEDS tracker/target state, never the baseline bridge tracker."""
    def __init__(self, parameters, *, state_factory, segment_clear, service_factory=TravelCostService):
        self.parameters = parameters
        self.state_factory, self.segment_clear = state_factory, segment_clear
        self.service_factory = service_factory
        self.tracker = PathTracker(parameters.path_tracking_threshold)
        self.clear()

    def clear(self):
        self.raw_target = self.accepted_raw_target = self.safe_target = self.tracking_target = None
        self.path = ()
        self.last_attempt = None
        self.route_length = None
        self.hold = self.slowdown = self.active = False
        self.residual_factor = 1.0
        self.details = {}
        self.tracker.reset()

    def prepare(self, state):
        if state.target_found or state.mission_complete:
            self.clear()
            return
        p = self.parameters
        agents = {a.agent_id: a for a in state.agents}
        current = np.asarray(agents[state.executor_id].position)
        self.active = True
        self.raw_target = tuple(compute_standby_target(current, [agents[i].position for i in state.searcher_ids]))
        shift = None if self.accepted_raw_target is None else float(np.linalg.norm(np.asarray(self.raw_target)-self.accepted_raw_target))
        due = self.last_attempt is None or state.step-self.last_attempt >= p.update_interval
        reason, updated = "RETAIN_TARGET", False
        # Audit the retained route against the *live* online planner each cycle.
        remaining = self.tracker.snapshot(state.executor_id).remaining_path_points if self.path else ()
        route_invalid = bool(self.path and not all(self.segment_clear(a, b) for a, b in
                                                   zip((tuple(current), *remaining), remaining)))
        if route_invalid:
            self.path, self.safe_target, self.route_length = (), None, None
            self.tracker.reset()
            reason = "PUBLIC_ROUTE_INVALID_HOLD"
        if due and (self.safe_target is None or shift is None or shift >= p.target_shift_threshold):
            self.last_attempt = int(state.step)
            try:
                fresh = self.state_factory()
                self.safe_target, query, reason = resolve_safe_route(
                    fresh, self.raw_target, p.safe_dist, self.segment_clear, self.service_factory)
            except (RuntimeError, ValueError) as exc:
                self.safe_target, query = None, None
                reason = f"PUBLIC_PLANNER_FAILURE:{type(exc).__name__}:{exc}"
            self.accepted_raw_target = self.raw_target
            self.path = () if query is None else tuple(tuple(float(v) for v in point) for point in query.path_points)
            self.route_length = None if query is None else float(query.path_length)
            self.tracker.reset()
            updated = True
        elif not due and reason == "RETAIN_TARGET":
            reason = "UPDATE_INTERVAL_GATE"
        elif due and reason == "RETAIN_TARGET":
            reason = "TARGET_SHIFT_GATE"
        self.hold = self.safe_target is None
        target = tuple(current) if self.hold else self.tracker.tracking_target(
            state.executor_id, current, self.path, self.safe_target)
        if not self.hold and (not self.segment_clear(current, target)
                              or not local_searcher_clearance(current, target, state, p.safe_dist)):
            self.hold, reason = True, "LOCAL_CLEARANCE_HOLD"
        distance = None if self.safe_target is None else float(np.linalg.norm(current-self.safe_target))
        # Only enter arrival hold at the final segment, not across an obstacle.
        if not self.hold and distance < p.hold_radius and target == self.safe_target:
            self.hold, reason = True, "ARRIVAL_HOLD"
        self.tracking_target = tuple(current) if self.hold else target
        delta = np.asarray(self.tracking_target)-current
        # The actual velocity feedback/slowdown remains in the environment prior.
        # Fade only the learned Executor residual, which otherwise survives hold.
        self.residual_factor = 0.0 if self.hold else float(max(
            min(1.0, np.linalg.norm(delta[:2])/p.prior_slow_radius_xy),
            min(1.0, abs(delta[2])/p.prior_slow_radius_z)))
        self.slowdown = self.residual_factor < 1.0
        self.details = dict(raw_standby_target=self.raw_target, safe_standby_target=self.safe_target,
                            target_update_event=updated, target_update_reason=reason, target_shift=shift,
                            route_available=bool(self.path), route_length=self.route_length,
                            route_point_count=len(self.path), tracking_target=self.tracking_target,
                            standby_distance=distance, tracking_target_distance=float(np.linalg.norm(delta)),
                            slowdown_active=self.slowdown, hold_active=self.hold, standby_active=self.active,
                            residual_factor=self.residual_factor, **vars(p))

    def apply_residual_safety(self, actions, state):
        if state.target_found:
            self.clear()
            return actions
        if not self.active or self.residual_factor == 1.0:
            return actions
        result = actions.clone() if hasattr(actions, "clone") else np.asarray(actions).copy()
        result[state.executor_id] *= self.residual_factor
        return result
