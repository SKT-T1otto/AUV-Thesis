"""Cached same-state plan scoring; query limits are independent of wall time."""
from dataclasses import dataclass, replace
import time

import numpy as np

from chapter3_bser.detection_model import detection_probability
from chapter3_bser.experiments.safe_search_v1.public_geometry import PublicGeometry
from core.mapping.planning_graph import EndpointConnectorSet, PlanningConnectorView
from core.mapping.travel_cost_service import TravelCostService
from .forecast import Incomparable, predict
from .options import SETTINGS


def route_key(item):
    return repr(item)


def allocation_key(allocation):
    return tuple(route_key(x) for x in allocation.search_assignments), route_key(allocation.executor_assignment)


def connected_times(state, point, geometry, *, limit):
    """Verified continuous-start connectors, then the unchanged graph service.

    Match the public adapter's linear connector travel metric. Never snap an
    arbitrary predicted position to a voxel center or call a truth planner.
    A bounded incomplete connector search is explicitly incomparable.
    """
    graph = state.planning_graph
    point = np.asarray(point, dtype=float)
    if not geometry.point_free(point):
        raise Incomparable("response_start_outside_public_free_geometry")
    centers = np.asarray(graph.cell_centers)
    valid = np.flatnonzero(graph.valid_mask)
    if not len(valid):
        raise Incomparable("response_graph_empty")
    delta = centers-point
    rates = graph.time_rates_for_role("executor")
    costs = np.linalg.norm(delta[:, :2], axis=1)*rates[0] + np.abs(delta[:, 2])*rates[1]
    order = valid[np.lexsort((valid, costs[valid]))]
    components = set(int(v) for v in graph.component_labels[valid] if v >= 0)
    best = {}
    tested = 0
    # One geometry batch avoids dozens of tiny NumPy allocations per endpoint.
    tested_indices = order[:limit]
    free = geometry._segments_free(np.repeat(point[None, :], len(tested_indices), axis=0), centers[tested_indices])
    for index, accepted in zip(tested_indices, free):
        tested += 1
        component = int(graph.component_labels[index])
        if accepted and component >= 0 and component not in best:
            best[component] = PlanningConnectorView(int(index), component, float(costs[index]), float(costs[index]))
        if set(best) == components:
            break
    if not best or (set(best) != components and len(order) > limit):
        raise Incomparable("response_connector_missing_or_budget_limited")
    endpoint = EndpointConnectorSet("F_forecast", "executor", tuple(float(x) for x in point),
        tuple(sorted(best.values(), key=lambda x: (x.planning_cost, x.component_id, x.cell_index))), True)
    # Put this verified connector first, so an old endpoint with the same point
    # cannot shadow it. No live snapshot or planner cache is modified.
    augmented = replace(state, planning_graph=replace(graph,
        endpoint_connectors=(endpoint, *graph.endpoint_connectors)))
    agent = next(a for a in state.agents if a.agent_id == state.executor_id)
    return TravelCostService(augmented).travel_times_from(point, agent), tested


@dataclass
class PlanScore:
    full: float
    near: float
    response: float | None
    risks: dict

    def record(self):
        return dict(full_search=self.full, near_search=self.near, response=self.response,
                    risks={str(i): list(v) for i, v in self.risks.items()})


class Scorer:
    def __init__(self, allocator, state, context, standbys, executor, arm):
        self.allocator, self.owner, self.state = allocator, allocator.owner, state
        self.context, self.standbys, self.executor, self.arm = context, standbys, executor, arm
        live = self.owner.planning_views.live()
        self.snapshot_compatible = (state.map_revision == live.map_revision
            and state.target_belief.revision == live.target_belief.revision)
        self.geometry = PublicGeometry(live, self.owner.planning_views.clearance)
        self.response_state = self.owner.planning_views.search(state)
        self.forecasts, self.detections, self.weights, self.scores = {}, {}, {}, {}
        self.counts = dict(forecasts=0, forecast_cache_hits=0, response_queries=0,
            response_cache_hits=0, connector_tests=0, combination_scores=0, combination_requests=0)
        self.timings = dict(forecast_seconds=0., response_seconds=0., score_seconds=0.)

    def coverage(self, allocation):
        complement = np.ones(self.context.belief.shape)
        for item in allocation.search_assignments:
            key = route_key(item)
            if key not in self.detections:
                candidate = self.allocator._frozen_search_candidate(item)
                self.detections[key] = detection_probability(candidate, self.state, self.allocator.config)
            complement *= 1-self.detections[key]
        return np.clip(1-complement, 0, 1)

    def full(self, allocation):
        return float(np.dot(self.context.belief, self.coverage(allocation)))

    def sequential(self, allocation):
        if self.executor is not None:
            return replace(allocation, executor_assignment=self.executor)
        q = self.coverage(allocation)
        best = min(self.standbys, key=lambda y: (
            -float(np.sum(self.context.belief*q*self.context.response_weight_by_id[y.candidate_id])), y.key))
        return replace(allocation, executor_assignment=self.allocator.execution.assign_standby(self.state, best))

    def forecast(self, allocation, agent_id):
        item = allocation.executor_assignment if agent_id == self.state.executor_id else next(
            (x for x in allocation.search_assignments if x.agent_id == agent_id), None)
        key = (agent_id, route_key(item))
        if key in self.forecasts:
            self.counts["forecast_cache_hits"] += 1
            if isinstance(self.forecasts[key], Incomparable):
                raise self.forecasts[key]
            return self.forecasts[key]
        if self.counts["forecasts"] >= SETTINGS["max_forecasts"]:
            raise Incomparable("forecast_budget")
        started = time.perf_counter()
        self.counts["forecasts"] += 1
        try:
            result = predict(self.owner, self.state, allocation, agent_id, self.geometry, self.allocator.config)
        except Incomparable as exc:
            self.forecasts[key] = exc
            raise
        finally:
            self.timings["forecast_seconds"] += time.perf_counter()-started
        self.forecasts[key] = result
        return result

    def response_weights(self, positions):
        output = []
        tau = self.allocator.config["objective"]["tau_executor"]
        for point in positions:
            key = tuple(float(x) for x in point)
            if key in self.weights:
                self.counts["response_cache_hits"] += 1
                if isinstance(self.weights[key], Incomparable):
                    raise self.weights[key]
            else:
                if self.counts["response_queries"] >= SETTINGS["max_response_queries"]:
                    raise Incomparable("response_query_budget")
                self.counts["response_queries"] += 1
                started = time.perf_counter()
                try:
                    times, tested = connected_times(self.response_state, point, self.geometry,
                                                    limit=SETTINGS["endpoint_test_limit"])
                    self.counts["connector_tests"] += tested
                    weights = np.zeros_like(times)
                    finite = np.isfinite(times)
                    weights[finite] = np.exp(-times[finite]/tau)
                    self.weights[key] = weights
                except Incomparable as exc:
                    self.weights[key] = exc
                    raise
                finally:
                    self.timings["response_seconds"] += time.perf_counter()-started
            output.append(self.weights[key])
        return np.asarray(output)

    def score(self, allocation, reference=None):
        if not self.snapshot_compatible:
            raise Incomparable("stale_public_planning_snapshot")
        key = allocation_key(allocation)
        if key in self.scores:
            return self.scores[key]
        started = time.perf_counter()
        forecasts = {i: self.forecast(allocation, i) for i in (*self.state.searcher_ids, self.state.executor_id)}
        complement = np.ones((SETTINGS["prediction_steps"]+1, len(self.context.belief)))
        for i in self.state.searcher_ids:
            complement *= 1-forecasts[i].cumulative_detection
        cumulative = np.clip(1-complement, 0, 1)
        incremental = np.maximum(np.diff(cumulative, axis=0), 0.)
        near = float(np.mean((cumulative[1:]-cumulative[0]) @ self.context.belief))
        if reference is not None:
            eps = SETTINGS["numerical_tolerance"]
            if self.arm["search_guard"] and near+eps < reference.near:
                raise Incomparable("near_search_regression")
            if self.arm["risk"]:
                for i, f in forecasts.items():
                    if any(a > b+eps for a, b in zip(f.risk, reference.risks[i])):
                        raise Incomparable("risk_regression_agent_"+str(i))
        response = None
        if self.arm["response"] == "dynamic":
            weights = self.response_weights(forecasts[self.state.executor_id].positions[1:])
            response = float(np.sum(incremental*weights*self.context.belief[None, :]))
        elif self.arm["response"] == "static":
            # Same filtered graph/connector model, original static full-path formula.
            weights = self.response_weights((allocation.executor_assignment.target_region,))[0]
            response = float(np.sum(self.context.belief*self.coverage(allocation)*weights))
        result = PlanScore(self.full(allocation), near, response, {i: f.risk for i, f in forecasts.items()})
        if not np.isfinite([result.full, result.near, *(v for v in [response] if v is not None)]).all():
            raise Incomparable("nonfinite_score")
        self.scores[key] = result
        self.counts["combination_scores"] += 1
        self.timings["score_seconds"] += time.perf_counter()-started
        return result
