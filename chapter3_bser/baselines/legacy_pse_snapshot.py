"""Read-only evaluation of current PSE navigation outputs."""

from __future__ import annotations

from chapter3_bser.objective import build_objective_context, evaluate_objective
from chapter3_bser.types import SearchCandidate, SolverResult, StandbyCandidate
from core.mapping.travel_cost_service import TravelCostService


def evaluate_legacy_pse_snapshot(state, config):
    if state.target_found:
        return SolverResult("legacy_pse_snapshot", (), None, 0.0, status="SKIPPED_TARGET_ALREADY_FOUND")
    service = TravelCostService(state)
    search = []
    for agent_id in state.searcher_ids:
        agent = state.agents[agent_id]
        if agent.current_navigation_target is None:
            return SolverResult("legacy_pse_snapshot", (), None, 0.0, status="PSE_BASELINE_DEFERRED_NO_READONLY_ACCESS")
        query = service.query(agent.position, agent.current_navigation_target, agent)
        if not query.reachable:
            return SolverResult("legacy_pse_snapshot", (), None, 0.0, status="PSE_SNAPSHOT_UNREACHABLE")
        search.append(SearchCandidate(
            agent_id, f"pse_s{agent_id}", agent.current_navigation_target,
            query.path_points, query.path_cell_indices, query.path_length,
            query.planning_cost, query.physical_travel_time,
            "current_pse_navigation_target",
        ))
    executor = state.agents[state.executor_id]
    if executor.current_navigation_target is None:
        return SolverResult("legacy_pse_snapshot", (), None, 0.0, status="PSE_BASELINE_DEFERRED_NO_READONLY_ACCESS")
    query = service.query(executor.position, executor.current_navigation_target, executor)
    if not query.reachable:
        return SolverResult("legacy_pse_snapshot", (), None, 0.0, status="PSE_SNAPSHOT_UNREACHABLE")
    standby = StandbyCandidate(
        "pse_y", executor.current_navigation_target, query.path_points,
        query.path_cell_indices, query.path_length, query.planning_cost,
        query.physical_travel_time, "current_pse_standby",
    )
    context = build_objective_context(state, search, (standby,), config)
    return SolverResult("legacy_pse_snapshot", tuple(search), standby, evaluate_objective(search, standby, context))
