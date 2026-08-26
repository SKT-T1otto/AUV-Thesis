"""Deterministic lazy marginal-gain greedy solver."""

from __future__ import annotations

import heapq
from typing import Optional, Sequence

from chapter3_bser.objective import ObjectiveContext, evaluate_objective, marginal_gain
from chapter3_bser.types import SearchCandidate, SolverResult, StandbyCandidate


def solve_fixed_standby_lazy(
    candidates: Sequence[SearchCandidate], standby: Optional[StandbyCandidate], context: ObjectiveContext, *, search_only: bool = False
) -> SolverResult:
    ordered = sorted(candidates, key=lambda candidate: candidate.key)
    heap = [(-marginal_gain((), candidate, standby, context, search_only=search_only), candidate.key, 0, candidate) for candidate in ordered]
    heapq.heapify(heap)
    selected = []
    used_agents = set()
    version = 0
    while heap:
        negative_bound, _, stamp, candidate = heapq.heappop(heap)
        if candidate.agent_id in used_agents:
            continue
        if stamp != version:
            gain = marginal_gain(selected, candidate, standby, context, search_only=search_only)
            heapq.heappush(heap, (-gain, candidate.key, version, candidate))
            continue
        gain = -negative_bound
        if gain <= 1e-15:
            break
        selected.append(candidate)
        used_agents.add(candidate.agent_id)
        version += 1
    selected.sort(key=lambda candidate: candidate.key)
    value = evaluate_objective(selected, standby, context, search_only=search_only)
    return SolverResult("search_only_lazy" if search_only else "fixed_y_lazy_greedy", tuple(selected), standby, value)


def solve_joint_lazy(candidates: Sequence[SearchCandidate], standby_candidates: Sequence[StandbyCandidate], context: ObjectiveContext) -> SolverResult:
    best = None
    for standby in sorted(standby_candidates, key=lambda item: item.key):
        current = solve_fixed_standby_lazy(candidates, standby, context)
        key = (tuple(candidate.key for candidate in current.selected), standby.key)
        best_key = None if best is None else (tuple(candidate.key for candidate in best.selected), best.standby.key)
        if best is None or current.objective > best.objective + 1e-15 or (abs(current.objective - best.objective) <= 1e-15 and key < best_key):
            best = current
    if best is None:
        return SolverResult("bser_lazy_greedy", (), None, 0.0, status="NO_STANDBY_CANDIDATE")
    return SolverResult("bser_lazy_greedy", best.selected, best.standby, best.objective)
