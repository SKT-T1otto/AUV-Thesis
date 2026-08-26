"""Complete finite enumeration for small BSER instances."""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Optional, Sequence, Tuple

from chapter3_bser.objective import ObjectiveContext, evaluate_objective
from chapter3_bser.types import SearchCandidate, SolverResult, StandbyCandidate


def partition_groups(candidates: Sequence[SearchCandidate]) -> Tuple[Tuple[SearchCandidate, ...], ...]:
    by_agent: Dict[int, list] = {}
    for candidate in candidates:
        by_agent.setdefault(candidate.agent_id, []).append(candidate)
    return tuple(tuple(sorted(by_agent[agent_id], key=lambda candidate: candidate.key)) for agent_id in sorted(by_agent))


def feasible_allocations(candidates: Sequence[SearchCandidate]):
    groups = partition_groups(candidates)
    for choices in itertools.product(*(tuple([None]) + group for group in groups)):
        yield tuple(candidate for candidate in choices if candidate is not None)


def exact_combination_count(candidates: Sequence[SearchCandidate], standby_candidates: Sequence[StandbyCandidate]) -> int:
    count = 1
    for group in partition_groups(candidates):
        count *= len(group) + 1
    return count * max(len(standby_candidates), 1)


def solve_fixed_standby_exact(
    candidates: Sequence[SearchCandidate], standby: Optional[StandbyCandidate], context: ObjectiveContext, *, search_only: bool = False
) -> SolverResult:
    best_selected = ()
    best_value = -1.0
    visited = 0
    for selected in feasible_allocations(candidates):
        visited += 1
        value = evaluate_objective(selected, standby, context, search_only=search_only)
        key = tuple(candidate.key for candidate in selected)
        best_key = tuple(candidate.key for candidate in best_selected)
        if value > best_value + 1e-15 or (abs(value - best_value) <= 1e-15 and key < best_key):
            best_value = value
            best_selected = selected
    return SolverResult("search_only_exact" if search_only else "fixed_y_exact", best_selected, standby, best_value, visited)


def solve_joint_exact(
    candidates: Sequence[SearchCandidate], standby_candidates: Sequence[StandbyCandidate], context: ObjectiveContext, *, combination_limit: int = 100000
) -> SolverResult:
    combinations = exact_combination_count(candidates, standby_candidates)
    if combinations > int(combination_limit):
        return SolverResult("joint_exact", (), None, 0.0, combinations, "EXACT_SKIPPED_COMBINATORIAL_LIMIT")
    best = None
    for standby in sorted(standby_candidates, key=lambda item: item.key):
        current = solve_fixed_standby_exact(candidates, standby, context)
        key = (tuple(candidate.key for candidate in current.selected), standby.key)
        best_key = None if best is None else (tuple(candidate.key for candidate in best.selected), best.standby.key)
        if best is None or current.objective > best.objective + 1e-15 or (abs(current.objective - best.objective) <= 1e-15 and key < best_key):
            best = current
    if best is None:
        return SolverResult("joint_exact", (), None, 0.0, combinations, "NO_STANDBY_CANDIDATE")
    return SolverResult("joint_exact", best.selected, best.standby, best.objective, combinations)
