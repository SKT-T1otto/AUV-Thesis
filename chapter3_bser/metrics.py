"""Feasibility and finite-instance mathematical checks."""

from __future__ import annotations

from chapter3_bser.exact_solver import feasible_allocations, solve_joint_exact
from chapter3_bser.greedy_solver import solve_joint_greedy
from chapter3_bser.lazy_greedy_solver import solve_joint_lazy
from chapter3_bser.objective import evaluate_objective, marginal_gain


def partition_feasible(selected) -> bool:
    ids = [candidate.agent_id for candidate in selected]
    return len(ids) == len(set(ids))


def validate_small_instance(candidates, standby_candidates, context, *, tolerance=1e-9):
    allocations = tuple(feasible_allocations(candidates))
    nonnegative = monotone = submodular = True
    for standby in standby_candidates:
        by_key = {frozenset(candidate.candidate_id for candidate in selected): selected for selected in allocations}
        values = {key: evaluate_objective(selected, standby, context) for key, selected in by_key.items()}
        nonnegative &= all(value >= -tolerance for value in values.values())
        # It is sufficient to check the Hasse edges: repeated one-element
        # extensions establish monotonicity and diminishing returns globally.
        for left_ids, left in by_key.items():
            used_left = {candidate.agent_id for candidate in left}
            for extension in candidates:
                if extension.agent_id in used_left:
                    continue
                right_ids = left_ids | {extension.candidate_id}
                right = by_key[right_ids]
                monotone &= values[left_ids] <= values[right_ids] + tolerance
                used_right = used_left | {extension.agent_id}
                for candidate in candidates:
                    if candidate.agent_id in used_right:
                        continue
                    left_gain = values[left_ids | {candidate.candidate_id}] - values[left_ids]
                    right_gain = values[right_ids | {candidate.candidate_id}] - values[right_ids]
                    submodular &= left_gain + tolerance >= right_gain
    exact = solve_joint_exact(candidates, standby_candidates, context)
    greedy = solve_joint_greedy(candidates, standby_candidates, context)
    lazy = solve_joint_lazy(candidates, standby_candidates, context)
    ratio = 1.0 if exact.objective <= tolerance else greedy.objective / exact.objective
    return {
        "nonnegative_pass": bool(nonnegative),
        "monotonicity_pass": bool(monotone),
        "submodularity_pass": bool(submodular),
        "partition_constraint_pass": partition_feasible(exact.selected) and partition_feasible(greedy.selected) and partition_feasible(lazy.selected),
        "greedy_bound_pass": ratio >= 0.5 - tolerance,
        "lazy_equivalence_pass": greedy.selected_ids == lazy.selected_ids and greedy.standby == lazy.standby and abs(greedy.objective - lazy.objective) <= tolerance,
        "greedy_exact_ratio": ratio,
        "allocation_count": len(allocations),
    }
