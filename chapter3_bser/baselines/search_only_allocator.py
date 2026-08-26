"""Search-only comparator using identical candidates and detection kernels."""

from chapter3_bser.greedy_solver import solve_fixed_standby_greedy


def solve_search_only_greedy(candidates, context):
    return solve_fixed_standby_greedy(candidates, None, context, search_only=True)
