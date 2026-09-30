"""Frozen intervention definitions for the manual paired experiment."""
ARMS = {
    "D0": dict(search_objective="detection", standby="initial_anchor"),
    "D1": dict(search_objective="response_weighted", standby="initial_anchor"),
    "D2": dict(search_objective="detection", standby="sequential_best_response"),
    "D3": dict(search_objective="response_weighted", standby="joint_greedy"),
}
SETTINGS = dict(parent_variant="V4", waypoint_threshold=0.75,
    residual_source="zeros_4x3", acceptance="eligible_feasible_changed_proposal",
    current_allocation_scoring="same_public_state_full_retained_paths",
    partial_solve="unaffected_searchers_fixed", post_found_overlay=False,
    braking_v2=False, hold_v2=False, recovery_v2=False)


def parse_arms(value):
    arms = value.split(",")
    if not arms or arms[0] != "D0" or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError("arms must be unique D0--D3 values, starting with D0")
    return arms
