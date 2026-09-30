"""Frozen F interventions. No evaluation-time hyperparameter search."""
ARMS = {
    "F0": dict(reference_only=True, search_guard=True, response="dynamic", risk=True, solver="enumerate"),
    "F1": dict(reference_only=False, search_guard=True, response="dynamic", risk=True, solver="enumerate"),
    "F2": dict(reference_only=False, search_guard=True, response="sequential", risk=True, solver="enumerate"),
    "F3": dict(reference_only=False, search_guard=False, response="dynamic", risk=True, solver="enumerate"),
    "F4": dict(reference_only=False, search_guard=True, response="static", risk=True, solver="enumerate"),
    "F5": dict(reference_only=False, search_guard=True, response="dynamic", risk=False, solver="enumerate"),
    "F6": dict(reference_only=False, search_guard=True, response="dynamic", risk=True, solver="coordinate_greedy"),
}
SETTINGS = dict(parent="bser_effect_v1_D2", parent_variant="V4", prediction_steps=20,
    waypoint_threshold=.75, minimum_search_gain=.03, minimum_response_gain=.05,
    score_floor=1e-10, numerical_tolerance=1e-12, disturbance_limit=.5,
    disturbance_uncertainty=.08, max_forecasts=24, max_response_queries=80,
    max_combinations=256, endpoint_test_limit=128,
    candidate_generation="unchanged_D2_pool", belief_forecast="frozen_current_public_belief",
    near_score="mean_incremental_cumulative_coverage_over_20_steps",
    response_score="incremental_coverage_at_t_times_exp_minus_response_from_executor_at_t_over_tau",
    repeated_observation="per_agent_temporal_max_then_cross_agent_union",
    selection="response_then_near_then_full; F2_near_then_full",
    greedy_seed="same_pool_D2_search",
    technical_fallback="same_state_installable_D2", budget="deterministic_work_counts",
    risk="per_agent_known_collision_exposure_unknown_length_max_cross_track",
    residual_source="zeros_4x3", post_found_overlay=False, braking_v2=False, hold_v2=False,
    training=False, checkpoint_loaded=False)
COMPARISONS = (("F1", "F0"), ("F1", "F2"), ("F2", "F0"),
               ("F1", "F3"), ("F1", "F4"), ("F1", "F5"), ("F1", "F6"))


def parse_arms(value):
    arms = value.split(",")
    if not arms or arms[0] != "F0" or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError("arms must be unique F0--F6 values starting with F0")
    return arms
