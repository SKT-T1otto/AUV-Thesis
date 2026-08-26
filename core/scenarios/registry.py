"""Supported scenario-profile registry derived from CH3 configuration."""

from __future__ import annotations

from .schema import ScenarioProfile


CANONICAL_THESIS_PROFILE = "M20_MOVING_UNKNOWN_MULTI"

PROFILES = {
    "M00_MOVING_CLEAR": ScenarioProfile("M00_MOVING_CLEAR", "clear_lower_boundary", "clear", "online_unknown"),
    "M10_MOVING_UNKNOWN_SINGLE": ScenarioProfile("M10_MOVING_UNKNOWN_SINGLE", "single_obstacle_medium", "random_single_aabb_v1", "online_unknown"),
    "M20_MOVING_UNKNOWN_MULTI": ScenarioProfile("M20_MOVING_UNKNOWN_MULTI", "canonical_thesis_profile", "random_multi_aabb_v1", "online_unknown"),
    "M90_MOVING_KNOWN_ORACLE": ScenarioProfile("M90_MOVING_KNOWN_ORACLE", "known_map_oracle_upper_boundary", "random_multi_aabb_v1", "oracle"),
}
