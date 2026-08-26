"""Code-traced Chapter-3 28-dimensional local-observation contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ObservationField:
    name: str
    start: in
    end: in
    normalization: str
    semantics: str
    target_unknown_behavior: str
    role_behavior: str


FIELDS: Tuple[ObservationField, ...] = (
    ObservationField("position", 0, 3, "none", "Agent world position (x,y,z).", "unchanged", "same computation for all roles"),
    ObservationField("velocity", 3, 6, "none", "Agent world velocity (vx,vy,vz).", "unchanged", "same computation for all roles"),
    ObservationField("navigation_target_delta", 6, 9, "none", "Current navigation target minus agent position.", "points to search waypoint or executor wait point", "role-specific navigation target"),
    ObservationField("navigation_target_direction", 9, 12, "divide by max(L2(delta),1e-6)", "Unit direction toward current navigation target.", "follows the unknown-phase navigation target", "role-specific navigation target"),
    ObservationField("known_target_delta", 12, 15, "none", "Known task-target estimate minus agent position.", "exact zeros", "populated only when that agent's task-known flag is true"),
    ObservationField("navigation_distance", 15, 16, "clip(L2(delta)/10,0,1)", "Distance to current navigation target.", "follows the unknown-phase navigation target", "same normalization; role-specific target"),
    ObservationField("speed", 16, 17, "clip(L2(velocity)/(role v_xy_max+1e-6),0,1)", "Normalized agent speed.", "unchanged", "role-specific horizontal speed limit"),
    ObservationField("closing_speed", 17, 18, "tanh(dot(velocity,nav_direction)/(role v_xy_max+1e-6))", "Signed velocity component toward navigation target.", "follows the unknown-phase navigation target", "role-specific horizontal speed limit"),
    ObservationField("nearest_obstacle_distance", 18, 19, "clip(nearest AABB distance/10,0,1)", "Distance to nearest configured obstacle; legacy no-obstacle sentinel is preserved.", "unchanged", "same computation for all roles"),
    ObservationField("waypoint_progress", 19, 20, "reached/max(1,total)", "Fraction of assigned waypoints reached.", "unchanged", "uses each agent's waypoint counts"),
    ObservationField("agent_finished", 20, 21, "bool cast to float", "Whether the agent completed its assigned phase.", "unchanged", "per-agent status"),
    ObservationField("hold_progress", 21, 22, "clip(counter/max(1,role hold steps),0,1)", "Progress through the role-specific hold requirement.", "unchanged", "search_hold_steps for ids 0-2; executor_hold_steps for id 3"),
    ObservationField("role_onehot", 22, 26, "one-hot", "Role identity in canonical order.", "unchanged", "search_fast, search_balanced, search_precise, executor"),
    ObservationField("target_knowledge_phase", 26, 28, "binary pair", "[1,0] unknown; [0,1] known for this agent.", "[1,0]", "uses each agent's task-known flag"),
)

OBSERVATION_DIM = 28
ROLE_ORDER = ("search_fast", "search_balanced", "search_precise", "executor")


def validate_contract() -> None:
    cursor = 0
    for field in FIELDS:
        if field.start != cursor or field.end <= field.start:
            raise RuntimeError(f"non-contiguous observation field: {field.name}")
        cursor = field.end
    if cursor != OBSERVATION_DIM:
        raise RuntimeError(f"observation contract ends at {cursor}, not 28")


validate_contract()
