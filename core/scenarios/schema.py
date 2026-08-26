"""Scenario schema declarations; numerical records remain CH3-authored dicts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioProfile:
    profile_id: str
    role: str
    obstacle_family: str
    obstacle_knowledge_mode: str
    target_motion_mode: str = "constant_velocity_reflect_v1"
