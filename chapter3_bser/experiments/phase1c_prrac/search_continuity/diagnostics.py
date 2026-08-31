"""One authoritative, information-bounded SEARCH diagnostic implementation."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import torch

from chapter3_bser.models.prrac.stage_mapping import PRRACStage

from .types import nullable_rate


def _array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _scalar(value: Any) -> float:
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1)[0].item()
    return float(np.asarray(value).reshape(-1)[0])


class SearchContinuityDiagnostics:
    """Observe public snapshots without retaining or mutating runtime objects."""

    def __init__(self) -> None:
        self.pre_found_step_count = 0
        self.collision_counts = [0, 0, 0]
        self.collision_streak = [0, 0, 0]
        self.collision_max_streak = [0, 0, 0]
        self.first_collision_step: int | None = None
        self.last_collision_step: int | None = None
        self.route_active = [0, 0, 0]
        self.hold = [0, 0, 0]
        self.assignment_missing = 0
        self.assignment_unreachable = 0
        self.assignment_switch_count = 0
        self.tracking_subgoal_switch_count = 0
        self.distance = [0.0, 0.0, 0.0]
        self.raw_norms: list[float] = []
        self.applied_norms: list[float] = []
        self.suppressed_env_step_count = 0
        self.suppressed_agent_step_count = 0
        self.negative_alignment_count = 0
        self.alignment_valid_count = 0
        self.alignment_zero_navigation_count = 0
        self.alignment_zero_residual_count = 0
        self.contribution_ratios: list[float] = []
        self._previous_assignment_identity: list[tuple[Any, ...] | None] = [None, None, None]
        self._previous_tracking_waypoint: list[tuple[float, float, float] | None] = [None, None, None]
        self._initial: tuple[float, float, float] | None = None
        self._latest: tuple[float, float, float] | None = None
        self._found_step: int | None = None

    @staticmethod
    def _map_belief(state: Any) -> tuple[float, float, float]:
        known = float(np.asarray(state.occupancy.known_mask, dtype=np.bool_).mean())
        return (
            known,
            float(state.target_belief.entropy),
            float(state.target_belief.peak_probability),
        )

    def begin_episode(self, initial_state: Any) -> None:
        self._initial = self._map_belief(initial_state)
        self._latest = self._initial

    def observe_transition(
        self,
        *,
        stage_before: int | PRRACStage,
        stage_after: int | PRRACStage,
        installed_guidance: Any,
        planning_state_before: Any,
        planning_state_after: Any,
        collision_flags: Iterable[Any],
        raw_actions: Any,
        applied_actions: Any,
        actor_outputs: Iterable[Any],
        residual_contribution_ratios: Any = None,
    ) -> None:
        if int(stage_before) != int(PRRACStage.SEARCH):
            return
        self.pre_found_step_count += 1
        if int(stage_after) != int(PRRACStage.SEARCH) and self._found_step is None:
            self._found_step = int(planning_state_after.step)
        before = {int(agent.agent_id): agent for agent in planning_state_before.agents}
        after = {int(agent.agent_id): agent for agent in planning_state_after.agents}
        collisions = np.asarray(tuple(collision_flags), dtype=np.bool_).reshape(-1)
        raw = _array(raw_actions).reshape(4, 3)
        applied = _array(applied_actions).reshape(4, 3)
        outputs = tuple(actor_outputs)
        scalar_ratio = None
        if residual_contribution_ratios is None:
            ratios = ()
        elif np.asarray(residual_contribution_ratios).ndim == 0:
            ratios = ()
            scalar_ratio = _scalar(residual_contribution_ratios)
        else:
            ratios = tuple(residual_contribution_ratios)

        suppressed_this_transition = False
        for agent_id in range(3):
            try:
                assignment = installed_guidance.assignment_for(agent_id)
            except KeyError:
                assignment = None
            missing = assignment is None or str(assignment.assignment_kind).upper() in {
                "", "NONE", "UNASSIGNED"
            }
            reachable = bool(assignment is not None and assignment.reachable)
            hold = bool(assignment is not None and assignment.hold_state)
            if missing:
                self.assignment_missing += 1
            if assignment is not None and not reachable:
                self.assignment_unreachable += 1
            if not missing and reachable and not hold:
                self.route_active[agent_id] += 1
            if hold:
                self.hold[agent_id] += 1
            if assignment is not None:
                identity = (
                    str(assignment.assignment_id),
                    str(assignment.assignment_kind),
                    tuple(float(v) for v in assignment.final_waypoint),
                )
                tracking = tuple(float(v) for v in assignment.tracking_waypoint)
                previous_identity = self._previous_assignment_identity[agent_id]
                previous_tracking = self._previous_tracking_waypoint[agent_id]
                if previous_identity is not None:
                    if identity != previous_identity:
                        self.assignment_switch_count += 1
                    elif tracking != previous_tracking:
                        self.tracking_subgoal_switch_count += 1
                self._previous_assignment_identity[agent_id] = identity
                self._previous_tracking_waypoint[agent_id] = tracking
            self.distance[agent_id] += float(
                np.linalg.norm(
                    np.asarray(after[agent_id].position, dtype=np.float64)
                    - np.asarray(before[agent_id].position, dtype=np.float64)
                )
            )
            collision = bool(agent_id < collisions.size and collisions[agent_id])
            if collision:
                collision_step = int(planning_state_after.step)
                if self.first_collision_step is None:
                    self.first_collision_step = collision_step
                self.last_collision_step = collision_step
                self.collision_counts[agent_id] += 1
                self.collision_streak[agent_id] += 1
                self.collision_max_streak[agent_id] = max(
                    self.collision_max_streak[agent_id], self.collision_streak[agent_id]
                )
            else:
                self.collision_streak[agent_id] = 0
            raw_norm = float(np.linalg.norm(raw[agent_id]))
            applied_norm = float(np.linalg.norm(applied[agent_id]))
            self.raw_norms.append(raw_norm)
            self.applied_norms.append(applied_norm)
            if not np.array_equal(raw[agent_id], applied[agent_id]):
                self.suppressed_agent_step_count += 1
                suppressed_this_transition = True
            route_active = bool(not missing and reachable and not hold)
            if route_active and agent_id < len(outputs):
                navigation_norm = float(
                    np.linalg.norm(
                        np.asarray(assignment.tracking_waypoint, dtype=np.float64)
                        - np.asarray(before[agent_id].position, dtype=np.float64)
                    )
                )
                residual_norm = float(
                    np.linalg.norm(_array(outputs[agent_id].residual_mix).reshape(-1, 3)[0])
                )
                if navigation_norm <= 1e-8:
                    self.alignment_zero_navigation_count += 1
                if residual_norm <= 1e-8:
                    self.alignment_zero_residual_count += 1
                alignment = _scalar(outputs[agent_id].alignment_cosine)
                if (
                    navigation_norm > 1e-8
                    and residual_norm > 1e-8
                    and math.isfinite(alignment)
                ):
                    self.alignment_valid_count += 1
                    self.negative_alignment_count += int(alignment < 0.0)
            if agent_id < len(ratios):
                ratio = _scalar(ratios[agent_id])
                if math.isfinite(ratio):
                    self.contribution_ratios.append(ratio)
        if suppressed_this_transition:
            self.suppressed_env_step_count += 1
        if scalar_ratio is not None and math.isfinite(scalar_ratio):
            self.contribution_ratios.append(scalar_ratio)
        self._latest = self._map_belief(planning_state_after)

    def summary(
        self,
        *,
        found: bool,
        max_steps: int,
        searcher_residual_off_enabled: bool = False,
    ) -> dict[str, Any]:
        initial = self._initial or (0.0, 0.0, 0.0)
        latest = self._latest or initial
        steps = self.pre_found_step_count
        agent_steps = 3 * steps
        found_step = self._found_step if found else None
        total_collisions = sum(self.collision_counts)
        total_active = sum(self.route_active)
        total_hold = sum(self.hold)
        raw_norm_mean = None if not self.raw_norms else float(np.mean(self.raw_norms))
        applied_norm_mean = None if not self.applied_norms else float(np.mean(self.applied_norms))
        waypoint_switch_count = (
            self.assignment_switch_count + self.tracking_subgoal_switch_count
        )
        result: dict[str, Any] = {
            "pre_found_step_count": int(steps),
            "found": bool(found),
            "found_step": found_step,
            "remaining_steps_after_found": None if found_step is None else max(0, int(max_steps) - found_step),
            "searcher_collision_episode_pre_found": bool(total_collisions),
            "searcher_collision_count_pre_found": int(total_collisions),
            "searcher_collision_count_pre_found_total": int(total_collisions),
            "searcher_collision_max_streak_pre_found": int(max(self.collision_max_streak, default=0)),
            "searcher_first_collision_step_pre_found": self.first_collision_step,
            "searcher_last_collision_step_pre_found": self.last_collision_step,
            "searcher_collision_agent_count_pre_found": int(sum(value > 0 for value in self.collision_counts)),
            "searcher_route_active_step_count_pre_found": int(total_active),
            "searcher_route_inactive_step_count_pre_found": int(agent_steps - total_active),
            "searcher_route_active_rate_pre_found": nullable_rate(total_active, agent_steps),
            "searcher_hold_step_count_pre_found": int(total_hold),
            "searcher_hold_rate_pre_found": nullable_rate(total_hold, agent_steps),
            "searcher_assignment_missing_step_count_pre_found": int(self.assignment_missing),
            "searcher_assignment_unreachable_step_count_pre_found": int(self.assignment_unreachable),
            "searcher_assignment_switch_count_pre_found": int(self.assignment_switch_count),
            "searcher_tracking_subgoal_switch_count_pre_found": int(self.tracking_subgoal_switch_count),
            "searcher_waypoint_switch_count_pre_found": int(waypoint_switch_count),
            "searcher_distance_travelled_pre_found": float(sum(self.distance)),
            "map_known_fraction_initial": float(initial[0]),
            "map_known_fraction_at_found_or_end": float(latest[0]),
            "map_known_fraction_gain_pre_found": float(latest[0] - initial[0]),
            "target_belief_entropy_initial": float(initial[1]),
            "target_belief_entropy_at_found_or_end": float(latest[1]),
            "target_belief_entropy_delta_pre_found": float(latest[1] - initial[1]),
            "target_belief_peak_initial": float(initial[2]),
            "target_belief_peak_at_found_or_end": float(latest[2]),
            "target_belief_peak_delta_pre_found": float(latest[2] - initial[2]),
            "searcher_residual_off_enabled": bool(searcher_residual_off_enabled),
            "searcher_raw_residual_norm_mean_pre_found": raw_norm_mean,
            "searcher_applied_residual_norm_mean_pre_found": applied_norm_mean,
            "searcher_raw_action_norm_pre_found": raw_norm_mean,
            "searcher_applied_action_norm_pre_found": applied_norm_mean,
            "searcher_residual_suppressed_env_step_count_pre_found": int(self.suppressed_env_step_count),
            "searcher_residual_suppressed_agent_step_count_pre_found": int(self.suppressed_agent_step_count),
            "searcher_residual_suppressed_step_count_pre_found": int(self.suppressed_env_step_count),
            "searcher_residual_negative_alignment_count_pre_found": int(self.negative_alignment_count),
            "searcher_residual_alignment_valid_count_pre_found": int(self.alignment_valid_count),
            "searcher_residual_alignment_zero_navigation_count_pre_found": int(self.alignment_zero_navigation_count),
            "searcher_residual_alignment_zero_residual_count_pre_found": int(self.alignment_zero_residual_count),
            "searcher_residual_negative_alignment_rate_pre_found": nullable_rate(self.negative_alignment_count, self.alignment_valid_count),
            "searcher_residual_contribution_ratio_mean_pre_found": None if not self.contribution_ratios else float(np.mean(self.contribution_ratios)),
        }
        for agent_id in range(3):
            result[f"searcher_collision_count_pre_found_agent_{agent_id}"] = int(self.collision_counts[agent_id])
            result[f"searcher_collision_max_streak_pre_found_agent_{agent_id}"] = int(self.collision_max_streak[agent_id])
            result[f"searcher_collision_streak_pre_found_agent_{agent_id}"] = int(self.collision_max_streak[agent_id])
            result[f"searcher_route_active_step_count_pre_found_agent_{agent_id}"] = int(self.route_active[agent_id])
            result[f"searcher_route_active_rate_pre_found_agent_{agent_id}"] = nullable_rate(self.route_active[agent_id], steps)
            result[f"searcher_hold_step_count_pre_found_agent_{agent_id}"] = int(self.hold[agent_id])
            result[f"searcher_hold_rate_pre_found_agent_{agent_id}"] = nullable_rate(self.hold[agent_id], steps)
            result[f"searcher_distance_travelled_pre_found_agent_{agent_id}"] = float(self.distance[agent_id])
        return result


__all__ = ("SearchContinuityDiagnostics",)
