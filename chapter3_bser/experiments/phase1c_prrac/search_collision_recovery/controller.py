"""Evaluation-only state machine for pre-found Searcher collision recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.models.prrac.stage_mapping import PRRACStage

from .detector import CollisionEdgeDetector
from .planner import plan_route_refresh, select_egress_route
from .types import RecoveryNavigationPlan, RecoveryStepSnapshot, SearchRecoveryMode, SearchRecoveryVariant


@dataclass
class _AgentRecovery:
    mode: SearchRecoveryMode = SearchRecoveryMode.NORMAL_SEARCH
    plan: RecoveryNavigationPlan | None = None
    attempt_id: int = 0
    trigger_reason: str | None = None
    failure_reason: str | None = None
    pending_refresh: bool = False
    pending_egress: bool = False
    failed_endpoint_cells: set[int] = field(default_factory=set)
    recovery_start_step: int | None = None
    last_duration: int = 0
    failed_direction: tuple[float, float, float] | None = None


class SearchCollisionRecoveryController:
    """Consumes current public collision outcomes and prepares next-step guidance."""

    def __init__(self, variant: SearchRecoveryVariant, *, path_tracker: PathTracker | None = None) -> None:
        if variant is SearchRecoveryVariant.S2A_C0_BASELINE:
            raise ValueError("C0 must not create SearchCollisionRecoveryController state")
        self.variant = variant
        self.detector = CollisionEdgeDetector()
        self.path_tracker = path_tracker or PathTracker()
        self.agents = {agent_id: _AgentRecovery() for agent_id in range(3)}
        self._step_refresh_attempted = {agent_id: False for agent_id in range(3)}
        self._step_egress_attempted = {agent_id: False for agent_id in range(3)}
        self._counts: dict[str, int] = {}
        self._agent_counts = {agent_id: {} for agent_id in range(3)}
        self._active_step_count = 0
        self._observed_search_steps = 0
        self._durations: list[int] = []
        self._recovery_collisions = 0
        self._recovery_max_collision_streak = 0
        self._last_step = 0
        self.rejoined_agent_ids: tuple[int, ...] = ()

    def _increment(self, name: str, agent_id: int | None = None) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1
        if agent_id is not None:
            bucket = self._agent_counts[int(agent_id)]
            bucket[name] = bucket.get(name, 0) + 1

    def _begin(self, agent_id: int, step: int) -> None:
        state = self.agents[agent_id]
        state.failed_endpoint_cells.clear()
        state.attempt_id += 1
        state.recovery_start_step = int(step)
        state.trigger_reason = "COLLISION_EDGE"
        state.failure_reason = None
        self._increment("search_recovery_entry_count", agent_id)

    def _end(self, agent_id: int, step: int, *, mode: SearchRecoveryMode = SearchRecoveryMode.NORMAL_SEARCH) -> None:
        state = self.agents[agent_id]
        if state.recovery_start_step is not None:
            duration = max(1, int(step) - state.recovery_start_step + 1)
            state.last_duration = duration
            self._durations.append(duration)
        state.mode = mode
        state.plan = None
        state.pending_refresh = False
        state.pending_egress = False
        state.recovery_start_step = None
        self.path_tracker.reset(agent_id)

    def terminate_all(self, step: int) -> None:
        for agent_id in range(3):
            self._end(agent_id, step)

    def observe_transition(
        self,
        *,
        stage_before: int | PRRACStage,
        planning_state_after: Any,
        collision_flags: Iterable[Any],
        planning_state_before: Any | None = None,
        installed_guidance_before: Any | None = None,
    ) -> None:
        self._step_refresh_attempted = {agent_id: False for agent_id in range(3)}
        self._step_egress_attempted = {agent_id: False for agent_id in range(3)}
        search_active = bool(
            int(stage_before) == int(PRRACStage.SEARCH)
            and not planning_state_after.target_found
            and not planning_state_after.mission_complete
        )
        step = int(planning_state_after.step)
        self._last_step = step
        rejoined: list[int] = []
        self.rejoined_agent_ids = ()
        collisions = np.asarray(tuple(collision_flags), dtype=np.bool_).reshape(-1)
        if not search_active:
            for agent_id in range(3):
                self.detector.observe(agent_id, False, search_active=False)
            self.terminate_all(step)
            return
        self._observed_search_steps += 1
        if any(state.mode is not SearchRecoveryMode.NORMAL_SEARCH for state in self.agents.values()):
            self._active_step_count += 1
        agents_by_id = {int(item.agent_id): item for item in planning_state_after.agents}
        for agent_id in range(3):
            collision = bool(agent_id < collisions.size and collisions[agent_id])
            edge = self.detector.observe(agent_id, collision, search_active=True)
            state = self.agents[agent_id]
            if collision and planning_state_before is not None and installed_guidance_before is not None:
                before_agents = {int(item.agent_id): item for item in planning_state_before.agents}
                tracking = installed_guidance_before.assignment_for(agent_id).tracking_waypoint
                direction = np.asarray(tracking, dtype=np.float64) - np.asarray(before_agents[agent_id].position, dtype=np.float64)
                state.failed_direction = tuple(float(value) for value in direction)
            if state.mode is not SearchRecoveryMode.NORMAL_SEARCH and collision:
                self._recovery_collisions += 1
                self._recovery_max_collision_streak = max(
                    self._recovery_max_collision_streak, self.detector.streak(agent_id)
                )
            if state.mode is SearchRecoveryMode.NORMAL_SEARCH and edge:
                self._begin(agent_id, step)
                state.mode = SearchRecoveryMode.COLLISION_EDGE_DETECTED
                state.pending_refresh = True
                continue
            if state.mode is SearchRecoveryMode.ROUTE_REFRESH:
                if collision and self.variant is SearchRecoveryVariant.S2A_C2_EGRESS_ROUTE:
                    state.pending_egress = True
                elif not collision:
                    self._end(agent_id, step)
                continue
            if state.mode is SearchRecoveryMode.EGRESS_ROUTE:
                if collision:
                    if state.plan is not None and state.plan.endpoint_cell_index is not None:
                        state.failed_endpoint_cells.add(int(state.plan.endpoint_cell_index))
                    state.failure_reason = "EGRESS_COLLISION"
                    self._increment("egress_failure_count", agent_id)
                    state.pending_egress = True
                elif state.plan is not None:
                    position = np.asarray(agents_by_id[agent_id].position, dtype=np.float64)
                    endpoint = np.asarray(state.plan.navigation_endpoint, dtype=np.float64)
                    if float(np.linalg.norm(position - endpoint)) < self.path_tracker.threshold:
                        self._increment("egress_success_count", agent_id)
                        self._increment("egress_rejoin_count", agent_id)
                        state.mode = SearchRecoveryMode.REJOIN_SEARCH
                        self._end(agent_id, step)
                        rejoined.append(agent_id)
                continue
            if state.mode is SearchRecoveryMode.RECOVERY_NO_EGRESS and not collision:
                self._end(agent_id, step)
        self.rejoined_agent_ids = tuple(rejoined)

    @staticmethod
    def _semantic(base_guidance: Any, agent_id: int):
        assignment = base_guidance.assignment_for(agent_id)
        return str(assignment.assignment_id), tuple(float(v) for v in assignment.final_waypoint), assignment

    def prepare_next_guidance(self, planning_state: Any, base_guidance: Any) -> None:
        """Resolve pending public plans. Overlay application is a separate pure step."""

        for agent_id, state in self.agents.items():
            if state.pending_refresh:
                state.pending_refresh = False
                self._step_refresh_attempted[agent_id] = True
                self._increment("route_refresh_attempt_count", agent_id)
                candidate_id, waypoint, _ = self._semantic(base_guidance, agent_id)
                plan, failure = plan_route_refresh(planning_state, agent_id, candidate_id, waypoint)
                if plan is not None:
                    state.plan = plan
                    state.mode = SearchRecoveryMode.ROUTE_REFRESH
                    self._increment("route_refresh_success_count", agent_id)
                else:
                    state.failure_reason = failure
                    self._increment("route_refresh_failure_count", agent_id)
                    if self.variant is SearchRecoveryVariant.S2A_C2_EGRESS_ROUTE:
                        state.pending_egress = True
                    else:
                        self._end(agent_id, int(planning_state.step))
            if state.pending_egress:
                state.pending_egress = False
                self._step_egress_attempted[agent_id] = True
                self._increment("egress_attempt_count", agent_id)
                candidate_id, waypoint, _ = self._semantic(base_guidance, agent_id)
                plan, failure = select_egress_route(
                    planning_state,
                    agent_id,
                    candidate_id,
                    waypoint,
                    failed_endpoint_cells=state.failed_endpoint_cells,
                    failed_direction=state.failed_direction,
                )
                if plan is None:
                    state.plan = None
                    state.mode = SearchRecoveryMode.RECOVERY_NO_EGRESS
                    state.failure_reason = failure
                    self._increment("egress_failure_count", agent_id)
                    self._increment("recovery_no_egress_count", agent_id)
                    self._increment("recovery_failed_endpoint_count", agent_id)
                else:
                    state.plan = plan
                    state.mode = SearchRecoveryMode.EGRESS_ROUTE
                    state.failure_reason = None
                    self.path_tracker.reset(agent_id)

    def snapshot(self) -> RecoveryStepSnapshot:
        def plan_value(agent_id: int, name: str):
            plan = self.agents[agent_id].plan
            return None if plan is None else getattr(plan, name)
        return RecoveryStepSnapshot(
            active_agent_ids=tuple(agent_id for agent_id, state in self.agents.items() if state.mode is not SearchRecoveryMode.NORMAL_SEARCH),
            mode_by_agent={agent_id: state.mode.value for agent_id, state in self.agents.items()},
            attempt_id_by_agent={agent_id: state.attempt_id for agent_id, state in self.agents.items()},
            collision_streak_by_agent={agent_id: self.detector.streak(agent_id) for agent_id in range(3)},
            semantic_candidate_id_by_agent={agent_id: plan_value(agent_id, "semantic_search_candidate_id") for agent_id in range(3)},
            semantic_waypoint_by_agent={agent_id: plan_value(agent_id, "semantic_search_waypoint") for agent_id in range(3)},
            navigation_endpoint_by_agent={agent_id: plan_value(agent_id, "navigation_endpoint") for agent_id in range(3)},
            endpoint_cell_index_by_agent={agent_id: plan_value(agent_id, "endpoint_cell_index") for agent_id in range(3)},
            trigger_reason_by_agent={agent_id: state.trigger_reason for agent_id, state in self.agents.items()},
            failure_reason_by_agent={agent_id: state.failure_reason for agent_id, state in self.agents.items()},
            route_refresh_attempted_by_agent=dict(self._step_refresh_attempted),
            egress_attempted_by_agent=dict(self._step_egress_attempted),
        )

    def summary(self) -> dict[str, Any]:
        durations = list(self._durations)
        for state in self.agents.values():
            if state.recovery_start_step is not None:
                durations.append(max(1, self._last_step - state.recovery_start_step + 1))
        result: dict[str, Any] = {
            "search_recovery_entry_count": self._counts.get("search_recovery_entry_count", 0),
            "search_recovery_active_step_count": self._active_step_count,
            "search_recovery_active_rate": None if self._observed_search_steps == 0 else self._active_step_count / self._observed_search_steps,
            "route_refresh_attempt_count": self._counts.get("route_refresh_attempt_count", 0),
            "route_refresh_success_count": self._counts.get("route_refresh_success_count", 0),
            "route_refresh_failure_count": self._counts.get("route_refresh_failure_count", 0),
            "egress_attempt_count": self._counts.get("egress_attempt_count", 0),
            "egress_success_count": self._counts.get("egress_success_count", 0),
            "egress_failure_count": self._counts.get("egress_failure_count", 0),
            "egress_rejoin_count": self._counts.get("egress_rejoin_count", 0),
            "recovery_collision_count": self._recovery_collisions,
            "recovery_max_collision_streak": self._recovery_max_collision_streak,
            "recovery_duration_sum": sum(durations),
            "recovery_duration_mean": None if not durations else float(np.mean(durations)),
            "recovery_duration_max": 0 if not durations else max(durations),
            "recovery_no_egress_count": self._counts.get("recovery_no_egress_count", 0),
            "recovery_failed_endpoint_count": self._counts.get("recovery_failed_endpoint_count", 0),
        }
        for agent_id in range(3):
            counts = self._agent_counts[agent_id]
            for name in ("search_recovery_entry_count", "route_refresh_attempt_count", "route_refresh_success_count", "egress_attempt_count", "egress_success_count"):
                result[f"{name}_agent_{agent_id}"] = counts.get(name, 0)
            result[f"last_recovery_mode_agent_{agent_id}"] = self.agents[agent_id].mode.value
            result[f"last_recovery_failure_reason_agent_{agent_id}"] = self.agents[agent_id].failure_reason
        return result


def build_search_recovery_controller(variant: SearchRecoveryVariant):
    from .types_v2 import SearchRecoveryVariantV2

    if variant in {SearchRecoveryVariant.S2A_C0_BASELINE, SearchRecoveryVariantV2.S2A1_C0_BASELINE}:
        return None
    if variant in {
        SearchRecoveryVariantV2.S2A1_C1_FORCED_REFRESH,
        SearchRecoveryVariantV2.S2A1_C2_LOCAL_CONNECTOR,
    }:
        from .controller_v2 import SearchCollisionRecoveryControllerV2

        return SearchCollisionRecoveryControllerV2(variant)
    return SearchCollisionRecoveryController(variant)


__all__ = ("SearchCollisionRecoveryController", "build_search_recovery_controller")
