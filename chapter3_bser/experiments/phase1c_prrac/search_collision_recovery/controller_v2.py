"""S2-A.1 forced-refresh and public local-connector recovery controller."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import numpy as np

from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.models.prrac.stage_mapping import PRRACStage
from core.mapping.travel_cost_service import TravelCostService

from .detector import CollisionEdgeDetector
from .planner import _plan
from .planner_v2 import canonical_path_hash, plan_forced_route_refresh, plan_local_connector, public_cell_index
from .types import RecoveryStepSnapshot
from .types_v2 import ActivationAuditStep, LastCollisionFreeState, LocalConnectorPlan, RecoveryModeV2, SearchRecoveryVariantV2


@dataclass
class _AgentRecoveryV2:
    mode: RecoveryModeV2 = RecoveryModeV2.NORMAL_SEARCH
    plan: Any | None = None
    attempt_id: int = 0
    trigger_reason: str | None = None
    failure_reason: str | None = None
    pending_route_refresh: bool = False
    pending_local_connector: bool = False
    pending_graph_reconnect: bool = False
    route_refresh_identical: bool = False
    failed_endpoint_cells: set[int] = field(default_factory=set)
    failed_direction: tuple[float, float, float] | None = None
    semantic_candidate_id: str | None = None
    semantic_assignment_kind: str | None = None
    semantic_waypoint: tuple[float, float, float] | None = None
    base_tracking_waypoint: tuple[float, float, float] | None = None
    base_planned_path: tuple[tuple[float, float, float], ...] = ()
    base_allocation_version: str | None = None
    base_allocation_hash: str | None = None
    collision_position_before: tuple[float, float, float] | None = None
    collision_position_after: tuple[float, float, float] | None = None
    recovery_start_step: int | None = None
    last_terminal_mode: str = RecoveryModeV2.NORMAL_SEARCH.value
    plan_install_pending: bool = False


class SearchCollisionRecoveryControllerV2:
    """Prepare next-transition Searcher guidance without modifying actions or Executor."""

    def __init__(self, variant: SearchRecoveryVariantV2, *, path_tracker: PathTracker | None = None) -> None:
        if variant not in {SearchRecoveryVariantV2.S2A1_C1_FORCED_REFRESH, SearchRecoveryVariantV2.S2A1_C2_LOCAL_CONNECTOR}:
            raise ValueError("the S2-A.1 controller accepts only C1/C2 v2 variants")
        self.variant = variant
        self.detector = CollisionEdgeDetector()
        self.path_tracker = path_tracker or PathTracker()
        self.agents = {agent_id: _AgentRecoveryV2() for agent_id in range(3)}
        self.last_collision_free = {agent_id: None for agent_id in range(3)}
        self._counts: dict[str, int] = {}
        self._agent_counts = {agent_id: {} for agent_id in range(3)}
        self._durations: list[int] = []
        self._recovery_collision_count = 0
        self._recovery_max_collision_streak = 0
        self._observed_search_steps = 0
        self._last_step = 0
        self._tracking_deltas: list[float] = []
        self._planning_audits: list[dict[str, Any]] = []
        self._activation_steps: list[dict[str, Any]] = []
        self._step_route_attempted = {agent_id: False for agent_id in range(3)}
        self._step_local_attempted = {agent_id: False for agent_id in range(3)}
        self.force_refresh_requested = False
        self.rejoined_agent_ids: tuple[int, ...] = ()

    def _increment(self, name: str, amount: int = 1, agent_id: int | None = None) -> None:
        self._counts[name] = self._counts.get(name, 0) + int(amount)
        if agent_id is not None:
            bucket = self._agent_counts[int(agent_id)]
            bucket[name] = bucket.get(name, 0) + int(amount)

    @staticmethod
    def _assignment(guidance: Any, agent_id: int):
        return guidance.assignment_for(int(agent_id))

    def _request_force_refresh(self) -> None:
        if not self.force_refresh_requested:
            self.force_refresh_requested = True
            self._increment("forced_public_refresh_count")

    def _begin(self, agent_id: int, step: int, state_before: Any, state_after: Any, guidance: Any) -> None:
        state = self.agents[agent_id]
        assignment = self._assignment(guidance, agent_id)
        before_agent = {int(item.agent_id): item for item in state_before.agents}[agent_id]
        after_agent = {int(item.agent_id): item for item in state_after.agents}[agent_id]
        state.attempt_id += 1
        state.trigger_reason = "COLLISION_EDGE"
        state.failure_reason = None
        state.failed_endpoint_cells.clear()
        state.semantic_candidate_id = str(assignment.assignment_id)
        state.semantic_assignment_kind = str(assignment.assignment_kind)
        state.semantic_waypoint = tuple(float(value) for value in assignment.final_waypoint)
        state.base_tracking_waypoint = tuple(float(value) for value in assignment.tracking_waypoint)
        state.base_planned_path = tuple(tuple(float(value) for value in point) for point in assignment.planned_path)
        state.base_allocation_version = str(guidance.allocation_version)
        state.base_allocation_hash = str(guidance.allocation_hash)
        state.failed_direction = tuple(
            float(value) for value in (
                np.asarray(assignment.tracking_waypoint, dtype=np.float64)
                - np.asarray(before_agent.position, dtype=np.float64)
            )
        )
        state.collision_position_before = tuple(float(value) for value in before_agent.position)
        state.collision_position_after = tuple(float(value) for value in after_agent.position)
        state.recovery_start_step = int(step)
        state.mode = RecoveryModeV2.COLLISION_EDGE_DETECTED
        state.pending_route_refresh = True
        # These are deliberately retained as diagnostics: rejected actions can
        # leave before/after positions identical.
        state.last_terminal_mode = (
            f"COLLISION_EDGE before={tuple(before_agent.position)} after={tuple(after_agent.position)}"
        )
        self._increment("search_recovery_entry_count", agent_id=agent_id)
        self._request_force_refresh()

    def _end(self, agent_id: int, step: int, terminal: RecoveryModeV2 = RecoveryModeV2.NORMAL_SEARCH) -> None:
        state = self.agents[agent_id]
        if state.recovery_start_step is not None:
            self._durations.append(max(1, int(step) - state.recovery_start_step + 1))
        state.last_terminal_mode = terminal.value
        state.mode = RecoveryModeV2.NORMAL_SEARCH
        state.plan = None
        state.pending_route_refresh = False
        state.pending_local_connector = False
        state.pending_graph_reconnect = False
        state.route_refresh_identical = False
        state.recovery_start_step = None
        state.plan_install_pending = False
        self.path_tracker.reset(agent_id)

    def _pass_through(self, agent_id: int, step: int, reason: str) -> None:
        state = self.agents[agent_id]
        state.failure_reason = str(reason)
        self._increment("recovery_failed_pass_through_count")
        self._end(agent_id, step, RecoveryModeV2.RECOVERY_FAILED_PASS_THROUGH)

    def terminate_all(self, step: int) -> None:
        for agent_id in range(3):
            self._end(agent_id, step)
        self.force_refresh_requested = False

    def _update_last_safe(self, planning_state: Any, guidance: Any, collisions: np.ndarray) -> None:
        agents = {int(item.agent_id): item for item in planning_state.agents}
        for agent_id in range(3):
            position = np.asarray(agents[agent_id].position, dtype=np.float64)
            if agent_id < collisions.size and bool(collisions[agent_id]):
                continue
            if not np.all(np.isfinite(position)):
                continue
            assignment = self._assignment(guidance, agent_id)
            self.last_collision_free[agent_id] = LastCollisionFreeState(
                position=tuple(float(value) for value in position),
                velocity=tuple(float(value) for value in agents[agent_id].velocity),
                step=int(planning_state.step),
                cell_index=public_cell_index(planning_state, position),
                tracking_waypoint=tuple(float(value) for value in assignment.tracking_waypoint),
            )

    def observe_transition(
        self,
        *,
        stage_before: int | PRRACStage,
        planning_state_after: Any,
        collision_flags: Iterable[Any],
        planning_state_before: Any | None = None,
        installed_guidance_before: Any | None = None,
    ) -> None:
        self.force_refresh_requested = False
        self._step_route_attempted = {agent_id: False for agent_id in range(3)}
        self._step_local_attempted = {agent_id: False for agent_id in range(3)}
        step = int(planning_state_after.step)
        self._last_step = step
        collisions = np.asarray(tuple(collision_flags), dtype=np.bool_).reshape(-1)
        search_active = bool(
            int(stage_before) == int(PRRACStage.SEARCH)
            and not planning_state_after.target_found
            and not planning_state_after.mission_complete
        )
        if not search_active:
            for agent_id in range(3):
                self.detector.observe(agent_id, False, search_active=False)
            self.terminate_all(step)
            return
        if planning_state_before is None or installed_guidance_before is None:
            raise ValueError("S2-A.1 requires before-state and installed guidance")
        self._observed_search_steps += 1
        if any(state.mode is not RecoveryModeV2.NORMAL_SEARCH for state in self.agents.values()):
            self._increment("recovery_state_non_normal_step_count")
        self._update_last_safe(planning_state_after, installed_guidance_before, collisions)
        agents_after = {int(item.agent_id): item for item in planning_state_after.agents}
        rejoined = []
        for agent_id in range(3):
            collision = bool(agent_id < collisions.size and collisions[agent_id])
            edge = self.detector.observe(agent_id, collision, search_active=True)
            state = self.agents[agent_id]
            if state.mode is not RecoveryModeV2.NORMAL_SEARCH and collision:
                self._recovery_collision_count += 1
                self._recovery_max_collision_streak = max(
                    self._recovery_max_collision_streak, self.detector.streak(agent_id)
                )
            if state.mode is RecoveryModeV2.NORMAL_SEARCH and edge:
                self._begin(agent_id, step, planning_state_before, planning_state_after, installed_guidance_before)
                continue
            if state.mode is RecoveryModeV2.ROUTE_REFRESH:
                if collision and self.variant is SearchRecoveryVariantV2.S2A1_C2_LOCAL_CONNECTOR:
                    state.pending_local_connector = True
                    state.plan = None
                    self.path_tracker.reset(agent_id)
                    self._request_force_refresh()
                elif collision:
                    self._pass_through(agent_id, step, "RECOVERY_FAILED_PASS_THROUGH")
                else:
                    self._end(agent_id, step, RecoveryModeV2.REJOIN_SEARCH)
                continue
            if state.mode is RecoveryModeV2.LOCAL_CONNECTOR_EGRESS:
                if collision:
                    self._record_local_egress_collision(state, planning_state_after, agent_id)
                    if state.plan is not None and state.plan.endpoint_cell_index is not None:
                        state.failed_endpoint_cells.add(int(state.plan.endpoint_cell_index))
                    state.failure_reason = "LOCAL_EGRESS_COLLISION"
                    state.plan = None
                    self.path_tracker.reset(agent_id)
                    self._increment("local_connector_collision_count", agent_id=agent_id)
                    state.pending_local_connector = True
                    self._request_force_refresh()
                elif state.plan is not None:
                    endpoint = np.asarray(state.plan.navigation_endpoint, dtype=np.float64)
                    position = np.asarray(agents_after[agent_id].position, dtype=np.float64)
                    if float(np.linalg.norm(position - endpoint)) < self.path_tracker.threshold:
                        self._increment("local_connector_reached_count", agent_id=agent_id)
                        state.plan = None
                        state.pending_graph_reconnect = True
                        state.mode = RecoveryModeV2.GRAPH_RECONNECT
                        self.path_tracker.reset(agent_id)
                        self._request_force_refresh()
                continue
            if state.mode is RecoveryModeV2.REJOIN_SEARCH:
                if collision and self.variant is SearchRecoveryVariantV2.S2A1_C2_LOCAL_CONNECTOR:
                    state.plan = None
                    state.pending_local_connector = True
                    self.path_tracker.reset(agent_id)
                    self._request_force_refresh()
                elif not collision:
                    self._end(agent_id, step, RecoveryModeV2.REJOIN_SEARCH)
                    rejoined.append(agent_id)
        self.rejoined_agent_ids = tuple(rejoined)

    def _record_audit(self, state: _AgentRecoveryV2, audit: Any) -> None:
        row = audit.scalar_row()
        row.update({"variant": self.variant.value, "attempt_id": state.attempt_id})
        self._planning_audits.append(row)

    def _record_local_egress_collision(
        self, state: _AgentRecoveryV2, planning_state: Any, agent_id: int
    ) -> None:
        """Capture the failed installed connector before its state is cleared."""

        plan = state.plan
        if not isinstance(plan, LocalConnectorPlan):
            return
        self._planning_audits.append(
            {
                "planning_state_step": int(planning_state.step),
                "step": int(planning_state.step),
                "planning_state_revision": int(planning_state.map_revision),
                "planning_stage": "LOCAL_CONNECTOR_EGRESS",
                "final_failure_stage": "LOCAL_CONNECTOR_EGRESS",
                "final_failure_reason": "LOCAL_EGRESS_COLLISION",
                "variant": self.variant.value,
                "agent_id": int(agent_id),
                "attempt_id": int(state.attempt_id),
                "plan_source": str(plan.source),
                "selected_endpoint": tuple(plan.local_endpoint),
                "endpoint_cell_index": plan.endpoint_cell_index,
                "selected_tier": int(plan.endpoint_tier),
                "segment_audit": dict(plan.public_segment_audit.__dict__),
                "base_path_hash": str(plan.base_path_hash),
                "overlay_path_hash": str(plan.overlay_path_hash),
                "guidance_changed": bool(plan.base_path_hash != plan.overlay_path_hash),
            }
        )

    def _semantic_values(self, state: _AgentRecoveryV2):
        if state.semantic_candidate_id is None or state.semantic_waypoint is None or state.base_tracking_waypoint is None:
            raise RuntimeError("recovery semantic identity was not captured at collision edge")
        return state.semantic_candidate_id, state.semantic_waypoint, state.base_tracking_waypoint

    def _install_plan(self, agent_id: int, plan: Any, mode: RecoveryModeV2) -> None:
        state = self.agents[agent_id]
        state.plan = plan
        state.mode = mode
        state.failure_reason = None
        state.plan_install_pending = True
        self.path_tracker.reset(agent_id)

    def prepare_next_guidance(self, planning_state: Any, base_guidance: Any) -> None:
        """Resolve requested plans from the evaluator's final post-transition public snapshot."""

        for agent_id, state in self.agents.items():
            if state.pending_route_refresh:
                state.pending_route_refresh = False
                state.mode = RecoveryModeV2.ROUTE_REFRESH
                self._step_route_attempted[agent_id] = True
                self._increment("route_refresh_attempt_count", agent_id=agent_id)
                candidate_id, semantic, base_tracking = self._semantic_values(state)
                plan, audit, failure = plan_forced_route_refresh(
                    planning_state, agent_id, candidate_id, semantic, base_tracking,
                    self.last_collision_free[agent_id], attempt_id=state.attempt_id,
                )
                if plan is None:
                    state.failure_reason = failure
                    self._increment("route_refresh_failure_count", agent_id=agent_id)
                    self._record_audit(state, audit)
                    if self.variant is SearchRecoveryVariantV2.S2A1_C2_LOCAL_CONNECTOR:
                        state.pending_local_connector = True
                    else:
                        self._pass_through(agent_id, int(planning_state.step), failure or "RECOVERY_FAILED_PASS_THROUGH")
                else:
                    self._increment("route_refresh_success_count", agent_id=agent_id)
                    agent = {int(item.agent_id): item for item in planning_state.agents}[agent_id]
                    probe = PathTracker(threshold=self.path_tracker.threshold)
                    tracking = probe.tracking_target(agent_id, agent.position, plan.path, plan.navigation_endpoint)
                    identical = (
                        canonical_path_hash(plan.path) == canonical_path_hash(state.base_planned_path)
                        and np.allclose(tracking, base_tracking, atol=1e-12, rtol=0.0)
                    )
                    if identical:
                        state.route_refresh_identical = True
                        state.failure_reason = "ROUTE_REFRESH_IDENTICAL_TO_BASE"
                        self._increment("route_refresh_identical_to_base_count")
                        self._record_audit(state, replace(audit, final_failure_stage="ROUTE_REFRESH",
                                                          final_failure_reason="ROUTE_REFRESH_IDENTICAL_TO_BASE",
                                                          base_path_hash=canonical_path_hash(state.base_planned_path),
                                                          overlay_path_hash=canonical_path_hash(plan.path)))
                        if self.variant is SearchRecoveryVariantV2.S2A1_C1_FORCED_REFRESH:
                            self._pass_through(agent_id, int(planning_state.step), "ROUTE_REFRESH_IDENTICAL_TO_BASE")
                    else:
                        self._record_audit(state, replace(audit,
                                                          base_path_hash=canonical_path_hash(state.base_planned_path),
                                                          overlay_path_hash=canonical_path_hash(plan.path)))
                        self._install_plan(agent_id, plan, RecoveryModeV2.ROUTE_REFRESH)
            if state.pending_local_connector:
                state.pending_local_connector = False
                self._step_local_attempted[agent_id] = True
                self._increment("local_connector_attempt_count", agent_id=agent_id)
                candidate_id, semantic, base_tracking = self._semantic_values(state)
                plan, audit, failure = plan_local_connector(
                    planning_state, agent_id, candidate_id, semantic, base_tracking,
                    state.base_planned_path, self.last_collision_free[agent_id],
                    failed_endpoint_cells=state.failed_endpoint_cells,
                    failed_direction=state.failed_direction, attempt_id=state.attempt_id,
                )
                self._record_audit(state, audit)
                if plan is None:
                    state.failure_reason = failure
                    self._pass_through(agent_id, int(planning_state.step), failure or "RECOVERY_FAILED_PASS_THROUGH")
                else:
                    self._increment("local_connector_plan_count", agent_id=agent_id)
                    self._increment(f"tier{plan.endpoint_tier}_count")
                    self._install_plan(agent_id, plan, RecoveryModeV2.LOCAL_CONNECTOR_EGRESS)
            if state.pending_graph_reconnect:
                state.pending_graph_reconnect = False
                self._increment("graph_reconnect_attempt_count", agent_id=agent_id)
                candidate_id, semantic, base_tracking = self._semantic_values(state)
                plan, audit, failure = plan_forced_route_refresh(
                    planning_state, agent_id, candidate_id, semantic, base_tracking,
                    self.last_collision_free[agent_id], attempt_id=state.attempt_id,
                )
                audit = replace(audit, planning_stage="GRAPH_RECONNECT")
                if plan is None:
                    self._increment("graph_reconnect_failure_count", agent_id=agent_id)
                    endpoint = public_cell_index(planning_state, {int(item.agent_id): item for item in planning_state.agents}[agent_id].position)
                    if endpoint is not None:
                        state.failed_endpoint_cells.add(endpoint)
                    state.failure_reason = "GRAPH_RECONNECT_UNREACHABLE"
                    self._record_audit(state, replace(audit, final_failure_stage="GRAPH_RECONNECT",
                                                      final_failure_reason="GRAPH_RECONNECT_UNREACHABLE"))
                    state.pending_local_connector = True
                    # Resolve the next deterministic endpoint in this same refreshed view.
                    self.prepare_next_guidance(planning_state, base_guidance)
                else:
                    self._increment("graph_reconnect_success_count", agent_id=agent_id)
                    self._record_audit(state, audit)
                    self._install_plan(agent_id, replace(plan, source="SEARCH_COLLISION_GRAPH_RECONNECT_V2"), RecoveryModeV2.REJOIN_SEARCH)

    def observe_activation(self, base_guidance: Any, overlay_guidance: Any) -> None:
        any_plan = any(self.agents[agent_id].plan is not None for agent_id in range(3))
        any_changed = False
        step_rows = []
        for agent_id in range(3):
            base = self._assignment(base_guidance, agent_id)
            overlay = self._assignment(overlay_guidance, agent_id)
            base_hash = canonical_path_hash(base.planned_path)
            overlay_hash = canonical_path_hash(overlay.planned_path)
            delta = float(np.linalg.norm(np.asarray(overlay.tracking_waypoint) - np.asarray(base.tracking_waypoint)))
            path_changed = base_hash != overlay_hash
            changed = bool(path_changed or delta > 1e-12 or tuple(base.final_waypoint) != tuple(overlay.final_waypoint))
            state = self.agents[agent_id]
            plan = state.plan
            effective = bool(state.plan_install_pending and plan is not None and changed)
            if effective:
                self._increment("recovery_effective_intervention_count", agent_id=agent_id)
                state.plan_install_pending = False
            any_changed = any_changed or changed
            if plan is not None or changed or state.mode is not RecoveryModeV2.NORMAL_SEARCH:
                row = ActivationAuditStep(
                    base_tracking_waypoint=tuple(base.tracking_waypoint), overlay_tracking_waypoint=tuple(overlay.tracking_waypoint),
                    tracking_waypoint_delta_norm=delta, base_final_waypoint=tuple(base.final_waypoint),
                    overlay_final_waypoint=tuple(overlay.final_waypoint), base_path_hash=base_hash,
                    overlay_path_hash=overlay_hash, path_changed=path_changed, guidance_changed=changed,
                    recovery_plan_installed=plan is not None,
                    recovery_plan_source=None if plan is None else str(plan.source),
                    recovery_endpoint_tier=None if not isinstance(plan, LocalConnectorPlan) else plan.endpoint_tier,
                    recovery_endpoint_cell_index=None if plan is None else plan.endpoint_cell_index,
                    recovery_effective_intervention=effective,
                    route_refresh_identical_to_base=state.route_refresh_identical,
                )
                step_rows.append(
                    {
                        "step": int(self._last_step),
                        "agent_id": agent_id,
                        "attempt_id": int(state.attempt_id),
                        "recovery_mode": state.mode.value,
                        **row.__dict__,
                    }
                )
                for planning_row in reversed(self._planning_audits):
                    if int(planning_row.get("agent_id", -1)) == agent_id and int(planning_row.get("attempt_id", -1)) == state.attempt_id:
                        planning_row["base_path_hash"] = base_hash
                        planning_row["overlay_path_hash"] = overlay_hash
                        planning_row["guidance_changed"] = changed
                        break
                if delta > 0.0:
                    self._tracking_deltas.append(delta)
        if any_plan:
            self._increment("recovery_plan_active_step_count")
        if any_changed:
            self._increment("recovery_guidance_changed_step_count")
        if any(row["path_changed"] for row in step_rows):
            self._increment("path_changed_step_count")
        self._activation_steps.extend(step_rows)

    def snapshot(self) -> RecoveryStepSnapshot:
        def plan_value(agent_id: int, name: str):
            plan = self.agents[agent_id].plan
            return None if plan is None else getattr(plan, name)
        return RecoveryStepSnapshot(
            active_agent_ids=tuple(agent_id for agent_id, state in self.agents.items() if state.mode is not RecoveryModeV2.NORMAL_SEARCH),
            mode_by_agent={agent_id: state.mode.value for agent_id, state in self.agents.items()},
            attempt_id_by_agent={agent_id: state.attempt_id for agent_id, state in self.agents.items()},
            collision_streak_by_agent={agent_id: self.detector.streak(agent_id) for agent_id in range(3)},
            semantic_candidate_id_by_agent={agent_id: self.agents[agent_id].semantic_candidate_id for agent_id in range(3)},
            semantic_waypoint_by_agent={agent_id: self.agents[agent_id].semantic_waypoint for agent_id in range(3)},
            navigation_endpoint_by_agent={agent_id: plan_value(agent_id, "navigation_endpoint") for agent_id in range(3)},
            endpoint_cell_index_by_agent={agent_id: plan_value(agent_id, "endpoint_cell_index") for agent_id in range(3)},
            trigger_reason_by_agent={agent_id: state.trigger_reason for agent_id, state in self.agents.items()},
            failure_reason_by_agent={agent_id: state.failure_reason for agent_id, state in self.agents.items()},
            route_refresh_attempted_by_agent=dict(self._step_route_attempted),
            egress_attempted_by_agent=dict(self._step_local_attempted),
        )

    def planning_failure_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._planning_audits]

    def activation_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._activation_steps]

    def summary(self) -> dict[str, Any]:
        deltas = self._tracking_deltas
        result = {
            "search_recovery_entry_count": self._counts.get("search_recovery_entry_count", 0),
            "forced_public_refresh_count": self._counts.get("forced_public_refresh_count", 0),
            "route_refresh_attempt_count": self._counts.get("route_refresh_attempt_count", 0),
            "route_refresh_success_count": self._counts.get("route_refresh_success_count", 0),
            "route_refresh_failure_count": self._counts.get("route_refresh_failure_count", 0),
            "route_refresh_identical_to_base_count": self._counts.get("route_refresh_identical_to_base_count", 0),
            "local_connector_attempt_count": self._counts.get("local_connector_attempt_count", 0),
            "local_connector_plan_count": self._counts.get("local_connector_plan_count", 0),
            "local_connector_reached_count": self._counts.get("local_connector_reached_count", 0),
            "local_connector_collision_count": self._counts.get("local_connector_collision_count", 0),
            "graph_reconnect_attempt_count": self._counts.get("graph_reconnect_attempt_count", 0),
            "graph_reconnect_success_count": self._counts.get("graph_reconnect_success_count", 0),
            "graph_reconnect_failure_count": self._counts.get("graph_reconnect_failure_count", 0),
            "recovery_state_non_normal_step_count": self._counts.get("recovery_state_non_normal_step_count", 0),
            "recovery_plan_active_step_count": self._counts.get("recovery_plan_active_step_count", 0),
            "recovery_guidance_changed_step_count": self._counts.get("recovery_guidance_changed_step_count", 0),
            "recovery_effective_intervention_episode": self._counts.get("recovery_effective_intervention_count", 0) > 0,
            "recovery_effective_intervention_count": self._counts.get("recovery_effective_intervention_count", 0),
            "effective_recovery_active_rate": None if self._observed_search_steps == 0 else self._counts.get("recovery_plan_active_step_count", 0) / self._observed_search_steps,
            "tracking_waypoint_delta_norm_sum": float(sum(deltas)),
            "tracking_waypoint_delta_norm_mean": None if not deltas else float(np.mean(deltas)),
            "tracking_waypoint_delta_norm_max": 0.0 if not deltas else float(max(deltas)),
            "path_changed_step_count": self._counts.get("path_changed_step_count", 0),
            "recovery_failed_pass_through_count": self._counts.get("recovery_failed_pass_through_count", 0),
            "recovery_duration_sum": sum(self._durations),
            "recovery_duration_mean": None if not self._durations else float(np.mean(self._durations)),
            "recovery_duration_max": 0 if not self._durations else max(self._durations),
            # v1 compatibility columns; v2 does not call local connectors egress routes.
            "search_recovery_active_step_count": self._counts.get("recovery_state_non_normal_step_count", 0),
            "search_recovery_active_rate": None if self._observed_search_steps == 0 else self._counts.get("recovery_state_non_normal_step_count", 0) / self._observed_search_steps,
            "egress_attempt_count": self._counts.get("local_connector_attempt_count", 0),
            "egress_success_count": self._counts.get("local_connector_reached_count", 0),
            "egress_failure_count": self._counts.get("local_connector_collision_count", 0),
            "egress_rejoin_count": self._counts.get("graph_reconnect_success_count", 0),
            "recovery_collision_count": self._recovery_collision_count,
            "recovery_max_collision_streak": self._recovery_max_collision_streak,
            "recovery_no_egress_count": 0,
            "recovery_failed_endpoint_count": len(set().union(*(state.failed_endpoint_cells for state in self.agents.values()))),
        }
        failures: dict[str, int] = {}
        for audit in self._planning_audits:
            reason = audit.get("final_failure_reason")
            if reason:
                failures[str(reason)] = failures.get(str(reason), 0) + 1
        result["failure_reason_distribution"] = failures
        result["candidate_tier_distribution"] = {
            f"tier{tier}": self._counts.get(f"tier{tier}_count", 0) for tier in range(4)
        }
        for tier in range(4):
            result[f"tier{tier}_count"] = self._counts.get(f"tier{tier}_count", 0)
        for agent_id, state in self.agents.items():
            counts = self._agent_counts[agent_id]
            result[f"last_recovery_mode_agent_{agent_id}"] = state.last_terminal_mode
            result[f"last_recovery_failure_reason_agent_{agent_id}"] = state.failure_reason
            for name in (
                "search_recovery_entry_count", "route_refresh_attempt_count",
                "route_refresh_success_count", "route_refresh_failure_count",
                "local_connector_attempt_count", "local_connector_plan_count",
                "local_connector_reached_count", "local_connector_collision_count",
                "graph_reconnect_attempt_count", "graph_reconnect_success_count",
                "graph_reconnect_failure_count", "recovery_effective_intervention_count",
            ):
                result[f"{name}_agent_{agent_id}"] = counts.get(name, 0)
            result[f"egress_attempt_count_agent_{agent_id}"] = counts.get("local_connector_attempt_count", 0)
            result[f"egress_success_count_agent_{agent_id}"] = counts.get("local_connector_reached_count", 0)
            result[f"egress_failure_count_agent_{agent_id}"] = counts.get("local_connector_collision_count", 0)
            result[f"egress_rejoin_count_agent_{agent_id}"] = counts.get("graph_reconnect_success_count", 0)
        return result


__all__ = ("SearchCollisionRecoveryControllerV2",)
