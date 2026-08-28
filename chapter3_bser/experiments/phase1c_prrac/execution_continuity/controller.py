"""Opt-in v3 controller overlay around the frozen Phase 1B controller."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping

import numpy as np

from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.controller import OnlineBSERController
from chapter3_bser.online.mission_context import OnlineMissionContext
from chapter3_bser.online.types import ExecutorAssignment, InitialBSERAllocation
from core.mapping.planning_state import PlanningStateView

from .config import overlay_config, parse_execution_variant
from .diagnostics import ExecutionContinuityDiagnostics
from .event_detector import ExecutionContinuityEventDetector
from .planner import plan_atomic_execution_continuity
from .types import ExecutionNavigationPlanV3, ExecutionVariant, NavigationMode


class ExecutionContinuityController:
    """Preserve legacy search behavior while replacing only execution navigation."""

    def __init__(
        self,
        legacy: OnlineBSERController,
        *,
        variant: str | ExecutionVariant,
        config: Mapping[str, Any],
    ) -> None:
        self.legacy = legacy
        self.variant = parse_execution_variant(variant)
        if self.variant is ExecutionVariant.B0_LEGACY_V2_1:
            raise ValueError("B0 must not be wrapped by the execution-continuity overlay")
        self.config = overlay_config(config)
        # This instance is private to the overlay.  Endpoint validity is handled
        # below so the legacy detector cannot substitute the semantic target.
        self.legacy.detector.dynamic_public_target_enabled = False
        self.detector = ExecutionContinuityEventDetector(
            executor_cost_increase_threshold=float(
                self.config["executor_cost_increase_threshold"]
            ),
            defer_stale_endpoint_invalid=bool(
                dict(config.get("execution_runtime", {})).get(
                    "defer_stale_endpoint_invalid", True
                )
            ),
        )
        self.current_plan: ExecutionNavigationPlanV3 | None = None
        self.last_detection = None
        self.last_plan_attempt_step: int | None = None
        self.last_semantic_update_step: int | None = None
        self.diagnostics = ExecutionContinuityDiagnostics(self.variant)

    @property
    def execution_target(self):
        return None if self.current_plan is None else self.current_plan.semantic_target

    @property
    def current_allocation(self):
        return self.legacy.current_allocation

    @property
    def replan_count(self) -> int:
        return int(self.legacy.replan_count)

    @staticmethod
    def _semantic_target(context: OnlineMissionContext):
        target = context.executor_navigation_target
        return None if target is None else tuple(float(value) for value in target)

    def _execution_active(self, context: OnlineMissionContext) -> bool:
        return bool(
            context.target_found
            and context.executor_knows_target
            and not context.mission_complete
            and context.executor_navigation_target is not None
        )

    def _retry_due(self, step: int) -> bool:
        if self.last_plan_attempt_step is None:
            return True
        interval = int(self.config["state_refresh_interval"])
        return bool(
            int(step) > int(self.last_plan_attempt_step)
            and int(step) % interval == 0
        )

    def _semantic_shift_due(self, semantic_target, step: int) -> bool:
        if self.current_plan is None:
            return True
        shift = float(
            np.linalg.norm(
                np.asarray(semantic_target, dtype=np.float64)
                - np.asarray(self.current_plan.semantic_target, dtype=np.float64)
            )
        )
        elapsed = (
            int(self.config["public_target_update_min_steps"])
            if self.last_semantic_update_step is None
            else int(step) - int(self.last_semantic_update_step)
        )
        return bool(
            shift >= float(self.config["public_target_update_distance"])
            and elapsed >= int(self.config["public_target_update_min_steps"])
        )

    @staticmethod
    def _executor_assignment(plan: ExecutionNavigationPlanV3, executor_id: int) -> ExecutorAssignment:
        return ExecutorAssignment(
            executor_id=int(executor_id),
            target_region=plan.navigation_endpoint,
            path=plan.path,
            estimated_arrival_time=float(plan.estimated_arrival_time),
            source=str(plan.source),
            reachable=bool(plan.reachable),
            path_cell_indices=plan.path_cell_indices,
            planning_cost=float(plan.planning_cost),
            failure_reason=plan.failure_reason,
        )

    def _overlay_allocation(self, allocation, plan: ExecutionNavigationPlanV3):
        return replace(
            allocation,
            executor_assignment=self._executor_assignment(
                plan, allocation.executor_assignment.executor_id
            ),
            response_time=float(plan.estimated_arrival_time),
            trigger_reason=str(plan.source),
            status="SAFE_HOLD" if plan.safe_hold else "OK",
            search_frozen=True,
        )

    def initialize(
        self, state: PlanningStateView, context: OnlineMissionContext
    ) -> InitialBSERAllocation:
        initialized = self.legacy.initialize(state, context)
        if not self._execution_active(context):
            return initialized
        target = self._semantic_target(context)
        assert target is not None
        plan = plan_atomic_execution_continuity(
            state, target, None, self.variant
        )
        self.current_plan = plan
        self.last_plan_attempt_step = int(state.step)
        self.last_semantic_update_step = int(state.step)
        self.diagnostics.observe_plan(plan, semantic_updated=True)
        allocation = self._overlay_allocation(initialized.allocation, plan)
        self.legacy.current_allocation = allocation
        return replace(initialized, allocation=allocation)

    def step(self, state: PlanningStateView, context: OnlineMissionContext):
        base = self.legacy.step(state, context)
        if not self._execution_active(context):
            self.current_plan = None
            self.last_detection = None
            return base

        target = self._semantic_target(context)
        assert target is not None
        retry_due = self._retry_due(state.step)
        semantic_updated = self._semantic_shift_due(target, state.step)
        detection = None
        if self.current_plan is not None:
            detection = self.detector.detect(
                state,
                self.current_plan,
                target,
                retry_due=retry_due,
            )
        self.last_detection = detection
        base_names = {
            str(getattr(event, "value", event)).upper() for event in base.events
        }
        event_trigger = bool(
            base_names
            & {
                "EXECUTOR_TARGET_RECEIVED",
                "OBSTACLE_DISCOVERED",
            }
        )
        should_plan = bool(
            self.current_plan is None
            or semantic_updated
            or event_trigger
            or (detection is not None and detection.route_invalid)
            or retry_due
        )
        previous = self.current_plan
        if should_plan:
            plan = plan_atomic_execution_continuity(
                state, target, previous, self.variant
            )
            self.current_plan = plan
            self.last_plan_attempt_step = int(state.step)
            if semantic_updated or previous is None:
                self.last_semantic_update_step = int(state.step)
            self.diagnostics.observe_plan(
                plan, semantic_updated=bool(semantic_updated or previous is None)
            )
        else:
            plan = self.current_plan
        assert plan is not None

        allocation = self._overlay_allocation(base.allocation, plan)
        self.legacy.current_allocation = allocation
        self.legacy.execution_target = target
        filtered = tuple(event for event in base.events if event is not BSEREvent.EXECUTOR_INVALID)
        invalid = bool(detection is not None and detection.route_invalid)
        if invalid and BSEREvent.EXECUTOR_INVALID not in filtered:
            filtered = tuple((*filtered, BSEREvent.EXECUTOR_INVALID))
        filtered = tuple(event for event in BSEREvent if event in filtered)
        reason = (
            "SAFE_HOLD_RETRY_PENDING"
            if plan.safe_hold and not should_plan
            else str(plan.source)
            if should_plan
            else str(base.decision_reason)
        )
        event_detection = base.event_detection
        if detection is not None:
            event_detection = replace(
                event_detection,
                events=filtered,
                executor_reachable=bool(plan.reachable and not invalid),
                executor_validity_evaluated=bool(detection.validity_evaluated),
                executor_validity_deferred=bool(detection.validity_deferred),
                executor_invalid_reason=str(detection.executor_invalid_reason),
                executor_query_failure_reason=str(detection.query_failure_reason),
                executor_assignment_reachable=bool(plan.reachable),
                executor_query_reachable=bool(not invalid),
                executor_installed_planning_cost=(
                    None if not math.isfinite(plan.planning_cost) else float(plan.planning_cost)
                ),
                executor_current_planning_cost=detection.current_planning_cost,
                executor_planning_cost_relative_change=float(
                    detection.planning_cost_relative_change
                ),
                executor_public_target_shift=float(detection.public_target_shift),
            )
        return replace(
            base,
            replanned=bool(should_plan or base.replanned),
            events=filtered,
            allocation=allocation,
            decision_reason=reason,
            event_detection=event_detection,
        )


__all__ = ("ExecutionContinuityController",)
