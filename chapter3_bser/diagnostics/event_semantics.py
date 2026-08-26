"""Read-only Phase 1B.3A event semantics over planner-visible state only.

This module classifies events after the existing detector and controller have
made their decisions.  Its outputs are evidence only: no diagnostic key or
classification is fed back into replanning, allocation, or action generation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np

from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.mission_context import OnlineMissionContext
from chapter3_bser.online.types import BSERActionAssignment, OnlineAllocation
from core.mapping.planning_state import PlanningStateView
from core.mapping.travel_cost_service import TravelCostService


EXECUTOR_HARD_FLAGS = frozenset(
    {"CURRENT_QUERY_UNREACHABLE", "PATH_CELL_BECAME_OCCUPIED", "COMPONENT_CHANGED"}
)
WAYPOINT_HARD_FLAGS = frozenset(
    {"WAYPOINT_QUERY_UNREACHABLE", "REMAINING_PATH_INVALIDATED", "COMPONENT_CHANGED"}
)


def mission_phase(context: OnlineMissionContext) -> str:
    """Derive phase solely from the public mission context."""

    if context.mission_complete:
        return "DONE"
    if not context.target_found:
        return "SEARCH"
    if not context.executor_knows_target:
        return "WAIT_PUBLIC_HANDOFF"
    return "EXECUTE_PUBLIC_TARGET"


def classify_executor_invalid(flags: Iterable[str]) -> str:
    values = frozenset(str(value) for value in flags)
    if "INSTALLED_ASSIGNMENT_REACHABLE_FALSE" in values:
        return "INSTALLED_ASSIGNMENT_INVALID"
    if values & EXECUTOR_HARD_FLAGS:
        return "HARD_ROUTE_INVALID"
    if "RELATIVE_COST_INCREASE" in values:
        return "SOFT_RESPONSE_DEGRADED_ONLY"
    return "DIAGNOSTIC_INCONSISTENCY"


def classify_waypoint_stale(flags: Iterable[str]) -> str:
    values = frozenset(str(value) for value in flags)
    if "FINAL_WAYPOINT_REACHED" in values:
        return "FINAL_WAYPOINT_REACHED"
    if values & WAYPOINT_HARD_FLAGS:
        return "HARD_PATH_INVALID"
    if "NO_ACTIVE_ASSIGNMENT" in values:
        return "NO_ACTIVE_ASSIGNMENT"
    return "DIAGNOSTIC_INCONSISTENCY"


class AssignmentVersionTracker:
    """Version allocation hashes without changing or suppressing allocations."""

    def __init__(self) -> None:
        self.assignment_version = 1
        self.allocation_sha256: str | None = None

    def observe(self, allocation_sha256: str) -> int:
        value = str(allocation_sha256)
        if self.allocation_sha256 is None:
            self.allocation_sha256 = value
        elif value != self.allocation_sha256:
            self.assignment_version += 1
            self.allocation_sha256 = value
        return self.assignment_version


def diagnostic_event_key(
    event_type: str,
    agent_id: int,
    assignment_version: int,
    map_revision: int,
    phase: str,
) -> tuple[str, int, int, int, str]:
    return (
        str(event_type),
        int(agent_id),
        int(assignment_version),
        int(map_revision),
        str(phase),
    )


def public_target_lock_violation(
    locked_target: Iterable[float] | None,
    current_target: Iterable[float] | None,
    source: str,
) -> bool:
    """Check a committed public-target assignment without reading private state."""

    if locked_target is None or current_target is None:
        return True
    return bool(
        str(source) != "PUBLIC_HANDOFF_TARGET"
        or not np.allclose(
            np.asarray(tuple(current_target), dtype=np.float64),
            np.asarray(tuple(locked_target), dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        )
    )


@dataclass(frozen=True)
class _InstalledRoute:
    path_cell_indices: tuple[int, ...]
    endpoint_components: tuple[int, ...]
    occupied_at_install: tuple[bool, ...]


def _route_snapshot(state: PlanningStateView, indices: Iterable[int]) -> _InstalledRoute:
    path = tuple(int(value) for value in indices)
    labels = np.asarray(state.planning_graph.component_labels, dtype=np.int64)
    occupied = np.asarray(state.occupancy.occupied_mask, dtype=np.bool_)
    valid_path = tuple(value for value in path if 0 <= value < len(labels))
    endpoints = () if not valid_path else (valid_path[0], valid_path[-1])
    return _InstalledRoute(
        path_cell_indices=valid_path,
        endpoint_components=tuple(int(labels[value]) for value in endpoints),
        occupied_at_install=tuple(bool(occupied[value]) for value in valid_path),
    )


def _current_components(state: PlanningStateView, route: _InstalledRoute) -> tuple[int, ...]:
    if not route.path_cell_indices:
        return ()
    labels = np.asarray(state.planning_graph.component_labels, dtype=np.int64)
    endpoints = (route.path_cell_indices[0], route.path_cell_indices[-1])
    return tuple(int(labels[value]) if 0 <= value < len(labels) else -1 for value in endpoints)


def _newly_occupied(state: PlanningStateView, route: _InstalledRoute, *, start: int = 0) -> bool:
    occupied = np.asarray(state.occupancy.occupied_mask, dtype=np.bool_)
    for offset, index in enumerate(route.path_cell_indices):
        if offset < int(start) or not (0 <= index < len(occupied)):
            continue
        installed = route.occupied_at_install[offset]
        if bool(occupied[index]) and not installed:
            return True
    return False


def _relative_cost_change(current: float, installed: float) -> float:
    if not math.isfinite(current):
        return math.inf
    if not math.isfinite(installed) or installed <= 1e-12:
        return 0.0
    return (float(current) - float(installed)) / float(installed)


def _distance(left: Iterable[float] | None, right: Iterable[float] | None) -> float:
    if left is None or right is None:
        return math.inf
    return float(
        np.linalg.norm(
            np.asarray(tuple(left), dtype=np.float64)
            - np.asarray(tuple(right), dtype=np.float64)
        )
    )


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Phase1B3ADiagnosticRecorder:
    """Collect event semantics without participating in control flow."""

    def __init__(self, scenario_seed: int, episode_index: int, config: Mapping[str, Any]):
        self.scenario_seed = int(scenario_seed)
        self.episode_index = int(episode_index)
        self.config = config
        self.version = AssignmentVersionTracker()
        self.event_rows: list[dict[str, Any]] = []
        self.collision_rows: list[dict[str, Any]] = []
        self.rejection_rows: list[dict[str, Any]] = []
        self._installed: dict[str, dict[int, _InstalledRoute]] = {}
        self.public_target_received_count = 0
        self.execute_phase_target_source_counts: Counter[str] = Counter()
        self.public_target_lock_violation_count = 0
        self.standby_source_after_public_handoff_count = 0
        self._locked_public_target: tuple[float, float, float] | None = None

    def initialize(self, state: PlanningStateView, allocation: OnlineAllocation) -> None:
        allocation_sha = allocation.allocation_sha256
        self.version.observe(allocation_sha)
        self._capture_install(state, allocation)

    def _capture_install(self, state: PlanningStateView, allocation: OnlineAllocation) -> None:
        allocation_sha = allocation.allocation_sha256
        if allocation_sha in self._installed:
            return
        routes = {
            item.agent_id: _route_snapshot(state, item.path_cell_indices)
            for item in allocation.search_assignments
        }
        executor = allocation.executor_assignment
        routes[executor.executor_id] = _route_snapshot(state, executor.path_cell_indices)
        self._installed[allocation_sha] = routes

    def _common(
        self,
        *,
        state: PlanningStateView,
        phase: str,
        event_type: str,
        agent_id: int,
        allocation_sha: str,
    ) -> dict[str, Any]:
        key = diagnostic_event_key(
            event_type,
            agent_id,
            self.version.assignment_version,
            state.map_revision,
            phase,
        )
        return {
            "method": "Event-BSER-phase1b2_corrected",
            "scenario_seed": self.scenario_seed,
            "episode_index": self.episode_index,
            "step": int(state.step),
            "event_type": event_type,
            "agent_id": int(agent_id),
            "assignment_version": int(self.version.assignment_version),
            "map_revision": int(state.map_revision),
            "belief_revision": int(state.target_belief.revision),
            "mission_phase": phase,
            "allocation_sha256": allocation_sha,
            "event_key": _json(key),
        }

    def _executor_row(
        self,
        state: PlanningStateView,
        context: OnlineMissionContext,
        allocation: OnlineAllocation,
    ) -> dict[str, Any]:
        phase = mission_phase(context)
        allocation_sha = allocation.allocation_sha256
        executor_assignment = allocation.executor_assignment
        executor = state.agents[executor_assignment.executor_id]
        query = TravelCostService(state).query(
            executor.position,
            executor_assignment.target_region,
            executor,
        )
        installed = self._installed[allocation_sha][executor_assignment.executor_id]
        current_components = _current_components(state, installed)
        component_changed = current_components != installed.endpoint_components
        path_occupied = _newly_occupied(state, installed)
        relative = _relative_cost_change(
            float(query.planning_cost),
            float(executor_assignment.planning_cost),
        )
        threshold = float(self.config["execution"]["executor_cost_increase_threshold"])
        flags = []
        if not executor_assignment.reachable:
            flags.append("INSTALLED_ASSIGNMENT_REACHABLE_FALSE")
        if not query.reachable:
            flags.append("CURRENT_QUERY_UNREACHABLE")
        if query.reachable and executor_assignment.reachable and relative > threshold:
            flags.append("RELATIVE_COST_INCREASE")
        if path_occupied:
            flags.append("PATH_CELL_BECAME_OCCUPIED")
        if component_changed:
            flags.append("COMPONENT_CHANGED")
        if not math.isfinite(float(executor_assignment.planning_cost)) or not math.isfinite(float(query.planning_cost)):
            flags.append("NONFINITE_COST")
        primary = classify_executor_invalid(flags)
        row = self._common(
            state=state,
            phase=phase,
            event_type=BSEREvent.EXECUTOR_INVALID.value,
            agent_id=executor_assignment.executor_id,
            allocation_sha=allocation_sha,
        )
        row.update(
            {
                "primary_classification": primary,
                "reason_flags": _json(flags),
                "installed_assignment_reachable": bool(executor_assignment.reachable),
                "current_query_reachable": bool(query.reachable),
                "installed_planning_cost": float(executor_assignment.planning_cost),
                "current_planning_cost": float(query.planning_cost),
                "relative_cost_change": float(relative),
                "executor_cost_increase_threshold": threshold,
                "executor_target_source": str(executor_assignment.source),
                "executor_target_region": _json(executor_assignment.target_region),
                "current_component_id": _json(current_components),
                "installed_path_component_id": _json(installed.endpoint_components),
                "path_cell_became_occupied": bool(path_occupied),
                "component_changed": bool(component_changed),
            }
        )
        return row

    def _waypoint_row(
        self,
        state: PlanningStateView,
        context: OnlineMissionContext,
        allocation: OnlineAllocation,
        agent_id: int,
        path_tracker: PathTracker | None,
    ) -> dict[str, Any]:
        phase = mission_phase(context)
        allocation_sha = allocation.allocation_sha256
        assignments = {item.agent_id: item for item in allocation.search_assignments}
        item = assignments.get(int(agent_id))
        agent = state.agents[int(agent_id)]
        tracker = None if path_tracker is None else path_tracker.snapshot(agent_id)
        final = None if item is None else item.waypoint
        local = None if tracker is None else tracker.current_target
        distance_final = _distance(agent.position, final)
        distance_local = _distance(agent.position, local)
        next_index = 0 if tracker is None else tracker.next_index
        remaining_count = 0 if tracker is None else tracker.remaining_point_count
        remaining_length = 0.0 if path_tracker is None else path_tracker.remaining_path_length(agent_id, agent.position)
        cross_track = 0.0 if path_tracker is None else path_tracker.cross_track_error(agent_id, agent.position)
        if item is None:
            query = None
            installed = _InstalledRoute((), (), ())
        else:
            query = TravelCostService(state).query(agent.position, item.waypoint, agent)
            installed = self._installed[allocation_sha][int(agent_id)]
        current_components = _current_components(state, installed)
        component_changed = current_components != installed.endpoint_components
        cell_start = max(0, int(next_index) - 1)
        remaining_invalidated = _newly_occupied(state, installed, start=cell_start)
        stale_threshold = float(self.config["events"]["waypoint_stale_distance"])
        tracking_threshold = float(self.config["execution"]["path_tracking_threshold"])
        flags = []
        if distance_final <= stale_threshold:
            flags.append("FINAL_WAYPOINT_REACHED")
        if distance_local <= tracking_threshold:
            flags.append("LOCAL_TRACKING_POINT_REACHED")
        if query is not None and not query.reachable:
            flags.append("WAYPOINT_QUERY_UNREACHABLE")
        if remaining_invalidated:
            flags.append("REMAINING_PATH_INVALIDATED")
        if component_changed:
            flags.append("COMPONENT_CHANGED")
        if item is None:
            flags.append("NO_ACTIVE_ASSIGNMENT")
        primary = classify_waypoint_stale(flags)
        row = self._common(
            state=state,
            phase=phase,
            event_type=BSEREvent.WAYPOINT_STALE.value,
            agent_id=agent_id,
            allocation_sha=allocation_sha,
        )
        row.update(
            {
                "primary_classification": primary,
                "reason_flags": _json(flags),
                "final_assignment_waypoint": "" if final is None else _json(final),
                "current_local_tracking_target": "" if local is None else _json(local),
                "distance_to_final_waypoint": float(distance_final),
                "distance_to_local_tracking_target": float(distance_local),
                "path_next_index": int(next_index),
                "remaining_path_point_count": int(remaining_count),
                "remaining_path_length": float(remaining_length),
                "cross_track_error_to_remaining_path": float(cross_track),
                "current_query_reachable": False if query is None else bool(query.reachable),
                "path_cell_became_occupied": bool(remaining_invalidated),
                "component_changed": bool(component_changed),
                "installed_path_component_id": _json(installed.endpoint_components),
                "current_component_id": _json(current_components),
            }
        )
        return row

    def _record_collisions(
        self,
        *,
        state: PlanningStateView,
        context: OnlineMissionContext,
        allocation: OnlineAllocation,
        path_tracker: PathTracker | None,
        collision_agent_ids: Iterable[int],
    ) -> None:
        search = {item.agent_id: item for item in allocation.search_assignments}
        for agent_id in collision_agent_ids:
            agent_id = int(agent_id)
            agent = state.agents[agent_id]
            item = search.get(agent_id)
            tracker = None if path_tracker is None else path_tracker.snapshot(agent_id)
            installed = self._installed[allocation.allocation_sha256].get(agent_id)
            invalidated = False if installed is None else _newly_occupied(
                state, installed, start=max(0, (0 if tracker is None else tracker.next_index) - 1)
            )
            self.collision_rows.append(
                {
                    "method": "Event-BSER-phase1b2_corrected",
                    "scenario_seed": self.scenario_seed,
                    "episode_index": self.episode_index,
                    "step": int(state.step),
                    "agent_id": agent_id,
                    "mission_phase": mission_phase(context),
                    "position": _json(agent.position),
                    "current_local_tracking_target": "" if tracker is None or tracker.current_target is None else _json(tracker.current_target),
                    "path_next_index": 0 if tracker is None else int(tracker.next_index),
                    "remaining_path_point_count": 0 if tracker is None else int(tracker.remaining_point_count),
                    "remaining_path_length": 0.0 if path_tracker is None else path_tracker.remaining_path_length(agent_id, agent.position),
                    "cross_track_error_to_remaining_path": 0.0 if path_tracker is None else path_tracker.cross_track_error(agent_id, agent.position),
                    "remaining_path_invalidated": bool(invalidated),
                    "final_assignment_waypoint": "" if item is None else _json(item.waypoint),
                    "allocation_sha256": allocation.allocation_sha256,
                }
            )

    def record_step(
        self,
        *,
        state: PlanningStateView,
        context: OnlineMissionContext,
        before: OnlineAllocation,
        result: BSERActionAssignment,
        path_tracker: PathTracker | None,
        collision_agent_ids: Iterable[int] = (),
    ) -> None:
        """Observe an already-computed step; never alter or replace its result."""

        before_sha = before.allocation_sha256
        self.version.observe(before_sha)
        if before_sha not in self._installed:
            self._capture_install(state, before)
        if BSEREvent.EXECUTOR_INVALID in result.events:
            self.event_rows.append(self._executor_row(state, context, before))
        if BSEREvent.WAYPOINT_STALE in result.events:
            stale_ids = tuple(result.event_detection.stale_searcher_ids)
            if not stale_ids:
                stale_ids = (-1,)
            for agent_id in stale_ids:
                if agent_id < 0:
                    common = self._common(
                        state=state,
                        phase=mission_phase(context),
                        event_type=BSEREvent.WAYPOINT_STALE.value,
                        agent_id=agent_id,
                        allocation_sha=before_sha,
                    )
                    common.update(
                        primary_classification="DIAGNOSTIC_INCONSISTENCY",
                        reason_flags="[]",
                    )
                    self.event_rows.append(common)
                else:
                    self.event_rows.append(
                        self._waypoint_row(state, context, before, agent_id, path_tracker)
                    )
        if BSEREvent.EXECUTOR_TARGET_RECEIVED in result.events:
            self.public_target_received_count += 1
        phase = mission_phase(context)
        after = result.allocation
        if phase == "EXECUTE_PUBLIC_TARGET":
            source = str(after.executor_assignment.source)
            self.execute_phase_target_source_counts[source] += 1
            committed_target = tuple(
                float(value) for value in after.executor_assignment.target_region
            )
            if self._locked_public_target is None and source == "PUBLIC_HANDOFF_TARGET":
                self._locked_public_target = tuple(
                    float(value) for value in committed_target
                )
            self.public_target_lock_violation_count += int(
                public_target_lock_violation(
                    self._locked_public_target,
                    committed_target,
                    source,
                )
            )
            self.standby_source_after_public_handoff_count += int(source.startswith("standby:"))
        if (
            result.diagnostics is not None
            and result.diagnostics.reject_reason
            and not result.replanned
            and (bool(result.events) or bool(result.diagnostics.optimizer_invoked))
        ):
            self.rejection_rows.append(
                {
                    "method": "Event-BSER-phase1b2_corrected",
                    "scenario_seed": self.scenario_seed,
                    "episode_index": self.episode_index,
                    "step": int(state.step),
                    "mission_phase": phase,
                    "allocation_scope": str(result.diagnostics.allocation_scope),
                    "rejection_reason": str(result.diagnostics.reject_reason),
                }
            )
        self._record_collisions(
            state=state,
            context=context,
            allocation=before,
            path_tracker=path_tracker,
            collision_agent_ids=collision_agent_ids,
        )
        after_sha = after.allocation_sha256
        if after_sha != before_sha:
            self.version.observe(after_sha)
            self._capture_install(state, after)

    def target_summary(self) -> dict[str, Any]:
        return {
            "public_target_received_count": int(self.public_target_received_count),
            "execute_phase_target_source_counts": dict(sorted(self.execute_phase_target_source_counts.items())),
            "public_target_lock_violation_count": int(self.public_target_lock_violation_count),
            "standby_source_after_public_handoff_count": int(self.standby_source_after_public_handoff_count),
        }
