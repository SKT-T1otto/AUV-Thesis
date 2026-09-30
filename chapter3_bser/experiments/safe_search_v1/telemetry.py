"""Read-only SafeSearch capture with explicit guidance/tracker route alignment.

A hold or a replaced guidance route need not advance/reset the original tracker.
Its retained cursor must never index a different guidance polyline. This adapter
keeps the original cursor and remaining-length proxy, but only reports active
segment geometry after checking the complete retained route and final target.
It reads controller diagnostics, never hidden world geometry, and changes no
tracker, guidance, controller, environment, or random state.
"""
from __future__ import annotations

from scripts.search_diagnostic_observer import (
    angle, distance, norm, segment_distance, subtract, vector,
)


def _path_geometry(tracker, agent_id, assignment, tracking, path, position, target):
    """Inspect the full private route signature without updating the tracker."""
    retained = getattr(tracker, "_state", {}).get(int(agent_id))
    signature = getattr(retained, "signature", None)
    route_matches = None
    if signature is not None:
        expected = (tuple(tuple(point) for point in path), tuple(assignment.final_waypoint))
        route_matches = signature == expected
    index = tracking.next_index
    diagnostics = dict(
        geometry_status=None,
        tracker_matches_guidance_route=route_matches,
        tracker_path_point_count=int(tracking.path_point_count),
        tracker_completed=bool(tracking.completed),
        remaining_path_length_source="retained_path_tracker_route",
    )
    if assignment.hold_state:
        status = "inactive_hold"
    elif not assignment.reachable:
        status = "inactive_unreachable"
    elif not path:
        status = "inactive_empty_guidance_path"
    elif route_matches is None:
        status = "unavailable_tracker_route_identity"
    elif not route_matches:
        status = "unavailable_guidance_tracker_route_mismatch"
    elif (type(index) is not int or not 0 <= index <= len(path)
          or tracking.path_point_count != len(path)
          or tracking.final_waypoint != tuple(assignment.final_waypoint)):
        status = "unavailable_tracker_snapshot_inconsistent"
    elif tracking.completed:
        status = "inactive_completed_route" if index == len(path) else "unavailable_tracker_snapshot_inconsistent"
    elif index >= len(path):
        status = "unavailable_tracker_snapshot_inconsistent"
    elif (tracking.current_target != tuple(target)
          or tuple(path[index]) != tuple(target)):
        status = "unavailable_guidance_tracker_target_mismatch"
    else:
        # The full signature already establishes the route identity; the
        # snapshot suffix additionally validates its cursor representation.
        remaining = tuple(tuple(point) for point in path[index:])
        final = tuple(assignment.final_waypoint)
        if not remaining or remaining[-1] != final:
            remaining = (*remaining, final)
        status = ("active_aligned" if tracking.remaining_path_points == remaining
                  else "unavailable_tracker_snapshot_inconsistent")
    diagnostics["geometry_status"] = status
    if status != "active_aligned":
        return None, None, diagnostics
    start, end = path[max(0, index - 1)], path[index]
    next_point = path[index + 1] if index + 1 < len(path) else None
    cross_track = segment_distance(position, start, end)
    turn = angle(subtract(end, start), subtract(next_point, end)) if next_point else None
    return cross_track, turn, diagnostics


def capture_runtime(runtime, live_public_state=None):
    """Capture the original fields plus route-alignment status, without mutation.

    For an active aligned route, all historical capture fields retain their
    exact values. Inactive/unaligned routes publish null segment/turn geometry.
    ``path_index`` and ``remaining_path_length`` retain the actual tracker's
    values, including a stale retained route, for the existing motion proxy.
    """
    state = runtime.state
    agents = runtime.env.get_agent_state()
    task = runtime.env.get_task_state()
    mapping = runtime.env.get_mapping_state()
    guidance = runtime.guidance
    tracker = runtime.bridge.path_tracker
    rows = []
    for agent in state.agents:
        agent_id = agent.agent_id
        assignment = guidance.assignment_for(agent_id)
        tracking = tracker.snapshot(agent_id)
        position, velocity = vector(agents.positions[agent_id]), vector(agents.velocities[agent_id])
        path = [vector(point) for point in assignment.planned_path]
        target = vector(assignment.tracking_waypoint)
        cross_track, turn, diagnostics = _path_geometry(
            tracker, agent_id, assignment, tracking, path, position, target)
        rows.append(dict(agent_id=agent_id, role=agent.role, position=position, velocity=velocity,
            speed=norm(velocity), assignment_id=assignment.assignment_id,
            assignment_kind=assignment.assignment_kind,
            semantic_waypoint=vector(assignment.final_waypoint), tracking_waypoint=target,
            hold=bool(assignment.hold_state), reachable=bool(assignment.reachable),
            path_index=tracking.next_index, path_point_count=len(path), planned_path=path,
            tracking_distance=distance(position, target), active_segment_cross_track_error=cross_track,
            velocity_to_tracking_angle_deg=angle(velocity, subtract(target, position)),
            next_turn_angle_deg=turn,
            remaining_path_length=tracker.remaining_path_length(agent_id, position),
            path_diagnostics=diagnostics))
    allocation = runtime.controller.current_allocation
    result = dict(step=int(task.step), found=bool(task.target_found),
        executor_knows_target=bool(task.executor_knows_target), agents=rows,
        allocation_hash=allocation.allocation_sha256, decision_reason=guidance.decision_reason,
        cached_map_revision=int(state.map_revision), live_map_revision=int(mapping.map_revision),
        cached_known_cells=int(sum(bool(value) for value in state.occupancy.known_mask)),
        cached_occupied_cells=int(sum(bool(value) for value in state.occupancy.occupied_mask)),
        last_full_refresh_step=int(runtime.provider._last_full_refresh_step),
        full_refresh=bool(runtime.provider.last_snapshot_was_full_refresh),
        replan_count=int(runtime.controller.replan_count), replan_steps=list(runtime.controller.replan_steps))
    if live_public_state is not None:
        result["live_public_map"] = dict(diagnostic_only=True, map_revision=int(live_public_state.map_revision),
            grid_shape=list(live_public_state.grid.shape), grid_origin=list(live_public_state.grid.origin),
            grid_spacing=list(live_public_state.grid.spacing),
            occupied_cell_indices=[i for i, value in enumerate(live_public_state.occupancy.occupied_mask) if value],
            unknown_cell_indices=[i for i, value in enumerate(live_public_state.occupancy.unknown_mask) if value])
    return result
