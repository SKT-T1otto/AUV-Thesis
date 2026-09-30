"""Pre-Found latched Hold and braking through the existing guidance fields."""
from dataclasses import replace
import math
import numpy as np

from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.experiments.safe_search_v1.safe_guidance import SafetyBridge
from chapter3_bser.experiments.safe_search_v1.public_geometry import PublicGeometry
from .motion import Vehicle, forecast_safe, vector


class SearchTracker(PathTracker):
    """Tighter search turns without changing executor or post-Found tracking."""
    def __init__(self, search_threshold):
        super().__init__()
        self.search_threshold = search_threshold
        self.search_enabled = True

    def tracking_target(self, agent_id, *args, **kwargs):
        original = self.threshold
        try:
            if self.search_enabled and agent_id in (0, 1, 2):
                self.threshold = self.search_threshold
            return super().tracking_target(agent_id, *args, **kwargs)
        finally:
            self.threshold = original


class FoundSafetyBridge(SafetyBridge):
    def __init__(self, owner):
        tracker = SearchTracker(owner.found_settings.tracking_threshold) if owner.found_options["braking"] else None
        super().__init__(owner, tracker)
        self.anchors = {}
        self.previous = {}
        self.next_query = {}
        self.last_found_safety = {}
        self.vehicles = {i: Vehicle.from_constants(owner.env.unwrapped, i) for i in (0, 1, 2)}
        self._compiled_step = None
        self._compiled_context = None

    def _anchor(self, agent_id, position, geometry):
        anchor = self.anchors.get(agent_id)
        if anchor is None:
            anchor = tuple(position)
            self.anchors[agent_id] = anchor
            self.owner.found_counts["hold_entries"] += 1
        elif not geometry.point_free(anchor) or not geometry.segment_free(position, anchor):
            # A newly discovered obstacle can invalidate a formerly safe anchor.
            # Do not invent an escape route through occupied public cells.
            anchor = tuple(position)
            self.anchors[agent_id] = anchor
            self.owner.found_counts["hold_anchor_invalidations"] += 1
        return vector(anchor)

    def _disturbance(self, agent, vehicle, step):
        previous = self.previous.get(agent.agent_id)
        if previous is None or previous[0] != step - 1:
            return np.zeros(3)
        _, position, velocity, target = previous
        _, nominal = vehicle.step(position, velocity, target, np.zeros(3))
        estimate = (vector(agent.velocity) - nominal) / vehicle.dt
        return np.clip(estimate, -self.owner.found_settings.disturbance_limit,
                       self.owner.found_settings.disturbance_limit)

    def _corner_target(self, item, position, vehicle):
        remaining = self.path_tracker.snapshot(item.agent_id).remaining_path_points
        if len(remaining) < 2:
            return vector(item.tracking_waypoint)
        incoming = vector(item.tracking_waypoint) - position
        outgoing = vector(remaining[1]) - vector(remaining[0])
        denominator = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
        if denominator < 1e-12:
            return vector(item.tracking_waypoint)
        angle = math.degrees(math.acos(float(np.clip(np.dot(incoming, outgoing) / denominator, -1, 1))))
        settings = self.owner.found_settings
        if angle < settings.corner_angle_degrees:
            return vector(item.tracking_waypoint)
        desired = vehicle.desired_velocity(position, item.tracking_waypoint)
        limit = math.sqrt(settings.corner_speed**2 + 2 * settings.corner_deceleration * np.linalg.norm(incoming))
        speed = np.linalg.norm(desired)
        if speed > limit:
            self.owner.found_counts["corner_speed_limits"] += 1
            desired *= limit / speed
        return vehicle.target_for_velocity(position, desired)

    def compile_guidance(self, allocation, planning_state, mission_context, *, decision_reason=None):
        if planning_state.target_found or mission_context.target_found or mission_context.mission_complete:
            if isinstance(self.path_tracker, SearchTracker):
                self.path_tracker.search_enabled = False
            self.anchors.clear()
            self.previous.clear()
            self.last_found_safety.clear()
            self._compiled_step = None
            return super().compile_guidance(allocation, planning_state, mission_context,
                                            decision_reason=decision_reason)
        step = int(planning_state.step)
        if self._compiled_step == step:
            # Recompilation must not advance the tracker twice for one command.
            if self._compiled_allocation != allocation or self._compiled_mission != mission_context:
                raise ValueError("conflicting guidance compilations in one physical step")
            return self._compiled_context
        owner, settings = self.owner, self.owner.found_settings
        suppressed = set()
        if owner.found_options["recovery"]:
            for agent_id, next_step in self.next_query.items():
                if step < next_step:
                    self._attempt_steps[agent_id] = step
                    suppressed.add(agent_id)
        context = super().compile_guidance(allocation, planning_state, mission_context,
                                           decision_reason=decision_reason)
        live = owner.planning_views.live()
        geometry = PublicGeometry(live, owner.planning_views.clearance)
        agents = {agent.agent_id: agent for agent in live.agents}
        output = []
        finished = dict(zip(planning_state.searcher_ids, mission_context.searcher_finished_flags))
        for item in context.agent_assignments:
            agent_id = item.agent_id
            if agent_id == planning_state.executor_id or finished.get(agent_id, False):
                output.append(item)
                continue
            if self._attempt_steps.get(agent_id) == step and agent_id not in suppressed:
                self.next_query[agent_id] = step + settings.recovery_retry_steps
            agent, vehicle = agents[agent_id], self.vehicles[agent_id]
            position, velocity = vector(agent.position), vector(agent.velocity)
            disturbance = self._disturbance(agent, vehicle, step)
            hold = item.hold_state
            anchor = self._anchor(agent_id, position, geometry) if hold else None
            target = anchor if hold else vector(item.tracking_waypoint)
            status, screened = ("LATCHED_HOLD" if hold else "TRACKING"), None
            if owner.found_options["braking"]:
                if hold:
                    target = vehicle.hold_target(position, velocity, anchor, settings.hold_velocity_feedback)
                    screened = forecast_safe(vehicle, geometry, position, velocity, target, disturbance, settings)
                    status = "DAMPED_HOLD" if screened else "HOLD_FORECAST_UNSAFE"
                else:
                    candidate = self._corner_target(item, position, vehicle)
                    desired = vehicle.desired_velocity(position, candidate)
                    screened = False
                    for scale in (1., .75, .5, .25):
                        proposed = vehicle.target_for_velocity(position, scale * desired)
                        if forecast_safe(vehicle, geometry, position, velocity, proposed, disturbance, settings):
                            target, screened = proposed, True
                            status = "TRACKING" if scale == 1. else "SPEED_LIMITED"
                            if scale != 1.:
                                owner.found_counts["forecast_speed_limits"] += 1
                            break
                    if not screened:
                        hold = True
                        anchor = self._anchor(agent_id, position, geometry)
                        target = vehicle.hold_target(position, velocity, anchor, settings.hold_velocity_feedback)
                        screened = forecast_safe(vehicle, geometry, position, velocity, target, disturbance, settings)
                        status = "PREDICTIVE_BRAKING" if screened else "BRAKING_FORECAST_UNSAFE"
                        owner.safe_pending.add(agent_id)
                        owner.found_counts["predictive_braking_steps"] += 1
                if not screened:
                    owner.found_counts["unsafe_forecast_steps"] += 1
            if hold:
                owner.safe_pending.add(agent_id)
                owner.found_counts["hold_agent_steps"] += 1
                item = replace(item, planned_path=(), tracking_waypoint=tuple(target),
                               hold_position=tuple(target), hold_state=True,
                               reachable=False, assignment_kind="hold")
            else:
                self.anchors.pop(agent_id, None)
                item = replace(item, tracking_waypoint=tuple(target))
            self.previous[agent_id] = (step, position.copy(), velocity.copy(), target.copy())
            self.last_found_safety[agent_id] = dict(status=status, anchor=None if anchor is None else anchor.tolist(),
                anchor_error=None if anchor is None else float(np.linalg.norm(position-anchor)),
                command_target=target.tolist(), disturbance_estimate=disturbance.tolist(),
                forecast_safe=screened, speed=float(np.linalg.norm(velocity)))
            output.append(item)
        result = replace(context, agent_assignments=tuple(output))
        self._compiled_step, self._compiled_context = step, result
        self._compiled_allocation, self._compiled_mission = allocation, mission_context
        return result
