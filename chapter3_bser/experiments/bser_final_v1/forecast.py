"""Read-only public-kinematics prediction, with the actual tracker/bridge preview.

No simulator rollout, random draw, truth obstacle, target or flow query. Future
map, target belief and high-level decisions are frozen over this short horizon.
"""
import copy
from collections import Counter
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.detection_model import _point_segment_distance
from chapter3_bser.experiments.safe_search_v1.safe_guidance import SafetyBridge
from chapter3_bser.experiments.safe_search_v2.motion import Vehicle
from chapter3_bser.online.mission_context import OnlineMissionContext
from .options import SETTINGS


class Incomparable(ValueError):
    """A declared model/budget failure, not a physical no-path assertion."""


def vehicle_constants(env, agent_id):
    # Correct the role-specific strength locally; the old v2 module is frozen.
    value = Vehicle.from_constants(env, agent_id)
    strength = env.prior_strength_executor if agent_id == 3 else env.prior_strength_search
    if not np.isfinite(strength) or strength <= 0:
        raise Incomparable("invalid_role_strength")
    return replace(value, strength=float(strength))


def observed_disturbance(vehicle, previous, agent, step):
    if previous is None:
        return np.zeros(3), "initial_zero_estimate"
    if previous["step"] + 1 != step:
        raise Incomparable("stale_kinematic_observation")
    i = agent.agent_id
    p, v, target = (np.asarray(previous[k][i], dtype=float) for k in ("positions", "velocities", "targets"))
    # Saturated velocities conceal part of flow. Do not claim it is identifiable.
    _, expected = vehicle.step(p, v, target, np.zeros(3))
    actual = np.asarray(agent.velocity)
    estimate = (actual - expected) / vehicle.dt
    estimate[2] = 0.0  # The unchanged physical dynamics apply flow only in XY.
    if not np.isfinite(estimate).all() or np.linalg.norm(estimate) > SETTINGS["disturbance_limit"]:
        raise Incomparable("unreliable_observed_disturbance")
    return estimate, "clamped_velocity_residual_estimate"


def preview_bridge(owner, state, allocation):
    """Clone only chapter guidance state; never deepcopy/restore a simulator."""
    live = owner.planning_views.live()
    if live.step != state.step:
        raise Incomparable("stale_public_geometry")
    views = SimpleNamespace(live=lambda: live, search=owner.planning_views.search,
                            clearance=owner.planning_views.clearance)
    stub = SimpleNamespace(state=state, planning_views=views, safe_options=owner.safe_options,
                           safe_counts=Counter(), safe_pending=set())
    original = getattr(owner, "bridge", None)
    probe = SafetyBridge(stub)
    if original is not None:
        if type(original) is not SafetyBridge:
            raise Incomparable("unexpected_guidance_bridge")
        for name in ("path_tracker", "_overrides", "_attempt_steps", "_counted", "last_safety_status"):
            setattr(probe, name, copy.deepcopy(getattr(original, name)))
    # Public mission context is needed for physically frozen finished searchers.
    from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.train_phase1c_v2 import _public_context
    mission = _public_context(owner.env, state)
    guidance = probe.compile_guidance(allocation, state, mission)
    return probe, guidance, mission


@dataclass
class Forecast:
    positions: np.ndarray
    cumulative_detection: np.ndarray | None
    risk: tuple
    next_position: np.ndarray
    next_velocity: np.ndarray
    disturbance_status: str
    hold: bool


def _unknown_length(points, state):
    # Segment midpoint exposure is a named proxy, not swept exact unknown volume.
    mid = (points[:-1] + points[1:]) / 2
    spacing = np.asarray(state.grid.spacing)
    cells = np.floor((mid - np.asarray(state.grid.origin)) / np.where(spacing > 0, spacing, 1)).astype(int)
    cells = np.clip(cells, 0, np.asarray(state.grid.shape)-1)
    indices = np.ravel_multi_index(cells.T, state.grid.shape)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1) * state.occupancy.unknown_mask[indices]))


def predict(owner, state, allocation, agent_id, geometry, config):
    probe, guidance, mission = preview_bridge(owner, state, allocation)
    item = next(x for x in guidance.agent_assignments if x.agent_id == agent_id)
    agent = next(x for x in state.agents if x.agent_id == agent_id)
    vehicle = owner.final_vehicles[agent_id]
    disturbance, status = observed_disturbance(vehicle, owner.final_previous, agent, state.step)
    frozen = dict(zip(state.searcher_ids, mission.searcher_finished_flags)).get(agent_id, False)
    h = SETTINGS["prediction_steps"]
    offsets = [np.zeros(3)] + [sign * np.eye(3)[i] * SETTINGS["disturbance_uncertainty"]
                              for i in (0, 1) for sign in (-1, 1)]
    paths, velocities, risks = [], [], []
    for offset in offsets:
        tracker = copy.deepcopy(probe.path_tracker)
        p, v = np.asarray(agent.position, dtype=float).copy(), np.asarray(agent.velocity, dtype=float).copy()
        points, speeds, departures = [p.copy()], [v.copy()], []
        for t in range(h):
            if item.hold_state:
                target = p  # Native Hold commands the current position every tick.
            elif t == 0:
                target = np.asarray(item.tracking_waypoint)
            else:
                target = tracker.tracking_target(agent_id, p, item.planned_path, item.final_waypoint)
            if frozen:
                next_p, next_v = p.copy(), np.zeros(3)
            else:
                next_p, next_v = vehicle.step(p, v, target, disturbance + offset)
            # Use remaining ordered segments, including the currently traversed
            # segment, rather than treating a finished waypoint as future coverage.
            snapshot = tracker.snapshot(agent_id)
            route = item.planned_path
            start_index = max(0, snapshot.next_index-1)
            remaining = tuple(route[start_index:]) + (item.final_waypoint,)
            if item.hold_state:
                departure = float(np.linalg.norm(next_p-np.asarray(agent.position)))
            elif len(remaining) > 1:
                departure = min(float(_point_segment_distance(next_p[None, :], np.asarray(a), np.asarray(b))[0])
                                for a, b in zip(remaining[:-1], remaining[1:]))
            else:
                departure = float(np.linalg.norm(next_p-np.asarray(item.final_waypoint)))
            departures.append(departure)
            p, v = next_p, next_v
            points.append(p.copy()); speeds.append(v.copy())
        points = np.asarray(points)
        risks.append((float(np.count_nonzero(~geometry._segments_free(points[:-1], points[1:]))),
                      _unknown_length(points, state), max(departures, default=0.)))
        paths.append(points); velocities.append(np.asarray(speeds))
    positions = paths[0]
    risk = tuple(float(x) for x in np.max(np.asarray(risks), axis=0))
    cumulative = None
    if agent_id in state.searcher_ids:
        centers = np.asarray(state.grid.cell_centers)
        params = config["detection_model"]
        sigma = float(params["sigma_sensor_radius_multiplier"]) * agent.sensor_radius
        scale = float(params["p_max_by_role"][agent.role]) * float(params.get("p_scale", 1.))
        q = 1. - np.asarray(state.occupancy.occupancy_probability)
        distance = np.linalg.norm(centers-positions[0], axis=1)
        sequence = [np.clip(scale*np.exp(-distance**2/(2*sigma**2))*q, 0, 1)]
        for a, b in zip(positions[:-1], positions[1:]):
            distance = np.minimum(distance, _point_segment_distance(centers, a, b))
            sequence.append(np.clip(scale*np.exp(-distance**2/(2*sigma**2))*q, 0, 1))
        cumulative = np.asarray(sequence)
    return Forecast(positions, cumulative, risk, positions[1], velocities[0][1], status, item.hold_state)
