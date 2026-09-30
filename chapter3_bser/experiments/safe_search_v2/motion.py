"""Public-state forecasts of the unchanged zero-residual waypoint controller.

This is a finite-horizon screening model, not a collision guarantee. It reads
only immutable vehicle constants and observed kinematics; never truth geometry,
hidden targets, simulator flow queries, rollouts, snapshots or random numbers.
"""
from dataclasses import dataclass
import numpy as np


def vector(value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("expected a finite three-vector")
    return result


@dataclass(frozen=True)
class Vehicle:
    dt: float
    kv_xy: float
    kv_z: float
    slow_xy: float
    slow_z: float
    strength: float
    v_xy: float
    v_z: float
    a_xy: float
    a_z: float
    drag_xy: float
    drag_z: float
    buoyancy: float

    @classmethod
    def from_constants(cls, env, agent_id):
        # A deliberately narrow read of configuration constants only.
        spec = env.agent_specs[agent_id]
        if not env.use_residual_prior:
            raise ValueError("braking requires the existing waypoint prior")
        result = cls(env.dt, env.prior_kv_xy, env.prior_kv_z,
                     env.prior_slow_radius_xy, env.prior_slow_radius_z,
                     env.prior_strength_search, spec["v_xy_max"], spec["v_z_max"],
                     spec["a_xy_max"], spec["a_z_max"], spec["drag_xy"],
                     spec["drag_z"], spec["buoyancy_bias"])
        if not all(np.isfinite(x) for x in vars(result).values()) or min(
                result.dt, result.kv_xy, result.kv_z, result.slow_xy,
                result.slow_z, result.strength, result.v_xy, result.v_z,
                result.a_xy, result.a_z) <= 0:
            raise ValueError("invalid vehicle constants")
        return result

    def clamp_velocity(self, velocity):
        result = vector(velocity).copy()
        result[:2] *= min(1., self.v_xy / max(np.linalg.norm(result[:2]), 1e-12))
        result[2] = np.clip(result[2], -self.v_z, self.v_z)
        return result

    def desired_velocity(self, position, target):
        relative = vector(target) - vector(position)
        desired = relative * np.array([self.v_xy / self.slow_xy] * 2 + [self.v_z / self.slow_z])
        return self.clamp_velocity(desired)

    def target_for_velocity(self, position, desired):
        desired = self.clamp_velocity(desired)
        return vector(position) + desired * np.array([self.slow_xy / self.v_xy] * 2 + [self.slow_z / self.v_z])

    def step(self, position, velocity, target, disturbance):
        desired = self.desired_velocity(position, target)
        limits = np.array([self.a_xy, self.a_xy, self.a_z])
        acceleration = np.clip(np.array([self.kv_xy, self.kv_xy, self.kv_z]) *
                               (desired - velocity), -limits, limits)
        acceleration = np.clip(self.strength * acceleration, -limits, limits)
        intermediate = velocity + acceleration * self.dt
        new_velocity = intermediate * (1 - self.dt * np.array([self.drag_xy, self.drag_xy, self.drag_z]))
        new_velocity += self.dt * (vector(disturbance) + np.array([0., 0., self.buoyancy]))
        new_velocity = self.clamp_velocity(new_velocity)
        return position + self.dt * new_velocity, new_velocity

    def hold_target(self, position, velocity, anchor, feedback):
        desired = self.desired_velocity(position, anchor) - feedback * velocity
        return self.target_for_velocity(position, desired)


def forecast_safe(vehicle, geometry, position, velocity, target, disturbance, settings):
    """One proposed step followed by latched, damped braking to rest.

Test nominal and six axis perturbations of the observed disturbance. These
samples are diagnostics, not a robust bound on every possible future flow.
"""
    starts, ends = [], []
    perturbations = [np.zeros(3)]
    for axis in np.eye(3):
        perturbations.extend((axis * settings.disturbance_uncertainty,
                              -axis * settings.disturbance_uncertainty))
    for offset in perturbations:
        p, v = vector(position).copy(), vector(velocity).copy()
        anchor = None
        for tick in range(settings.prediction_steps):
            command = target if tick == 0 else vehicle.hold_target(p, v, anchor, settings.hold_velocity_feedback)
            next_p, v = vehicle.step(p, v, command, disturbance + offset)
            starts.append(p)
            ends.append(next_p)
            p = next_p
            if anchor is None:
                anchor = p.copy()
    return bool(np.all(geometry._segments_free(starts, ends)))
