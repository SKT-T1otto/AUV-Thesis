"""Deterministic weighted geometric median and opt-in normalized P action."""
from __future__ import annotations

import math
import numpy as np


def resolve_executor_standby(config=None):
    result = {"enabled": False, "gain": 0.5, **dict(config or {})}
    if type(result["enabled"]) is not bool:
        raise ValueError("executor_standby.enabled must be boolean")
    result["gain"] = float(result["gain"])
    if not math.isfinite(result["gain"]) or result["gain"] < 0:
        raise ValueError("executor_standby.gain must be finite and nonnegative")
    if set(result)-{"enabled", "gain"}:
        raise ValueError("unknown executor_standby option")
    return result


def normalized_weights(optional_weights=None):
    if optional_weights is not None:
        weights = np.asarray(optional_weights, float)
        if weights.shape == (3,) and np.isfinite(weights).all() and np.all(weights >= 0) and weights.max() > 0:
            weights = weights/weights.max()
            return weights/weights.sum()
    return np.full(3, 1/3, dtype=float)


def compute_standby_target(executor_position, searcher_positions, optional_weights=None):
    """Modified Weiszfeld, including coincident sites and collinear minimizers.

    The weighted mean is only an initialization, not the general solution.
    With no audited per-searcher values the caller uses uniform weights.
    """
    executor = np.asarray(executor_position, float)
    points = np.asarray(searcher_positions, float)
    if executor.shape != (3,) or points.shape != (3, 3) or not np.isfinite(executor).all() or not np.isfinite(points).all():
        raise ValueError("finite executor 3D and exactly three searcher 3D positions required")
    weights = normalized_weights(optional_weights)
    # Normalize coordinates to avoid overflow in norms for finite large inputs.
    scale = max(1.0, float(np.abs(points).max()))
    points = points/scale
    point = weights@points
    for _ in range(512):
        distances = np.linalg.norm(points-point, axis=1)
        away = distances > 1e-10
        if not np.any(away & (weights > 0)):
            return point.copy()*scale
        weighted = weights[away]/distances[away]
        proposal = (weighted@points[away])/weighted.sum()
        coincident_weight = float(weights[~away].sum())
        residual = np.linalg.norm(((points[away]-point)*(weights[away]/distances[away])[:, None]).sum(axis=0))
        if residual <= coincident_weight:
            return point.copy()*scale
        fraction = coincident_weight/residual
        updated = (1-fraction)*proposal+fraction*point
        if np.linalg.norm(updated-point) <= 1e-9:
            return updated*scale
        point = updated
    return point.copy()*scale


def standby_controller(current_executor_position, standby_position, gain=0.5):
    current, target = np.asarray(current_executor_position, float), np.asarray(standby_position, float)
    if current.shape != (3,) or target.shape != (3,) or not np.isfinite([current, target]).all():
        raise ValueError("finite 3D positions required")
    gain = resolve_executor_standby({"gain": gain})["gain"]
    return np.clip(gain*(target-current), -1.0, 1.0)


def apply_standby_action(actions, executor_position, standby_position, *, enabled=False, target_found=False, gain=0.5):
    if not enabled or target_found:
        return actions
    command = standby_controller(executor_position, standby_position, gain)
    # Import torch only on the active action boundary; preserve input dtype/device.
    import torch
    if torch.is_tensor(actions):
        if tuple(actions.shape) != (4, 3):
            raise ValueError("expected frozen 4-agent/3-action contract")
        result = actions.clone()
        result[3] = torch.as_tensor(command, dtype=result.dtype, device=result.device)
    else:
        result = np.asarray(actions).copy()
        if result.shape != (4, 3):
            raise ValueError("expected frozen 4-agent/3-action contract")
        result[3] = command
    return result
