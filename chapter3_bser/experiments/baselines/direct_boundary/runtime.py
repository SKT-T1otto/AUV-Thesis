"""B3 runtime and boundary-conditioned core MADDPG model factory."""
from chapter3_bser.experiments.baselines.common.model import (
    BoundaryEncoder, BoundaryConditionedMADDPG, boundary_features, build_model as _build_model,
)
from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime


MissionRuntime = BaselineMissionRuntime


def build_model(config, env=None, *, device="cpu"):
    return _build_model(config, env, algorithm="direct_boundary_maddpg", device=device)


__all__ = ("MissionRuntime", "BoundaryEncoder", "BoundaryConditionedMADDPG", "boundary_features", "build_model")
