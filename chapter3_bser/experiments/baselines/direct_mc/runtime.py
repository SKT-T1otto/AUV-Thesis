"""B2 runtime and the ordinary core MADDPG model factory."""
from chapter3_bser.experiments.baselines.common.model import build_model as _build_model
from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime


MissionRuntime = BaselineMissionRuntime


def build_model(config, env=None, *, device="cpu"):
    return _build_model(config, env, algorithm="maddpg", device=device)


__all__ = ("MissionRuntime", "build_model")
