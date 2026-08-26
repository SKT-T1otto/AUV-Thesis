"""Opt-in Phase 1C interfaces between online BSER and residual RMADDPG."""

from .control_context import (
    AgentAssignmentContextV1,
    BSERControlContextV1,
    ExecutorAssignmentContextV1,
)
from .guided_env import GuidedEnv
from .rmaddpg_bridge import (
    RMADDPGGuidanceBridge,
    compile_guidance,
    get_tracking_targets,
)

__all__ = (
    "AgentAssignmentContextV1",
    "BSERControlContextV1",
    "ExecutorAssignmentContextV1",
    "GuidedEnv",
    "RMADDPGGuidanceBridge",
    "compile_guidance",
    "get_tracking_targets",
)
