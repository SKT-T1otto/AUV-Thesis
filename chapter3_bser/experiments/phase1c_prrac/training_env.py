"""Transparent PRRAC metadata wrapper over Phase1CV2TrainingEnv."""

from __future__ import annotations

from typing import Any

from chapter3_bser.experiments.phase1c_common import TransitionPhase
from chapter3_bser.experiments.phase1c_prrac.transition_protocol import (
    PRRACTransitionMetadata,
)
from chapter3_bser.models.prrac.stage_mapping import (
    transition_phase_to_prrac_stage,
)


class PRRACTrainingEnv:
    def __init__(self, env: Any) -> None:
        self.env = env
        self._previous_phase = TransitionPhase.PRE_FOUND
        self.last_prrac_transition_metadata: PRRACTransitionMetadata | None = None
        self._closed = False

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def installed_context(self):
        return getattr(self.env, "installed_context", None)

    @property
    def last_reward_breakdown(self):
        return getattr(self.env, "last_reward_breakdown", {})

    @property
    def last_transition_metadata(self):
        return getattr(self.env, "last_transition_metadata", None)

    def reset(self, *args, **kwargs):
        self._previous_phase = TransitionPhase.PRE_FOUND
        self.last_prrac_transition_metadata = None
        return self.env.reset(*args, **kwargs)

    def step(self, actions):
        stage_before = transition_phase_to_prrac_stage(self._previous_phase)
        result = self.env.step(actions)
        base = self.last_transition_metadata
        if base is None:
            raise RuntimeError("Phase1CV2TrainingEnv did not emit transition metadata")
        current_phase = base.phase
        stage_after = transition_phase_to_prrac_stage(current_phase)
        self.last_prrac_transition_metadata = PRRACTransitionMetadata(
            base=base,
            stage_before=stage_before,
            stage_after=stage_after,
        )
        self._previous_phase = current_phase
        return result

    def finalize_episode(self):
        return self.env.finalize_episode()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str):
        return getattr(self.env, name)


__all__ = ("PRRACTrainingEnv",)
