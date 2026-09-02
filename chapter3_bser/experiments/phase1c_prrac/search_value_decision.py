"""Hysteretic decision sidecar for Search Value guided BSER re-evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


SEARCH_VALUE_DECISION_SCHEMA = "bser.phase1c.prrac.search_value_decision.v1"

DEFAULT_SEARCH_VALUE_DECISION_CONFIG = {
    "enabled": False,
    "threshold": 0.35,
    "patience": 20,
    "cooldown": 50,
}


def resolve_search_value_decision_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a validated configuration with baseline-preserving defaults."""

    resolved = dict(DEFAULT_SEARCH_VALUE_DECISION_CONFIG)
    resolved.update(dict(config or {}))
    resolved["enabled"] = bool(resolved["enabled"])
    resolved["threshold"] = float(resolved["threshold"])
    resolved["patience"] = int(resolved["patience"])
    resolved["cooldown"] = int(resolved["cooldown"])
    if not 0.0 <= resolved["threshold"] <= 1.0:
        raise ValueError("search_value_decision.threshold must be in [0, 1]")
    if resolved["patience"] <= 0:
        raise ValueError("search_value_decision.patience must be positive")
    if resolved["cooldown"] < 0:
        raise ValueError("search_value_decision.cooldown must be non-negative")
    return resolved


@dataclass(frozen=True)
class SearchValueDecision:
    search_value: float
    low_value: bool
    continuous_low_value_steps: int
    cooldown_remaining: int
    trigger_replan: bool


class SearchValueDecisionController:
    """Convert SearchValueHead probabilities into rate-limited re-evaluations.

    This class does not plan, allocate, modify guidance, or inspect privileged
    environment state.  It only decides whether the existing public BSER
    controller should receive a forced fresh planning snapshot.
    """

    SCHEMA = SEARCH_VALUE_DECISION_SCHEMA

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = resolve_search_value_decision_config(config)
        self.enabled = bool(self.config["enabled"])
        self.threshold = float(self.config["threshold"])
        self.patience = int(self.config["patience"])
        self.cooldown = int(self.config["cooldown"])
        self.reset()

    def reset(self) -> None:
        self.continuous_low_value_steps = 0
        self.low_value_steps = 0
        self.trigger_count = 0
        self.replan_after_trigger = 0
        self.last_trigger_step: int | None = None
        self.last_step: int | None = None
        self.trigger_values: list[float] = []
        self.observed_values: list[float] = []

    def _cooldown_remaining(self, step: int) -> int:
        if self.last_trigger_step is None or self.cooldown == 0:
            return 0
        elapsed = int(step) - int(self.last_trigger_step)
        return max(0, self.cooldown - elapsed + 1)

    def observe(
        self,
        search_value: float,
        *,
        step: int,
        search_active: bool = True,
    ) -> SearchValueDecision:
        value = float(search_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("search_value must be finite and in [0, 1]")
        current_step = int(step)
        if self.last_step is not None and current_step < self.last_step:
            raise ValueError("search-value decision steps must be monotonic")
        self.last_step = current_step
        if not self.enabled:
            return SearchValueDecision(value, False, 0, 0, False)
        if not bool(search_active):
            self.continuous_low_value_steps = 0
            return SearchValueDecision(
                value,
                False,
                0,
                self._cooldown_remaining(current_step),
                False,
            )

        self.observed_values.append(value)
        low_value = value < self.threshold
        if low_value:
            self.low_value_steps += 1
            self.continuous_low_value_steps += 1
        else:
            self.continuous_low_value_steps = 0
        remaining = self._cooldown_remaining(current_step)
        trigger = bool(
            low_value
            and self.continuous_low_value_steps >= self.patience
            and remaining == 0
        )
        if trigger:
            self.trigger_count += 1
            self.trigger_values.append(value)
            self.last_trigger_step = current_step
            self.continuous_low_value_steps = 0
            remaining = self.cooldown
        return SearchValueDecision(
            value,
            low_value,
            int(self.continuous_low_value_steps),
            int(remaining),
            trigger,
        )

    def observe_replan_result(
        self, decision: SearchValueDecision, *, replanned: bool
    ) -> None:
        if decision.trigger_replan and bool(replanned):
            self.replan_after_trigger += 1

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "trigger_count": int(self.trigger_count),
            "low_value_steps": int(self.low_value_steps),
            "mean_trigger_value": (
                None
                if not self.trigger_values
                else float(sum(self.trigger_values) / len(self.trigger_values))
            ),
            "replan_after_trigger": int(self.replan_after_trigger),
            "search_value_mean": (
                None
                if not self.observed_values
                else float(sum(self.observed_values) / len(self.observed_values))
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "config": dict(self.config),
            "continuous_low_value_steps": int(self.continuous_low_value_steps),
            "low_value_steps": int(self.low_value_steps),
            "trigger_count": int(self.trigger_count),
            "replan_after_trigger": int(self.replan_after_trigger),
            "last_trigger_step": self.last_trigger_step,
            "last_step": self.last_step,
            "trigger_values": list(self.trigger_values),
            "observed_values": list(self.observed_values),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any] | None,
        *,
        strict: bool = True,
    ) -> bool:
        """Restore state; return True when missing state was freshly initialized."""

        if state is None:
            if strict:
                raise KeyError("missing SearchValueDecisionController state")
            self.reset()
            return True
        if state.get("schema") != self.SCHEMA:
            raise ValueError("unsupported search-value decision state schema")
        stored_config = resolve_search_value_decision_config(state.get("config"))
        if stored_config != self.config:
            raise ValueError("search-value decision configuration mismatch")
        required = {
            "continuous_low_value_steps",
            "low_value_steps",
            "trigger_count",
            "replan_after_trigger",
            "last_trigger_step",
            "last_step",
            "trigger_values",
            "observed_values",
        }
        missing = required.difference(state)
        if missing and strict:
            raise KeyError(
                "missing SearchValueDecisionController keys: "
                + ", ".join(sorted(missing))
            )
        self.continuous_low_value_steps = int(
            state.get("continuous_low_value_steps", 0)
        )
        self.low_value_steps = int(state.get("low_value_steps", 0))
        self.trigger_count = int(state.get("trigger_count", 0))
        self.replan_after_trigger = int(state.get("replan_after_trigger", 0))
        self.last_trigger_step = (
            None
            if state.get("last_trigger_step") is None
            else int(state["last_trigger_step"])
        )
        self.last_step = (
            None if state.get("last_step") is None else int(state["last_step"])
        )
        self.trigger_values = [float(value) for value in state.get("trigger_values", ())]
        self.observed_values = [
            float(value) for value in state.get("observed_values", ())
        ]
        return bool(missing)


__all__ = (
    "DEFAULT_SEARCH_VALUE_DECISION_CONFIG",
    "SEARCH_VALUE_DECISION_SCHEMA",
    "SearchValueDecision",
    "SearchValueDecisionController",
    "resolve_search_value_decision_config",
)
