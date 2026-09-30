"""Explicit descendants of B0 V5 / B1 V4; R0 returns the original runtime."""
from collections import Counter

from chapter3_bser.experiments.safe_search_v1 import runtime as parent
from .options import ARMS, PARENTS, SETTINGS
from .guidance import FoundSafetyBridge
from .recovery import RecoveryController


class FoundMixin:
    def _setup_safe(self, scenario, variant):
        super()._setup_safe(scenario, variant)
        if self.found_options["recovery"]:
            # Start preflight is part of the new recovery arm for B0 too.
            self.safe_options["failure_policy"] = True

    def _build_prior_controller(self, phase, allocator):
        wrapped = super()._build_prior_controller(phase, allocator)
        return RecoveryController(wrapped.inner, self) if self.found_options["recovery"] else wrapped

    def _build_online_controller(self, phase_config, config):
        wrapped = super()._build_online_controller(phase_config, config)
        return RecoveryController(wrapped.inner, self) if self.found_options["recovery"] else wrapped

    def _build_guidance_bridge(self):
        return FoundSafetyBridge(self)

    def controller_diagnostics(self):
        result = super().controller_diagnostics()
        result["found_search_v2"] = dict(arm=self.found_arm, parent_variant=self.safe_variant,
            options=self.found_options, settings=self.found_settings.record(),
            counts=dict(self.found_counts), guidance=self.bridge.last_found_safety,
            recovery=getattr(self.controller, "last_recovery", {}),
            forecast_is_guarantee=False, post_found_overlay_enabled=False)
        return result


class FoundPriorRuntime(FoundMixin, parent.SafePriorRuntime):
    pass


class FoundJointRuntime(FoundMixin, parent.SafeJointRuntime):
    pass


def make_runtime(config, scenario, *, baseline, arm, seed, episode_id=0):
    if baseline not in PARENTS or arm not in ARMS:
        raise ValueError("expected a B0/B1 baseline and an R0--R4 arm")
    if arm == "R0":
        return parent.make_runtime(config, scenario, baseline=baseline,
            variant=PARENTS[baseline], seed=seed, episode_id=episode_id)
    cls = FoundPriorRuntime if baseline == "B0_search_prior" else FoundJointRuntime
    runtime = cls.__new__(cls)
    runtime.found_arm = arm
    runtime.found_options = dict(ARMS[arm])
    runtime.found_settings = SETTINGS
    runtime.found_counts = Counter()
    try:
        cls.__init__(runtime, config, scenario, variant=PARENTS[baseline], seed=seed, episode_id=episode_id)
    except BaseException:
        if hasattr(runtime, "env"):
            runtime.close()
        raise
    return runtime
