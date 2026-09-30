"""All F arms preserve V4 physics/events; F0 delegates to untouched D2."""
from collections import Counter
import hashlib
import time
import numpy as np

from chapter3_bser.experiments.bser_effect_v1.runtime import EffectRuntime
from chapter3_bser.experiments.bser_effect_v1.controller import EffectController, EffectPolicy
from chapter3_bser.experiments.safe_search_v1.runtime import SearchController
from .allocator import FinalAllocator
from .forecast import vehicle_constants
from .options import ARMS, SETTINGS


class FinalRuntime(EffectRuntime):
    def __init__(self, config, scenario, *, arm, seed, episode_id=0):
        if arm not in ARMS:
            raise ValueError("unknown F arm")
        self.final_arm, self.final_previous = arm, None
        self.final_vehicles = {}
        self.final_error_records = []
        self.final_planning_timings = []
        super().__init__(config, scenario, arm="D2", seed=seed, episode_id=episode_id)

    def _build_online_controller(self, phase_config, config):
        wrapped = super()._build_online_controller(phase_config, config)
        if self.final_arm == "F0":
            return self._timed(wrapped)
        self.final_vehicles = {i: vehicle_constants(self.env.unwrapped, i) for i in range(4)}
        parent = wrapped.inner
        allocator = FinalAllocator(self, parent.allocator.config, self.final_arm, parent.allocator.anchor)
        inner = EffectController(parent.config, allocator)
        inner.prrac_runtime_contract = parent.prrac_runtime_contract
        wrapped = SearchController(inner, self)
        inner.policy = EffectPolicy(inner.config)
        return self._timed(wrapped)

    def _timed(self, wrapped):
        # Timing surrounds the whole solve, including candidate generation, for
        # F0 as well as F1--F6. It never affects selection or a work-budget gate.
        for name in ("allocate", "allocate_partial"):
            original = getattr(wrapped.allocator, name)
            def timed(state, *args, _method=original, **kwargs):
                started = time.perf_counter()
                try:
                    return _method(state, *args, **kwargs)
                finally:
                    if not state.target_found:
                        self.final_planning_timings.append(time.perf_counter()-started)
            setattr(wrapped.allocator, name, timed)
        return wrapped

    def advance(self):
        record = None
        searching = not self.env.get_task_state().target_found
        if self.final_arm != "F0" and searching:
            agents = self.env.get_agent_state()
            self.final_previous = dict(step=self.step, positions=np.asarray(agents.positions).copy(),
                velocities=np.asarray(agents.velocities).copy(), targets=np.asarray(agents.navigation_targets).copy())
            audits = self.controller.allocator.final_audit
            if audits and audits[-1]["step"] == self.step:
                record = audits[-1]
                # Only compare a proposal forecast with the actually installed
                # allocation; a rejected proposal is not a model prediction error.
                current = self.controller.current_allocation
                from .scoring import allocation_key
                installed = record["selected_route_signature"] == hashlib.sha256(
                    repr(allocation_key(current)).encode()).hexdigest()
                record["selected_assignment_installed"] = installed
                if not installed:
                    record = None
        result = super().advance()
        if record is not None:
            agents = self.env.get_agent_state()
            errors = {i: dict(position=float(np.linalg.norm(np.asarray(pred["position"])-agents.positions[int(i)])),
                velocity=float(np.linalg.norm(np.asarray(pred["velocity"])-agents.velocities[int(i)])))
                for i, pred in record["next_step_prediction"].items()}
            self.final_error_records.append(dict(prediction_step=record["step"], observed_step=self.step,
                terminal=self.terminal, errors=errors))
        return result

    def controller_diagnostics(self):
        result = super().controller_diagnostics()
        records = getattr(self.controller.allocator, "final_audit", [])
        fallback = Counter(x["fallback_reason"] for x in records if x["fallback_reason"])
        times = list(self.final_planning_timings)
        result["bser_final_v1"] = dict(arm=self.final_arm, options=ARMS[self.final_arm], settings=SETTINGS,
            original_D2_delegate=self.final_arm == "F0", proposal_count=len(records),
            nonreference_proposals=sum(not r["selected_is_reference"] for r in records),
            installed_nonreference_proposals=sum(not r["selected_is_reference"]
                and r.get("selected_assignment_installed", False) for r in records),
            fallback_counts=dict(fallback), planning_seconds=times,
            total_response_queries=sum(r["counts"]["response_queries"] for r in records),
            prediction_error_records=len(self.final_error_records),
            one_step_position_errors=[v["position"] for r in self.final_error_records for v in r["errors"].values()],
            one_step_velocity_errors=[v["velocity"] for r in self.final_error_records for v in r["errors"].values()])
        return result


def make_runtime(config, scenario, *, arm, seed, episode_id=0):
    instance = FinalRuntime.__new__(FinalRuntime)
    try:
        FinalRuntime.__init__(instance, config, scenario, arm=arm, seed=seed, episode_id=episode_id)
    except BaseException:
        if hasattr(instance, "env"):
            instance.close()
        raise
    return instance
