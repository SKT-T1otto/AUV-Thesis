"""The four opt-in arms share one V4 runtime and unchanged post-Found execution."""
from collections import Counter
import numpy as np
from chapter3_bser.experiments.safe_search_v1.runtime import SafeJointRuntime, SearchController
from chapter3_bser.online.controller import OnlineBSERController
from .allocator import EffectAllocator
from .controller import EffectController, EffectPolicy
from .options import ARMS, SETTINGS


class EffectRuntime(SafeJointRuntime):
    def __init__(self, config, scenario, *, arm, seed, episode_id=0):
        if arm not in ARMS:
            raise ValueError("unknown D arm")
        self.effect_arm = arm
        self.effect_counts = Counter()
        self.executor_pre_found_distance = 0.0
        self.executor_anchor_max_distance = 0.0
        self.executor_standby_arrived_steps = 0
        super().__init__(config, scenario, variant="V4", seed=seed, episode_id=episode_id)

    def _build_online_controller(self, phase_config, config):
        parent = super()._build_online_controller(phase_config, config).inner
        if type(parent) is not OnlineBSERController:
            raise ValueError("D protocol requires the pinned reference controller contract")
        anchor = self.env.get_agent_state().positions[3]
        allocator = EffectAllocator(self, parent.allocator.config, self.effect_arm, anchor)
        inner = EffectController(parent.config, allocator)
        inner.prrac_runtime_contract = parent.prrac_runtime_contract
        wrapped = SearchController(inner, self)
        inner.policy = EffectPolicy(inner.config)
        return wrapped

    def advance(self):
        searching = not self.env.get_task_state().target_found
        before = np.asarray(self.env.get_agent_state().positions[3], dtype=float)
        standby = np.asarray(self.controller.current_allocation.executor_assignment.target_region)
        result = super().advance()
        if searching:
            after = np.asarray(self.env.get_agent_state().positions[3], dtype=float)
            self.executor_pre_found_distance += float(np.linalg.norm(after-before))
            self.executor_anchor_max_distance = max(self.executor_anchor_max_distance,
                float(np.linalg.norm(after-self.controller.allocator.anchor)))
            self.executor_standby_arrived_steps += int(np.linalg.norm(after-standby) <= 0.75)
            # Count arrival at the standby commanded during this transition;
            # the allocation after a Found event may already be a pursuit target.
            self.effect_counts["pre_found_executor_steps"] += 1
        return result

    def controller_diagnostics(self):
        result = super().controller_diagnostics()
        allocator = self.controller.allocator
        result["bser_effect_v1"] = dict(arm=self.effect_arm, options=ARMS[self.effect_arm], settings=SETTINGS,
            anchor=list(allocator.anchor), counts=dict(self.effect_counts),
            executor_pre_found_distance=self.executor_pre_found_distance,
            executor_anchor_max_distance=self.executor_anchor_max_distance,
            executor_standby_arrived_steps=self.executor_standby_arrived_steps,
            audit_record_count=len(allocator.audit),
            response_weight_search_change_proposals=sum(x.get("search_differs_from_pure", False) for x in allocator.audit),
            proposal_count=sum(x["kind"] == "proposal" for x in allocator.audit))
        return result


def make_runtime(config, scenario, *, arm, seed, episode_id=0):
    instance = EffectRuntime.__new__(EffectRuntime)
    try:
        EffectRuntime.__init__(instance, config, scenario, arm=arm, seed=seed, episode_id=episode_id)
    except BaseException:
        if hasattr(instance, "env"):
            instance.close()
        raise
    return instance
