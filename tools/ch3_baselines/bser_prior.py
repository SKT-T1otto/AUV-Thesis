"""B1: the original MissionRuntime and BSER planner, with zero residuals."""
import torch

from chapter3_bser.experiments.hgr.runtime import MissionRuntime
from chapter3_bser.online.allocator import BSEROnlineAllocator


class ZeroResidualSource:
    """An action source, not a learned policy or a stochastic sampler."""
    def __init__(self):
        self.command_calls = 0

    def actions(self, observations, *, suffix, active, generator, deterministic):
        actions = torch.zeros((4, 3), dtype=torch.float32)
        if len(observations) != 4 or len(active) != 4:
            raise ValueError("zero residual source requires four agents")
        self.command_calls += 1
        return actions, [None] * 4


class BSERPriorRuntime(MissionRuntime):
    def __init__(self, config, scenario, *, seed, episode_id=0):
        # No copied constructor or allocator injection: even initialization uses
        # exactly the production controller factory and its joint BSER allocator.
        super().__init__(config, scenario, seed=seed, episode_id=episode_id)
        if type(self.controller.allocator) is not BSEROnlineAllocator:
            self.close()
            raise ValueError("B1 requires the unchanged BSER joint allocator")
        self.zero_source = ZeroResidualSource()
        self.diagnostics = dict(actor_forward_calls=0, action_sampling_calls=0,
            optimizer_update_count=0, residual_steps_checked=0, residual_action_max_abs=0.0,
            physical_residual_acceleration_max_abs=0.0, physical_prior_acceleration_max_abs=0.0)

    def advance(self):
        if not self.env.unwrapped.use_residual_prior:
            raise RuntimeError("prior control disabled")
        record = super().advance(self.zero_source)
        for value in (torch.as_tensor(record["actions"]), self.env.unwrapped._last_residual_acc):
            if value.shape != (4, 3) or not torch.isfinite(value).all() or torch.count_nonzero(value):
                raise RuntimeError("B1 requires strictly zero command and physical residual")
        self.diagnostics["residual_steps_checked"] += 1
        self.diagnostics["physical_prior_acceleration_max_abs"] = max(
            self.diagnostics["physical_prior_acceleration_max_abs"], float(self.env.unwrapped._last_prior_acc.abs().max()))
        return record

    def controller_diagnostics(self):
        return dict(self.diagnostics, zero_command_calls=self.zero_source.command_calls,
            allocator_class=type(self.controller.allocator).__module__ + "." + type(self.controller.allocator).__name__,
            accepted_replans=self.controller.replan_count, replan_steps=list(self.controller.replan_steps))
