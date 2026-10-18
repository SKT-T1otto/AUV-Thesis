"""Execution-aware masking with original networks, optimizers and Bellman settings."""
import copy
import torch

from core.algorithms.maddpg import MADDPG
from chapter3_bser.experiments.baselines.common.model import (
    _contract_environment, BoundaryConditionedMADDPG, BoundaryConditionedPolicy,
)
from .config import VERSION, validate


class ExecutionRepair:
    def _augment_sample(self, batch, sample):
        if not hasattr(sample, "enabled") or not hasattr(sample, "next_enabled"):
            raise ValueError("repair learner requires replay execution gates")
        for name, raw, obs in (("enabled", sample.enabled, batch["obs"]),
                               ("next_enabled", sample.next_enabled, batch["next_obs"])):
            value = torch.as_tensor(raw, device=self.device)
            if value.shape != (obs[0].shape[0], 4) or not bool(((value == 0) | (value == 1)).all()):
                raise ValueError("invalid replay execution gates")
            batch[name] = value.to(torch.float32)
        for i, action in enumerate(batch["acs"]):
            if bool((action[batch["enabled"][:, i] == 0] != 0).any()):
                raise ValueError("critic replay contains an unapplied residual")

    def _project_policy_actions(self, batch, actions, *, next_state=False):
        gates = batch["next_enabled" if next_state else "enabled"]
        return [action*gates[:, i:i+1] for i, action in enumerate(actions)]

    def _actor_sample_weights(self, batch, agent_i):
        weights = batch["enabled"][:, agent_i:agent_i+1]
        if self.repairs["inactive_actor"]:
            weights = weights * (batch["obs"][agent_i][:, 20:21] < .5)
        return weights

    def update(self, sample, agent_i, **kwargs):
        result = super().update(sample, agent_i, **kwargs)
        self.actor_updates[agent_i] += int(self.last_actor_updated)
        return result

    def _compute_target_q(self, *args):
        target = super()._compute_target_q(*args)
        self.last_target_diagnostics = dict(mean=float(target.mean()), abs_max=float(target.abs().max()),
                                            clipped_fraction=float((target.abs() == 10).float().mean()))
        return target

    def training_state_dict(self):
        return dict(**super().training_state_dict(), repair_version=VERSION,
                    repairs=copy.deepcopy(self.repairs), actor_updates=list(self.actor_updates))

    def load_training_state_dict(self, state):
        if state.get("repair_version") != VERSION or state.get("repairs") != self.repairs:
            raise ValueError("historical or different repair weights cannot be relabeled")
        super().load_training_state_dict(state)
        counts = state["actor_updates"]
        if len(counts) != 4 or any(type(x) is not int or not 0 <= x <= self.niter for x in counts):
            raise ValueError("invalid actor update counts")
        self.actor_updates = list(counts)


class RepairedMADDPG(ExecutionRepair, MADDPG):
    pass


class RepairedBoundaryMADDPG(ExecutionRepair, BoundaryConditionedMADDPG):
    pass


def build_model(config, device="cpu"):
    c = validate(config)
    rl = c["rl"]
    boundary = c["algorithm"] == "direct_boundary_maddpg"
    cls = RepairedBoundaryMADDPG if boundary else RepairedMADDPG
    model = cls.init_from_env(_contract_environment(), gamma=rl["gamma"], tau=rl["tau"],
                lr_actor=rl["lr_actor"], lr_critic=rl["lr_critic"], hidden_dim=rl["hidden_dim"], residual_action_reg=0.)
    if boundary:
        for agent in model.agents:
            agent.policy = BoundaryConditionedPolicy(agent.policy, c.get("boundary_encoder", {}).get("hidden_dim", 32))
            agent.target_policy = copy.deepcopy(agent.policy)
            agent.policy_optimizer = torch.optim.Adam(agent.policy.parameters(), lr=model.lr_actor)
    model.repairs = copy.deepcopy(c["repairs"])
    model.actor_updates = [0]*4
    model.last_target_diagnostics = {}
    model.prep_training(device=device)
    return model
