"""Chapter-local learner repairs with unchanged external tensor dimensions."""
import copy
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from core.algorithms.maddpg import MADDPG
from core.algorithms.misc import soft_update
from core.env.observation_contract import ROLE_ORDER


class FieldScale(nn.Module):
    """Fixed units per semantic field; never normalize actions together with positions."""
    def __init__(self, joint=False):
        super().__init__()
        scale = torch.ones(28)
        scale[0:3] = torch.tensor([20., 20., 8.])
        scale[3:6] = torch.tensor([2.8, 2.8, 1.2])
        scale[6:9] = torch.tensor([20., 20., 8.])
        scale[12:15] = torch.tensor([20., 20., 8.])
        if joint:
            scale = torch.cat([scale.repeat(4), torch.ones(12)])
        self.register_buffer('scale', scale)

    def forward(self, values):
        if values.shape[-1] != self.scale.numel():
            raise ValueError('network input dimension mismatch')
        return values / self.scale


class ScaledMLP(nn.Module):
    def __init__(self, *, actor, hidden_dim):
        super().__init__()
        self.in_fn = FieldScale(joint=not actor)
        self.fc1 = nn.Linear(28 if actor else 124, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 if actor else 1)
        self.actor = actor
        if actor:
            nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
            nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, values):
        values = F.relu(self.fc1(self.in_fn(values)))
        values = F.relu(self.fc2(values))
        values = self.fc3(values)
        return torch.tanh(values) if self.actor else values


class LongHorizonMADDPG(MADDPG):
    """Reuse frozen core infrastructure, overriding learning contracts explicitly."""
    def configure(self, config):
        self.repair = copy.deepcopy(config['rl'])
        self.exploration = copy.deepcopy(config['exploration'])
        self.actor_updates = [0] * 4
        self.base_sigmas = [a.noise.sigma for a in self.agents]
        if self.repair['input_scaling'] == 'field_scale':
            for agent in self.agents:
                agent.policy = ScaledMLP(actor=True, hidden_dim=self.hidden_dim)
                agent.critic1 = ScaledMLP(actor=False, hidden_dim=2*self.hidden_dim)
                agent.critic2 = ScaledMLP(actor=False, hidden_dim=2*self.hidden_dim)
                agent.target_policy = copy.deepcopy(agent.policy)
                agent.target_critic1 = copy.deepcopy(agent.critic1)
                agent.target_critic2 = copy.deepcopy(agent.critic2)
                agent.policy_optimizer = torch.optim.Adam(agent.policy.parameters(), lr=self.lr_actor)
                agent.critic1_optimizer = torch.optim.Adam(agent.critic1.parameters(), lr=self.lr_critic)
                agent.critic2_optimizer = torch.optim.Adam(agent.critic2.parameters(), lr=self.lr_critic)
        for agent in self.agents:
            for net in (agent.target_policy, agent.target_critic1, agent.target_critic2):
                net.requires_grad_(False)
        self._clear_sample_cache()
        return self

    def exploration_at(self, environment_step):
        progress = min(1., max(0., environment_step / self.exploration['decay_steps']))
        factor = 1. - progress * (1. - self.exploration['end_factor'])
        for agent, sigma in zip(self.agents, self.base_sigmas):
            agent.scale_noise(sigma*factor)
        return [a.noise.sigma for a in self.agents]

    def _active(self, obs):
        return (obs[..., 20:21] < .5).to(obs.dtype)

    def _canonical(self, action, obs):
        return action * self._active(obs) if self.repair['mask_inactive_actions'] else action

    def step(self, observations, explore=False):
        obs = [self._ensure_obs_tensor(o) for o in observations]
        actions = super().step(obs, explore=explore)
        return [self._canonical(a, o) for a, o in zip(actions, obs)]

    def _build_target_actions(self, next_obs):
        actions = super()._build_target_actions(next_obs)
        return [self._canonical(a, o) for a, o in zip(actions, next_obs)]

    def _prepare_sample(self, sample):
        # Do not cache by id(sample): identity can be recycled by independent callers.
        self._clear_sample_cache()
        batch = super()._prepare_sample(sample)
        batch['acs'] = [self._canonical(a, o) for a, o in zip(batch['acs'], batch['obs'])]
        batch['joint_vf_in'] = torch.cat((*batch['obs'], *batch['acs']), dim=1)
        return batch

    def _compute_target_q(self, ag, target_vf_in, rew_i, done_i):
        with torch.no_grad():
            next_q = torch.minimum(ag.target_critic1(target_vf_in), ag.target_critic2(target_vf_in))
            # torch.where prevents terminal 0*NaN from poisoning a terminal reward.
            target = rew_i.view(-1, 1) + torch.where(done_i.view(-1, 1).bool(),
                    torch.zeros_like(next_q), self.gamma*next_q)
            if not torch.isfinite(target).all():
                raise FloatingPointError('nonfinite critic target')
            return target

    def update(self, sample, agent_i, parallel=False, logger=None):
        batch = self._prepare_sample(sample)
        ag = self.agents[agent_i]
        target = self._compute_target_q(ag, batch['target_vf_in'], batch['rews'][agent_i], batch['dones'][agent_i])
        q1, q2 = ag.critic1(batch['joint_vf_in']), ag.critic2(batch['joint_vf_in'])
        td_error = (target-q1).detach().flatten()
        loss = ((F.smooth_l1_loss(q1, target, reduction='none')
                + F.smooth_l1_loss(q2, target, reduction='none')) * batch['weights']).mean()
        for opt in (ag.critic1_optimizer, ag.critic2_optimizer):
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for net, opt in ((ag.critic1, ag.critic1_optimizer), (ag.critic2, ag.critic2_optimizer)):
            nn.utils.clip_grad_norm_(net.parameters(), self.repair['critic_grad_clip'], error_if_nonfinite=True)
            opt.step()
            opt.zero_grad(set_to_none=True)
        active = self._active(batch['obs'][agent_i])
        if not self.repair['mask_inactive_actions']:
            active = torch.ones_like(active)
        eligible = self.niter >= self.repair['critic_warmup_updates'] and (self.niter+1) % self.repair['actor_delay'] == 0
        actor_loss = None
        ag.policy_optimizer.zero_grad(set_to_none=True)
        if eligible and bool(active.sum()):
            actions=[]
            for j, (policy, obs) in enumerate(zip(self.policies, batch['obs'])):
                if j == agent_i:
                    action = self._canonical(policy(obs), obs)
                elif self.repair['actor_use_replay_teammates']:
                    action = batch['acs'][j].detach()
                else:
                    with torch.no_grad():
                        action = self._canonical(policy(obs), obs)
                actions.append(action)
            for parameter in ag.critic1.parameters():
                parameter.requires_grad_(False)
            try:
                value = ag.critic1(torch.cat((*batch['obs'], *actions), dim=1))
                actor_loss = -(value*active).sum()/active.sum()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(ag.policy.parameters(), self.repair['actor_grad_clip'], error_if_nonfinite=True)
                ag.policy_optimizer.step()
                self.actor_updates[agent_i] += 1
            finally:
                for parameter in ag.critic1.parameters():
                    parameter.requires_grad_(True)
        self.last_update = dict(agent=agent_i, target_mean=float(target.mean()), target_abs_max=float(target.abs().max()),
             q_mean=float(q1.detach().mean()), td_abs_mean=float(td_error.abs().mean()),
             active_fraction=float(active.mean()), actor_updated=actor_loss is not None,
             critic_loss=float(loss.detach()), actor_loss=None if actor_loss is None else float(actor_loss.detach()))
        return float(loss.detach()), self.last_update['actor_loss'], td_error

    def update_all_targets(self, compute_diff=False):
        # Targets move after all four critic/actor updates, never midway through a batch.
        for ag in self.agents:
            for target, source in ((ag.target_critic1, ag.critic1), (ag.target_critic2, ag.critic2)):
                soft_update(target, source, self.tau)
            if self.niter >= self.repair['critic_warmup_updates'] and (self.niter+1) % self.repair['actor_delay'] == 0:
                soft_update(ag.target_policy, ag.policy, self.tau)
        self.niter += 1
        self._clear_sample_cache()

    def training_state_dict(self):
        return dict(**super().training_state_dict(), repair=copy.deepcopy(self.repair),
                    exploration=copy.deepcopy(self.exploration), actor_updates=list(self.actor_updates))

    def load_training_state_dict(self, state):
        if state['repair'] != self.repair or state['exploration'] != self.exploration:
            raise ValueError('long-horizon learner settings mismatch')
        super().load_training_state_dict(state)
        self.actor_updates = list(state['actor_updates'])


def build_model(config):
    from .config import validate
    c=validate(config)
    env=SimpleNamespace(num_agents=4, role_names=list(ROLE_ORDER),
        agent_specs=[dict(name=role) for role in ROLE_ORDER],
        observation_space={f'agent_{i}':SimpleNamespace(shape=(28,)) for i in range(4)},
        action_space={f'agent_{i}':SimpleNamespace(shape=(3,)) for i in range(4)})
    rl=c['rl']
    return LongHorizonMADDPG.init_from_env(env, gamma=rl['gamma'], tau=rl['tau'],
            hidden_dim=rl['hidden_dim'], lr_actor=rl['lr_actor'], lr_critic=rl['lr_critic'],
            residual_action_reg=0.).configure(c)
