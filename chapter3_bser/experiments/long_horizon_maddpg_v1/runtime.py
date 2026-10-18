"""D2 rollout adapter independent of the frozen B2/B3 training package."""
import copy
import random

import numpy as np
import torch

from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from chapter3_bser.experiments.d2_v1 import assembly
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.training_env import Phase1CV2TrainingEnv
from chapter3_bser.experiments.phase1c_prrac.runtime_factory import build_prrac_online_controller
from chapter3_bser.experiments.phase1c_prrac.training_env import PRRACTrainingEnv
from chapter3_bser.integration.guided_env import GuidedEnv
from chapter3_bser.online.config import execution_runtime_config, load_phase1b2_config
from chapter3_bser.online.mission_context import OnlineMissionContext
from core.config.ch3_config import build_ch3_config
from core.env.mission_env import MissionCoreEnv, environment_kwargs_from_config
from core.env.task_protocol import PROTOCOL_FIELDS, strict_terminal

from .config import validate


def seed_all(seed):
    random.seed(seed)
    np.random.seed(np.random.SeedSequence(seed).generate_state(4))
    torch.manual_seed(seed)


def observations(values):
    result = [torch.as_tensor(v).detach().cpu().numpy().copy() for v in values]
    if np.asarray(result).shape != (4, 28) or not np.isfinite(result).all():
        raise ValueError('finite four-agent 28D observations required')
    return result


class MissionRuntime:
    def __init__(self, config, scenario, *, seed, episode_id=0):
        self.config=validate(config)
        self.scenario=copy.deepcopy(scenario)
        c=self.config
        assembly.prepare(self,c,scenario)
        seed_all(seed)
        settings=build_ch3_config(c['base_candidate'],c['profile'])
        settings.update({key:c[key] for key in (*PROTOCOL_FIELDS,'collision_terminal_reward')})
        base=MissionCoreEnv(**environment_kwargs_from_config(settings, device='cpu', max_steps=400, return_numpy=False))
        self.env=PRRACTrainingEnv(Phase1CV2TrainingEnv(GuidedEnv(base,enabled=True),reward_config=c['reward']),
                                  reward_objective_config=c,gamma=c['rl']['gamma'])
        self.found_step=None
        self.handoff_event_step=None
        self.handoff_decision_step=None
        try:
            self.env.reset(scenario=scenario,episode_id=episode_id,episode_index=episode_id)
            seed_all(seed)
            phase=load_phase1b2_config()
            options=execution_runtime_config(c)
            phase['execution_runtime']=copy.deepcopy(options)
            self.provider=OnlinePlanningStateProvider(self.env,refresh_interval=int(phase['online']['state_refresh_interval']),
                **{k:options[k] for k in ('refresh_on_executor_handoff','refresh_on_public_target_shift',
                                        'public_target_update_distance','public_target_update_min_steps')})
            self.state=self.provider.initialize()
            context=self._context()
            self.controller=assembly.build_controller(self,build_prrac_online_controller(phase,c))
            initial=self.controller.initialize(self.state,context)
            self.bridge=assembly.build_bridge(self)
            self.guidance=self.bridge.compile_guidance(initial.allocation,self.state,context,decision_reason='INITIALIZE')
            self.env.install_guidance(self.guidance)
            self.observations=self.env.refresh_observation_after_guidance()
            observations(self.observations)
            self._record_events()
        except BaseException:
            self.env.close()
            raise

    def _context(self):
        return OnlineMissionContext.from_public_views(self.env.get_task_state(),self.env.get_search_execution_state(),self.state)

    @property
    def step(self):
        return int(self.env.get_task_state().step)

    @property
    def terminal(self):
        return strict_terminal(self.env)

    def _record_events(self):
        task=self.env.get_task_state()
        if task.target_found and self.found_step is None:
            self.found_step=int(self.env.unwrapped.found_step)
        if task.executor_knows_target and self.handoff_event_step is None:
            self.handoff_event_step=task.handoff_step
        if bool(self.env.get_search_execution_state().target_known_by_agent[3]) and not self.terminal and self.handoff_decision_step is None:
            self.handoff_decision_step=self.step

    def advance(self,model,*,explore=True):
        if self.terminal:
            raise RuntimeError('cannot step a terminal mission')
        before=self.step
        obs=observations(self.observations)
        actions=torch.cat(model.step(obs,explore=explore),dim=0).detach().cpu()
        if actions.shape != (4,3) or not torch.isfinite(actions).all():
            raise ValueError('finite 4x3 actions required')
        step_obs,rewards,dones=self.env.step(actions)
        if self.terminal:
            self.observations=step_obs
        else:
            self.state=self.provider.snapshot(force=False)
            result=self.controller.step(self.state,self._context())
            self.env.observe_controller_result(result,controller=self.controller,state_provider=self.provider)
            self.guidance=self.bridge.compile_guidance(result.allocation,self.state,self._context(),decision_reason=result.decision_reason)
            self.env.install_guidance(self.guidance)
            self.observations=self.env.refresh_observation_after_guidance()
        self._record_events()
        dones=[bool(d) for d in dones]
        if self.step != before+1 or self.terminal != all(dones):
            raise RuntimeError('mission clock/done mismatch')
        return dict(t=before,observations=obs,actions=actions.numpy().copy(),
                    rewards=torch.as_tensor(rewards).cpu().numpy().copy(),dones=dones,
                    next_observations=observations(self.observations),
                    **copy.deepcopy(self.env.reward_accounting.last))

    def summary(self):
        return dict(self.env.finalize_episode(),planner=assembly.diagnostics(self),
                    scenario_id=self.scenario['scenario_id'],scenario_seed=self.scenario['scenario_seed'],
                    actual_length=self.step,found_step=self.found_step,
                    handoff_event_step=self.handoff_event_step,handoff_decision_step=self.handoff_decision_step)

    def close(self):
        self.env.close()
