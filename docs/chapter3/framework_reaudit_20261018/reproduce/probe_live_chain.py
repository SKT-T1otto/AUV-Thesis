"""Fresh policies and bounded real transitions, independent of regression helpers."""
import copy
import json
from pathlib import Path
import torch
import numpy as np
from chapter3_bser.experiments.d2_repair_v1.model import build_model
from chapter3_bser.experiments.d2_repair_v1.replay import ExecutionReplay
from chapter3_bser.experiments.d2_repair_v1.runtime import RepairedMissionRuntime
from chapter3_bser.experiments.baselines.common.checkpoint import state_digest

ROOT = Path.cwd()
OUT = ROOT/'tmp/framework_reaudit_20261018'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
scene = json.loads((ROOT/'configs/scenarios/e0_equivalence/M20_MOVING_UNKNOWN_MULTI.json').read_text())['scenarios'][0]
results = []

for label in ('b2','b3'):
    c = json.loads((ROOT/f'configs/chapter3/d2_repair_v1/{label}_train.json').read_text())
    # Actual network widths, but a bounded task horizon and replay allocation.
    c['max_steps'] = 16
    torch.manual_seed(7901)
    model = build_model(c)
    all_transitions = []
    for near in (False, True):
        task = copy.deepcopy(scene)
        task['max_steps'] = c['max_steps']
        if near:
            point = (np.asarray(task['initial_agent_positions'][0])+[.4,.4,.2]).tolist()
            task.update(target_position=point,target_initial_position=point,target_initial_velocity=[.05,0,0])
        rt = RepairedMissionRuntime(c,task,seed=7902)
        saved = state_digest(model.training_state_dict())
        gates, previous = [], None
        try:
            for t in range(16):
                if rt.terminal: break
                before = np.asarray(rt.observations).copy()
                row = rt.advance(model,explore=False)
                np.testing.assert_array_equal(row['observations'],before)
                if previous is not None:
                    np.testing.assert_array_equal(previous,row['observations'])
                np.testing.assert_array_equal(row['rewards'],rt.env.reward_accounting.last['final_reward_by_agent'])
                assert np.all(row['rewards']==row['team_reward'])
                assert np.isfinite(row['actions']).all() and np.abs(row['actions']).max()<=1
                assert np.all(row['actions'][~row['action_enabled']]==0)
                np.testing.assert_array_equal(rt.env.unwrapped._last_residual_acc[~row['action_enabled']],0.)
                if rt.terminal:
                    assert not row['next_action_enabled'].any() and all(row['dones'])
                else:
                    assert not any(row['dones'])
                gates.append(row['action_enabled'].astype(int).tolist())
                previous = row['next_observations']
                all_transitions.append(row)
            assert saved == state_digest(model.training_state_dict())
            results.append(dict(model=label,near_target=near,steps=rt.step,found_step=rt.found_step,
                terminal=rt.terminal,outcome=rt.env.unwrapped.episode_outcome.termination_reason,
                gates=gates,rollout_parameters_unchanged=True,transition_alignment=True,
                stored_action_matches_gated_command=True,shared_reward_exact=True))
        finally:
            rt.close()
    replay = ExecutionReplay(64)
    for row in all_transitions: replay.push_transition(row)
    s = replay.sample(len(all_transitions))
    batch = model._prepare_sample(s)
    # Independent raw concatenation and Bellman oracle.
    with torch.no_grad():
        next_actions = model._build_target_actions(batch['next_obs'])
        projected = [a*batch['next_enabled'][:,i:i+1] for i,a in enumerate(next_actions)]
        joint = torch.cat(batch['next_obs']+projected,dim=1)
        for i,ag in enumerate(model.agents):
            q = torch.minimum(ag.target_critic1(joint),ag.target_critic2(joint))
            expected = (batch['rews'][i].view(-1,1)+.95*(1-batch['dones'][i].view(-1,1))*q).clamp(-10,10)
            actual = model._compute_target_q(ag,batch['target_vf_in'],batch['rews'][i],batch['dones'][i])
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    # Verify soft update arithmetic separately from its implementation.
    for i in range(4):
        model.update(s,i)
    pairs = []
    for ag in model.agents:
        for target,source in ((ag.target_policy,ag.policy),(ag.target_critic1,ag.critic1),(ag.target_critic2,ag.critic2)):
            for a,b in zip(target.parameters(),source.parameters()):
                pairs.append((a,a.detach().clone(),b.detach().clone()))
    model.update_all_targets()
    for target,old,online in pairs:
        torch.testing.assert_close(target,old*(1-.005)+online*.005,atol=6e-8,rtol=2e-6)
    results.append(dict(model=label,sampled_real_transitions=len(s[5]),bellman_oracle=True,
        soft_update_oracle=True,actor_steps=model.actor_updates,critic_rounds=model.niter,
        parameters_finite=all(torch.isfinite(p).all().item() for a in model.agents for net in
            (a.policy,a.target_policy,a.critic1,a.critic2,a.target_critic1,a.target_critic2) for p in net.parameters())))

result = dict(scope='bounded fresh 16-step tasks; not training or performance evidence',results=results)
(OUT/'live_chain.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result,indent=2))
