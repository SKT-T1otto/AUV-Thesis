"""Independent bounded rollout/replay/update observer; production is unpatched."""
import copy
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from core.replay.ch3_buffer import CH3ReplayBuffer
from chapter3_bser.experiments.baselines.common.model import build_model as legacy_model
from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime
from chapter3_bser.experiments.d2_repair_v2.model import build_model
from chapter3_bser.experiments.d2_repair_v2.replay import ExecutionReplay,ExecutionSample
from chapter3_bser.experiments.d2_repair_v2.runtime import RepairedMissionRuntime
from tests.d2_v2_test_utils import ROOT,config,state,delta,state_digest,evidence_dir
from tests.d2_v2_scenarios import scenario

def plain(x):
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.integer,np.floating,np.bool_)): return x.item()
    if isinstance(x,dict): return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    return x

def subset(s,mask):
    values=tuple([v[mask] for v in x] if isinstance(x,list) else x[mask] for x in s)
    return ExecutionSample(values,s.enabled[mask],s.next_enabled[mask]) if hasattr(s,'enabled') else values

def probe(label,repair):
    started=time.perf_counter(); torch.set_num_threads(1); torch.manual_seed(31729)
    c=config(label)
    if not repair:
        stem='direct_mc' if label=='b2' else 'direct_boundary'
        c=json.loads((ROOT/f'configs/chapter3/d2_v1/{stem}_train.json').read_text(encoding='utf-8'))
    c['max_steps']=32
    model=(build_model if repair else legacy_model)(c)
    replay=ExecutionReplay(c['rl']['replay_size']) if repair else CH3ReplayBuffer(c['rl']['replay_size'],4,[28]*4,[3]*4,success_priority=1.,alpha=0.,beta_start=0.)
    records=[]; cases=[]; legacy_ineffective=0; coverage=set()
    for number,kind in enumerate(('normal','normal','normal','normal','near','success','collision')):
        horizon=32 if kind=='normal' else 12
        cc=copy.deepcopy(c); cc['max_steps']=horizon
        rt=(RepairedMissionRuntime if repair else BaselineMissionRuntime)(cc,scenario(kind,horizon,31729+number),seed=31729)
        actor_raw={}; noise={}; hooks=[]; previous=None; case_rows=[]
        for i,a in enumerate(model.agents):
            hooks.append(a.policy.register_forward_hook(lambda n,args,out,i=i:actor_raw.__setitem__(i,out.detach().cpu().clone())))
            original=a.noise.sample
            def observed_noise(i=i,original=original):
                value=original(); noise[i]=value.detach().cpu().clone(); return value
            a.noise.sample=observed_noise
        originals=[(a,a.noise.sample.__defaults__[1]) for a in model.agents]
        received=[]; original_step=rt.env.step
        def observed_step(actions): received.append(actions.detach().cpu().clone()); return original_step(actions)
        rt.env.step=observed_step
        try:
            while not rt.terminal:
                before=rt.env.unwrapped.agent_finished.clone()
                obs=np.asarray(rt.observations).copy()
                before_pos=rt.env.unwrapped._agent_pos.clone()
                before_vel=rt.env.unwrapped._agent_vel.clone()
                row=rt.advance(model,explore=True); u=rt.env.unwrapped
                raw=torch.cat([actor_raw[i] for i in range(4)])
                exploration=torch.stack([noise[i] for i in range(4)])
                clipped=(raw+exploration).clamp(-1,1)
                gates=torch.as_tensor(row['action_enabled']) if repair else ~before
                sent=clipped*gates[:,None] if repair else clipped
                np.testing.assert_array_equal(row['observations'],obs)
                if previous is not None: np.testing.assert_array_equal(previous,row['observations'])
                previous=row['next_observations']
                torch.testing.assert_close(torch.as_tensor(row['actions']),sent,rtol=0,atol=0)
                torch.testing.assert_close(received[-1],sent,rtol=0,atol=0)
                effective=sent.clone(); effective[before]=0
                expected=u._actions_to_residual_acc(effective)
                if u.use_residual_prior: expected=expected*u._residual_scale[:,None]
                torch.testing.assert_close(u._last_residual_acc,expected,rtol=0,atol=0)
                final=u._last_prior_acc*u._prior_strength[:,None]+expected if u.use_residual_prior else expected.clone()
                final[:,:2]=torch.clamp(final[:,:2],-u._a_xy_max[:,None],u._a_xy_max[:,None])
                final[:,2]=torch.clamp(final[:,2],-u._a_z_max,u._a_z_max)
                final[before]=0
                torch.testing.assert_close(u._agent_acc,final,rtol=0,atol=0)
                if repair: assert not bool((sent[~gates] != 0).any())
                else: legacy_ineffective+=int(bool((sent[before] != 0).any()))
                np.testing.assert_array_equal(row['rewards'],u._rewards_to_public(torch.full((4,),row['team_reward'])))
                assert row['dones']==[rt.terminal]*4
                task=rt.env.get_task_state()
                if not task.target_found: coverage.add('pre_found')
                if row['t']==0 and task.target_found: coverage.add('discovery')
                if task.target_found and not task.executor_knows_target: coverage.add('waiting_handoff')
                if task.executor_knows_target: coverage.add('executor_informed')
                if bool(before[:3].all()): coverage.add('frozen_searchers')
                record=dict(**row,actor_raw=raw,exploration_noise=exploration,clipped=clipped,received=received[-1],
                    effective_normalized=effective,physical_residual=u._last_residual_acc.clone(),
                    physical_prior=u._last_prior_acc.clone(),acceleration_command=u._agent_acc.clone(),
                    physical_net_acceleration=(u._agent_vel-before_vel)/u.dt,
                    physical_velocity=(u._agent_pos-before_pos)/u.dt,dt=u.dt,
                    raw_reward_components={k:v.clone() for k,v in u.last_reward_components.items()},
                    target_found=task.target_found,executor_informed=task.executor_knows_target,
                    delivery_step=u.executor_received_target_step,hold_counter=u._capture_hold_counter)
                if repair: replay.push_transition(row)
                else: replay.push(row['observations'],row['actions'],row['rewards'],row['next_observations'],row['dones'],[False]*4)
                records.append(record); case_rows.append(record)
            outcome=u.episode_outcome.termination_reason
            coverage.add(outcome)
            cases.append(dict(kind=kind,steps=rt.step,found=rt.found_step,delivery=u.executor_received_target_step,outcome=outcome))
            assert rt.step<=32
            print(f'chain {label} repair={repair} {kind} steps={rt.step} {outcome}',flush=True)
        finally:
            rt.close()
            for h in hooks:h.remove()
            for a,original in originals:a.noise.sample=original
    expected={'pre_found','discovery','waiting_handoff','executor_informed','frozen_searchers','success','obstacle_collision','timeout'}
    assert expected<=coverage,(label,repair,coverage,cases)
    assert len(records)>=c['rl']['batch_size']
    all_sample=replay.sample(len(records),norm_rews=False)
    for k,index in enumerate(all_sample[6].tolist()):
        source=records[index]
        for part,name in ((0,'observations'),(1,'actions'),(2,'rewards'),(3,'next_observations'),(4,'dones')):
            for i in range(4): np.testing.assert_array_equal(all_sample[part][i][k],np.asarray(source[name][i]))
        if repair:
            np.testing.assert_array_equal(all_sample.enabled[k],source['action_enabled'])
            np.testing.assert_array_equal(all_sample.next_enabled[k],source['next_action_enabled'])
    frozen=subset(all_sample,all_sample[0][0][:,20]>=.5)
    active=subset(all_sample,all_sample[0][0][:,20]<.5)
    assert len(frozen[5]) and len(active[5])
    mixed=replay.sample(c['rl']['batch_size'],norm_rews=False)
    counts=[0]*4; frozen_deltas=[]; target_checks=0
    for round_no,batch in enumerate((mixed,mixed,frozen,active)):
        for i,a in enumerate(model.agents):
            own=state(a.policy); opt=state_digest(a.policy_optimizer.state_dict())
            targets=state(a.target_policy); other=[state(x.policy) for j,x in enumerate(model.agents) if j!=i]
            if round_no==0:
                model.update_critic_only(batch,i)
                assert delta(own,a.policy)==0 and opt==state_digest(a.policy_optimizer.state_dict())
            else:
                model.update(batch,i); counts[i]+=int(model.last_actor_updated)
            assert delta(targets,a.target_policy)==0
            assert all(delta(s,x.policy)==0 for s,x in zip(other,[x for j,x in enumerate(model.agents) if j!=i]))
            if round_no==2 and i<3:
                frozen_deltas.append(delta(own,a.policy))
                if repair:
                    assert delta(own,a.policy)==0 and opt==state_digest(a.policy_optimizer.state_dict())
            assert all(torch.isfinite(p).all() for x in model.agents for net in (x.policy,x.critic1,x.critic2,x.target_policy,x.target_critic1,x.target_critic2) for p in net.parameters())
        pairs=[]
        for a in model.agents:
            for t,o in ((a.target_policy,a.policy),(a.target_critic1,a.critic1),(a.target_critic2,a.critic2)):
                pairs.extend((p,p.detach().clone(),q.detach().clone()) for p,q in zip(t.parameters(),o.parameters()))
        model.update_all_targets()
        for actual,old,online in pairs: torch.testing.assert_close(actual,old*.995+online*.005,rtol=2e-6,atol=6e-8)
        target_checks+=len(pairs)
    assert model.niter==4
    if repair: assert model.actor_updates==counts
    for i,a in enumerate(model.agents):
        for opt,n in ((a.policy_optimizer,counts[i]),(a.critic1_optimizer,4),(a.critic2_optimizer,4)):
            assert all(int(s['step'])==n for s in opt.state.values())
    out=evidence_dir()
    (out/f'trace_{label}_{repair}.json').write_bytes((json.dumps(plain(records))+'\n').encode())
    result=dict(algorithm=label,repair=repair,scenarios=cases,physical_steps=len(records),coverage=sorted(coverage),
        rollout_replay_alignment=True,team_reward_exact=True,update_rounds=4,actor_steps=counts,critic_steps=32,
        target_parameter_checks=target_checks,frozen_actor_deltas=frozen_deltas,
        legacy_ineffective_commands=legacy_ineffective,wall_seconds=time.perf_counter()-started,
        reward_weights_changed=False,networks_changed=False,performance_evidence=False)
    del model,replay,records; gc.collect()
    return result
