"""Independent interface counterexamples, on disposable fresh CPU objects only."""
import copy
import json
from pathlib import Path
import torch

from core.algorithms.maddpg import MADDPG
from core.replay.ch3_buffer import CH3ReplayBuffer
from chapter3_bser.experiments.baselines.common.model import build_model as parent_model
from chapter3_bser.experiments.baselines.common.checkpoint import state_digest
from chapter3_bser.experiments.d2_repair_v1.model import build_model
from chapter3_bser.experiments.d2_repair_v1.replay import ExecutionSample, ExecutionReplay

torch.set_num_threads(1)
ROOT = Path.cwd()
OUT = ROOT / 'tmp/framework_reaudit_20261018'
OUT.mkdir(parents=True, exist_ok=True)
results = {}

def config(label):
    return json.loads((ROOT/f'configs/chapter3/d2_repair_v1/{label}_train.json').read_text())

def batch(n=7):
    obs = torch.randn(n,4,28)
    obs[:,:,20] = 0
    obs[:,:,22:26] = torch.eye(4)
    obs[:,:,26:28] = torch.tensor([1.,0.])
    nxt = obs + .1
    nxt[:,:,20] = 0
    actions = torch.randn(n,4,3).tanh()
    values = ([obs[:,i].clone() for i in range(4)], [actions[:,i].clone() for i in range(4)],
              [torch.linspace(-.1,.2,n) for _ in range(4)], [nxt[:,i].clone() for i in range(4)],
              [torch.zeros(n) for _ in range(4)],torch.ones(n),torch.arange(n),torch.zeros(n,dtype=torch.bool))
    return ExecutionSample(values,torch.ones(n,4),torch.ones(n,4))

for label in ('b2','b3'):
    torch.manual_seed(5901)
    m = build_model(config(label))
    sample = batch()
    fields = list(sample)
    fields[2] = [torch.tensor([.75]) for _ in range(4)]
    fields[5] = torch.tensor([1.])
    bad = ExecutionSample(fields,sample.enabled,sample.next_enabled)
    before = state_digest(m.training_state_dict())
    losses = m.update(bad,0)
    after = state_digest(m.training_state_dict())
    results[label+'_broadcast_reward'] = dict(accepted=True, obs_batch=7, reward_batch=1,
        weight_batch=1, parameters_changed=before!=after, losses=[losses[0],losses[1]],
        reachability='malformed external caller only; current trainer supplies correct vectors')
    assert before != after

torch.manual_seed(5901)
m = build_model(config('b2'))
sample = batch()
snapshot = m.training_state_dict()
before = state_digest(snapshot)
m.update(sample,0)
after = state_digest(snapshot)
results['cpu_state_alias'] = dict(returned_snapshot_changed_without_edit=before!=after,
    same_policy_storage=snapshot['agent_params'][0]['policy']['fc1.weight'].data_ptr()==m.agents[0].policy.fc1.weight.data_ptr(),
    reachability='in-memory retained states only; immediate synchronous torch.save is unaffected')
assert before != after

def push(buffer,k):
    buffer.push([torch.full((28,),float(k)) for _ in range(4)], [torch.zeros(3) for _ in range(4)],
                [float(k)]*4,[torch.full((28,),float(k+1)) for _ in range(4)],[False]*4,[False]*4)

small = CH3ReplayBuffer(3,4,[28]*4,[3]*4,alpha=0.,beta_start=0.)
for k in range(5): push(small,k)
large = CH3ReplayBuffer(5,4,[28]*4,[3]*4,alpha=0.,beta_start=0.)
for buff in large.obs_buffs: buff.fill_(-9876.)
large.load_state_dict(small.state_dict())
push(large,5)
sampled = sorted(large.sample(4,norm_rews=False)[0][0][:,0].tolist())
results['replay_capacity_change'] = dict(saved_capacity=3,loaded_capacity=5,filled=len(large),sampled_ids=sampled,
    read_unwritten_slot=-9876. in sampled,
    reachability='generic restore to a different capacity; B2/B3 and repair trainers are fresh-only')
assert -9876. in sampled

r = ExecutionReplay(8)
r.enabled.fill_(False)
r.next_enabled.fill_(False)
push(r,1)
s = r.sample(1)
m = build_model(config('b2'))
m.update(s,0)
results['execution_replay_inherited_push'] = dict(accepted_push_without_execution_metadata=True,
    sampled_gates=s.enabled.tolist(),actor_updated=m.last_actor_updated,
    reachability='inherited .push bypasses new contract; current trainer uses .push_transition')
assert m.last_actor_updated is False

torch.manual_seed(5901)
m = build_model(config('b2'))
short = m.step([torch.zeros(28) for _ in range(3)], explore=False)
results['core_rollout_agent_count'] = dict(accepted_agents=3,returned_actions=len(short),
    reachability='raw core step only; runtime checks the final 4 x 3 action shape')
assert len(short)==3

m.update(batch(),0)
m.update_critic_only(batch(),0)
results['critic_only_diagnostic_flag'] = dict(last_actor_updated=m.last_actor_updated,
    reachability='diagnostic stale flag only; current repair trainer always calls update')
assert m.last_actor_updated is True

result = dict(counterexamples=results, production_changed=False, user_models_loaded=False,
              long_training=False, torch=torch.__version__, device='cpu')
(OUT/'interfaces.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result,indent=2))
