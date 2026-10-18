"""Manual train/evaluate entrypoints; defaults only preview, never resume."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from core.replay.ch3_buffer import CH3ReplayBuffer
from .config import METHOD, SCHEMA, discount_report, validate
from .model import build_model
from .provenance import ROOT, digest, preflight, require_ready, sha
from .runtime import MissionRuntime, seed_all

DEFAULT=ROOT/'configs/chapter3/long_horizon_maddpg_v1.json'
CHECKPOINT=SCHEMA+'.checkpoint.v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def population(scenes):
    return [dict(id=s['scenario_id'],seed=s['scenario_seed'],content=digest({k:v for k,v in s.items()
            if k not in ('scenario_id','scenario_seed','scenario_split','scenario_role','pair_group_id','protocol')})) for s in scenes]


def manifest(path,split):
    value=read(path)
    scenes=value.get('scenarios',[])
    if not scenes:
        raise ValueError('nonempty frozen manifest required')
    for s in scenes:
        if (not isinstance(s.get('scenario_id'),str) or not s['scenario_id']
                or type(s.get('scenario_seed')) is not int or s['scenario_seed'] < 0
                or s.get('scenario_profile') != 'M20_MOVING_UNKNOWN_MULTI'
                or s.get('scenario_split') != split or s.get('scenario_role') != split
                or s.get('max_steps') != 400
                or s.get('protocol') not in ('ch3_unknown_map_v1','CH3_UNKNOWN_MAP_V1')):
            raise ValueError('scenario profile/split/protocol/horizon mismatch')
    pop=population(scenes)
    if any(len({s[k] for s in pop})!=len(pop) for k in ('id','seed','content')):
        raise ValueError('duplicate scenario identity or content')
    return scenes


def check_disjoint(train, evaluation):
    if any({s[k] for s in train}&{s[k] for s in evaluation} for k in ('id','seed','content')):
        raise ValueError('training/evaluation scenario overlap')


def new_output(path):
    path=Path(path).resolve()
    if 'collision_terminal' not in path.parts or 'long_horizon_maddpg_v1' not in path.parts:
        raise ValueError('new method output needs its own long_horizon_maddpg_v1/collision_terminal path')
    if path.exists():
        raise FileExistsError('existing outputs are retained: '+str(path))
    return path


def state_digest(value):
    hashed=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            t=v.detach().cpu().contiguous()
            hashed.update(str((str(t.dtype),tuple(t.shape))).encode())
            hashed.update(t.numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=lambda k:(type(k).__name__,str(k))):
                hashed.update(repr(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            hashed.update(type(v).__name__.encode())
            for item in v:visit(item)
        else:hashed.update(repr(v).encode())
    visit(value)
    return hashed.hexdigest()


def load_checkpoint(path):
    payload=torch.load(Path(path),map_location='cpu',weights_only=True)
    if payload.get('schema') != CHECKPOINT or payload.get('method') != METHOD:
        raise ValueError('new-method checkpoint required; historical weights cannot be relabeled')
    config=validate(payload['config'])
    if (config!=payload['config'] or digest(config)!=payload['config_sha256']
            or payload['source']!=require_ready()
            or payload['state_sha256']!=state_digest(payload['state'])):
        raise ValueError('checkpoint config/source/state identity mismatch')
    if payload.get('episode_complete') is not True or payload.get('environment_steps',0)<1:
        raise ValueError('completed training checkpoint required')
    with torch.random.fork_rng():
        model=build_model(config)
        model.load_training_state_dict(payload['state'])
    model.prep_rollouts(device='cpu')
    return model,payload


def train(config,manifest_path,output,environment_steps,*,execute=False):
    c=validate(config)
    if type(environment_steps) is not int or environment_steps<1:
        raise ValueError('explicit positive environment-step budget required')
    scenes=manifest(manifest_path,'train')
    output=new_output(output)
    status=preflight()
    if not execute:
        return dict(executed=False,source_ready=status['ready'],unreviewed_additions=status['unreviewed_additions'],
                    discount=discount_report(c),environment_step_budget=environment_steps,output=str(output))
    source=require_ready()
    output.mkdir(parents=True)
    write(output/'config.json',c)
    write(output/'train_manifest.json',dict(scenarios=scenes))
    seed_all(c['seed'])
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model=build_model(c)
        rl=c['rl']
        replay=CH3ReplayBuffer(rl['replay_size'],4,[28]*4,[3]*4,success_priority=1.,alpha=0.,beta_start=0.)
        step=episode=0
        rows=[]
        started=time.perf_counter()
        with (output/'episodes.jsonl').open('x',encoding='utf-8',buffering=1) as episodes, \
             (output/'training_metrics.jsonl').open('x',encoding='utf-8',buffering=1) as metrics:
            while step<environment_steps:
                episode+=1
                seed=c['seed']+episode
                scene=copy.deepcopy(scenes[int(np.random.default_rng(seed).integers(len(scenes)))])
                runtime=MissionRuntime(c,scene,seed=seed,episode_id=episode)
                try:
                    model.reset_noise()
                    while not runtime.terminal:
                        model.exploration_at(step)
                        transition=runtime.advance(model,explore=True)
                        replay.push(transition['observations'],transition['actions'],transition['rewards'],
                                    transition['next_observations'],transition['dones'],[False]*4)
                        step+=1
                        if step>=rl['warmup_steps'] and step%rl['update_frequency']==0 and len(replay)>=rl['batch_size']:
                            model.prep_training(device='cpu')
                            for _ in range(rl['updates_per_train']):
                                sample=replay.sample(rl['batch_size'],norm_rews=False,device='cpu')
                                for i in range(4):
                                    model.update(sample,i)
                                    metrics.write(json.dumps(dict(step=step,critic_round=model.niter,**model.last_update),allow_nan=False)+'\n')
                                model.update_all_targets()
                            model.prep_rollouts(device='cpu')
                    row=runtime.summary()
                finally:
                    runtime.close()
                if require_ready()!=source:
                    raise ValueError('source changed during training')
                row.update(method=METHOD,episode=episode,global_step=step,environment_innovation_seed=seed)
                rows.append(row)
                episodes.write(json.dumps(row,allow_nan=False)+'\n')
                print(f'{METHOD} episode={episode} steps={step}/{environment_steps}',flush=True)
        state=model.training_state_dict()
        payload=dict(schema=CHECKPOINT,method=METHOD,config=c,config_sha256=digest(c),source=source,
                     state=state,state_sha256=state_digest(state),environment_steps=step,
                     episode_complete=True,episodes=episode,training_population=population(scenes))
        checkpoint=output/'final.pt'
        torch.save(payload,checkpoint.with_suffix('.tmp'))
        checkpoint.with_suffix('.tmp').replace(checkpoint)
        load_checkpoint(checkpoint)
        result=dict(method=METHOD,training_complete=True,actual_environment_steps=step,
                    budget_environment_steps=environment_steps,budget_policy='finish_started_episode',
                    completed_episodes=episode,critic_rounds=model.niter,actor_updates=model.actor_updates,
                    checkpoint=str(checkpoint),wall_seconds=time.perf_counter()-started,
                    discount=discount_report(c),performance_passed=False)
        write(output/'summary.json',result)
        return result
    finally:
        torch.set_num_threads(previous_threads)


def evaluate(checkpoint,manifest_path,output,*,seed=12729,execute=False):
    if type(seed) is not int or seed<0:
        raise ValueError('nonnegative evaluation seed required')
    scenes=manifest(manifest_path,'validation')
    output=new_output(output)
    if not execute:
        status=preflight()
        return dict(executed=False,checkpoint_loaded=False,source_ready=status['ready'],
                    unreviewed_additions=status['unreviewed_additions'],episodes=len(scenes),output=str(output))
    source=require_ready()
    model,payload=load_checkpoint(checkpoint)
    check_disjoint(payload['training_population'],population(scenes))
    c=payload['config']
    frozen=state_digest(model.training_state_dict())
    output.mkdir(parents=True)
    write(output/'identity.json',dict(method=METHOD,source=source,checkpoint_sha256=sha(checkpoint),
                                    manifest_sha256=sha(manifest_path),seed=seed,optimizer_updates=0))
    rows=[]
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (output/'episodes.jsonl').open('x',encoding='utf-8',buffering=1) as journal:
            for index,scene in enumerate(scenes):
                runtime=MissionRuntime(c,scene,seed=seed+index,episode_id=index)
                try:
                    while not runtime.terminal:
                        runtime.advance(model,explore=False)
                    row=runtime.summary()
                    if runtime.env.reward_accounting.steps!=runtime.step:
                        raise ValueError('reward accounting clock mismatch')
                finally:
                    runtime.close()
                collision=row['first_collision_step']
                found=row['found_step']
                row.update(method=METHOD,environment_innovation_seed=seed+index,
                    found_penalized_steps=found if row['found'] else 400,
                    pre_found_collision=collision is not None and (found is None or collision<=found))
                rows.append(row)
                journal.write(json.dumps(row,allow_nan=False)+'\n')
                if require_ready()!=source or state_digest(model.training_state_dict())!=frozen:
                    raise ValueError('source or policy changed during evaluation')
        result=dict(method=METHOD,evaluation_complete=True,episodes=len(rows),optimizer_updates=0,
                    found_rate=float(np.mean([r['found'] for r in rows])),
                    success_rate=float(np.mean([r['success'] for r in rows])),
                    collision_rate=float(np.mean([r['collision_episode'] for r in rows])),
                    pre_found_collision_rate=float(np.mean([r['pre_found_collision'] for r in rows])),
                    found_penalized_steps=float(np.mean([r['found_penalized_steps'] for r in rows])))
        write(output/'episodes.json',rows)
        write(output/'summary.json',result)
        return result
    finally:
        torch.set_num_threads(previous_threads)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('check')
    q.add_argument('--config',default=str(DEFAULT))
    q=sub.add_parser('train')
    q.add_argument('--config',default=str(DEFAULT))
    q.add_argument('--manifest',required=True)
    q.add_argument('--output',required=True)
    q.add_argument('--environment-steps',type=int,required=True)
    q.add_argument('--execute',action='store_true')
    q=sub.add_parser('evaluate')
    q.add_argument('--checkpoint',required=True)
    q.add_argument('--manifest',required=True)
    q.add_argument('--output',required=True)
    q.add_argument('--seed',type=int,default=12729)
    q.add_argument('--execute',action='store_true')
    args=vars(p.parse_args(argv))
    command=args.pop('command')
    if command=='check':
        result=dict(discount=discount_report(read(args['config'])),**preflight())
        result.pop('source')
    elif command=='train':
        config=read(args.pop('config'))
        args['manifest_path']=args.pop('manifest')
        result=train(config,**args)
    else:
        args['manifest_path']=args.pop('manifest')
        result=evaluate(**args)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return result


if __name__=='__main__':main()
