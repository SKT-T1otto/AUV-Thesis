"""Legal initial conditions for bounded mechanisms, never reward/done edits."""
import copy
import json
from tests.d2_v2_test_utils import ROOT

def scenario(kind="normal",horizon=32,seed=31729):
    value=copy.deepcopy(json.loads((ROOT/'tests/fixtures/hgr/handoff_manifest.json').read_text(encoding='utf-8'))['scenarios'][0])
    value.update(scenario_id=f'v2_mechanism_{kind}_{seed}',scenario_seed=seed,planner_seed=seed,max_steps=horizon)
    value['target_initial_velocity']=[0.,0.,0.]
    point=[16.,16.,2.] if kind in ('normal','collision') else [2.6,2.6,2.]
    if kind=='success':
        value['initial_agent_positions']=[[2.,5.2,2.],[2.,8.4,2.],[2.,2.,2.],[4.8,2.,2.]]
        point=[4.7,2.,2.]
        value['initial_executor_wait_point']=point
    value['target_position']=list(point); value['target_initial_position']=list(point)
    if kind=='collision':
        center=value['initial_agent_positions'][0]
        obstacles=[]
        for axis in (0,1):
            for sign in (-1,1):
                p=list(center); p[axis]+=sign*.004
                size=[.05,.05,.05]; size[axis]=.002
                obstacles.append(dict(center=p,size=size))
        value.update(obstacles=obstacles,obstacle_layout_id='custom_aabb_v1',flow_phase_x=1.)
    return value
