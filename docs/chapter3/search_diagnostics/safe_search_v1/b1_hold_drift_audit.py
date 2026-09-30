"""Offline one-step dynamics reconciliation, not a new simulator episode."""
import argparse
import inspect
import math
from pathlib import Path

from chapter3_bser.experiments.safe_search_v1.run_paired import ROOT,read_json,write_json
from docs.chapter3.search_diagnostics.safe_search_v1.development_baseline_audit import Inputs
from docs.chapter3.search_diagnostics.safe_search_v1.development_trace_audit import trace_rows


def audit(output):
    # Resolve the actual runtime's configuration without constructing an env,
    # advancing physics, loading a policy, or restoring a checkpoint.
    from core.config.ch3_config import build_ch3_config
    from core.env.mission_env import environment_kwargs_from_config
    from core.env.uav_env import _BaseUAVEnv
    output=Path(output)
    if output.exists():raise ValueError('Output must be new')
    inputs=Inputs()
    for f in (Path(__file__),ROOT/'core/env/uav_env.py',ROOT/'core/env/mission_env.py',
              ROOT/'chapter3_bser/integration/guided_env.py',ROOT/'chapter3_bser/experiments/safe_search_v1/safe_guidance.py'):
        inputs.watch(f)
    directory=ROOT/'runs/safe_search_v1/development_B1_20260928_v1/scene_0046_V5'
    cfg=inputs.read(directory/'resolved_config.json')
    manifest=inputs.read(directory/'evaluation_manifest.json')
    scenario=manifest['scenarios'][46]
    kwargs=environment_kwargs_from_config(build_ch3_config(cfg['base_candidate'],cfg['profile']))
    defaults=inspect.signature(_BaseUAVEnv.__init__).parameters
    params={k:kwargs.get(k,defaults[k].default) for k in ('dt','prior_kv_xy','prior_strength_search')}
    # Agent 0 and flow constants are explicit in the protected environment.
    # Keep this small replay pinned to the reviewed values, never guess others.
    assert params==dict(dt=.2,prior_kv_xy=1.1,prior_strength_search=1.)
    params.update(agent_id=0,drag_xy=.10,a_xy_max=1.30,flow_gain=.18,
        flow_phase_x=scenario['flow_phase_x'],flow_phase_y=scenario['flow_phase_y'])
    path=directory/'episode_0046/V5/step_trace.jsonl';inputs.watch(path)
    rows=list(trace_rows(path));records=[]
    for r in rows:
        if not 305<=r['step_before']<=309:continue
        a=r['before']['agents'][0];b=r['after']['agents'][0]
        assert a['hold'] is True and a['tracking_waypoint']==a['position']
        x,y=a['position'][:2];vx,vy=a['velocity'][:2]
        flow=[.18*math.sin(.25*y+params['flow_phase_x']),.18*math.cos(.22*x+params['flow_phase_y'])]
        command=[max(-1.3,min(1.3,-1.1*v)) for v in (vx,vy)]
        intermediate=[v+.2*c for v,c in zip((vx,vy),command)]
        predicted=[v+.2*(-.10*v+f) for v,f in zip(intermediate,flow)]
        actual=b['velocity'][:2]
        error=max(abs(a-b) for a,b in zip(actual,predicted))
        if error>2e-6:raise ValueError('Hold dynamics do not reconcile')
        records.append(dict(step_before=r['step_before'],step_after=r['step_after'],
            speed_before=a['speed'],actual_speed_after=b['speed'],flow_xy=flow,
            prior_acceleration_xy=command,predicted_next_velocity_xy=predicted,
            recorded_next_velocity_xy=actual,max_abs_velocity_error=error))
    assert len(records)==5
    inputs.verify()
    write_json(output,dict(schema='ch3.safe_search.b1_hold_drift_audit.v1',passed=True,
        original_episode_index=46,variant='V5',agent_id=0,parameters=params,
        no_simulator_episode_executed=True,source_hashes=inputs.hashes,transitions=records,
        max_abs_velocity_error=max(r['max_abs_velocity_error'] for r in records),
        finding='Moving-current-position Hold has zero position error. Its velocity feedback balances nonzero flow at nonzero velocity; it is not station keeping.',
        limits='Five recorded nonterminal transitions are reconciled, not a new closed-loop counterfactual experiment. Terminal collision remains the actual step-311 result.'))
    print('HOLD_DYNAMICS_RECONCILED',max(r['max_abs_velocity_error'] for r in records))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args().output)
