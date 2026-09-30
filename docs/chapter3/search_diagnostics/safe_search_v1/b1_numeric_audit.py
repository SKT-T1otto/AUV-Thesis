"""Independent recomputation from terminal JSON, without analysis helpers."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(analysis_path, run, output):
    run,output=Path(run),Path(output)
    if output.exists():raise ValueError('Output must be new')
    a=read(analysis_path);identity=read(run/'identity.json')
    assert identity['experiment_complete'] is True
    assert identity['sources_before']==identity['sources_after']
    assert len(identity['trace_sha256'])==60
    actual=read(run/'episodes.json')
    d2=Path(identity['references']['B0_D2']['directory'])
    ac=Path(identity['references']['B0_AC']['directory'])
    historical=[r for r in read(d2/'episodes.json') if r['variant'] in ('V0','V3')]+read(ac/'episodes.json')
    checks=[]
    for baseline,rows in (('B0',historical),('B1',actual)):
        assert len(rows)==60
        assert len({(r['original_episode_index'],r['variant']) for r in rows})==60
        for v in ('V0','V3','V5'):
            records=[r for r in rows if r['variant']==v]
            assert len(records)==20
            s=a['results'][baseline]['variants'][v]
            found=sum(r['found_step'] is not None for r in records)
            success=sum(r['stop_reason']=='success' for r in records)
            collisions=sum(r['episode_result']['first_collision_step'] is not None and
                (r['found_step'] is None or r['episode_result']['first_collision_step']<=r['found_step']) for r in records)
            exposure=sum(r['found_step'] if r['found_step'] is not None else r['physical_steps'] for r in records)
            stall=sum(r['searcher_motion_stall_proxy_agent_steps'] for r in records)
            hold=sum(r['searcher_hold_agent_steps'] for r in records)
            effective=sum(r['search_coverage']['effective_observation_steps'] for r in records)
            assert (found,success,collisions,exposure,stall,hold)==(s['found_count'],s['success_count'],s['pre_found_collision_count'],s['pre_found_exposure_steps'],s['searcher_motion_stall_proxy_agent_steps'],s['searcher_hold_agent_steps'])
            assert math.isclose((stall+hold)/(3*exposure),s['stall_plus_hold_fraction'],abs_tol=1e-12)
            assert math.isclose(effective/exposure,s['actual_belief_footprint']['effective_observation_fraction'],abs_tol=1e-12)
            checks.append(dict(baseline=baseline,variant=v,found=found,success=success,pre_found_collision=collisions,
                exposure=exposure,stall=stall,hold=hold,effective=effective))
        by_key={(r['original_episode_index'],r['variant']):r for r in rows}
        for new,old in (('V5','V3'),('V5','V0'),('V3','V0')):
            ds=[]
            for scene in identity['selected']:
                i=scene['original_episode_index']
                ds.append(int(by_key[i,new]['found_step'] is not None)-int(by_key[i,old]['found_step'] is not None))
            reported=a['results'][baseline]['comparisons'][f'{new}_minus_{old}']['found']
            assert math.isclose(sum(ds)/20,reported['mean_delta'],abs_tol=1e-12)
            assert ds.count(1)==len(reported['gained_indices'])
            assert ds.count(-1)==len(reported['lost_indices'])
    for r in actual:
        assert r['baseline']=='B1_bser_prior'
        assert r['terminal'] is True and r['full_episode_completed'] is True
        assert r['complete_episode_row']['method']=='ch3_baseline_bser_prior'
        assert r['complete_episode_row']['reference_runtime_method']=='ch3_baseline_bser_prior'
        for f in ('actor_forward_calls','action_sampling_calls','optimizer_update_count','residual_action_max_abs','physical_residual_acceleration_max_abs'):
            assert r['controller'][f]==0
    s=a['results']['B1']['variants'];new,old=s['V5'],s['V3']
    expected=(new['found_count']>=old['found_count'] and new['pre_found_collision_count']<=old['pre_found_collision_count'] and
        new['stall_plus_hold_fraction']<old['stall_plus_hold_fraction'] and
        new['actual_belief_footprint']['effective_observation_fraction']>=old['actual_belief_footprint']['effective_observation_fraction'])
    assert a['b1_gate']['passed'] is expected
    assert a['b1_physical_steps']==sum(r['physical_steps'] for r in actual)
    result=dict(schema='ch3.safe_search.b1_independent_numeric_audit.v1',passed=True,
        recomputed_variants=checks,b1_complete_episodes=60,independent_scenarios=20,
        b1_physical_steps=a['b1_physical_steps'],development_gate=expected,
        analysis_sha256=hashlib.sha256(Path(analysis_path).read_bytes()).hexdigest(),
        audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',required=True,type=Path);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();audit(a.analysis,a.run,a.output)
