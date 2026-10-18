"""Read-only recomputation of compact shadow results; creates a new analysis file.

Run: python -B -m docs.chapter3_map_stp.analyze_shadow --results <new score directory>
Scene bootstrap describes these six reused development scenes, not confirmation.
"""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import statistics


def mean(values):
    return statistics.mean(values) if values else None


def interval(values):
    ordered=sorted(values)
    def quantile(q):
        index=(len(ordered)-1)*q; left=int(index); fraction=index-left
        return ordered[left]*(1-fraction)+ordered[min(left+1,len(ordered)-1)]*fraction
    return [quantile(.025),quantile(.975)]


def analyze(directory):
    rows=json.loads((directory/'cases.json').read_text(encoding='utf-8'))['cases']
    teams=[r for r in rows if r['team'] and r['team']['combinations']>1]
    scenes=sorted({r['identity']['scenario_index'] for r in teams})
    result=dict(statistical_unit='scenario; snapshots nested within scenario',
        scene_count=len(scenes),team_snapshots=len(teams),methods={},paired_comparisons={},
        timing_seconds=[r['scoring_seconds'] for r in rows],generation={},case_table=[])
    methods=('static','arrival_discount','time_marginal','correlated_first')
    for method in methods:
        values={s:mean([r['team']['validation_scores'][method]['correlated_first'] for r in teams
                       if r['identity']['scenario_index']==s]) for s in scenes}
        result['methods'][method]=dict(scene_weighted_validation_first=mean(list(values.values())),
            snapshot_weighted_validation_first=mean([r['team']['validation_scores'][method]['correlated_first'] for r in teams]),
            scene_weighted_hit_probability=mean([mean([r['team']['validation_scores'][method]['dynamic_hit_probability']
                for r in teams if r['identity']['scenario_index']==s]) for s in scenes]))
    for method in methods[:-1]:
        values=[mean([r['team']['comparisons'][method]['validation_delta'] for r in teams
                     if r['identity']['scenario_index']==s]) for s in scenes]
        distribution=[mean(draw) for draw in product(values,repeat=len(values))]
        result['paired_comparisons'][method]=dict(scene_weighted_delta=mean(values),
            descriptive_scene_bootstrap95=interval(distribution),bootstrap_draws=len(distribution),
            positive_scenes=sum(v>1e-12 for v in values),negative_scenes=sum(v < -1e-12 for v in values))
    selected=[r['team']['validation_scores']['time_marginal'] for r in teams]
    result['marginal_calibration_under_shared_model']=dict(
        mean_independence_cumulative_probability=mean([v['marginal_hit_probability'] for v in selected]),
        mean_correlated_cumulative_probability=mean([v['dynamic_hit_probability'] for v in selected]),
        caveat='same model sample, not calibration against real environment outcomes')
    for r in rows:
        for g in r['generation']:
            for key,value in g.items():
                if key!='agent_id' and isinstance(value,(int,float)):
                    result['generation'][key]=result['generation'].get(key,0)+value
    for r in teams:
        result['case_table'].append(dict(scenario=r['identity']['scenario_index'],step=r['identity']['step'],
            combinations=r['team']['combinations'],choices=r['team']['choices'],
            validation_first={m:r['team']['validation_scores'][m]['correlated_first'] for m in methods},
            delta_vs_static=r['team']['comparisons']['static']['validation_delta'],
            particle_interval=r['team']['comparisons']['static']['particle_bootstrap95']))
    all_agents=[a for r in rows for a in r['agents']]
    result['agent_ranking_changes']={method:sum(a['choices'][method]!=a['choices']['correlated_first'] for a in all_agents)
                                     for method in methods[:-1]}
    result['agent_ranking_denominator']=len(all_agents)
    result['cost']=dict(total_scoring_seconds=sum(result['timing_seconds']),
                       median_snapshot_seconds=statistics.median(result['timing_seconds']),
                       max_snapshot_seconds=max(result['timing_seconds']))
    result['input_sha256']=hashlib.sha256((directory/'cases.json').read_bytes()).hexdigest()
    result['analysis_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True)
    args=parser.parse_args(); directory=Path(args.results)
    result=analyze(directory)
    with (directory/'analysis.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('case_table','timing_seconds')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
