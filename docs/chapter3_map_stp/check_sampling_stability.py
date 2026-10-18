"""Post-hoc stability recomputation; does not retune the frozen decision gate."""
import argparse
import json
from pathlib import Path
import statistics


def analyze(root):
    rows=json.loads((root/'cases.json').read_text(encoding='utf-8'))['cases']
    methods=('static','arrival_discount','time_marginal','correlated_first')
    cases=[]
    for row in rows:
        t=row['team']
        if not t or t['combinations']<2:continue
        scores=t['all_scores']; choices={}
        for method in methods:
            best=max(x['validation'][method] for x in scores)
            choices[method]=next(x for x in scores if x['validation'][method]>=best-1e-12)
        cases.append(dict(scenario=row['identity']['scenario_index'],step=row['identity']['step'],
            same_choice={m:choices[m]['routes']==t['choices'][m] for m in methods},
            reverse_delta=choices['correlated_first']['selection']['correlated_first']-choices['static']['selection']['correlated_first'],
            forward_delta=t['comparisons']['static']['validation_delta']))
    scenes=sorted({c['scenario'] for c in cases})
    scene_delta={s:statistics.mean(c['forward_delta'] for c in cases if c['scenario']==s) for s in scenes}
    return dict(kind='posthoc_sampling_stability_not_new_confirmation_or_retuned_gate',cases=cases,
        same_team_choice_counts={m:sum(c['same_choice'][m] for c in cases) for m in methods},denominator=len(cases),
        reverse_scene_weighted_delta=statistics.mean(statistics.mean(c['reverse_delta'] for c in cases if c['scenario']==s) for s in scenes),
        leave_one_scene_out_delta={str(s):statistics.mean(v for other,v in scene_delta.items() if other!=s) for s in scenes})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True); parser.add_argument('--verify',action='store_true')
    args=parser.parse_args(); root=Path(args.results); result=analyze(root)
    path=root/'sampling_stability.json'
    if args.verify:
        if result!=json.loads(path.read_text(encoding='utf-8')):raise ValueError('stability recomputation differs')
        print('Sampling stability exactly reproduced.')
    else:
        with path.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':main()
