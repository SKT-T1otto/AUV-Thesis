"""Static thesis-development figures from completed B1 transfer analysis."""
import argparse
from pathlib import Path

from chapter3_bser.experiments.safe_search_v1.run_paired import read_json,write_json,file_hash


def render(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    source,output=Path(source),Path(output)
    a=read_json(source)
    if not a['complete'] or a['new_b1_episodes']!=60 or a['independent_scenarios']!=20:
        raise ValueError('Full verified analysis required')
    if output.exists():raise ValueError('Use a new figure directory')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
    variants=['V0','V3','V5'];colors=['#737B87','#D18E40','#217B83']
    fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    fields=[('found_rate','Found (%)'),('pre_found_collision_rate','Pre-Found collision (%)'),
            ('stall_plus_hold_fraction','Search stall + Hold (%)')]
    for row,base in enumerate(('B0','B1')):
        for col,(field,title) in enumerate(fields):
            ax=axs[row,col]
            vals=[100*a['results'][base]['variants'][v][field] for v in variants]
            bars=ax.bar(variants,vals,color=colors,width=.58)
            ax.set_ylim(0,100);ax.set_ylabel(title);ax.set_title(base,loc='left',fontweight='bold')
            ax.bar_label(bars,labels=[f'{y:.1f}' for y in vals],padding=4)
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Transfer of fixed repairs: same 20 development scenarios',fontsize=15)
    fig.supxlabel('V0: native   |   V3: C   |   V5: A + C; pooled rates; no training')
    for ext in ('png','pdf'):fig.savefig(output/f'b1_transfer_overview.{ext}')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    labels=['V5 - V3','V5 - V0','V3 - V0']
    for ax,(field,title) in zip(axs,[('found','Found change (percentage points)'),('pre_found_collision','Pre-Found collision change (pp)')]):
        for offset,base,color,marker in ((-.1,'B0','#737B87','o'),(.1,'B1','#217B83','s')):
            xs=[];lo=[];hi=[]
            for pair in labels:
                r=a['results'][base]['comparisons'][pair.replace(' - ','_minus_')][field]
                x=100*r['mean_delta'];l,u=[100*k for k in r['bootstrap_95_percentile_ci']]
                xs.append(x);lo.append(x-l);hi.append(u-x)
            ax.errorbar(xs,np.arange(3)+offset,xerr=[lo,hi],fmt=marker,color=color,capsize=4,label=base)
        ax.axvline(0,color='#888888',lw=1,ls=':');ax.set_yticks(range(3),labels)
        ax.invert_yaxis();ax.set_xlabel(title);ax.legend(frameon=False);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Paired scene differences with descriptive 95% bootstrap intervals',fontsize=13)
    fig.supxlabel('20 paired scenes; 10,000 resamples; development evidence, no multiplicity adjustment')
    for ext in ('png','pdf'):fig.savefig(output/f'b1_transfer_paired.{ext}')
    plt.close(fig)
    write_json(output/'sources.json',dict(analysis_sha256=file_hash(source),generator_sha256=file_hash(Path(__file__)),
        data_bindings=['results.B0.variants','results.B1.variants','results.*.comparisons.*'],
        figures={p.name:file_hash(p) for p in output.glob('*') if p.suffix in ('.png','.pdf')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();render(a.analysis,a.output)
