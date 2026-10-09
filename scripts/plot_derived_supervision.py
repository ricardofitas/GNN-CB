#!/usr/bin/env python
"""Figures/tables for the complete registered study; no model selection."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.train_derived_supervision import ROOT, OUT, ARMS, register

LABELS={
    'primary_replay':'Original GNN (exact replay)',
    'temporal_pairs':'Temporal differences',
    'temporal_balanced':'Temporal, visibility balanced',
    'spatial_pairs':'Edge differences',
    'joint_pairs':'Temporal + edge differences',
    'mc_geometry_jitter':'Monte Carlo geometry noise',
    'global_readout':'Global + local readout',
    'global_joint':'Global readout + both differences',
    'joint_video14':'Both differences + videos at 1/7',
}


def main():
    register()
    summary=json.loads((OUT/'summary.json').read_text())['arms']
    stats=json.loads((OUT/'statistics.json').read_text())
    if set(summary)!=set(ARMS):raise ValueError('All arms must finish before plotting')
    rows=[];names=list(ARMS);specimens=stats['specimen_order']
    for arm in names:
        r=summary[arm];s=stats['new_arms_vs_original'].get(arm)
        rows.append({'arm':arm,'description':LABELS[arm],'rmse_percent':r['macro_rmse_percent'],
            'seed_sd_pp':r['seed_sd_pp'],'delta_vs_original_pp':s['delta_rmse_pp'] if s else 0.,
            'relative_change_percent':s['relative_change_percent'] if s else 0.,
            'nominal_two_sided_sign_flip_p':s['nominal_two_sided_sign_flip_p'] if s else '',
            'nominal_holm_p':s['nominal_holm_p'] if s else '',
            'conditional_case_bootstrap_low_pp':s['conditional_case_bootstrap_95_percent_interval_pp'][0] if s else '',
            'conditional_case_bootstrap_high_pp':s['conditional_case_bootstrap_95_percent_interval_pp'][1] if s else '',
            **r['per_specimen']})
    with (OUT/'comparison.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys(),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'gnn-cb-derived-supervision-v1'})
    folder=ROOT/'reports/figures';folder.mkdir(exist_ok=True)
    fig,(left,right)=plt.subplots(1,2,figsize=(13,6),sharey=True,layout='constrained',
        gridspec_kw={'width_ratios':[1.1,1.4]})
    y=np.arange(len(names));mean=[summary[a]['macro_rmse_percent'] for a in names]
    sd=[summary[a]['seed_sd_pp'] for a in names]
    left.errorbar(mean,y,xerr=sd,fmt='o',color='#25758b',capsize=3,ms=5)
    left.axvline(summary['primary_replay']['macro_rmse_percent'],color='#52616c',ls='--',lw=1,label='Original GNN mean')
    left.set_yticks(y,[LABELS[a] for a in names]);left.set_ylim(len(names)-.5,-.5)
    left.set(xlabel='Coordinate RMSE / initial height (%)',title='A. Same five tests; mean ± optimizer-seed SD')
    left.grid(axis='x',alpha=.2);left.legend(fontsize=8,loc='lower right')
    values=np.array([[summary[a]['per_specimen'][s]-summary['primary_replay']['per_specimen'][s] for s in specimens] for a in names])
    limit=max(.1,float(abs(values).max()))
    im=right.imshow(values,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto',origin='upper')
    for i in range(len(names)):
        for j in range(len(specimens)):
            right.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',fontsize=8,
                color='white' if abs(values[i,j])>.63*limit else '#15202b')
    right.set_xticks(range(5),['Exp3','Exp4','Exp5','Probe1_22','Probe2_7'])
    right.tick_params(axis='y',left=False,labelleft=False)
    right.set(xlabel='Physical test specimen',title='B. Difference from original GNN (percentage points)')
    right.set_ylim(len(names)-.5,-.5)
    fig.colorbar(im,ax=right,label='Negative = lower error',shrink=.8)
    fig.suptitle('GNN-CB: derived supervision and Monte Carlo input noise',fontsize=13)
    fig.savefig(folder/'derived_supervision_results.svg',metadata={'Date':None})
    fig.savefig(OUT/'cache/derived_supervision_results.png',dpi=160);plt.close(fig)
    arms=names[1:];rs=[stats['new_arms_vs_original'][a] for a in arms]
    delta=np.array([r['delta_rmse_pp'] for r in rs]);interval=np.array([r['conditional_case_bootstrap_95_percent_interval_pp'] for r in rs])
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    ax.errorbar(delta,np.arange(len(arms)),xerr=np.stack([delta-interval[:,0],interval[:,1]-delta]),
        fmt='o',color='#25758b',capsize=4)
    ax.axvline(0,color='#52616c',ls='--',lw=1)
    ax.set_yticks(np.arange(len(arms)),[f"{LABELS[a]}   [p = {r['nominal_two_sided_sign_flip_p']:.4f}]" for a,r in zip(arms,rs)])
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    ax.set(xlabel='Change in normalized RMSE versus original GNN (percentage points)',
        title='Paired five-case sensitivity: mean and empirical 95% case-bootstrap interval')
    fig.suptitle('Intervals and nominal sign-flip p-values are not population guarantees\nShared LOSO training, five cases; all eight interventions reported',fontsize=11)
    fig.savefig(folder/'derived_supervision_uncertainty.svg',metadata={'Date':None})
    fig.savefig(OUT/'cache/derived_supervision_uncertainty.png',dpi=160);plt.close(fig)
    # Stable LF text and no trailing whitespace in versioned SVGs.
    for path in folder.glob('derived_supervision_*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    print(OUT/'comparison.csv')


if __name__=='__main__':main()
