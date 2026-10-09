#!/usr/bin/env python
"""Publish figures and tables from recorded results; never fits or selects models."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/diagnosis'
LABELS={
 'augmented_replay':'Both videos, reference weights',
 'only8_mass40':'Only Probe1_8, video mass 40%',
 'only9_mass40':'Only Probe1_9, video mass 40%',
 'both_mass14':'Both videos, video mass 14.3%',
 'strict_jump':'Strict jump filter',
 'matched_dropout':'Matched count/phase/y dropout',
 'equal_frames':'Equal observed frame weight',
 'phase_earlier':'Video phase warp −0.4 p(1−p)',
 'phase_later':'Video phase warp +0.4 p(1−p)',
 'primary_affine':'Freeze primary affine field',
 'primary_residual_scale':'Freeze primary residual scale',
 'no_edges_primary':'No edges, primary specimens',
 'no_edges_augmented':'No edges, augmented specimens',
 'censor9_like8':'Censor Probe1_9 above phase 0.830',
}


def main():
    summary=json.loads((OUT/'summary.json').read_text())['arms']
    if set(summary)!=set(LABELS): raise ValueError('All registered arms must finish before publication')
    audit=json.loads((OUT/'input_audit.json').read_text())
    comparison=json.loads((ROOT/'results/expanded/fixed_test_comparison.json').read_text())
    original=comparison['models']['gnn']['five_rmse_percent_initial_height']
    rows=[]
    for arm,label in LABELS.items():
        r=summary[arm]; rows.append({'arm':arm,'description':label,
            'gnn_rmse_percent':r['macro_rmse_percent'],'seed_sd_pp':r['seed_sd_pp'],
            'delta_vs_original_pp':r['macro_rmse_percent']-original,
            'delta_vs_augmented_pp':r['macro_rmse_percent']-summary['augmented_replay']['macro_rmse_percent'],
            'affine_rmse_percent':r['affine_rmse_percent'],**r['per_specimen']})
    with (OUT/'ablation_table.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    figures=ROOT/'reports/figures';figures.mkdir(exist_ok=True);(OUT/'cache').mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row in audit['specimens']:
        video=row['specimen_id'] in ['Probe1_8','Probe1_9'];col={'Probe1_8':'#bb6232','Probe1_9':'#398b72'}.get(row['specimen_id'],'#65768a')
        axes[0,0].scatter(row['aspect_ratio'],row['normalized_edge_length']['median'],s=row['nodes']*1.3,color=col,alpha=.8)
        axes[0,0].annotate(row['specimen_id'],(row['aspect_ratio'],row['normalized_edge_length']['median']),xytext=(4,5),textcoords='offset points',fontsize=8)
        bins=row['phase_quintiles'];phase=[(v['lo']+v['hi'])/2 for v in bins]
        for ax,key,factor in [(axes[0,1],'coverage',100),(axes[1,0],'rms_percent',1)]:
            ax.plot(phase,[v[key]*factor if v[key] is not None else np.nan for v in bins],label=row['specimen_id'],color=col,alpha=1 if video else .6,lw=2.3 if video else 1,marker='o' if video else None)
    axes[0,0].set(xlabel='Initial width / height',ylabel='Median edge length / initial height',title='A. Geometry shift (marker area = initial nodes)')
    axes[0,1].set(xlabel='Protocol phase (bin centre)',ylabel='Observed node-frame coverage (%)',ylim=(0,103),title='B. Missing observations are phase dependent')
    axes[1,0].set(xlabel='Protocol phase (bin centre)',ylabel='Observed displacement RMS / height (%)',title='C. Targets differ in amplitude and phase')
    axes[1,0].legend(ncol=2,fontsize=8)
    per=comparison['per_test_specimen'];values=[r['gnn']['change_percentage_points'] for r in per]
    axes[1,1].barh([r['physical_specimen_id'] for r in per],values,color=['#b56238' if v>0 else '#357f93' for v in values])
    axes[1,1].axvline(0,color='black',lw=.8);axes[1,1].invert_yaxis()
    axes[1,1].set(xlabel='Augmented − original GNN RMSE (percentage points)',title='D. Original paired result: heterogeneous transfer')
    fig.suptitle('GNN-CB: observed dataset differences and fixed-test regression',fontsize=13)
    fig.savefig(figures/'augmentation_input_diagnosis.svg');fig.savefig(OUT/'cache'/'augmentation_input_diagnosis.png',dpi=160);plt.close(fig)
    labels=list(LABELS);names=list(LABELS.values());means=[summary[k]['macro_rmse_percent'] for k in labels];sd=[summary[k]['seed_sd_pp'] for k in labels]
    fig,(left,right)=plt.subplots(1,2,figsize=(14,8),gridspec_kw={'width_ratios':[1,1]},layout='constrained')
    ypos=np.arange(len(names));left.errorbar(means,ypos,xerr=sd,fmt='o',color='#285b80',capsize=3)
    left.set_yticks(ypos,names,fontsize=9);left.invert_yaxis();left.axvline(original,color='#398b72',ls='--',label=f'Original GNN {original:.3f}%')
    left.axvline(comparison['models']['phase_affine']['five_rmse_percent_initial_height'],color='#777777',ls=':',label='Original affine')
    left.set(xlabel='Macro RMSE / initial height (%)',title='All prespecified interventions; three seeds each');left.legend(loc='lower right',fontsize=8)
    specimens=list(summary[labels[0]]['per_specimen']);base={r['physical_specimen_id']:r['gnn']['five'] for r in per}
    matrix=np.asarray([[summary[k]['per_specimen'][sp]-base[sp] for sp in specimens] for k in labels])
    bound=max(1,float(np.abs(matrix).max()));im=right.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-bound,vmax=bound)
    right.set_yticks(ypos,[]);right.set_xticks(range(len(specimens)),specimens,rotation=35,ha='right',fontsize=8)
    for i in range(len(names)):
        for j in range(len(specimens)):
            right.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(matrix[i,j])>.58*bound else '#202020')
    right.set_title('Change versus original GNN by test specimen (pp)')
    fig.colorbar(im,ax=right,shrink=.7,label='Positive = higher error')
    fig.suptitle('Exploratory ablations — original benchmark remains unchanged\nError bars: optimizer-seed standard deviation, not population confidence intervals',fontsize=12)
    fig.savefig(figures/'augmentation_ablation_results.svg');fig.savefig(OUT/'cache'/'augmentation_ablation_results.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
