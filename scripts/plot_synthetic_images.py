#!/usr/bin/env python
"""Exact scientific figures; native input photos are preserved unchanged."""
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection
from PIL import Image
import torch

from gnn_cb.image_graph import extract_graph, ExtractionConfig
from gnn_cb.model import load_checkpoint
from gnn_cb.train import tensor_data
from gnn_cb.train_expanded import physical_id
from scripts.diagnose_augmentation import ROOT, load_data
from scripts.synthetic_image_geometry import DATA, OUT, procedural, render

FIG=ROOT/'reports/figures'
NAVY='#142c43'; TEAL='#168d91'; ORANGE='#e59a35'; GRAY='#9aa6b1'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'svg.hashsalt':'gnn-cb-synthetic-images-20261010'})


def edges(ax,positions,edge_index,color,lw=.8,alpha=1.,mask=None):
    e=edge_index[:,edge_index[0]<edge_index[1]].T
    if mask is not None: e=e[mask[e].all(1)]
    ax.add_collection(LineCollection(positions[e],colors=color,linewidths=lw,alpha=alpha))


def native_figure():
    names=['ai_regular','ai_asymmetric','ai_crushed','ai_contact_requested']
    titles=['Regular control','Small asymmetries · ends cropped','Requested defects · visible local crush','Requested open contact · not unequivocally realized']
    fig,axes=plt.subplots(4,2,figsize=(15,7.5),layout='constrained')
    for row,(name,title) in enumerate(zip(names,titles)):
        im=cv2.imread(str(DATA/'native'/f'{name}.png'),0)
        graph=extract_graph(im,ExtractionConfig(threshold_mode='otsu')); xy=graph['nodes_xy']
        lo=xy.min(0)-15; hi=xy.max(0)+15
        for ax in axes[row]:
            ax.imshow(im,cmap='gray',vmin=0,vmax=255); ax.set_xlim(lo[0],hi[0]); ax.set_ylim(hi[1],lo[1]); ax.axis('off')
        axes[row,0].set_title(title,loc='left',fontsize=10,color=NAVY,pad=7)
        for e in graph['edges']:
            p=np.asarray(e['points_xy']); axes[row,1].plot(p[:,0],p[:,1],color='#53d0c4',lw=.65)
        axes[row,1].scatter(*xy.T,s=8,color='#ffb556',zorder=5)
        axes[row,1].set_title(f"Otsu graph · {len(xy)} nodes / {len(graph['edges'])} traced paths",loc='left',fontsize=10,color=NAVY,pad=7)
    fig.suptitle('GENERATED IMAGES → EXTRACTED GRAPHS',weight='bold',size=17,color=NAVY)
    fig.supxlabel('Text-only image_gen inputs. Graphs are algorithmic outputs; zero independent mechanical labels.',fontsize=10,color=NAVY)
    fig.savefig(FIG/'synthetic_native_graphs.jpg',dpi=155,bbox_inches='tight',pil_kwargs={'quality':88}); plt.close(fig)


def topology_figure():
    truth=procedural(18,defective=True); im=render(truth,0,False,0)
    graph=extract_graph(im,ExtractionConfig(threshold_mode='otsu'))
    fig,axes=plt.subplots(2,2,figsize=(13,5.8),layout='constrained')
    regions=[(280,570,'A bonded contact'),(440,640,'A prescribed open contact')]
    # Use a bonded top contact and the known opened j=12 contact region.
    regions=[(350,530,'Bonded contact: split raster junctions'),(480,650,'Open contact: gap is specified in the renderer')]
    for col,(xmin,xmax,title) in enumerate(regions):
        for row in range(2):
            ax=axes[row,col]; ax.imshow(im,cmap='gray',vmin=0,vmax=255); ax.set_xlim(xmin,xmax); ax.set_ylim(155,93); ax.axis('off')
            if row==0:
                p=truth['nodes_xy']; ax.scatter(*p.T,s=28,facecolors='none',edgecolors='#ffb556',lw=1.5)
            else:
                p=graph['nodes_xy']; ax.scatter(*p.T,s=18,color='#40d7cb')
        axes[0,col].set_title(title,fontsize=10,color=NAVY)
    axes[0,0].text(-.04,.5,'Ideal junctions',rotation=90,transform=axes[0,0].transAxes,va='center')
    axes[1,0].text(-.04,.5,'Extracted junctions',rotation=90,transform=axes[1,0].transAxes,va='center')
    fig.suptitle('A CLEAN IMAGE DOES NOT GUARANTEE THE SAME NODE CONVENTION',size=14,weight='bold',color=NAVY)
    fig.supxlabel('Orange: analytic contact nodes. Cyan: skeleton nodes. Coordinate truth is prescribed geometry, not material-test data.',fontsize=9)
    fig.savefig(FIG/'synthetic_contact_topology.png',dpi=150,bbox_inches='tight'); plt.close(fig)


def architecture():
    fig,ax=plt.subplots(figsize=(14,5.4)); ax.set_xlim(0,14); ax.set_ylim(0,5.4); ax.axis('off')
    def box(x,y,w,h,title,body,color=TEAL):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',facecolor='#f4f7fa',edgecolor=color,lw=1.5))
        ax.text(x+w/2,y+h-.32,title,ha='center',va='top',weight='bold',color=NAVY,fontsize=11)
        ax.text(x+w/2,y+.17,body,ha='center',va='bottom',color=NAVY,fontsize=9,linespacing=1.5)
    box(.3,2.2,2.1,1.55,'INITIAL GEOMETRY','Node coordinates\nInitial edges\nCaller-supplied phase')
    box(3.05,2.2,2.3,1.55,'NODE ENCODER','[x − x̄, y − ȳ, p, p²]\n4 → 32 → 32\nSiLU activations')
    box(6.05,2.2,2.7,1.55,'3 MESSAGE ROUNDS','Neighbour states + edge geometry\nΔx, Δy and straight length\nMean aggregation + residual update\nLayerNorm')
    box(9.45,2.2,2.1,1.55,'NODE DECODER','32 → 32 → 2\nResidual displacement\nTrain-fit scalar scale')
    box(5.3,.2,3.7,1.15,'AFFINE PRIOR','Quadratic phase × initial coordinates\nCoefficients fitted to 3 training specimens',ORANGE)
    ax.text(12.5,3.0,'+',fontsize=24,ha='center',color=NAVY)
    ax.text(12.5,1.65,'2D displacement\nfor every initial node',ha='center',color=NAVY,fontsize=10)
    def arrow(a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color=NAVY,lw=1.4))
    for a,b in [((2.5,3),(2.9,3)),((5.5,3),(5.9,3)),((8.9,3),(9.3,3)),((11.7,3),(12.15,3)),((9.15,.8),(12.5,2.85))]: arrow(a,b)
    ax.text(.3,4.8,'HOW THE ORIGINAL GNN MAKES A PREDICTION',weight='bold',fontsize=18,color=NAVY)
    ax.text(.3,4.35,'21,634 trainable parameters · geometry + phase only · no observed displacement is an inference input',fontsize=11,color=NAVY)
    fig.savefig(FIG/'gnn_architecture.svg',bbox_inches='tight'); fig.savefig(OUT/'cache/architecture_qa.png',dpi=120,bbox_inches='tight'); plt.close(fig)


def heldout_figure():
    primary,_=load_data(); torch.set_num_threads(2)
    fig,axes=plt.subplots(5,1,figsize=(15,9),layout='constrained')
    for ax,d in zip(axes,primary):
        case=str(d['specimen_id']); j=json.loads((ROOT/'web_demo/data'/f'{case}.json').read_text())
        imeta=j['images'][2]; image=np.array(Image.open(ROOT/imeta['source_path']))
        idx=int(np.where(d['frame']==imeta['frame'])[0][0]); t=tensor_data(d)
        model,_=load_checkpoint(str(ROOT/j['checkpoint_path']))
        with torch.no_grad(): pred=model(t['x0'],t['edge_index'],t['progress'][idx:idx+1]).numpy()[0]
        initial=d['initial_pos_px']; predicted=initial+pred*float(d['scale_px'])
        observed=initial+d['y'][idx]*float(d['scale_px']); mask=d['mask'][idx]
        ax.imshow(image,cmap='gray'); edges(ax,predicted,d['edge_index'],'#ffb65b',lw=1)
        ax.scatter(*predicted.T,s=9,color='#ffb65b',label='GNN prediction',zorder=5)
        ax.scatter(*observed[mask].T,s=11,facecolors='none',edgecolors='#40d7cb',lw=.9,label='Observed tracking',zorder=6)
        if (~mask).any(): ax.scatter(*predicted[~mask].T,s=25,facecolors='none',edgecolors='#e785e0',lw=.8,label='Unobserved node',zorder=7)
        low=np.minimum(initial.min(0),predicted.min(0)); high=np.maximum(initial.max(0),predicted.max(0))
        ax.set_xlim(low[0]-20,high[0]+20); ax.set_ylim(high[1]+45,low[1]-12); ax.axis('off')
        ax.set_title(f"{physical_id(d)} · held out · phase {float(d['progress'][idx]):.3f} · {mask.sum()}/{len(mask)} observed nodes",loc='left',fontsize=11,color=NAVY)
    fig.supxlabel('Orange: GNN prediction · Cyan: observed tracking · Magenta: unobserved node · Original held-out checkpoints, seed 11',fontsize=10)
    fig.suptitle('REAL HELD-OUT IMAGES + ORIGINAL GNN PREDICTIONS',weight='bold',size=17,color=NAVY)
    fig.savefig(FIG/'gnn_heldout_image_predictions.jpg',dpi=160,bbox_inches='tight',pil_kwargs={'quality':89}); plt.close(fig)


def result_figures():
    if not (OUT/'statistics.json').exists(): return
    j=json.loads((OUT/'statistics.json').read_text()); s=json.loads((OUT/'summary.json').read_text())
    names=['zero_displacement','phase_affine','mlp','random_forest','rbf_ridge','gnn']
    labels=['Zero displacement','Affine phase/geometry','Node MLP','Random forest + affine','RBF ridge + affine','Original GNN']
    fig,(ax,per)=plt.subplots(1,2,figsize=(13,5.3),layout='constrained',gridspec_kw={'width_ratios':[1,1.3]})
    values=[j['model_rmse_percent'][n]['mean'] for n in names]
    ax.barh(labels,values,color=[GRAY]*5+[TEAL]); ax.invert_yaxis(); ax.set_xlim(0,31)
    for i,v in enumerate(values): ax.text(v+.4,i,f'{v:.3f}%',va='center',fontsize=10)
    ax.set_xlabel('Macro coordinate RMSE / initial height (%)'); ax.set_title('Same five real test specimens',loc='left')
    ids=['Exp3','Exp4','Exp5','P1_22','P2_7']; x=np.arange(5)
    for n,label,c in zip(['gnn','mlp','phase_affine','random_forest','rbf_ridge'],['GNN','MLP','Affine','Random forest','RBF ridge'],[TEAL,GRAY,ORANGE,'#8054a8','#4450b2']):
        per.plot(x,j['model_rmse_percent'][n]['per_specimen_seed_mean'],'o-',label=label,color=c,lw=1.5,ms=5)
    per.set_xticks(x,ids); per.set_ylabel('Coordinate RMSE / initial height (%)'); per.legend(fontsize=9,ncol=2,frameon=False)
    per.set_title('Seed means by physical specimen',loc='left'); per.grid(axis='y',alpha=.2)
    fig.suptitle('GNN VERSUS COMPETING MODEL CLASSES',fontsize=17,weight='bold',color=NAVY)
    fig.supxlabel('Nominal exact paired p-values and Holm adjustment are reported at specimen level; n = 5, not frames or seeds.',fontsize=9)
    fig.savefig(FIG/'gnn_competing_models.svg',bbox_inches='tight'); fig.savefig(OUT/'cache/competitors_qa.png',dpi=120,bbox_inches='tight'); plt.close(fig)
    arms=[a for a in s['arms'] if a not in ['random_forest','rbf_ridge']]
    # Explicit fixed display order, never sorted to promote an apparent winner.
    arms=['real_mass90','ai_affine','ai_teacher','procedural_affine','procedural_teacher',
          'ai_defects_affine','ai_defects_teacher','procedural_defects_affine','procedural_defects_teacher']
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained'); ref=j['model_rmse_percent']['gnn']['mean']
    for i,a in enumerate(arms):
        stats=j['synthetic_minus_original'][a]; lo,hi=stats['conditional_case_bootstrap_95_percent_interval_pp']; mid=stats['delta_rmse_pp']
        ax.plot([lo,hi],[i,i],color=GRAY,lw=2); ax.scatter(mid,i,color=TEAL if 'defects' in a else ORANGE,s=45,zorder=3)
        ax.text(max(hi,.45)+.04,i,f"{s['arms'][a]['macro_rmse_percent']:.3f}%   p={stats['nominal_two_sided_sign_flip_p']:.4f}",va='center',fontsize=9)
    ax.axvline(0,color=NAVY,lw=1); ax.set_yticks(range(len(arms)),[a.replace('_',' ') for a in arms]); ax.invert_yaxis()
    ax.set_xlabel('Change from original GNN (percentage points; left = lower error)')
    ax.set_xlim(min(-.7,ax.get_xlim()[0]),max(1.5,ax.get_xlim()[1])); ax.grid(axis='x',alpha=.15)
    ax.set_title('GENERATED GEOMETRY + SYNTHETIC TARGETS',loc='left',weight='bold',color=NAVY,fontsize=16)
    fig.supxlabel('Bars: conditional empirical case-bootstrap sensitivity, not population confidence. All registered outcomes retained.',fontsize=9)
    fig.savefig(FIG/'gnn_synthetic_ablation.svg',bbox_inches='tight'); fig.savefig(OUT/'cache/ablations_qa.png',dpi=120,bbox_inches='tight'); plt.close(fig)


if __name__=='__main__':
    FIG.mkdir(exist_ok=True); native_figure(); topology_figure(); architecture(); heldout_figure(); result_figures()
