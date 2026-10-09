"""Plot recorded held-out predictions; no training or model selection occurs here."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gnn_cb.train import fit_affine,partition,tensor_data
from gnn_cb.model import affine_features

root=Path(__file__).resolve().parents[1];out=root/'reports/figures';out.mkdir(parents=True,exist_ok=True)
summary=json.loads((root/'results/loso/metrics.json').read_text())
if len(summary['folds'])!=2 or 'aggregate' not in summary: raise RuntimeError('Benchmark not complete')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'affine':'#b3965d','gnn':'#246b98','mlp':'#92509a'}
fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
loaded=[]
for ax,fold in zip(axs,summary['folds']):
 vals=[fold['baselines']['phase_affine']['compression_window']['rmse_px']]
 errs=[0]
 for kind in ['mlp','gnn']:
  a=[r['test']['compression_window']['rmse_px'] for r in fold['models'][kind]]; vals.append(np.mean(a));errs.append(np.std(a,ddof=1))
 ax.bar(['Affine','Node MLP','GNN'],vals,yerr=errs,capsize=5,color=[colors[k] for k in ['affine','mlp','gnn']])
 ax.set(title='Held out: '+fold['held_out_specimen'],ylabel='Coordinate RMSE (pixels)');ax.grid(axis='y',alpha=.2)
 for i,v in enumerate(vals):ax.text(i,v+max(vals)*.035,f'{v:.2f}',ha='center',fontweight='bold')
 ax.set_ylim(0,max(vals)*1.23)
 test={k:v for k,v in np.load(root/'data/processed'/f"{fold['held_out_specimen']}.npz").items()}
 train={k:v for k,v in np.load(root/'data/processed'/f"{fold['training_specimen']}.npz").items()}
 folder=root/'results/loso'/f"train_{fold['training_specimen']}__test_{fold['held_out_specimen']}"
 pred=np.stack([np.load(folder/f'gnn_seed{s}_predictions.npz')['prediction'] for s in summary['protocol']['seeds']]).mean(0)
 affine=(affine_features(tensor_data(test)['x0'],tensor_data(test)['progress'])@fit_affine(train,partition(train['progress'])['train'])).numpy()
 loaded.append((fold,test,pred,affine))
fig.savefig(out/'benchmark.png',dpi=190);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
for ax,(fold,d,pred,affine) in zip(axs,loaded):
 keep=(d['progress']>=0)&(d['progress']<=1);scale=float(d['scale_px'])
 for label,p,col in [('Affine',affine,colors['affine']),('GNN ensemble',pred,colors['gnn'])]:
  err=[np.sqrt(np.mean((p[i][d['mask'][i]]-d['y'][i][d['mask'][i]])**2))*scale for i in np.flatnonzero(keep)]
  ax.plot(d['progress'][keep],err,label=label,color=col,lw=1.8)
 ax.axvspan(.75,1,color='#dddddd',alpha=.6,label='Beyond fit phase');ax.set(title=fold['held_out_specimen'],xlabel='Compression protocol phase',ylabel='Coordinate RMSE (pixels)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(out/'error_by_phase.png',dpi=190);plt.close(fig)
fig,axs=plt.subplots(4,1,figsize=(11,7.5),layout='constrained')
for row,(fold,d,pred,_) in enumerate(loaded):
 for offset,phase in enumerate([.5,1.]):
  ax=axs[row*2+offset];i=int(np.argmin(abs(d['progress']-phase)));p0=d['initial_pos_px'];obs=p0+d['displacement_px'][i];estimated=p0+pred[i]*float(d['scale_px']);mask=d['mask'][i]
  for u,v in d['edge_index'].T:
   if u>=v:continue
   if mask[u] and mask[v]: ax.plot(obs[[u,v],0],obs[[u,v],1],color='#333333',lw=1.1,alpha=.6)
   ax.plot(estimated[[u,v],0],estimated[[u,v],1],color=colors['gnn'],lw=1.2,ls='--')
  ax.scatter(obs[mask,0],obs[mask,1],s=12,c='#333333',label='Image tracking',zorder=4)
  ax.scatter(estimated[:,0],estimated[:,1],s=10,c=colors['gnn'],label='Held-out GNN ensemble')
  ax.invert_yaxis();ax.set_ylabel('y (px)');ax.set_title(f"{fold['held_out_specimen']} | phase {d['progress'][i]:.2f} | {mask.sum()}/{len(mask)} observed nodes",loc='left',fontsize=10)
  ax.grid(alpha=.15);ax.set_xlim(0,1030);ax.set_ylim(max(obs[mask,1].max(),estimated[:,1].max())+8,min(obs[mask,1].min(),estimated[:,1].min())-8)
  if row==0 and offset==0:ax.legend(loc='upper right',fontsize=8,ncol=2)
axs[-1].set_xlabel('x (pixels); y expanded for legibility')
fig.savefig(out/'deformation_comparison.png',dpi=190);plt.close(fig)
