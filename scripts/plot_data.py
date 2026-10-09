"""Create the image-to-graph data audit figure from exact source snapshots."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

root=Path(__file__).resolve().parents[1]
data=root/'data/processed'; out=root/'reports/figures';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,1,figsize=(11,3.5),layout='constrained')
for ax,name in zip(axs,['Probe2_7','Probe1_22']):
 d=np.load(data/f'{name}.npz');p=d['initial_pos_px'];edges=d['edge_index'];im=Image.open(data/f'images/{name}_frame0000.jpg')
 ax.imshow(im)
 for u,v in edges.T:
  if u<v:ax.plot(p[[u,v],0],p[[u,v],1],color='#e3b954',lw=1.1,alpha=.9)
 ax.scatter(p[:,0],p[:,1],s=8,c='#21acde',zorder=4)
 ax.set_ylim(p[:,1].max()+15,p[:,1].min()-15);ax.set_xlim(0,1024)
 ax.set_title(f'{name}: {len(p)} nodes, {edges.shape[1]//2} edges',loc='left',fontweight='bold',color='#18324d')
 ax.set_xticks([]);ax.set_yticks([])
fig.savefig(out/'image_graphs.png',dpi=190);plt.close(fig)
manifest=json.loads((data/'manifest.json').read_text());(root/'reports/data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
for ax,name in zip(axs,['Probe2_7','Probe1_22']):
 d=np.load(data/f'{name}.npz');valid=(d['progress']>=0)&(d['progress']<=1)
 ax.plot(d['progress'][valid],d['mask'][valid].sum(1),color='#246b98',lw=2)
 ax.set(xlabel='Compression protocol phase',ylabel='Observed nodes',title=name,ylim=(0,d['mask'].shape[1]+4));ax.grid(alpha=.2)
fig.savefig(out/'tracking_coverage.png',dpi=180)
