#!/usr/bin/env python
"""Post-prediction descriptive errors: original task and derived differences.

This diagnostic uses held-out labels only after predictions are frozen. It
cannot select models or supply label-based corrections at inference time.
"""
import json
import numpy as np
import torch

from gnn_cb.train_expanded import compression_selection, physical_id
from scripts.train_derived_supervision import OUT, ARMS, SEEDS, register
from scripts.diagnose_augmentation import load_data, save
from scripts.derived_objectives import temporal_pair_mse, edge_pair_mse
from scripts.decompose_prediction_errors import decompose


def main():
    protocol=register();torch.set_num_threads(1)
    primary,_=load_data();lookup={physical_id(d):d for d in primary};rows=[]
    for arm in ARMS:
        for i,fold in enumerate(protocol['folds']):
            data=lookup[fold['held_out_physical_specimen']];select=compression_selection(data)
            truth=torch.as_tensor(data['y'][select]);mask=torch.as_tensor(data['mask'][select]);edges=torch.as_tensor(data['edge_index'])
            for seed in SEEDS:
                with np.load(OUT/'cache'/f'{arm}__fold{i}__seed{seed}.npz',allow_pickle=False) as z:
                    for key in ['mask','node_ids','frame']:np.testing.assert_array_equal(z[key],data[key])
                    prediction=z['prediction'];p=torch.as_tensor(prediction[select])
                    values=decompose(data,prediction)
                error=(prediction[select]-data['y'][select])[data['mask'][select]]
                values.update({'temporal_difference_rmse_percent':100*float(temporal_pair_mse(p,truth,mask).sqrt()),
                    'spatial_difference_rmse_percent':100*float(edge_pair_mse(p,truth,mask,edges).sqrt()),
                    'rmse_x_percent':100*float(np.sqrt((error[:,0]**2).mean())),
                    'rmse_y_percent':100*float(np.sqrt((error[:,1]**2).mean()))})
                rows.append({'arm':arm,'physical_specimen':physical_id(data),'seed':seed,**values})
    keys=[k for k in rows[0] if k not in ['arm','physical_specimen','seed','identity_absolute_residual']]
    macro={a:{k:float(np.mean([r[k] for r in rows if r['arm']==a])) for k in keys} for a in ARMS}
    per={a:{sp:{k:float(np.mean([r[k] for r in rows if r['arm']==a and r['physical_specimen']==sp])) for k in keys}
            for sp in lookup} for a in ARMS}
    output={'warning':'Post-hoc descriptive audit of all frozen predictions; no adjustment of test predictions. Common error is the mean over observed nodes, not independently measured rigid motion. Auxiliary metrics are derived from the same noisy tracking labels.',
        'rows':rows,'macro':macro,'per_specimen':per,
        'macro_change_vs_original':{a:{k:v-macro['primary_replay'][k] for k,v in m.items()} for a,m in macro.items()}}
    save(OUT/'prediction_audit.json',output)
    print(json.dumps(macro,indent=2))


if __name__=='__main__':main()
