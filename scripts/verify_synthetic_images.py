#!/usr/bin/env python
"""Audit every selected GNN and recompute all original-mask test metrics."""
import json
import numpy as np
import torch
from gnn_cb.model import load_checkpoint
from gnn_cb.train import tensor_data
from gnn_cb.train_expanded import evaluate_expanded, physical_id
from scripts.diagnose_augmentation import digest, save, load_data, ROOT
from scripts.train_synthetic_images import OUT, ARMS, SEEDS, register


def main():
    torch.set_num_threads(2); p=register(); primary,_=load_data(); count=0; forwards=0
    fingerprint=digest(OUT/'protocol.json'); max_error=0.
    for arm,options in ARMS.items():
        for i,fold in enumerate(p['folds']):
            d=next(d for d in primary if physical_id(d)==fold['held_out_physical_specimen'])
            for seed in SEEDS:
                stem=f'{arm}__fold{i}__seed{seed}'; row=json.loads((OUT/'runs'/f'{stem}.json').read_text())
                assert row['protocol_sha256']==fingerprint
                for k,v in fold.items(): assert row[k]==v
                array=OUT/'cache'/f'{stem}.npz'; assert digest(array)==row['prediction_sha256']
                with np.load(array,allow_pickle=False) as z:
                    pred=z['prediction']
                    for key in ['frame','mask','node_ids']: np.testing.assert_array_equal(z[key],d[key])
                assert np.isfinite(pred).all()
                checked=evaluate_expanded(d,pred)['compression_window']
                assert checked==row['test']['compression_window']
                if not options.get('competitor'):
                    ck=OUT/'cache'/f'{stem}.pt'; assert digest(ck)==row['checkpoint_sha256']
                    model,meta=load_checkpoint(str(ck)); t=tensor_data(d)
                    with torch.no_grad(): fresh=model(t['x0'],t['edge_index'],t['progress']).numpy()
                    max_error=max(max_error,float(np.max(np.abs(fresh-pred))))
                    np.testing.assert_array_equal(fresh,pred); forwards+=1
                count+=1
    save(OUT/'verification.json',{'status':'PASS','complete_selected_runs':count,'gnn_checkpoint_forwards':forwards,
        'maximum_saved_prediction_difference':max_error,'protocol_sha256':fingerprint,
        'checks':['Frozen code/input hashes match','Every physical fit/validation/test group matches original protocol',
                  'All predictions finite','Every frame ID/node ID/mask equals original data',
                  'Every score recomputed from complete prediction arrays and original masks',
                  'Every GNN checkpoint reloaded and predictions reproduced bit for bit'],
        'competitor_limitation':'Selected competitor predictions/validation grids retained. Exact rerun regenerates the estimator; no unsafe sklearn pickle is distributed.'})
    print(f'PASS: {count} prediction audits, {forwards} exact GNN forward replays')


if __name__=='__main__': main()
