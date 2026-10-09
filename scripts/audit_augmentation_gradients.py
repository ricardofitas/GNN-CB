#!/usr/bin/env python
"""Fitting-only local gradient conflict at each original GNN checkpoint.

No held-out test target is used. These local derivatives are descriptive, not
proof that a data source causes population test degradation.
"""
import json
from pathlib import Path
import numpy as np
import torch
from gnn_cb.model import load_checkpoint
from gnn_cb.train import tensor_data
from gnn_cb.train_expanded import load_expanded,physical_id,model_mse

ROOT=Path(__file__).resolve().parents[1]


def gradient(model, data):
    value=sum(model_mse(model,d,tensor_data(d)) for d in data)/len(data)
    grads=torch.autograd.grad(value,list(model.parameters()))
    return torch.cat([g.detach().reshape(-1) for g in grads]).double(),float(value.detach())


def cosine(a,b):
    norm=float(a.norm()*b.norm())
    return float(a@b)/norm if norm>0 else None


def main():
    torch.set_num_threads(1)
    primary=load_expanded([str(ROOT/'data/expanded')]);extra=load_expanded([str(ROOT/'data/automatic/training')])
    protocol=json.loads((ROOT/'results/expanded/five_specimens/protocol.json').read_text())
    output={'meaning':'Local gradient alignment at original validation-selected GNN; fit labels only. A negative dot means the extra-data descent direction raises primary fit loss to first order, not necessarily test error.',
        'test_targets_used':False,'validation_targets_used_for_gradient':False,'rows':[]}
    for fold in protocol['folds']:
        fit=[d for d in primary if physical_id(d) in fold['training_physical_specimens']]
        for seed in protocol['seeds']:
            path=ROOT/'results/expanded/five_specimens'/f"test_{fold['held_out_physical_specimen']}"/f'gnn_seed{seed}.pt'
            model,_=load_checkpoint(str(path))
            gp,loss=gradient(model,fit); g8,l8=gradient(model,[extra[0]]);g9,l9=gradient(model,[extra[1]])
            ge=(g8+g9)/2; mix=.6*gp+.4*ge
            output['rows'].append({'test_group_id_only':fold['held_out_physical_specimen'],'seed':seed,
                'primary_fit_mse':loss,'video8_mse':l8,'video9_mse':l9,
                'primary_gradient_norm':float(gp.norm()),'video8_gradient_norm':float(g8.norm()),'video9_gradient_norm':float(g9.norm()),
                'primary_video8_cosine':cosine(gp,g8),'primary_video9_cosine':cosine(gp,g9),
                'primary_videos_cosine':cosine(gp,ge),'video8_video9_cosine':cosine(g8,g9),
                'primary_fit_directional_derivative_under_video_descent':-float(gp@ge),
                'primary_fit_directional_derivative_under_mixed_descent':-float(gp@mix)})
    output['summary']={key: {'median':float(np.median([r[key] for r in output['rows']])),
                         'negative_count':sum(r[key]<0 for r in output['rows'])}
                       for key in ['primary_video8_cosine','primary_video9_cosine','primary_videos_cosine','video8_video9_cosine']}
    path=ROOT/'results/diagnosis/gradient_audit.json';path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output['summary'],indent=2))


if __name__=='__main__':main()
