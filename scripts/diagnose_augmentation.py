#!/usr/bin/env python
"""Prespecified exploratory ablations of the fixed five-test augmentation result.

No benchmark file is overwritten. Every intervention is selected before its
outcomes, changes fitting data/model only, and preserves the registered whole
validation/test specimens. Tests are scored only after validation selection.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from gnn_cb.data import load_specimen
from gnn_cb.model import DeformationModel, ModelConfig, affine_features
from gnn_cb.train import tensor_data
from gnn_cb.train_expanded import (compression_selection, fit_affine_balanced,
    load_expanded, physical_id, model_mse, residual_scale_for, evaluate_expanded)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/diagnosis'
REFERENCE = ROOT / 'results/expanded/five_specimens'
AUGMENTED = ROOT / 'results/expanded/seven_specimens_fixed_tests'
# All arms and thresholds are committed to protocol.json before fitting.
ARMS = {
    'augmented_replay': {},
    'only8_mass40': {'extra': ['Probe1_8'], 'mass': .4},
    'only9_mass40': {'extra': ['Probe1_9'], 'mass': .4},
    'both_mass14': {'mass': 1/7},
    'strict_jump': {'quality': 'strict'},
    'matched_dropout': {'quality': 'matched'},
    'equal_frames': {'frame_balance': True},
    'phase_earlier': {'warp': -.4},
    'phase_later': {'warp': .4},
    'primary_affine': {'affine_primary': True},
    'primary_residual_scale': {'scale_primary': True},
    'no_edges_primary': {'extra': [], 'no_edges': True},
    'no_edges_augmented': {'no_edges': True},
    'censor9_like8': {'censor9': True},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_data():
    return (load_expanded([str(ROOT / 'data/expanded')]),
            load_expanded([str(ROOT / 'data/automatic/training')]))


def strict_mask(data):
    """Stop a raw track permanently at its first large *observed* step.

    This heuristic does not certify correspondence. Use raw (not stride-10)
    distances, with 10% of median initial edge length as the fixed threshold.
    """
    raw = load_specimen(ROOT / 'data/automatic/tracks' / (str(data['specimen_id']) + '.npz'))
    edge = data['edge_index']
    typical_px = float(np.median(np.linalg.norm(data['x0'][edge[0]]-data['x0'][edge[1]], axis=1)) * data['scale_px'])
    bad = raw['mask'] & (raw['matching_distance_px'] > .1 * typical_px)
    alive = ~np.maximum.accumulate(bad, axis=0)
    indices = np.searchsorted(raw['frame'], data['frame'])
    if not np.array_equal(raw['frame'][indices], data['frame']):
        raise ValueError('Sampled and raw frame IDs do not align')
    return data['mask'] & alive[indices]


def matched_mask(data, target_mask):
    """Same counts per frame and initial vertical third, randomized identities.

    One fixed mask seed is independent of the three optimizer seeds. This
    controls count/phase/vertical support, not whole-trajectory continuity.
    """
    rng = np.random.default_rng(20261009)
    output = np.zeros_like(data['mask'])
    bins = np.minimum((data['x0'][:, 1] * 3).astype(int), 2)
    for i in range(len(output)):
        for band in range(3):
            candidates = np.flatnonzero(data['mask'][i] & (bins == band))
            count = int((target_mask[i] & (bins == band)).sum())
            if count:
                output[i, rng.choice(candidates, count, replace=False)] = True
    return output


def transform_extra(extra, options):
    transformed = []
    cutoff = max(d['progress'][d['mask'].any(1)].max() for d in extra if str(d['specimen_id']) == 'Probe1_8')
    for source in extra:
        if str(source['specimen_id']) not in options.get('extra', ['Probe1_8','Probe1_9']):
            continue
        d = copy.deepcopy(source)
        if options.get('quality'):
            strict = strict_mask(d)
            d['mask'] = strict if options['quality'] == 'strict' else matched_mask(d, strict)
        if options.get('censor9') and str(d['specimen_id']) == 'Probe1_9':
            d['mask'][d['progress'] > cutoff] = False
        if options.get('warp'):
            p = d['progress']
            d['progress'] = (p + options['warp'] * p * (1-p)).astype(np.float32)
        # No interpolation, re-labelling of y, or revival of missing observations.
        if np.any(d['mask'] & ~source['mask']):
            raise AssertionError('Intervention revived unobserved targets')
        transformed.append(d)
    return transformed


def split_data(primary, extra, split, options):
    train = [copy.deepcopy(d) for d in primary if physical_id(d) in split['training_physical_specimens']]
    validation = [copy.deepcopy(d) for d in primary if physical_id(d) in split['validation_physical_specimens']]
    test = [copy.deepcopy(d) for d in primary if physical_id(d) == split['held_out_physical_specimen']]
    n_primary = len(train)
    train += transform_extra(extra, options)
    groups = [{physical_id(d) for d in part} for part in (train, validation, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError('Physical specimens overlap between partitions')
    if len(validation) != 1 or len(test) != 1 or n_primary != 3:
        raise ValueError('Fixed protocol must have 3 primary fit, 1 validation, 1 test')
    if options.get('no_edges'):
        for d in train + validation + test:
            d['edge_index'] = np.empty((2, 0), dtype=np.int64)
    mass = options.get('mass', .4) if len(train) > n_primary else 0
    weights = [(1-mass)/n_primary]*n_primary + ([mass/(len(train)-n_primary)]*(len(train)-n_primary) if mass else [])
    return train, validation, test, weights


def observation_weights(data, frame_balance=False):
    mask = data['mask'][compression_selection(data)]
    if not frame_balance:
        return np.ones(int(mask.sum()), dtype=np.float64) / mask.sum()
    count = mask.sum(1)
    per_frame = np.divide(1., count, out=np.zeros(len(count)), where=count>0)/(count>0).sum()
    return np.broadcast_to(per_frame[:, None], mask.shape)[mask]


def fit_weighted(train, weights, frame_balance=False):
    # Exact original code path for equal-specimen reference/replay arms.
    if not frame_balance and np.allclose(weights, np.ones(len(train))/len(train), atol=0, rtol=1e-14):
        return fit_affine_balanced(train)
    xs, ys = [], []
    for d, weight in zip(train, weights):
        selected = compression_selection(d); mask = d['mask'][selected]
        x = affine_features(torch.as_tensor(d['x0'], dtype=torch.float64),
                           torch.as_tensor(d['progress'][selected], dtype=torch.float64)).numpy()[mask]
        factor = np.sqrt(weight * observation_weights(d, frame_balance))[:,None]
        xs.append(x*factor); ys.append(d['y'][selected][mask].astype(np.float64)*factor)
    return torch.as_tensor(np.linalg.lstsq(np.concatenate(xs), np.concatenate(ys), rcond=1e-10)[0], dtype=torch.float32)


def loss(model, data, tensors, frame_balance=False):
    if not frame_balance:
        return model_mse(model, data, tensors)
    selected = compression_selection(data); mask = tensors['mask'][selected]
    pred = model(tensors['x0'], tensors['edge_index'], tensors['progress'][selected])
    squared = (pred[mask]-tensors['y'][selected][mask]).square().mean(-1)
    w = torch.as_tensor(observation_weights(data, True), dtype=squared.dtype)
    return (w*squared).sum()


def measure(model, data):
    t = tensor_data(data)
    with torch.no_grad():
        pred = model(t['x0'], t['edge_index'], t['progress']).numpy()
    evaluation = evaluate_expanded(data, pred)
    # Keep actual original test masks and output compact phase aggregates.
    framewise = evaluation.pop('framewise')
    evaluation['phase_quintiles'] = []
    for k in range(5):
        selected = (data['progress']>=k/5) & (data['progress']<=1 if k==4 else data['progress']<(k+1)/5)
        mask = data['mask'][selected]; error = (pred[selected]-data['y'][selected])[mask]
        evaluation['phase_quintiles'].append({'lo':k/5,'hi':(k+1)/5,'n':int(mask.sum()),
            'rmse_percent':float(np.sqrt((error**2).mean())*100) if len(error) else None})
    return evaluation, pred


def train_model(train, validation, weights, options, seed, fixed_epoch=None, epochs=300, patience=75):
    """Training and epoch selection have no access to a test object or label."""
    if {physical_id(d) for d in train} & {physical_id(d) for d in validation}:
        raise ValueError('Train/validation overlap')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    primary = [d for d in train if not physical_id(d).startswith('ML2023_')]
    fb = options.get('frame_balance', False)
    coefficients = fit_affine_balanced(primary) if options.get('affine_primary') else fit_weighted(train, weights, fb)
    train_tensors = [tensor_data(d) for d in train]
    if options.get('scale_primary'):
        scale = residual_scale_for(primary, fit_affine_balanced(primary))
    else:
        baseline = DeformationModel(ModelConfig(kind='mlp'), coefficients)
        with torch.no_grad():
            scale = max(float(sum(w*loss(baseline,d,t,fb) for w,d,t in zip(weights,train,train_tensors)).sqrt()), .01)
    torch.manual_seed(seed)
    model = DeformationModel(ModelConfig(), coefficients, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    vt = [tensor_data(d) for d in validation]
    history=[]; best=float('inf'); best_epoch=0
    best_state=copy.deepcopy(model.state_dict()); fixed_state=None
    start=time.monotonic()
    for epoch in range(epochs+1):
        if epoch:
            model.train(); optimizer.zero_grad(set_to_none=True)
            for w,d,t in zip(weights, train, train_tensors):
                (w*loss(model,d,t,fb)/scale**2).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
        if fixed_epoch is not None and epoch==fixed_epoch:
            fixed_state=copy.deepcopy(model.state_dict())
        if epoch%5==0 or epoch==epochs:
            model.eval()
            with torch.no_grad():
                tm=float(sum(w*loss(model,d,t,fb) for w,d,t in zip(weights,train,train_tensors)))
                vm=float(sum(model_mse(model,d,t) for d,t in zip(validation,vt))/len(validation))
            history.append({'epoch':epoch,'train_mse':tm,'validation_mse':vm})
            if vm < best-1e-10:
                best=vm; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict())
            if epoch-best_epoch>=patience:
                break
    # Continue only for the separately declared fixed-epoch probe, without
    # reopening validation selection or changing the replay checkpoint.
    if fixed_epoch is not None and fixed_state is None and fixed_epoch > epoch:
        for _ in range(epoch + 1, fixed_epoch + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            for w,d,t in zip(weights, train, train_tensors):
                (w*loss(model,d,t,fb)/scale**2).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
        fixed_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); model.eval()
    return model, fixed_state, {'seed':seed,'selected_epoch':best_epoch,'completed_epochs':epoch,
        'validation_mse':best,'residual_scale':scale,'runtime_seconds':time.monotonic()-start,'history':history}


def input_files():
    fixed = [REFERENCE/'protocol.json',REFERENCE/'metrics.json',AUGMENTED/'metrics.json',
             ROOT/'results/expanded/fixed_test_comparison.json']
    for folder in ['data/expanded','data/automatic/training']:
        fixed += list((ROOT/folder).glob('*.npz')) + list((ROOT/folder).glob('*.json'))
    for sp in ['Probe1_8','Probe1_9']:
        fixed += [ROOT/'data/automatic/tracks'/f'{sp}.npz']
    fixed += list((ROOT/'gnn_cb').glob('*.py'))
    return sorted(set(fixed))


def protocol():
    reference=json.loads((REFERENCE/'protocol.json').read_text())
    return {'status':'exploratory diagnosis; never a replacement benchmark',
        'source_commit':'8b0f2a2c9ff8a971d4b7349163b8306495ed74ac',
        'registered_before_new_results':True, 'script_sha256':digest(__file__),
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in input_files()},
        'seeds':[11,29,47], 'maximum_epochs':300, 'patience':75,'validation_every_epochs':5,
        'learning_rate':.002,'weight_decay':.001,'gradient_clip':5.,'threads_per_fit':2,
        'folds':reference['folds'], 'arms':ARMS,
        'primary_metric':'equal-test-specimen mean coordinate RMSE / own initial graph height; average over seeds',
        'video_mass':'0.40 except explicitly specified; single-video arms keep 0.40 to isolate composition at fixed domain mass',
        'quality_threshold':'terminate raw observed track after first matching step >0.1 median initial edge length in px',
        'matched_control':'same kept count per frame and initial-y third; fixed mask seed 20261009; no physical accuracy claim',
        'phase_warp':'only fitting videos: p + c*p*(1-p), c=+-0.4; endpoints fixed; maximum phase change 0.1; sensitivity, not calibrated correction',
        'censor9':'mask Probe1_9 observations above original Probe1_8 maximum observed phase; do not rescale phases',
        'fixed_epoch_probe':'augmented replay also scores epoch previously selected ONLY by original whole-specimen validation',
        'no_selection_on_test':True, 'no_refit_of_primary_benchmark':True,
        'interpretation':'All outcomes reported. Three optimizer seeds and correlated LOSO folds do not establish population significance. Video labels remain automated pseudo-targets.'}


def register():
    p=protocol(); path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=p:
        raise ValueError('Code/input/protocol changed: use a fresh versioned investigation directory')
    if not path.exists(): save(path,p)
    return p


def execute_one(task):
    arm,fold_index,seed=task
    p=register(); fingerprint=digest(OUT/'protocol.json')
    dest=OUT/'runs'/f'{arm}__fold{fold_index}__seed{seed}.json'
    if dest.exists():
        obj=json.loads(dest.read_text())
        if obj['protocol_sha256']!=fingerprint: raise ValueError('Stale cached run')
        return {'cached':True,'arm':arm,'fold':fold_index,'seed':seed}
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    primary,extra=load_data(); split=p['folds'][fold_index]; options=ARMS[arm]
    train,validation,test,weights=split_data(primary,extra,split,options)
    fixed_epoch=None
    if arm=='augmented_replay':
        reference=json.loads((REFERENCE/'metrics.json').read_text())
        fixed_epoch=next(r['selected_epoch'] for r in reference['folds'][fold_index]['models']['gnn'] if r['seed']==seed)
    model,fixed_state,info=train_model(train,validation,weights,options,seed,fixed_epoch)
    evaluations,pred=measure(model,test[0])
    tensors=tensor_data(test[0])
    with torch.no_grad():
        affine=(affine_features(tensors['x0'],tensors['progress'])@model.affine_coefficients).numpy()
    affine_evaluation=evaluate_expanded(test[0],affine); affine_evaluation.pop('framewise')
    record={'arm':arm, 'protocol_sha256':fingerprint, **split,**info,
        'test_specimen_id':str(test[0]['specimen_id']), 'test_source_sha256':test[0]['_source_sha256'],
        'test':evaluations,'affine':affine_evaluation,
        'training':[{'specimen_id':str(d['specimen_id']),'physical_id':physical_id(d),
            'weight':w,'observations':int(d['mask'][compression_selection(d)].sum())} for d,w in zip(train,weights)],
        'fixed_original_validation_epoch':fixed_epoch,
        'fixed_original_validation_epoch_reached':fixed_state is not None}
    if fixed_state is not None:
        chosen=copy.deepcopy(model.state_dict()); model.load_state_dict(fixed_state)
        record['fixed_original_validation_epoch_test']=measure(model,test[0])[0]
        model.load_state_dict(chosen)
    cache=OUT/'cache'; cache.mkdir(exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'config':asdict(model.config),'seed':seed,
                'selected_epoch':info['selected_epoch'],**split},cache/f'{arm}__fold{fold_index}__seed{seed}.pt')
    np.savez_compressed(cache/f'{arm}__fold{fold_index}__seed{seed}.npz',prediction=pred,
                        frame=test[0]['frame'],node_ids=test[0]['node_ids'],mask=test[0]['mask'])
    record['checkpoint_sha256']=digest(cache/f'{arm}__fold{fold_index}__seed{seed}.pt')
    record['prediction_sha256']=digest(cache/f'{arm}__fold{fold_index}__seed{seed}.npz')
    save(dest,record)
    return {'arm':arm,'fold':fold_index,'seed':seed,'epoch':info['selected_epoch'],
            'rmse_percent':100*evaluations['compression_window']['rmse_normalized'],'seconds':info['runtime_seconds']}


def summarize():
    p=register(); fingerprint=digest(OUT/'protocol.json'); result={'protocol_sha256':fingerprint,'arms':{}}
    for name in ARMS:
        rows=[json.loads(f.read_text()) for f in sorted((OUT/'runs').glob(name+'__*.json'))]
        if any(r['protocol_sha256']!=fingerprint for r in rows): raise ValueError('Stale result')
        if len(rows)!=15: continue
        seedmeans=[100*np.mean([r['test']['compression_window']['rmse_normalized'] for r in rows if r['seed']==seed]) for seed in p['seeds']]
        per_spec={sp:float(100*np.mean([r['test']['compression_window']['rmse_normalized'] for r in rows if r['held_out_physical_specimen']==sp])) for sp in [f['held_out_physical_specimen'] for f in p['folds']]}
        result['arms'][name]={'neural_fits':len(rows),'macro_rmse_percent':float(np.mean(seedmeans)),
            'seed_sd_pp':float(np.std(seedmeans,ddof=1)), 'seed_means':dict(zip(map(str,p['seeds']),seedmeans)),
            'per_specimen':per_spec,'affine_rmse_percent':float(100*np.mean([r['affine']['compression_window']['rmse_normalized'] for r in rows])),
            'epochs':[r['selected_epoch'] for r in rows], 'runtime_seconds':sum(r['runtime_seconds'] for r in rows)}
        fixed=[r for r in rows if r.get('fixed_original_validation_epoch_reached')]
        if len(fixed)==15:
            result['arms'][name]['fixed_original_validation_epochs_macro_rmse_percent']=float(100*np.mean([r['fixed_original_validation_epoch_test']['compression_window']['rmse_normalized'] for r in rows]))
    save(OUT/'summary.json', result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--arms',nargs='+',choices=list(ARMS))
    args=parser.parse_args(); register()
    if args.register_only: print(OUT/'protocol.json'); return
    if args.summarize_only: print(json.dumps(summarize(),indent=2)); return
    tasks=[(arm,fold,seed) for arm in (args.arms or ARMS) for fold in range(5) for seed in [11,29,47]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute_one, task) for task in tasks]
        for future in as_completed(futures): print(json.dumps(future.result()),flush=True)
    print(json.dumps(summarize(),indent=2))


if __name__=='__main__': main()
