#!/usr/bin/env python
"""Registered follow-up: correlated derived targets, graph readout and input noise.

No original dataset, code, score or checkpoint is modified. All nine arms are
reported, using the original five whole-specimen tests and validation mapping.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from gnn_cb.model import DeformationModel, ModelConfig, affine_features
from gnn_cb.train import tensor_data
from gnn_cb.train_expanded import compression_selection, model_mse, physical_id
from scripts.diagnose_augmentation import (ROOT, REFERENCE, digest, save, load_data,
    split_data, fit_weighted, measure, input_files)
from scripts.derived_objectives import GlobalReadoutModel, objective, jitter_geometry, label_inventory

OUT = ROOT / 'results/derived_supervision'
ARMS = {
    'primary_replay': {},
    'temporal_pairs': {'temporal': .5},
    'temporal_balanced': {'temporal': .5, 'balance_nodes': True},
    'spatial_pairs': {'spatial': .5},
    'joint_pairs': {'temporal': .5, 'spatial': .5},
    'mc_geometry_jitter': {'jitter': .005},
    'global_readout': {'global_readout': True},
    'global_joint': {'global_readout': True, 'temporal': .5, 'spatial': .5},
    'joint_video14': {'temporal': .5, 'spatial': .5, 'videos': True},
}
SEEDS = [11, 29, 47]


def partition_options(options):
    return {'mass': 1/7} if options.get('videos') else {'extra': []}


def train(train_data, validation, weights, options, seed, epochs=300, patience=75):
    """No test object or labels are accepted by this fitting/selection function."""
    if {physical_id(d) for d in train_data} & {physical_id(d) for d in validation}:
        raise ValueError('Fitting and validation physical groups overlap')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    coefficients = fit_weighted(train_data, weights)
    tt = [tensor_data(d) for d in train_data]
    baseline = DeformationModel(ModelConfig(kind='mlp'), coefficients)
    with torch.no_grad():
        scale = max(float(sum(w * model_mse(baseline, d, t)
                        for w, d, t in zip(weights, train_data, tt)).sqrt()), .01)
    torch.manual_seed(seed)
    cls = GlobalReadoutModel if options.get('global_readout') else DeformationModel
    model = cls(ModelConfig(), coefficients, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    # Independent stream: geometry noise does not change parameter initialization.
    generator = torch.Generator().manual_seed(20261010 + seed)
    vt = [tensor_data(d) for d in validation]
    selected = [compression_selection(d) for d in train_data]
    best = float('inf'); best_epoch = 0; best_state = copy.deepcopy(model.state_dict())
    history = []; start = time.monotonic()
    for epoch in range(epochs + 1):
        if epoch:
            model.train(); optimizer.zero_grad(set_to_none=True)
            for w, t, select in zip(weights, tt, selected):
                x = jitter_geometry(t['x0'], generator, options['jitter']) if options.get('jitter') else t['x0']
                pred = model(x, t['edge_index'], t['progress'][select])
                value = objective(pred, t['y'][select], t['mask'][select], t['edge_index'],
                    options.get('temporal', 0.), options.get('spatial', 0.), options.get('balance_nodes', False))
                (w * value / scale**2).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                tm = float(sum(w * model_mse(model, d, t) for w, d, t in zip(weights, train_data, tt)))
                vm = float(sum(model_mse(model, d, t) for d, t in zip(validation, vt)) / len(validation))
            history.append({'epoch': epoch, 'clean_train_mse': tm, 'validation_mse': vm})
            if vm < best - 1e-10:
                best = vm; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model, {'seed': seed, 'selected_epoch': best_epoch, 'completed_epochs': epoch,
        'validation_mse': best, 'residual_scale': scale, 'history': history,
        'runtime_seconds': time.monotonic() - start,
        'trainable_parameters': sum(p.numel() for p in model.parameters())}


def protocol():
    source = input_files() + [Path(__file__), ROOT/'scripts/derived_objectives.py',
        ROOT/'scripts/diagnose_augmentation.py', ROOT/'results/diagnosis/summary.json']
    return {'status': 'exploratory follow-up, original benchmark remains authoritative',
        'source_commit': 'eba8d358de1e4b6a045c8c318bd2ddac7f7e9406',
        'registered_before_new_fits': True,
        'prior_outcomes_known': 'Original and fourteen diagnostic arms informed this design; tests are not newly untouched.',
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(source))},
        'arms': ARMS, 'seeds': SEEDS,
        'folds': json.loads((REFERENCE/'protocol.json').read_text())['folds'],
        'maximum_epochs': 300, 'patience': 75, 'validation_every_epochs': 5,
        'learning_rate': .002, 'weight_decay': .001, 'gradient_clip': 5., 'threads_per_fit': 2,
        'fitting': '3 original specimens; only joint_video14 adds both admitted videos with total weight 1/7.',
        'normalization': 'Original per-specimen initial graph height, affine fitting and scalar residual normalization unchanged.',
        'objective': 'node_MSE + temporal_weight * temporal_difference_MSE / 2 + spatial_weight * edge_difference_MSE / 2',
        'temporal': 'All unordered pairs of observed endpoints with same node ID, within compression window; no interpolation or division by phase gap.',
        'spatial': 'One copy of each initial undirected edge, only when both endpoints are observed in the same frame.',
        'temporal_balanced': 'Node mass proportional to observation count k, instead of number of pairs k(k-1)/2; singletons excluded from auxiliary loss only.',
        'mc_geometry_jitter': 'One fixed-stream random view per update; Gaussian sigma 0.005 initial graph height clipped at 3 sigma then centered. Topology, units, labels and masks unchanged. Hypothetical input noise, not simulated mechanics.',
        'global_readout': 'Learn graph-wide residual from mean hidden state, plus all-node-centered local residual. No observed deformation, mask or label is an inference input.',
        'selection': 'Only original whole validation specimen node MSE selects epochs; epoch zero eligible. No test selects arm, coefficient, phase, ensemble member or synthetic perturbation.',
        'primary_metric': 'Mean of five per-specimen normalized coordinate RMSEs, then mean over three seeds. Same original observed test masks.',
        'primary_scientific_contrast': 'joint_pairs versus primary_replay; all eight intervention contrasts are reported with Holm adjustment.',
        'statistics': {'unit': 'physical test specimen, after averaging its three seed scores',
            'nominal_test': 'Two-sided exact sign flips of five paired specimen differences, 32 assignments, plus Holm over eight interventions.',
            'significance_level': .05,
            'uncertainty': 'Exact empirical case bootstrap over 5^5 ordered resamples; percentile 95% sensitivity interval, conditional on these fitted models and five cases.',
            'dependence_warning': 'LOSO fits share training specimens. Independent sign symmetry / exchangeable cases are unverified; neither nominal p nor bootstrap is a calibrated population guarantee.',
            'secondary': 'Paired seed differences; leave-one-case sensitivity; fixed arithmetic three-seed prediction ensemble compared with the same ensemble of original models.',
            'previous_study': 'Apply same descriptive procedure retrospectively to all fourteen prior arms versus original, with separate Holm family and also combined 22-contrast adjustment.'},
        'label_warning': 'More deterministic target relationships do not increase independent physical measurements. Original tracks remain automatic image-derived supervision.',
        'generative_images': 'Not trained or used. A photorealistic generator supplies no validated displacement field. This study tests derived objectives and graph input-noise Monte Carlo only.'}


def register():
    value = protocol(); path = OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Frozen code/input/protocol changed; use a fresh versioned study')
    if not path.exists(): save(path, value)
    return value


def execute_one(task):
    arm, fold_index, seed = task
    p = register(); fingerprint = digest(OUT/'protocol.json')
    dest = OUT/'runs'/f'{arm}__fold{fold_index}__seed{seed}.json'
    if dest.exists():
        if json.loads(dest.read_text())['protocol_sha256'] != fingerprint:
            raise ValueError('Stale cached run')
        return {'cached': True, 'arm': arm, 'fold': fold_index, 'seed': seed}
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    primary, extra = load_data(); split = p['folds'][fold_index]; options = ARMS[arm]
    fit, validation, test, weights = split_data(primary, extra, split, partition_options(options))
    model, info = train(fit, validation, weights, options, seed)
    evaluation, prediction = measure(model, test[0])
    # Test is scored only after restoring the validation-selected model.
    cache = OUT/'cache'; cache.mkdir(exist_ok=True)
    stem = f'{arm}__fold{fold_index}__seed{seed}'
    membership = {'training_physical_specimens': [physical_id(d) for d in fit],
        'validation_physical_specimens': [physical_id(d) for d in validation],
        'held_out_physical_specimen': physical_id(test[0])}
    checkpoint = {'state_dict': model.state_dict(), 'config': asdict(model.config),
        'model_class': type(model).__name__, 'arm': arm, 'options': options,
        'seed': seed, 'selected_epoch': info['selected_epoch'], 'protocol_sha256': fingerprint,
        **membership, 'warning': p['label_warning']}
    torch.save(checkpoint, cache/f'{stem}.pt')
    np.savez_compressed(cache/f'{stem}.npz', prediction=prediction, frame=test[0]['frame'],
        node_ids=test[0]['node_ids'], mask=test[0]['mask'])
    record = {'arm': arm, 'fold_index': fold_index, 'protocol_sha256': fingerprint,
        **membership, **info, 'test_specimen_id': str(test[0]['specimen_id']),
        'test_source_sha256': test[0]['_source_sha256'], 'test': evaluation,
        'training': [{'physical_id': physical_id(d), 'weight': w, **label_inventory(d)} for d,w in zip(fit,weights)],
        'checkpoint_sha256': digest(cache/f'{stem}.pt'),
        'prediction_sha256': digest(cache/f'{stem}.npz')}
    save(dest, record)
    return {'arm': arm, 'fold': fold_index, 'seed': seed, 'epoch': info['selected_epoch'],
        'rmse_percent': 100 * evaluation['compression_window']['rmse_normalized'], 'seconds': info['runtime_seconds']}


def summarize():
    p = register(); fingerprint = digest(OUT/'protocol.json'); result = {'protocol_sha256': fingerprint, 'arms': {}}
    for name in ARMS:
        rows = [json.loads(f.read_text()) for f in sorted((OUT/'runs').glob(name+'__*.json'))]
        if any(r['protocol_sha256'] != fingerprint for r in rows): raise ValueError('Stale result')
        if len(rows) != 15: continue
        sm = [100 * np.mean([r['test']['compression_window']['rmse_normalized'] for r in rows if r['seed'] == seed]) for seed in SEEDS]
        per = {fold['held_out_physical_specimen']: float(100 * np.mean([r['test']['compression_window']['rmse_normalized'] for r in rows if r['fold_index'] == i])) for i,fold in enumerate(p['folds'])}
        result['arms'][name] = {'neural_fits': len(rows), 'macro_rmse_percent': float(np.mean(sm)),
            'seed_sd_pp': float(np.std(sm, ddof=1)), 'seed_means': dict(zip(map(str, SEEDS), sm)),
            'per_specimen': per, 'epochs': [r['selected_epoch'] for r in rows],
            'runtime_seconds': sum(r['runtime_seconds'] for r in rows)}
    save(OUT/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--arms', nargs='+', choices=list(ARMS))
    args = parser.parse_args(); register()
    if args.register_only: print(OUT/'protocol.json'); return
    if args.summarize_only: print(json.dumps(summarize(), indent=2)); return
    tasks = [(a,f,s) for a in (args.arms or ARMS) for f in range(5) for s in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(execute_one,t) for t in tasks]):
            print(json.dumps(future.result()), flush=True)
    print(json.dumps(summarize(), indent=2))


if __name__ == '__main__': main()
