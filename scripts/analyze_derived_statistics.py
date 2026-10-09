#!/usr/bin/env python
"""Paired, physical-specimen-level sensitivity analyses; never select/fits models.

LOSO training sets overlap. The exact enumeration is computationally exact under
independent sign symmetry, NOT a guarantee that this assumption holds here.
Case bootstrap intervals are descriptive sensitivity intervals, not certified
confidence intervals for generalization to a population of corrugated boards.
"""
from __future__ import annotations
import argparse
import itertools
import json

import numpy as np

from gnn_cb.train import metrics
from gnn_cb.train_expanded import compression_selection, physical_id
from scripts.diagnose_augmentation import ROOT, REFERENCE, save, load_data
from scripts.derived_objectives import label_inventory
from scripts.train_derived_supervision import OUT, ARMS, SEEDS, register

WARNING = ('Five physical test cases, three optimizer seeds, overlapping LOSO fitting sets. '
    'Nominal exact sign-flip p assumes independent sign symmetry of paired case differences; '
    'this is unverified. Bootstrap is conditional empirical case sensitivity, not a '
    'calibrated population confidence interval. Frames, pairs and seeds are not physical replicates.')


def paired_statistics(candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError('Expected matching [physical specimen, seed] arrays')
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite metric')
    delta = (candidate - reference).mean(1)
    n = len(delta)
    signs = np.array(list(itertools.product([-1., 1.], repeat=n)))
    null = (signs * delta).mean(1)
    p = float(np.mean(np.abs(null) >= abs(delta.mean()) - 1e-12))
    indices = np.array(list(itertools.product(range(n), repeat=n)))
    bootstrap = delta[indices].mean(1)
    return {'delta_rmse_pp': float(delta.mean()), 'relative_change_percent': float(100 * delta.mean() / reference.mean()),
        'paired_specimen_delta_pp': delta.tolist(),
        'paired_seed_macro_delta_pp': (candidate - reference).mean(0).tolist(),
        'improved_specimens': int((delta < 0).sum()), 'test_specimens': n,
        'nominal_two_sided_sign_flip_p': p, 'sign_assignments': len(signs),
        'minimum_attainable_two_sided_p_with_nonzero_differences': 2 / 2**n,
        'conditional_case_bootstrap_95_percent_interval_pp': np.quantile(bootstrap,[.025,.975]).tolist(),
        'case_bootstrap_ordered_resamples': len(indices),
        'leave_one_case_out_delta_pp': [float(np.delete(delta,i).mean()) for i in range(n)],
        'population_significance_established': False}


def holm(pvalues):
    p = np.asarray(pvalues, dtype=np.float64)
    if np.any((p < 0) | (p > 1)): raise ValueError('Invalid p-value')
    order = np.argsort(p, kind='stable'); answer = np.empty_like(p)
    answer[order] = np.minimum(1., np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return answer.tolist()


def adjust(rows, key='nominal_holm_p'):
    values = holm([v['nominal_two_sided_sign_flip_p'] for v in rows.values()])
    for row, value in zip(rows.values(), values): row[key] = value


def benchmark_matrix(folder, kind):
    benchmark = json.loads((folder/'metrics.json').read_text())
    result = []
    for fold in benchmark['folds']:
        extraction = fold['test_extractions'][0]
        if kind in ['gnn','mlp']:
            row = [100 * next(r for r in fold['models'][kind] if r['seed']==s)['tests'][extraction]['compression_window']['rmse_normalized'] for s in SEEDS]
        else:
            # Deterministic baseline repeated for paired array shape, not extra n.
            row = [100 * fold['baselines'][kind][extraction]['compression_window']['rmse_normalized']] * len(SEEDS)
        result.append(row)
    return np.asarray(result)


def recorded_matrix(folder, arm):
    return np.asarray([[100 * json.loads((folder/'runs'/f'{arm}__fold{i}__seed{s}.json').read_text())['test']['compression_window']['rmse_normalized']
        for s in SEEDS] for i in range(5)])


def previous_analysis():
    original = benchmark_matrix(REFERENCE, 'gnn')
    old = json.loads((ROOT/'results/diagnosis/summary.json').read_text())
    rows = {arm: paired_statistics(recorded_matrix(ROOT/'results/diagnosis',arm), original) for arm in old['arms']}
    adjust(rows)
    output = {'warning': WARNING, 'interpretation': 'Retrospective sensitivity analysis of already inspected diagnostic outcomes.',
        'contrast': 'each previous arm minus original GNN', 'family_size': len(rows), 'arms': rows}
    save(OUT/'previous_statistics.json', output)
    return output


def inventory():
    primary, extra = load_data()
    def summarize(data):
        records = {physical_id(d): label_inventory(d) for d in data}
        total = {k: sum(r[k] for r in records.values()) for k in ['absolute_vector_targets',
            'temporal_difference_vector_targets','spatial_difference_vector_targets',
            'combined_vector_targets','temporal_incidence_rank_per_coordinate','independent_additional_measurements']}
        total['combined_to_absolute_ratio'] = total['combined_vector_targets'] / total['absolute_vector_targets']
        return {'specimens': records, 'total': total}
    output = {'unit': '2D vector target; multiply by two for scalar coordinate targets',
        'warning': 'These are dataset-wide descriptive counts. Each fold fits only its registered three primary groups and optional two training videos.',
        'primary': summarize(primary), 'admitted_videos': summarize(extra)}
    save(OUT/'label_inventory.json', output)
    return output


def ensembles(primary, arm):
    values = []
    for i,d in enumerate(primary):
        predictions = []
        for seed in SEEDS:
            with np.load(OUT/'cache'/f'{arm}__fold{i}__seed{seed}.npz', allow_pickle=False) as z:
                for key in ['frame','mask','node_ids']: np.testing.assert_array_equal(z[key],d[key])
                predictions.append(z['prediction'])
        # Registered arithmetic mean, no fitted or validation/test-tuned weights.
        prediction = np.mean(np.stack(predictions), axis=0)
        select = compression_selection(d)
        values.append(100 * metrics(prediction[select],d['y'][select],d['mask'][select],float(d['scale_px']))['rmse_normalized'])
    return np.asarray(values)[:,None]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-only',action='store_true')
    args=parser.parse_args(); p=register(); previous=previous_analysis(); inventory()
    if args.previous_only:
        print(json.dumps({a:{k:r[k] for k in ['delta_rmse_pp','nominal_two_sided_sign_flip_p','nominal_holm_p']} for a,r in previous['arms'].items()},indent=2)); return
    summary=json.loads((OUT/'summary.json').read_text())
    if set(summary['arms']) != set(ARMS): raise ValueError('All registered outcomes must complete')
    original=benchmark_matrix(REFERENCE,'gnn')
    matrices={arm:recorded_matrix(OUT,arm) for arm in ARMS}
    np.testing.assert_array_equal(matrices['primary_replay'],original)
    new={a:paired_statistics(m,original) for a,m in matrices.items() if a!='primary_replay'}
    adjust(new)
    combined={**{'previous/'+k:v for k,v in previous['arms'].items()}, **{'new/'+k:v for k,v in new.items()}}
    adjust(combined,'nominal_holm_p_all_22')
    matched={
        'joint_video14_minus_video14_node_loss':paired_statistics(matrices['joint_video14'],recorded_matrix(ROOT/'results/diagnosis','both_mass14')),
        'temporal_balanced_minus_temporal_pairs':paired_statistics(matrices['temporal_balanced'],matrices['temporal_pairs']),
        'global_joint_minus_global_readout':paired_statistics(matrices['global_joint'],matrices['global_readout']),
        'global_joint_minus_joint_pairs':paired_statistics(matrices['global_joint'],matrices['joint_pairs'])}
    primary,_=load_data()
    by_id={physical_id(d):d for d in primary}
    ordered=[by_id[f['held_out_physical_specimen']] for f in p['folds']]
    ensemble_matrices={arm:ensembles(ordered,arm) for arm in ARMS}
    ensemble_stats={arm:paired_statistics(m,ensemble_matrices['primary_replay']) for arm,m in ensemble_matrices.items() if arm!='primary_replay'}
    adjust(ensemble_stats)
    baselines={kind:benchmark_matrix(REFERENCE,kind) for kind in ['mlp','phase_affine','zero_displacement']}
    augmented_gnn=benchmark_matrix(ROOT/'results/expanded/seven_specimens_fixed_tests','gnn')
    output={'warning':WARNING,'specimen_order':[physical_id(d) for d in ordered], 'seed_order':SEEDS,
        'difference_direction':'candidate minus comparator; negative means lower error',
        'primary_contrast':'joint_pairs minus original GNN', 'new_family_size':len(new),
        'new_arms_vs_original':new,'previous_arms_vs_original':previous['arms'],
        'all_22_nominal_holm':{k:v['nominal_holm_p_all_22'] for k,v in combined.items()},
        'matched_secondary_contrasts':matched,
        'original_baseline_rmse_percent':{k:float(m.mean()) for k,m in baselines.items()},
        'original_gnn_minus_baselines':{k:paired_statistics(original,m) for k,m in baselines.items()},
        'each_new_arm_minus_existing_baselines':{a:{k:paired_statistics(m,b) for k,b in baselines.items()} for a,m in matrices.items()},
        'each_new_arm_minus_augmented_gnn':{a:paired_statistics(m,augmented_gnn) for a,m in matrices.items() if a!='primary_replay'},
        'ensembles':{'definition':'Equal average of predictions from all three fixed seeds; one ensemble per fold. Separate secondary estimator, not original mean-of-seed-RMSE benchmark.',
            'rmse_percent':{a:float(m.mean()) for a,m in ensemble_matrices.items()},
            'per_specimen_rmse_percent':{a:m[:,0].tolist() for a,m in ensemble_matrices.items()},
            'contrasts_vs_original_ensemble':ensemble_stats},
        'no_population_significance_claim':True}
    save(OUT/'statistics.json',output)
    print(json.dumps({'new_arms':{a:{k:r[k] for k in ['delta_rmse_pp','improved_specimens','nominal_two_sided_sign_flip_p','nominal_holm_p']} for a,r in new.items()},
        'ensemble_rmse_percent':output['ensembles']['rmse_percent']},indent=2))


if __name__=='__main__':main()
