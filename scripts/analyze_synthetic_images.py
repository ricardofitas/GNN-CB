#!/usr/bin/env python
"""All competing-model contrasts and generated-image arms, without selection."""
import json
import numpy as np
from scripts.diagnose_augmentation import REFERENCE, save
from scripts.train_synthetic_images import OUT, ARMS, register, summarize
from scripts.analyze_derived_statistics import (benchmark_matrix, recorded_matrix,
    paired_statistics, adjust, WARNING)


def main():
    register(); summary=summarize()
    if set(summary['arms'])!=set(ARMS): raise ValueError('Report only after every registered arm completes')
    original=benchmark_matrix(REFERENCE,'gnn')
    competitors={kind:benchmark_matrix(REFERENCE,kind) for kind in ['mlp','phase_affine','zero_displacement']}
    competitors.update({kind:recorded_matrix(OUT,kind) for kind in ['random_forest','rbf_ridge']})
    primary={name:paired_statistics(original,values) for name,values in competitors.items()}; adjust(primary)
    matrices={name:recorded_matrix(OUT,name) for name in ARMS if not ARMS[name].get('competitor')}
    secondary={name:paired_statistics(values,original) for name,values in matrices.items()}; adjust(secondary)
    control={name:paired_statistics(values,matrices['real_mass90']) for name,values in matrices.items() if name!='real_mass90'}; adjust(control)
    defects={f'{source}_{target}':paired_statistics(matrices[f'{source}_defects_{target}'],matrices[f'{source}_{target}'])
        for source in ['ai','procedural'] for target in ['affine','teacher']}; adjust(defects)
    teacher={f'{source}':paired_statistics(matrices[f'{source}_teacher'],matrices[f'{source}_affine'])
        for source in ['ai','procedural','ai_defects','procedural_defects']}; adjust(teacher)
    all_models={'gnn':original,**competitors}
    output={'warning':WARNING,'primary_family':'Original GNN minus competing model classes; five contrasts with Holm.',
        'units':'RMSE in percent of initial graph height; deltas in percentage points.',
        'original_gnn_minus_competitors':primary,
        'model_rmse_percent':{name:{'mean':float(values.mean()),'seed_sd':float(values.mean(0).std(ddof=1)),
            'per_specimen_seed_mean':values.mean(1).tolist()} for name,values in all_models.items()},
        'synthetic_minus_original':secondary,'synthetic_minus_real_mass90':control,
        'defect_requested_minus_regular':defects,'teacher_minus_affine_targets':teacher,
        'all_new_arms_minus_competing_classes':{arm:{name:paired_statistics(values,other) for name,other in competitors.items()} for arm,values in matrices.items()},
        'interpretation':'Synthetic images and pseudo-labels add zero independent mechanical observations. No primary benchmark replacement or test-based arm selection.'}
    save(OUT/'statistics.json',output)
    print(json.dumps({'competitors':output['model_rmse_percent'],'synthetic':summary['arms']},indent=2))


if __name__=='__main__': main()
