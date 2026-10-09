#!/usr/bin/env python
"""Exact evaluation-only MSE decomposition of unchanged saved predictions.

Per-frame mean error uses held-out labels ONLY to describe error. It is never
fed into inference, fitting, normalization or model selection, and is not a
new deployable accuracy score or a measured physical rigid-body motion.
"""
import json
from pathlib import Path
import numpy as np
from gnn_cb.train_expanded import load_expanded,compression_selection,physical_id

ROOT=Path(__file__).resolve().parents[1]


def decompose(data,prediction):
    selection=compression_selection(data);mask=data['mask'][selection]
    error=prediction[selection].astype(float)-data['y'][selection].astype(float)
    n=mask.sum(); total=(error[mask]**2).mean();common=0.;relative=0.
    for e,m in zip(error,mask):
        if not m.any():continue
        values=e[m];mean=values.mean(0)
        common+=len(values)/n * (mean**2).mean()
        relative+=len(values)/n * ((values-mean)**2).mean()
    if abs(total-common-relative)>1e-12:raise AssertionError('MSE decomposition failed')
    return {'mse_percent_height_squared':total*10000,
            'frame_mean_error_mse_percent_height_squared':common*10000,
            'within_frame_centered_error_mse_percent_height_squared':relative*10000,
            'identity_absolute_residual':abs(total-common-relative)}


def main():
    rows=[]
    for d in load_expanded([str(ROOT/'data/expanded')]):
        for variant in ['five_specimens','seven_specimens_fixed_tests']:
            for seed in [11,29,47]:
                path=ROOT/'results/expanded'/variant/f'test_{physical_id(d)}'/f"gnn_seed{seed}_{d['specimen_id']}_predictions.npz"
                with np.load(path,allow_pickle=False) as z:
                    assert np.array_equal(z['mask'],d['mask'])
                    result=decompose(d,z['prediction'])
                rows.append({'physical_specimen':physical_id(d),'variant':variant,'seed':seed,**result})
    summary=[]
    for sp in sorted({r['physical_specimen'] for r in rows}):
        a=[r for r in rows if r['physical_specimen']==sp and r['variant']=='five_specimens']
        b=[r for r in rows if r['physical_specimen']==sp and r['variant']=='seven_specimens_fixed_tests']
        r={'physical_specimen':sp}
        for key in ['mse_percent_height_squared','frame_mean_error_mse_percent_height_squared','within_frame_centered_error_mse_percent_height_squared']:
            r[key]={'original':float(np.mean([v[key] for v in a])),'augmented':float(np.mean([v[key] for v in b]))}
            r[key]['change']=r[key]['augmented']-r[key]['original']
        summary.append(r)
    output={'scope':__doc__,'rows':rows,'per_specimen_seed_means':summary,
            'macro_mse_changes':{key:float(np.mean([r[key]['change'] for r in summary])) for key in ['mse_percent_height_squared','frame_mean_error_mse_percent_height_squared','within_frame_centered_error_mse_percent_height_squared']},
            'warning':'MSE components, not additive RMSEs. Mean over observed nodes is not physical mass-weighted centre motion. Original RMSE benchmark is unchanged.'}
    (ROOT/'results/diagnosis/error_decomposition.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'summary':summary,'macro':output['macro_mse_changes']},indent=2))


if __name__=='__main__':main()
