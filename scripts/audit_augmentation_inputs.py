#!/usr/bin/env python
"""Read-only diagnostics; never transforms benchmark labels or fits a predictor."""
from pathlib import Path
import json
import hashlib
import numpy as np
import networkx as nx
from gnn_cb.train_expanded import load_expanded,compression_selection,physical_id,conservative_continuity_mask
from gnn_cb.data import load_specimen
from scripts.diagnose_augmentation import strict_mask, matched_mask

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/diagnosis'


def stats(values):
    a=np.asarray(values); a=a[np.isfinite(a)]
    return dict(zip(['min','p25','median','p75','p95','max'],map(float,np.percentile(a,[0,25,50,75,95,100])))) if len(a) else {}


def audit_one(d):
    sel=compression_selection(d); mask=d['mask'][sel]; y=d['y'][sel]; phase=d['progress'][sel]; x=d['x0'];ei=d['edge_index']
    n=len(x);count=mask.sum(1);g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from(ei.T.tolist())
    typical=np.linalg.norm(x[ei[0]]-x[ei[1]],axis=1); full=d['mask']; frame=d['frame'];px=d['initial_pos_px'];scale=float(d['scale_px'])
    expected=(px-px.min(0))/scale
    position_error=float(np.max(np.abs(x-expected)))
    target_error=float(np.max(np.abs(d['y'][full]-d['displacement_px'][full]/scale)))
    offset=next(i for i in range(len(phase)) if mask[i].any())
    bins=np.minimum((x[:,1]*3).astype(int),2)
    val={'specimen_id':str(d['specimen_id']),'physical_specimen_id':physical_id(d),
         'npz_sha256':d['_source_sha256'],'nodes':n,'undirected_edges':g.number_of_edges(),
         'connected_components':nx.number_connected_components(g),'degree':stats(list(dict(g.degree()).values())),
         'graph_height_px':scale,'aspect_ratio':float(np.ptp(x[:,0])),
         'normalized_edge_length':stats(typical),'initial_y_distribution':stats(x[:,1]),
         'three_hop_node_reach_mean':float(np.mean([len(nx.single_source_shortest_path_length(g,node,cutoff=3))/n for node in g])),
         'max_coordinate_normalization_error':position_error,'max_target_normalization_error':target_error,
         'initial_height_relative_error':float(abs(np.ptp(px[:,1])/scale-1)),
         'compression_states':int(sel.sum()),'supervised_states':int((count>0).sum()),
         'observations':int(mask.sum()),'coverage':float(mask.mean()),
         'complete_nodes_full_window':int(mask.all(0).sum()),
         'complete_nodes_observed_frames_only':int(mask[count>0].all(0).sum()),
         'phase_range_observed':list(map(float,phase[count>0][[0,-1]])),
         'observation_weighted_phase':float(np.average(phase,weights=count)),
         'zero_rmse_percent':float(np.sqrt((y[mask]**2).mean())*100),
         'first_observed_phase':float(phase[offset]),
         'first_observed_displacement_rms_percent':float(np.sqrt((y[offset,mask[offset]]**2).mean())*100),
         'first_observed_mean_xy_percent':(y[offset,mask[offset]].mean(0)*100).tolist(),
         'mean_displacement_xy_percent':(y[mask].mean(0)*100).tolist(),
         'phase_convention':d['_metadata'].get('progress_definition',d['_metadata'].get('phase_definition','see source metadata')),
         'vertical_thirds':[], 'phase_quintiles':[]}
    for band in range(3):
        m=mask[:,bins==band];v=y[:,bins==band]
        val['vertical_thirds'].append({'band':band,'nodes':int((bins==band).sum()),'coverage':float(m.mean()) if m.size else None,
            'observation_mass':float(m.sum()/mask.sum()),'rms_percent':float(np.sqrt((v[m]**2).mean())*100) if m.any() else None})
    for k in range(5):
        pick=(phase>=k/5)&(phase<=1 if k==4 else phase<(k+1)/5); m=mask[pick];v=y[pick]
        val['phase_quintiles'].append({'lo':k/5,'hi':(k+1)/5,'observations':int(m.sum()),
            'observation_mass':float(m.sum()/mask.sum()),'coverage':float(m.mean()),
            'mean_xy_percent':(v[m].mean(0)*100).tolist() if m.any() else None,
            'rms_percent':float(np.sqrt((v[m]**2).mean())*100) if m.any() else None})
    # Sample-to-sample jumps are reported with phase gap; no uniform-fps assumption.
    adjacent=mask[1:]&mask[:-1];dy=np.diff(y,axis=0)
    val['observed_sample_step_norm_percent']=stats(np.linalg.norm(dy,axis=-1)[adjacent]*100)
    val['sampling_frame_gap']=stats(np.diff(d['frame'][sel]))
    if physical_id(d).startswith('ML2023_'):
        raw=load_specimen(ROOT/'data/automatic/tracks'/f"{d['specimen_id']}.npz")
        clean=conservative_continuity_mask(raw);indices=np.searchsorted(raw['frame'],d['frame'])
        strict=strict_mask(d); matched=matched_mask(d,strict)
        val['raw_gap_continuity_retained']=int((d['mask']&clean[indices]).sum())
        val['strict_jump_threshold_px']=.1*float(np.median(typical))*scale
        val['strict_jump_retained']=int(strict.sum())
        val['matched_dropout_retained']=int(matched.sum())
        val['strict_matched_counts_by_frame_equal']=bool(np.array_equal(strict.sum(1),matched.sum(1)))
        val['raw_observed_match_distance_px']=stats(raw['matching_distance_px'][raw['mask']])
    return val


def score_existing():
    primary=load_expanded([str(ROOT/'data/expanded')]);lookup={physical_id(d):d for d in primary}
    summaries={k:json.loads((ROOT/'results/expanded'/k/'metrics.json').read_text()) for k in ['five_specimens','seven_specimens_fixed_tests']}
    results=[]
    for i,fold in enumerate(summaries['five_specimens']['folds']):
        sp=fold['held_out_physical_specimen'];d=lookup[sp];sel=compression_selection(d);mask=d['mask'][sel]
        r={'physical_specimen':sp,'variants':{}}
        for variant,summary in summaries.items():
            f=summary['folds'][i];source=ROOT/'results/expanded'/variant/f'test_{sp}'
            models={}
            for kind in ['gnn','mlp']:
                rows=[]
                for row in f['models'][kind]:
                    with np.load(source/f"{kind}_seed{row['seed']}_{d['specimen_id']}_predictions.npz",allow_pickle=False) as z:
                        assert np.array_equal(z['mask'],d['mask']) and np.array_equal(z['node_ids'],d['node_ids'])
                        pred=z['prediction'][sel]
                    errors=(pred-d['y'][sel])[mask]
                    entry={'seed':row['seed'],'selected_epoch':row['selected_epoch'],
                        'rmse_percent':100*row['tests'][str(d['specimen_id'])]['compression_window']['rmse_normalized'],
                        'coordinate_xy_rmse_percent':(np.sqrt((errors**2).mean(0))*100).tolist(),
                        'coordinate_xy_bias_percent':(errors.mean(0)*100).tolist(),
                        'early_rmse_percent':100*row['tests'][str(d['specimen_id'])]['early_phase']['rmse_normalized'],
                        'late_rmse_percent':100*row['tests'][str(d['specimen_id'])]['late_phase']['rmse_normalized']}
                    rows.append(entry)
                models[kind]=rows
            r['variants'][variant]={'models':models,'affine_rmse_percent':100*f['baselines']['phase_affine'][str(d['specimen_id'])]['compression_window']['rmse_normalized']}
        results.append(r)
    delta=[]
    for seed in [11,29,47]:
        change=[]
        for r in results:
            a=next(v for v in r['variants']['five_specimens']['models']['gnn'] if v['seed']==seed)
            b=next(v for v in r['variants']['seven_specimens_fixed_tests']['models']['gnn'] if v['seed']==seed)
            change.append(b['rmse_percent']-a['rmse_percent'])
        delta.append({'seed':seed,'macro_delta_pp':float(np.mean(change)),'specimen_deltas_pp':change})
    # Descriptive case sensitivity, not sample-level pseudo-replicated inference.
    means=np.mean([r['specimen_deltas_pp'] for r in delta],axis=0)
    return {'per_specimen':results,'paired_seed_changes':delta,
            'leave_one_test_case_out_macro_delta_pp':{sp:float(np.delete(means,i).mean()) for i,sp in enumerate(lookup)},
            'uncertainty':'Only five physical tests; correlated folds share fitting specimens; optimizer seeds are not independent experiments.'}


def main():
    datasets=load_expanded([str(ROOT/'data/expanded')])+load_expanded([str(ROOT/'data/automatic/training')])
    result={'scope':'Descriptive input/mask/prediction audit; no test-based fitting or hyperparameter choice',
            'specimens':[audit_one(d) for d in datasets], 'existing_predictions':score_existing()}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'input_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for r in result['specimens']:
        print(r['specimen_id'],{k:r[k] for k in ['nodes','aspect_ratio','coverage','observation_weighted_phase','zero_rmse_percent','first_observed_displacement_rms_percent','max_target_normalization_error','three_hop_node_reach_mean']})
    print(result['existing_predictions']['paired_seed_changes'])


if __name__=='__main__':main()
