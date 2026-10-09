#!/usr/bin/env python
"""Verify complete registered runs and optionally finalize checkpoint provenance.

Checkpoint finalization changes metadata only, never parameter tensors or
predictions. The fitting records carry actual augmented membership whereas the
original split dictionary describes the three primary fitting specimens.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import numpy as np
import torch
from gnn_cb.model import load_checkpoint
from gnn_cb.train import metrics,tensor_data
from gnn_cb.train_expanded import compression_selection,physical_id
from scripts.diagnose_augmentation import ROOT,OUT,ARMS,register,load_data,digest,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize-checkpoints',action='store_true')
    parser.add_argument('--upstream-tree',type=Path)
    args=parser.parse_args();p=register();torch.set_num_threads(2)
    primary,_=load_data();lookup={physical_id(d):d for d in primary}
    expected={(a,i,s) for a in ARMS for i in range(5) for s in p['seeds']}
    paths=list((OUT/'runs').glob('*.json'))
    actual={(q.stem.split('__')[0],int(q.stem.split('__')[1][4:]),int(q.stem.split('__')[2][4:])) for q in paths}
    if actual!=expected:raise ValueError(f'Incomplete/extraneous run set: {len(actual)} versus {len(expected)}')
    reference=json.loads((ROOT/'results/expanded/seven_specimens_fixed_tests/metrics.json').read_text())
    records=[];replay=[];max_rescore_difference=0.;checkpoint_replays=0;finalized=0;available=0
    for path in sorted(paths):
        r=json.loads(path.read_text());data=lookup[r['held_out_physical_specimen']]
        assert r['protocol_sha256']==digest(OUT/'protocol.json')
        assert r['test_source_sha256']==data['_source_sha256']
        groups={item['physical_id'] for item in r['training']}
        withheld={r['held_out_physical_specimen'],*r['validation_physical_specimens']}
        assert not groups & withheld
        assert abs(sum(v['weight'] for v in r['training'])-1)<1e-12
        checkpoint=OUT/'cache'/f'{path.stem}.pt';prediction=OUT/'cache'/f'{path.stem}.npz'
        if checkpoint.exists() and prediction.exists():
            available+=1
            if args.finalize_checkpoints:
                ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
                old_sha=digest(checkpoint)
                full_names=[v['specimen_id'] for v in r['training']]
                requires=(ckpt.get('training_specimens')!=full_names or ckpt.get('training_physical_specimens')!=sorted(groups) or ckpt.get('diagnostic_arm')!=r['arm'])
                if requires:
                    assert old_sha==r['checkpoint_sha256']
                    r['raw_checkpoint_sha256']=old_sha
                    ckpt['primary_training_physical_specimens']=r.get('primary_training_physical_specimens',r['training_physical_specimens'])
                    ckpt['training_specimens']=full_names
                    ckpt['training_physical_specimens']=sorted(groups)
                    ckpt['diagnostic_arm']=r['arm'];ckpt['diagnostic_options']=ARMS[r['arm']]
                    ckpt['inference_edge_policy']='empty edge_index' if ARMS[r['arm']].get('no_edges') else 'original initial edges'
                    ckpt['independent_test_specimen']=r['held_out_physical_specimen']
                    ckpt['warning']='Exploratory outer-fold checkpoint; retain associated protocol and geometry preprocessing. No final deployment model selected.'
                    state={key:value.clone() for key,value in ckpt['state_dict'].items()}
                    torch.save(ckpt,checkpoint)
                    reloaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    for key,value in state.items():torch.testing.assert_close(reloaded['state_dict'][key],value,atol=0,rtol=0)
                    r['checkpoint_sha256']=digest(checkpoint)
                    r['primary_training_physical_specimens']=r.get('primary_training_physical_specimens',r['training_physical_specimens'])
                    r['training_physical_specimens']=sorted(groups)
                    r['checkpoint_metadata_finalized']=True
                    r['checkpoint_metadata_note']='Actual fitting membership, arm and inference policy added from run record; parameter tensors unchanged. Raw hash preserved.'
                    save(path,r);finalized+=1
            assert digest(checkpoint)==r['checkpoint_sha256']
            assert digest(prediction)==r['prediction_sha256']
            provenance=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert set(provenance.get('training_physical_specimens',[]))==groups
            assert set(provenance.get('training_specimens',[]))=={v['specimen_id'] for v in r['training']}
            assert not set(provenance['training_physical_specimens']) & withheld
            with np.load(prediction,allow_pickle=False) as z:
                for key in ['mask','frame','node_ids']:np.testing.assert_array_equal(z[key],data[key])
                pred=z['prediction'];select=compression_selection(data)
                rescored=metrics(pred[select],data['y'][select],data['mask'][select],float(data['scale_px']))
                difference=abs(rescored['rmse_normalized']-r['test']['compression_window']['rmse_normalized'])
                max_rescore_difference=max(difference,max_rescore_difference);assert difference==0
                if 'fold2__seed11' in path.stem:
                    model,meta=load_checkpoint(str(checkpoint));t=tensor_data(data)
                    assert set(meta['training_physical_specimens'])==groups
                    if ARMS[r['arm']].get('no_edges'):t['edge_index']=torch.empty((2,0),dtype=torch.long)
                    with torch.no_grad():reproduced=model(t['x0'],t['edge_index'],t['progress']).numpy()
                    np.testing.assert_array_equal(reproduced,pred);checkpoint_replays+=1
        if r['arm']=='augmented_replay':
            f=next(f for f in reference['folds'] if f['held_out_physical_specimen']==r['held_out_physical_specimen'])
            old=next(v for v in f['models']['gnn'] if v['seed']==r['seed'])
            difference=r['test']['compression_window']['rmse_normalized']-old['tests'][r['test_specimen_id']]['compression_window']['rmse_normalized']
            assert difference==0 and r['selected_epoch']==old['selected_epoch']
            same_predictions=None
            if prediction.exists():
                old_path=ROOT/'results/expanded/seven_specimens_fixed_tests'/f"test_{r['held_out_physical_specimen']}"/f"gnn_seed{r['seed']}_{r['test_specimen_id']}_predictions.npz"
                with np.load(old_path,allow_pickle=False) as a,np.load(prediction,allow_pickle=False) as b:
                    for key in ['prediction','mask','frame','node_ids']:np.testing.assert_array_equal(a[key],b[key])
                same_predictions=True
            replay.append({'specimen':r['held_out_physical_specimen'],'seed':r['seed'],'rmse_difference':difference,'selected_epoch_identical':True,'full_predictions_bitwise_identical':same_predictions})
        records.append(r)
    source_audit=None
    if args.upstream_tree:
        tree=json.loads(args.upstream_tree.read_text());unchanged=[];changed=[]
        for entry in tree['tree']:
            if entry['type']!='blob':continue
            file=ROOT/entry['path'];blob=file.read_bytes()
            sha=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
            (unchanged if sha==entry['sha'] else changed).append(entry['path'])
        assert set(changed)<={'.gitignore','README.md','reports/SCIENTIFIC_REVIEW.md'}
        source_audit={'upstream_commit':tree['sha'],'original_blob_count':len(unchanged)+len(changed),'unchanged_count':len(unchanged),'intentionally_updated_documentation':changed,
                      'all_original_data_results_models_and_core_code_unchanged':True}
    test_log=(OUT/'all_tests.log').read_text()
    match=re.search(r'Ran (\d+) tests?',test_log)
    skipped_match=re.search(r'OK \(skipped=(\d+)\)',test_log)
    if not match or not re.search(r'^OK(?: \(skipped=\d+\))?$',test_log,re.M):
        raise ValueError('A successful unittest log is required')
    n_tests=int(match.group(1));n_skipped=int(skipped_match.group(1)) if skipped_match else 0
    output={'source_commit':p['source_commit'],'registered_protocol_sha256':digest(OUT/'protocol.json'),
        'registered_training_script_sha256':p['script_sha256'],'fits':len(records),'arms':len(ARMS),'physical_test_specimens':5,
        'complete_seed_fold_matrix':True,'no_fit_test_validation_identity_overlap':True,
        'unchanged_test_source_hashes':True,'replays_identical_to_published':replay,
        'cached_prediction_rescores':available,'maximum_rescore_rmse_difference':max_rescore_difference,
        'representative_bitwise_checkpoint_replays':checkpoint_replays,
        'checkpoint_provenance_finalizations_this_call':finalized,
        'checkpoint_metadata_note':'Metadata finalization includes all actual fitting specimens and records pre/post hashes; model parameter tensors are verified unchanged.',
        'total_training_seconds':sum(r['runtime_seconds'] for r in records),
        'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'torch':str(torch.__version__),'threads_per_fit':2,'device':'cpu'},
        'upstream_preservation':source_audit,
        'test_results':{'unittest_count':n_tests,'passed':n_tests-n_skipped,'skipped':n_skipped,'log':'results/diagnosis/all_tests.log','original_source_zip_audit_executed':bool(re.search(r'^test_original_tracking_rows_exactly_define_mask_and_targets.* \.\.\. ok$',test_log,re.M))},
        'numerical_artifact_policy':'Individual run records/histories and tables are committed; checkpoints/full predictions are reproducible local cache. Original benchmark checkpoints remain committed.'}
    save(OUT/'verification.json',output);print(json.dumps({k:v for k,v in output.items() if k not in ['replays_identical_to_published','environment']},indent=2))


if __name__=='__main__':main()
