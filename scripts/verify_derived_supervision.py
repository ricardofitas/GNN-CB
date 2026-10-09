#!/usr/bin/env python
"""Verify all saved outcomes, unchanged tests, complete provenance and source files."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

import numpy as np
import torch

from gnn_cb.model import DeformationModel, ModelConfig
from gnn_cb.train import metrics, tensor_data
from gnn_cb.train_expanded import compression_selection, physical_id
from scripts.derived_objectives import GlobalReadoutModel, label_inventory
from scripts.diagnose_augmentation import load_data, digest, save, split_data
from scripts.train_derived_supervision import ROOT, OUT, REFERENCE, ARMS, SEEDS, register, partition_options
from scripts.analyze_derived_statistics import recorded_matrix, benchmark_matrix


def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    cls={'DeformationModel':DeformationModel,'GlobalReadoutModel':GlobalReadoutModel}[checkpoint['model_class']]
    model=cls(ModelConfig(**checkpoint['config']))
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-tree',type=Path)
    args=parser.parse_args();protocol=register();torch.set_num_threads(2)
    primary,extra=load_data();lookup={physical_id(d):d for d in primary}
    expected={f'{a}__fold{i}__seed{s}' for a in ARMS for i in range(5) for s in SEEDS}
    paths=list((OUT/'runs').glob('*.json'))
    if {p.stem for p in paths}!=expected:raise ValueError('Incomplete or extraneous run matrix')
    source_audit=None
    if args.upstream_tree:
        tree=json.loads(args.upstream_tree.read_text());changed=[];unchanged=[]
        for item in tree['tree']:
            if item['type']!='blob':continue
            data=(ROOT/item['path']).read_bytes()
            sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            (unchanged if sha==item['sha'] else changed).append(item['path'])
        assert set(changed)<={'.gitignore','README.md','reports/AUGMENTATION_DIAGNOSIS.md'}
        source_audit={'source_commit':protocol['source_commit'],'original_blob_count':len(changed)+len(unchanged),
            'unchanged_count':len(unchanged),'intentionally_updated_documents':changed,
            'all_original_data_models_results_and_core_code_unchanged':True}
    original=json.loads((REFERENCE/'metrics.json').read_text());replays=[];records=[]
    max_score_error=0.;full_model_replays=0
    for path in sorted(paths):
        row=json.loads(path.read_text());records.append(row)
        assert row['protocol_sha256']==digest(OUT/'protocol.json')
        split=protocol['folds'][row['fold_index']]
        train,validation,test,weights=split_data(primary,extra,split,partition_options(ARMS[row['arm']]))
        data=test[0];assert row['test_source_sha256']==data['_source_sha256']
        assert row['held_out_physical_specimen']==physical_id(data)
        expected_training=[{'physical_id':physical_id(d),'weight':w,**label_inventory(d)} for d,w in zip(train,weights)]
        assert row['training']==expected_training
        assert row['training_physical_specimens']==[physical_id(d) for d in train]
        assert row['validation_physical_specimens']==[physical_id(d) for d in validation]
        history=row['history'];selected=min(history,key=lambda x:x['validation_mse'])
        assert row['selected_epoch']==selected['epoch']
        assert row['validation_mse']==selected['validation_mse']
        cp=OUT/'cache'/f'{path.stem}.pt';npz=OUT/'cache'/f'{path.stem}.npz'
        assert digest(cp)==row['checkpoint_sha256'] and digest(npz)==row['prediction_sha256']
        model,meta=load_model(cp)
        for key in ['training_physical_specimens','validation_physical_specimens','held_out_physical_specimen','arm','seed','selected_epoch','protocol_sha256']:
            assert meta[key]==row[key]
        assert meta['options']==ARMS[row['arm']]
        assert not set(meta['training_physical_specimens']) & {meta['held_out_physical_specimen'],*meta['validation_physical_specimens']}
        with np.load(npz,allow_pickle=False) as z:
            for key in ['mask','node_ids','frame']:np.testing.assert_array_equal(z[key],data[key])
            pred=z['prediction'];select=compression_selection(data)
            rescored=metrics(pred[select],data['y'][select],data['mask'][select],float(data['scale_px']))
            error=abs(rescored['rmse_normalized']-row['test']['compression_window']['rmse_normalized'])
            max_score_error=max(max_score_error,error);assert error==0
            t=tensor_data(data)
            with torch.no_grad(): replay=model(t['x0'],t['edge_index'],t['progress']).numpy()
            np.testing.assert_array_equal(replay,pred);full_model_replays+=1
            if row['arm']=='primary_replay':
                fold=original['folds'][row['fold_index']]
                old=next(r for r in fold['models']['gnn'] if r['seed']==row['seed'])
                assert old['selected_epoch']==row['selected_epoch']
                old_path=REFERENCE/f"test_{physical_id(data)}"/f"gnn_seed{row['seed']}_{str(data['specimen_id'])}_predictions.npz"
                with np.load(old_path,allow_pickle=False) as before:
                    for key in ['prediction','mask','node_ids','frame']:np.testing.assert_array_equal(before[key],z[key])
                replays.append({'specimen':physical_id(data),'seed':row['seed'],'selected_epoch_identical':True,'full_predictions_bitwise_identical':True})
    np.testing.assert_array_equal(recorded_matrix(OUT,'primary_replay'),benchmark_matrix(REFERENCE,'gnn'))
    log=(OUT/'all_tests.log').read_text();count=re.search(r'Ran (\d+) tests?',log)
    if not count or not re.search(r'^OK$',log,re.M):raise ValueError('Successful test log with no skips is required')
    statistics=json.loads((OUT/'statistics.json').read_text())
    assert statistics['new_family_size']==len(ARMS)-1
    assert len(statistics['all_22_nominal_holm'])==22
    assert all(r['test_specimens']==5 and r['sign_assignments']==32 and r['case_bootstrap_ordered_resamples']==3125
        for r in statistics['new_arms_vs_original'].values())
    output={'source_commit':protocol['source_commit'],
        'remote_registration_commit':'061a38f46c9bc0e0b51a364ec24f186ae6d63b91',
        'protocol_sha256':digest(OUT/'protocol.json'),'all_protocol_input_and_code_hashes_match':True,
        'fits':len(records),'arms':len(ARMS),'physical_test_specimens':5,'optimizer_seeds':SEEDS,
        'original_replays':replays,'bitwise_checkpoint_prediction_replays':full_model_replays,
        'saved_prediction_rescores':len(records),'maximum_rescore_rmse_error':max_score_error,
        'complete_registered_fold_seed_matrix':True,'all_train_validation_test_physical_ids_disjoint':True,
        'all_epochs_selected_only_from_original_validation_cases':True,
        'all_original_test_masks_frames_and_node_ids_unchanged':True,
        'statistical_unit_is_physical_case_not_frame_or_seed':True,
        'training_seconds_summed_over_fits':sum(r['runtime_seconds'] for r in records),
        'tests_passed':int(count.group(1)),'tests_skipped':0,'test_log_sha256':digest(OUT/'all_tests.log'),
        'upstream_preservation':source_audit,
        'environment':{'python':sys.version,'platform':platform.platform(),'torch':str(torch.__version__),'numpy':np.__version__,'device':'cpu','threads_per_fit':2}}
    save(OUT/'verification.json',output)
    print(json.dumps({k:v for k,v in output.items() if k not in ['original_replays','environment']},indent=2))


if __name__=='__main__':main()
