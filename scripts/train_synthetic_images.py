#!/usr/bin/env python
"""Registered generated-image follow-up and competing model classes.

Training/selection functions accept fit and whole-validation data only.
Synthetic graphs NEVER carry independent physical observations.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import platform
import random
import time

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
import torch

from gnn_cb.model import DeformationModel, ModelConfig, affine_features, load_checkpoint
from gnn_cb.train import tensor_data
from gnn_cb.train_expanded import compression_selection, model_mse, physical_id, evaluate_expanded
from scripts.diagnose_augmentation import (ROOT, REFERENCE, digest, save, load_data,
    split_data, fit_weighted, measure, input_files)
from scripts.synthetic_image_geometry import DATA, OUT

SEEDS = [11,29,47]
ARMS = {
    'real_mass90': {'geometry':None,'target':None},
    'ai_affine': {'geometry':'ai','target':'affine'},
    'ai_teacher': {'geometry':'ai','target':'teacher'},
    'procedural_affine': {'geometry':'procedural','target':'affine'},
    'procedural_teacher': {'geometry':'procedural','target':'teacher'},
    'ai_defects_affine': {'geometry':'ai_defects','target':'affine'},
    'ai_defects_teacher': {'geometry':'ai_defects','target':'teacher'},
    'procedural_defects_affine': {'geometry':'procedural_defects','target':'affine'},
    'procedural_defects_teacher': {'geometry':'procedural_defects','target':'teacher'},
    'random_forest': {'competitor':True},
    'rbf_ridge': {'competitor':True},
}


def registered_protocol():
    sources = input_files() + [ROOT/'scripts/train_synthetic_images.py',
        ROOT/'scripts/synthetic_image_geometry.py', ROOT/'scripts/diagnose_augmentation.py',
        OUT/'geometry_bank.json',DATA/'native/provenance.json',DATA/'native/defect_provenance.json'] + list((DATA/'native').glob('*.png'))
    return {'status':'exploratory; original five-specimen benchmark retained',
        'source_commit':'50f888bcddffe3284dae77cfba0769cbd54a62fb',
        'registered_before_new_fits':True,
        'known_before_registration':'All earlier diagnosis/derived-objective results; initial extraction/render checks and user request to emphasize geometric defects. No new model outcomes.',
        'sources_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted(set(sources))},
        'folds':json.loads((REFERENCE/'protocol.json').read_text())['folds'],
        'arms':ARMS,'seeds':SEEDS,'epochs':300,'patience':75,'validation_every':5,
        'learning_rate':.002,'weight_decay':.001,'gradient_clip':5.,'threads_per_fit':2,
        'synthetic_mass':.1,'real_mass':.9,'synthetic_phases':33,
        'geometry':'4 native text-only image_gen photos -> Otsu extraction: two roughly regular, two prompted for defects. 2 ideal sinusoidal controls (14/18 waves) plus two prescribed defective variants. Each source family has two base graphs with x stretches .9,1,1.1,1.2: eight graphs per family. Defects are predeclared pitch/lean/local crush and one open contact in analytic 18-wave control. Native prompt compliance is imperfect and documented.',
        'synthetic_sampling':'Seeded permutation of eight geometries, cycled one per update; equal long-run graph mass. All 33 equally spaced phases used.',
        'targets':'Either same-fold 3-real-fit affine field, or frozen same-fold/same-seed original GNN teacher. No synthetic physical displacement truth. Image source (native/analytic) x regular/defect-requested x target family forms eight arms. Native defects are not calibrated to analytic ones.',
        'mass_control':'real_mass90 scales the original supervised loss by .9 without auxiliary data, controlling the reduced real gradient scale.',
        'normalization':'Affine coefficients and scalar residual scale fitted on original real fitting specimens ONLY. Initial height normalizes geometry. No test/validation scaler fitting.',
        'selection':'Whole original validation specimen observed coordinate MSE only; epoch zero eligible; all arms reported, none promoted based on test score.',
        'competitors':{'inputs':'Same four local input features as MLP: graph-mean-centered x,y,phase,phase^2. No future observations. Both learn residual on identical real-fit affine prior; no synthetic data.',
            'sample_weights':'Each real fit specimen receives 1/3 total weight; observed compression node-frames only. Weights sum to one.',
            'random_forest':{'n_estimators':64,'max_depth':16,'min_samples_leaf':[8,32,128],'max_features':1.0,'bootstrap':True,'n_jobs':1},
            'rbf_ridge':{'features':128,'gamma':[.125,.5,2.],'alpha':[.0001,.001,.01],'fit_intercept':True,'scaling':'fit-only weighted feature mean/variance','solver':'cholesky'},
            'tuning':'Select the candidate with minimum same whole validation specimen MSE; fixed iteration order breaks ties. No refit on validation. These finite searches do not exhaust either model class.'},
        'statistics':{'primary_family':'Original GNN versus MLP, affine, zero, random forest and RBF ridge (five contrasts, Holm).',
            'secondary_family':'Nine synthetic interventions versus original GNN (Holm); matched factorial/control contrasts reported separately.',
            'unit':'Five physical test specimens; average three seeds within each specimen before paired testing.',
            'nominal_test':'Exact two-sided sign flips of five paired errors (32 assignments); minimum nonzero p .0625. Shared LOSO training makes independent-sign symmetry unverified.',
            'interval':'Enumerate 5^5 case resamples: conditional 95% empirical sensitivity interval, not calibrated population CI.',
            'significance_level':.05},
        'labels_warning':'Zero extra independent physical measurements. Renderer displacements are prescribed kinematics; affine targets and teacher targets are synthetic model outputs, never physical measurements.',
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'scikit_learn':sklearn.__version__}}


def register():
    value = registered_protocol(); path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Registered inputs/code changed; use a separate study')
    if not path.exists(): save(path,value)
    return value


def validate_teacher_membership(meta, training, validation, heldout):
    fit_ids = {physical_id(d) for d in training}; val_ids={physical_id(d) for d in validation}
    if set(meta['training_physical_specimens']) != fit_ids or set(meta['validation_physical_specimens']) != val_ids:
        raise ValueError('Teacher partition does not match original fit/validation')
    if meta['held_out_physical_specimen'] != heldout or heldout in fit_ids|val_ids or fit_ids & val_ids:
        raise ValueError('Teacher/test partition contamination')


def make_synthetic(options, coefficients, teacher=None):
    if options['geometry'] is None: return []
    bank = json.loads((OUT/'geometry_bank.json').read_text())[options['geometry']]
    result=[]; p=torch.linspace(0,1,33)
    for d in bank:
        x=torch.tensor(d['x0'],dtype=torch.float32); e=torch.tensor(d['edge_index'],dtype=torch.long)
        with torch.no_grad():
            y = teacher(x,e,p) if options['target']=='teacher' else affine_features(x,p) @ coefficients
        result.append({'name':d['name'],'x0':x,'edge_index':e,'progress':p,'y':y.detach()})
    return result


def train_gnn(training, validation, options, seed, teacher=None, epochs=300):
    if {physical_id(d) for d in training} & {physical_id(d) for d in validation}: raise ValueError('Train/validation overlap')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    weights=[1/len(training)]*len(training); coef=fit_weighted(training,weights)
    tt=[tensor_data(d) for d in training]; vt=[tensor_data(d) for d in validation]
    baseline=DeformationModel(ModelConfig(kind='mlp'),coef)
    with torch.no_grad(): scale=max(float(sum(w*model_mse(baseline,d,t) for w,d,t in zip(weights,training,tt)).sqrt()),.01)
    torch.manual_seed(seed); model=DeformationModel(ModelConfig(),coef,scale)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    synthetic=make_synthetic(options,coef,teacher)
    order=np.random.default_rng(20261010+seed).permutation(len(synthetic))
    best=float('inf'); best_epoch=0; best_state=copy.deepcopy(model.state_dict()); history=[]
    start=time.monotonic()
    for epoch in range(epochs+1):
        if epoch:
            model.train(); optimizer.zero_grad(set_to_none=True)
            for w,d,t in zip(weights,training,tt): (.9*w*model_mse(model,d,t)/scale**2).backward()
            if synthetic:
                d=synthetic[int(order[(epoch-1)%len(order)])]
                pred=model(d['x0'],d['edge_index'],d['progress'])
                (.1*(pred-d['y']).square().mean()/scale**2).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
        if epoch%5==0 or epoch==epochs:
            model.eval()
            with torch.no_grad(): vm=float(sum(model_mse(model,d,t) for d,t in zip(validation,vt))/len(vt))
            history.append({'epoch':epoch,'validation_mse':vm})
            if vm<best-1e-10: best=vm; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict())
            if epoch-best_epoch>=75: break
    model.load_state_dict(best_state); model.eval()
    return model,{'seed':seed,'selected_epoch':best_epoch,'completed_epochs':epoch,'validation_mse':best,
        'history':history,'runtime_seconds':time.monotonic()-start,'residual_scale':scale,
        'synthetic_graphs':len(synthetic),'synthetic_vector_targets':sum(d['y'].numel()//2 for d in synthetic),
        'synthetic_targets_are_independent_measurements':False}


def local_features(data, selected=None):
    p=np.asarray(data['progress'] if selected is None else data['progress'][selected],dtype=np.float64)
    xy=np.asarray(data['x0'],dtype=np.float64); xy=xy-xy.mean(0)
    return np.concatenate((np.broadcast_to(xy,(len(p),len(xy),2)),
        np.broadcast_to(p[:,None,None],(len(p),len(xy),1)),
        np.broadcast_to(p[:,None,None]**2,(len(p),len(xy),1))),axis=-1)


def base_prediction(data,coef,selected=None):
    p=data['progress'] if selected is None else data['progress'][selected]
    return (affine_features(torch.tensor(data['x0'],dtype=torch.float32),torch.tensor(p,dtype=torch.float32)) @ coef).numpy()


def fit_table(training,coef):
    xs=[]; ys=[]; ws=[]
    for d in training:
        select=compression_selection(d); mask=d['mask'][select]; n=int(mask.sum())
        xs.append(local_features(d,select)[mask]); ys.append((d['y'][select]-base_prediction(d,coef,select))[mask])
        ws.append(np.full(n,1/(len(training)*n)))
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)


def train_competitor(training,validation,kind,seed):
    if len(validation)!=1 or {physical_id(d) for d in training} & {physical_id(d) for d in validation}: raise ValueError('Invalid fitting partition')
    start=time.monotonic(); coef=fit_weighted(training,[1/len(training)]*len(training))
    x,y,w=fit_table(training,coef); val=validation[0]; select=compression_selection(val); mask=val['mask'][select]
    vx=local_features(val,select)[mask]; vy=val['y'][select][mask]; vb=base_prediction(val,coef,select)[mask]
    mu=np.sum(w[:,None]*x,axis=0); sd=np.sqrt(np.sum(w[:,None]*(x-mu)**2,axis=0)); sd=np.maximum(sd,1e-8)
    best=float('inf'); winner=None; selected_config=None; candidates=[]
    if kind=='random_forest':
        configs=[{'min_samples_leaf':leaf} for leaf in [8,32,128]]
        for cfg in configs:
            fit=RandomForestRegressor(n_estimators=64,max_depth=16,max_features=1.,bootstrap=True,n_jobs=1,random_state=seed,**cfg)
            fit.fit(x,y,sample_weight=w); mse=float(np.mean((fit.predict(vx)+vb-vy)**2))
            candidates.append({**cfg,'validation_mse':mse})
            if mse<best: best=mse; winner=fit; selected_config=cfg
        def predict(data):
            f=local_features(data); return (winner.predict(f.reshape(-1,4)).reshape(*f.shape[:-1],2)+base_prediction(data,coef)).astype(np.float32)
    else:
        for gamma in [.125,.5,2.]:
            sampler=RBFSampler(gamma=gamma,n_components=128,random_state=seed)
            z=sampler.fit_transform((x-mu)/sd); vz=sampler.transform((vx-mu)/sd)
            for alpha in [.0001,.001,.01]:
                fit=Ridge(alpha=alpha,fit_intercept=True,solver='cholesky'); fit.fit(z,y,sample_weight=w)
                mse=float(np.mean((fit.predict(vz)+vb-vy)**2)); cfg={'gamma':gamma,'alpha':alpha}
                candidates.append({**cfg,'validation_mse':mse})
                if mse<best: best=mse; winner=(sampler,fit); selected_config=cfg
        def predict(data):
            f=local_features(data); sampler,fit=winner
            pred=fit.predict(sampler.transform((f.reshape(-1,4)-mu)/sd)).reshape(*f.shape[:-1],2)
            return (pred+base_prediction(data,coef)).astype(np.float32)
    return predict,{'seed':seed,'selected_config':selected_config,'validation_mse':best,'candidates':candidates,
        'runtime_seconds':time.monotonic()-start,'fit_observed_node_frames':len(x),'fit_weight_sum':float(w.sum()),
        'fit_weighted_feature_mean':mu.tolist(),'fit_weighted_feature_sd':sd.tolist()}


def execute_one(task):
    arm,fold,seed=task; p=register(); fingerprint=digest(OUT/'protocol.json')
    stem=f'{arm}__fold{fold}__seed{seed}'; dest=OUT/'runs'/f'{stem}.json'; cache=OUT/'cache'
    if dest.exists():
        old=json.loads(dest.read_text()); array=cache/f'{stem}.npz'
        if old['protocol_sha256']!=fingerprint: raise ValueError('Stale cached run')
        if array.exists() and digest(array)==old['prediction_sha256']:
            if ARMS[arm].get('competitor') or (cache/f'{stem}.pt').exists(): return {'cached':True,'arm':arm,'fold':fold,'seed':seed}
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    primary,extra=load_data(); split=p['folds'][fold]
    train,val,test,weights=split_data(primary,extra,split,{'extra':[]}); options=ARMS[arm]
    cache.mkdir(parents=True,exist_ok=True)
    teacher_path=None
    if options.get('competitor'):
        predict,info=train_competitor(train,val,arm,seed)
        # Held-out data first enter the selected prediction function here.
        pred=predict(test[0]); evaluation=evaluate_expanded(test[0],pred); evaluation.pop('framewise')
    else:
        teacher=None
        if options['target']=='teacher':
            teacher_path=REFERENCE/f"test_{split['held_out_physical_specimen']}"/f'gnn_seed{seed}.pt'
            teacher,meta=load_checkpoint(str(teacher_path)); validate_teacher_membership(meta,train,val,split['held_out_physical_specimen'])
            for parameter in teacher.parameters(): parameter.requires_grad_(False)
        model,info=train_gnn(train,val,options,seed,teacher)
        evaluation,pred=measure(model,test[0])
        torch.save({'state_dict':model.state_dict(),'config':asdict(model.config),'arm':arm,
            'protocol_sha256':fingerprint,**split,'seed':seed,'selected_epoch':info['selected_epoch']},cache/f'{stem}.pt')
    np.savez_compressed(cache/f'{stem}.npz',prediction=pred,frame=test[0]['frame'],node_ids=test[0]['node_ids'],mask=test[0]['mask'])
    record={'arm':arm,'fold_index':fold,'protocol_sha256':fingerprint,**split,**info,
        'test_specimen_id':str(test[0]['specimen_id']),'test_source_sha256':test[0]['_source_sha256'],'test':evaluation,
        'prediction_sha256':digest(cache/f'{stem}.npz'),
        'teacher_sha256':digest(teacher_path) if teacher_path else None}
    if not options.get('competitor'): record['checkpoint_sha256']=digest(cache/f'{stem}.pt')
    save(dest,record)
    return {'arm':arm,'fold':fold,'seed':seed,'rmse_percent':100*evaluation['compression_window']['rmse_normalized'],'seconds':info['runtime_seconds']}


def summarize():
    p=register(); result={'protocol_sha256':digest(OUT/'protocol.json'),'arms':{}}
    for arm in ARMS:
        rows=[json.loads(f.read_text()) for f in sorted((OUT/'runs').glob(arm+'__*.json'))]
        if len(rows)!=15: continue
        if any(r['protocol_sha256']!=result['protocol_sha256'] for r in rows): raise ValueError('Stale result')
        sm=[100*np.mean([r['test']['compression_window']['rmse_normalized'] for r in rows if r['seed']==s]) for s in SEEDS]
        per={f['held_out_physical_specimen']:float(100*np.mean([r['test']['compression_window']['rmse_normalized'] for r in rows if r['fold_index']==i])) for i,f in enumerate(p['folds'])}
        result['arms'][arm]={'macro_rmse_percent':float(np.mean(sm)),'seed_sd_pp':float(np.std(sm,ddof=1)),
            'seed_means':dict(zip(map(str,SEEDS),sm)),'per_specimen':per,'fits':len(rows),
            'runtime_seconds':sum(r['runtime_seconds'] for r in rows)}
    save(OUT/'summary.json',result); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--register-only',action='store_true')
    parser.add_argument('--summarize-only',action='store_true'); parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--arms',nargs='+',choices=list(ARMS)); args=parser.parse_args(); register()
    if args.register_only: print(OUT/'protocol.json'); return
    if args.summarize_only: print(json.dumps(summarize(),indent=2)); return
    tasks=[(arm,fold,seed) for arm in (args.arms or ARMS) for fold in range(5) for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(execute_one,task) for task in tasks]): print(json.dumps(future.result()),flush=True)
    print(json.dumps(summarize(),indent=2))


if __name__=='__main__': main()
