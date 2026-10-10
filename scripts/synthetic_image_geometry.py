#!/usr/bin/env python
"""Image-generation inputs and exact-coordinate rendering controls.

Native generated photos have no physical labels. Procedural renders have known
kinematics by construction, not measured or mechanically simulated deformation.
"""
from dataclasses import asdict
from pathlib import Path
import json

import cv2
import numpy as np

from gnn_cb.image_graph import ExtractionConfig, extract_graph, gated_assignment, track_graphs
from scripts.diagnose_augmentation import ROOT, save, digest

DATA = ROOT / 'data/synthetic_images'
OUT = ROOT / 'results/synthetic_images'
STRETCHES = [.9, 1., 1.1, 1.2]


def procedural(waves, height=80., width=1600., defective=False):
    """Ideal liner-contact topology and sinusoidal centerline polylines."""
    x = np.linspace(40., 40.+width, 2*waves+1)
    nodes = np.column_stack((x, 150. - height/2 * (-1.)**np.arange(len(x))))
    # Complete the bottom liner with endpoints below the two top contacts.
    nodes = np.vstack((nodes, [x[0], 150.+height/2], [x[-1], 150.+height/2]))
    edges = []
    for j in range(2*waves):
        q = np.linspace(0, 1, 65)
        points = np.column_stack((x[j]+q*(x[j+1]-x[j]),
                                  150.-height/2*np.cos(np.pi*(j+q))))
        edges.append({'source': j, 'target': j+1, 'points_xy': points.tolist()})
    for ids in (list(range(0, 2*waves+1, 2)),
                [2*waves+1]+list(range(1,2*waves,2))+[2*waves+2]):
        for u,v in zip(ids[:-1],ids[1:]):
            edges.append({'source':u,'target':v,'points_xy':nodes[[u,v]].tolist()})
    graph = {'nodes_xy': nodes.astype(np.float32), 'edges': edges,
             'metadata': {'origin':'analytic geometry; not an experimental specimen','defective':defective}}
    if defective:
        # Fixed manufacturing-shape perturbation, independent of all measured targets.
        def shape(points):
            p=np.asarray(points,dtype=np.float64).copy(); x=p[:,0]; y=p[:,1]
            bump=np.exp(-((x-930.)/200.)**2)
            p[:,0] += 10*np.sin(2*np.pi*x/260.) + .20*(y-150.)*bump
            p[:,1] = 190.+(y-190.)*(1-.28*bump)
            return p
        graph['nodes_xy']=shape(graph['nodes_xy']).astype(np.float32)
        for e in edges: e['points_xy']=shape(e['points_xy']).tolist()
        if waves==18:
            # Open one upper contact by 12 px: fuse both liner segments and both
            # flute paths separately, then remove their former junction node.
            j=12; left=edges[j-1]; right=edges[j]
            a=np.asarray(left['points_xy']); b=np.asarray(right['points_xy'])
            a[:,1]+=12*np.linspace(0,1,len(a))**3
            b[:,1]+=12*np.linspace(1,0,len(b))**3
            flute={'source':j-1,'target':j+1,'points_xy':np.vstack([a,b[1:]]).tolist()}
            liner=[e for e in edges[2*waves:] if j in (e['source'],e['target'])]
            u=next(e['source'] for e in liner if e['target']==j)
            v=next(e['target'] for e in liner if e['source']==j)
            joined={'source':u,'target':v,'points_xy':graph['nodes_xy'][[u,j,v]].tolist()}
            remove={id(left),id(right),*(id(e) for e in liner)}
            graph['edges']=[e for e in edges if id(e) not in remove]+[flute,joined]
            graph['nodes_xy']=np.delete(graph['nodes_xy'],j,axis=0)
            for e in graph['edges']:
                e['source']-=e['source']>j; e['target']-=e['target']>j
            graph['metadata']['open_contact_gap_px']=12
    return graph


def adjacency(graph):
    pairs = {(e['source'],e['target']) for e in graph['edges'] if e['source'] != e['target']}
    pairs |= {(v,u) for u,v in list(pairs)}
    return np.asarray(sorted(pairs), dtype=np.int64).T.reshape(2,-1)


def normalized_graph(graph, name, stretch=1.):
    nodes = np.asarray(graph['nodes_xy'], dtype=np.float32)
    scale = float(np.ptp(nodes[:,1])); x0 = (nodes-nodes.min(0))/scale
    x0[:,0] *= stretch
    return {'name': name, 'x0': x0, 'edge_index': adjacency(graph),
            'source_nodes': len(nodes), 'source_height_px': scale,
            'aspect': float(np.ptp(x0[:,0]))}


def geometry_bank():
    bank = {'ai': [], 'procedural': [],'ai_defects': [],'procedural_defects': []}
    for name in ['ai_regular','ai_asymmetric','ai_crushed','ai_contact_requested']:
        image = cv2.imread(str(DATA/'native'/f'{name}.png'),cv2.IMREAD_GRAYSCALE)
        if image is None: raise FileNotFoundError(name)
        graph = extract_graph(image, ExtractionConfig(threshold_mode='otsu'))
        if not graph['metadata']['valid_geometry']: raise ValueError('Invalid generated graph')
        for stretch in STRETCHES:
            family='ai' if name in ['ai_regular','ai_asymmetric'] else 'ai_defects'
            bank[family].append(normalized_graph(graph,f'{name}_x{stretch}',stretch))
    for defective in [False,True]:
        for waves in [14,18]:
            for stretch in STRETCHES:
                family='procedural_defects' if defective else 'procedural'
                bank[family].append(normalized_graph(procedural(waves,defective=defective),f'analytic{waves}_defect{defective}_x{stretch}',stretch))
    return bank


def warp(points, phase):
    p = np.asarray(points, dtype=np.float64).copy()
    p[:,0] = 840. + (p[:,0]-840.)*(1+.02*phase)
    p[:,1] = 190. + (p[:,1]-190.)*(1-.35*phase)
    return p


def render(graph, phase, noisy, seed):
    """Draw a NEW exact-coordinate raster, never edit a native AI photograph."""
    canvas = np.zeros((300,1680), dtype=np.uint8)
    for e in graph['edges']:
        pts = np.rint(warp(e['points_xy'],phase)).astype(np.int32)
        cv2.polylines(canvas,[pts],False,230,3,cv2.LINE_AA)
    if noisy:
        canvas = cv2.GaussianBlur(canvas,(7,7),1.2).astype(np.float32)
        lighting = np.linspace(.60,1.15,canvas.shape[1])[None,:]
        rng = np.random.default_rng(seed)
        canvas = np.clip(canvas*lighting+rng.normal(4.,5.,canvas.shape),0,255).astype(np.uint8)
    return canvas


def extraction_score(truth, graph, phase):
    true = warp(truth['nodes_xy'],phase)
    assigned, distance, _ = gated_assignment(true,graph['nodes_xy'],12.,ambiguity_margin=1.5)
    valid = assigned >= 0; matched = int(valid.sum())
    mapping = {int(obs):int(i) for i,obs in enumerate(assigned) if obs>=0}
    edges_true = {tuple(sorted((e['source'],e['target']))) for e in truth['edges']}
    extracted = {tuple(sorted((e['source'],e['target']))) for e in graph['edges']}
    correct = sum(u in mapping and v in mapping and tuple(sorted((mapping[u],mapping[v]))) in edges_true
                  for u,v in extracted)
    return {'matched_nodes':matched,'true_nodes':len(true),'extracted_nodes':len(graph['nodes_xy']),
        'node_precision':matched/max(1,len(graph['nodes_xy'])),'node_recall':matched/len(true),
        'node_f1':2*matched/max(1,len(true)+len(graph['nodes_xy'])),
        'node_localization_rmse_px':float(np.sqrt(np.mean(distance[valid]**2))) if matched else None,
        'edge_f1':2*correct/max(1,len(edges_true)+len(extracted)), 'assignment':assigned.tolist()}


def image_audit():
    result = {'warning':'Synthetic rendering diagnostics, not physical accuracy or independent extra specimens.',
              'matching_tolerance_px':12,'native':{},'rendered_sequences':[]}
    for name in ['ai_regular','ai_asymmetric','ai_crushed','ai_contact_requested']:
        path = DATA/'native'/f'{name}.png'; image = cv2.imread(str(path),0)
        result['native'][name] = {'sha256':digest(path),'shape':list(image.shape),'thresholds':{}}
        for mode in ['fixed','otsu']:
            g = extract_graph(image,ExtractionConfig(threshold_mode=mode))
            result['native'][name]['thresholds'][mode] = g['metadata']
        if name=='ai_asymmetric': result['native'][name]['manual_flag']='Object cropped at left and right image borders'
    for waves,defective in [(14,False),(18,False),(14,True),(18,True)]:
        truth = procedural(waves,defective=defective)
        for noisy in [False,True]:
            phases = np.linspace(0,1,21)
            images = [render(truth,p,noisy,20261010+100*waves+i) for i,p in enumerate(phases)]
            folder = DATA/'rendered'/f'waves{waves}_defect{defective}_{"noisy" if noisy else "clean"}'
            folder.mkdir(parents=True,exist_ok=True)
            for i in [0,10,20]: cv2.imwrite(str(folder/f'frame{i:02d}.png'),images[i])
            for mode in ['fixed','otsu']:
                config = ExtractionConfig(threshold_mode=mode)
                graphs = [extract_graph(im,config) for im in images]
                rows = [extraction_score(truth,g,p) for g,p in zip(graphs,phases)]
                tracks = track_graphs(graphs,config)
                initial_proxy=graphs[0]['nodes_xy']
                proxy_truth=np.stack([warp(initial_proxy,p)-initial_proxy for p in phases])
                proxy_mask=tracks['mask'] & tracks['frame_valid'][:,None]
                proxy_error=tracks['displacement_px']-proxy_truth
                mapping = np.asarray(rows[0]['assignment']); ok = mapping>=0
                initial_truth_ids = np.flatnonzero(ok); extracted_ids = mapping[ok]
                observed = tracks['mask'][:,extracted_ids] & tracks['frame_valid'][:,None]
                true_displacement = np.stack([warp(truth['nodes_xy'],p)-truth['nodes_xy'] for p in phases])[:,initial_truth_ids]
                error = tracks['displacement_px'][:,extracted_ids]-true_displacement
                metrics = {'waves':waves,'defective':defective,'noisy':noisy,'threshold_mode':mode,'frames':len(phases),
                    'mean_node_f1':float(np.mean([r['node_f1'] for r in rows])),
                    'mean_edge_f1':float(np.mean([r['edge_f1'] for r in rows])),
                    'initial_true_nodes_matched':int(ok.sum()),'initial_true_nodes':len(truth['nodes_xy']),
                    'track_coverage_of_initial_matched_nodes':float(observed.mean()),
                    'track_coverage_of_all_true_nodes':float(observed.sum()/(len(phases)*len(truth['nodes_xy']))),
                    'tracked_displacement_endpoint_rmse_px':float(np.sqrt(np.mean(np.sum(error[observed]**2,axis=-1)))) if observed.any() else None,
                    'initial_extracted_proxy_nodes':len(initial_proxy),
                    'proxy_warp_tracking_coverage':float(proxy_mask.mean()),
                    'proxy_warp_endpoint_rmse_px':float(np.sqrt(np.mean(np.sum(proxy_error[proxy_mask]**2,axis=-1)))) if proxy_mask.any() else None,
                    'framewise':rows}
                result['rendered_sequences'].append(metrics)
    result['unique_generated_renders'] = 168
    result['extraction_calls_on_renders'] = 336
    result['render_definition'] = '21 phases per 2 geometries x regular/defective x 2 appearances; bottom fixed, height reduced 35%, width increased 2%. Initial defects: uneven pitch, lean, local 28% crush; 18-wave case additionally opens one contact by 12px. Prescribed kinematics, not FEM.'
    result['topology_caution']='Strict ideal-contact matching exposes representation mismatch: raster skeleton junctions may split around a tangential contact. Proxy-warp metrics instead measure consistency at all frame0 extracted nodes, and are not mechanical accuracy.'
    save(OUT/'image_audit.json',result)
    bank = geometry_bank()
    save(OUT/'geometry_bank.json',{key:[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in d.items()} for d in rows] for key,rows in bank.items()})
    print(json.dumps({'native':{k:{m:v['n_nodes'] for m,v in d['thresholds'].items()} for k,d in result['native'].items()},
                      'sequences':[{k:v for k,v in row.items() if k!='framewise'} for row in result['rendered_sequences']]},indent=2))


if __name__ == '__main__': image_audit()
