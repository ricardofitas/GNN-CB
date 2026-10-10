#!/usr/bin/env python
"""Quantify initial domains and a directly testable graph representation limit."""
import copy
from collections import Counter
import json
import networkx as nx
import numpy as np
import torch
from gnn_cb.model import load_checkpoint
from scripts.diagnose_augmentation import ROOT, REFERENCE, load_data, save, digest
from scripts.synthetic_image_geometry import OUT, procedural, normalized_graph


def describe(name,x0,edge_index,origin):
    unique={tuple(sorted((int(a),int(b)))) for a,b in edge_index.T if a!=b}
    g=nx.Graph();g.add_nodes_from(range(len(x0)));g.add_edges_from(unique)
    lengths=np.array([np.linalg.norm(x0[a]-x0[b]) for a,b in unique])
    components=nx.number_connected_components(g)
    return {'id':name,'origin':origin,'nodes':len(x0),'undirected_edges':len(unique),'components':components,
        'simple_cycle_rank':len(unique)-len(x0)+components,'width_over_height':float(np.ptp(x0[:,0])/np.ptp(x0[:,1])),
        'degree_histogram':dict(Counter(dict(g.degree()).values())),
        'median_edge_length_over_height':float(np.median(lengths)),
        'edge_length_coefficient_of_variation':float(lengths.std()/lengths.mean())}


def main():
    torch.set_num_threads(2); primary,_=load_data(); rows=[]
    for d in primary: rows.append(describe(str(d['specimen_id']),d['x0'],d['edge_index'],'original physical specimen'))
    bank=json.loads((OUT/'geometry_bank.json').read_text())
    for family,items in bank.items():
        for d in items:
            if d['name'].endswith('_x1.0'):
                rows.append(describe(d['name'],np.array(d['x0']),np.array(d['edge_index']),family))
    original=procedural(18,defective=True); altered=copy.deepcopy(original)
    # The free flute's internal curve changes; its endpoints and all junctions do not.
    path=np.array(altered['edges'][-2]['points_xy']); delta=12*np.sin(np.linspace(0,np.pi,len(path)))
    path[:,1]+=delta; altered['edges'][-2]['points_xy']=path.tolist()
    a=normalized_graph(original,'open12');b=normalized_graph(altered,'larger_gap')
    np.testing.assert_array_equal(a['x0'],b['x0']);np.testing.assert_array_equal(a['edge_index'],b['edge_index'])
    checkpoint=REFERENCE/'final/model.pt'; model,_=load_checkpoint(str(checkpoint));phase=torch.linspace(0,1,33)
    with torch.no_grad():
        pa=model(torch.tensor(a['x0']),torch.tensor(a['edge_index']),phase)
        pb=model(torch.tensor(b['x0']),torch.tensor(b['edge_index']),phase)
    assert torch.equal(pa,pb)
    pairs=[tuple(sorted((e['source'],e['target']))) for e in original['edges']]
    output={'geometry_domains':rows,'identifiability_probe':{
        'source':'Analytic 18-wave defective graph; open flute path versus same endpoints and larger internal gap',
        'maximum_internal_polyline_change_px':float(np.max(delta)),
        'material_polyline_edges':len(pairs),'simple_adjacency_edges':len(set(pairs)),
        'parallel_material_path_pairs':[list(pair) for pair,n in Counter(pairs).items() if n>1],
        'initial_node_coordinates_identical':True,'initial_adjacency_identical':True,
        'predicted_coordinates_compared':pa.numel(),'maximum_prediction_difference':float((pa-pb).abs().max()),
        'checkpoint_sha256':digest(checkpoint),
        'conclusion':'Interior edge curvature/gap changes that preserve node coordinates and simple adjacency are unidentifiable by this model input. This is an exact representation limitation, not evidence that it caused the original augmentation regression.'}}
    save(OUT/'representation_audit.json',output); print(json.dumps(output['identifiability_probe'],indent=2))


if __name__=='__main__':main()
