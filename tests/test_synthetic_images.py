"""Leakage/masking and source-semantics gates for the image experiment."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch

from gnn_cb.model import DeformationModel, ModelConfig
from gnn_cb.train_expanded import physical_id
from scripts.synthetic_image_geometry import procedural, adjacency, warp, normalized_graph
from scripts.train_synthetic_images import (train_gnn, train_competitor, fit_table,
    validate_teacher_membership, make_synthetic, ARMS, local_features)
from tests.test_expanded_training import synthetic_specimen


class SyntheticImageIntegrity(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data=[synthetic_specimen('S'+str(i),.03*i) for i in range(5)]

    def test_renderer_has_known_topology_and_motion(self):
        g=procedural(14); e=adjacency(g); nodes=g['nodes_xy']
        self.assertEqual(len(nodes),31)
        self.assertEqual(e.shape[1]//2-len(nodes)+1,27)
        np.testing.assert_array_equal(warp(nodes,0),nodes)
        self.assertAlmostEqual(np.ptp(warp(nodes,1)[:,1]),.65*np.ptp(nodes[:,1]))
        self.assertTrue(all((v,u) in set(map(tuple,e.T)) for u,v in e.T))

    def test_teacher_from_wrong_fold_is_rejected(self):
        ids=[physical_id(d) for d in self.data]
        meta={'training_physical_specimens':ids[:3],'validation_physical_specimens':ids[3:4],
              'held_out_physical_specimen':ids[4]}
        validate_teacher_membership(meta,self.data[:3],self.data[3:4],ids[4])
        meta['training_physical_specimens']=[ids[0],ids[1],ids[4]]
        with self.assertRaises(ValueError): validate_teacher_membership(meta,self.data[:3],self.data[3:4],ids[4])

    def test_local_inputs_do_not_read_observations(self):
        a=local_features(self.data[0]); d=copy.deepcopy(self.data[0]); del d['y']; del d['mask']
        np.testing.assert_array_equal(a,local_features(d))

    def test_competitor_table_preserves_specimen_mass_and_excludes_nan_masks(self):
        d=copy.deepcopy(self.data[:3]); d[0]['mask'][2:,0]=False; d[0]['y'][~d[0]['mask']]=np.nan
        x,y,w=fit_table(d,torch.zeros(9,2))
        self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())
        offset=0
        for source in d:
            n=int(source['mask'].sum()); self.assertAlmostEqual(w[offset:offset+n].sum(),1/3); offset+=n

    def test_training_and_selection_are_invariant_to_masked_nan(self):
        a=copy.deepcopy(self.data); b=copy.deepcopy(self.data)
        for x,y in zip(a,b):
            x['mask'][2:,0]=False; y['mask'][2:,0]=False; y['y'][~y['mask']]=np.nan
        ma,ia=train_gnn(a[:3],a[3:4],ARMS['real_mass90'],11,epochs=5)
        mb,ib=train_gnn(b[:3],b[3:4],ARMS['real_mass90'],11,epochs=5)
        self.assertEqual(ia['history'],ib['history'])
        for k,v in ma.state_dict().items(): torch.testing.assert_close(v,mb.state_dict()[k],atol=0,rtol=0)
        for kind in ['random_forest','rbf_ridge']:
            pa,ia=train_competitor(a[:3],a[3:4],kind,11)
            pb,ib=train_competitor(b[:3],b[3:4],kind,11)
            self.assertEqual(ia['candidates'],ib['candidates'])
            np.testing.assert_array_equal(pa(a[4]),pb(b[4]))

    def test_synthetic_targets_are_frozen_model_outputs(self):
        model=DeformationModel(ModelConfig(),torch.ones(9,2)*.01)
        graphs=make_synthetic(ARMS['ai_teacher'],torch.zeros(9,2),model)
        self.assertEqual(len(graphs),8)
        for d in graphs:
            self.assertFalse(d['y'].requires_grad)
            torch.testing.assert_close(d['y'],model(d['x0'],d['edge_index'],d['progress']))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_duplicate_physical_groups_are_rejected_before_fitting(self):
        with self.assertRaises(ValueError): train_gnn(self.data[:3],self.data[1:2],ARMS['real_mass90'],11,epochs=0)
        with self.assertRaises(ValueError): train_competitor(self.data[:3],self.data[1:2],'random_forest',11)


if __name__=='__main__': unittest.main()
