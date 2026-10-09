"""Scientific integrity gates for extra correlated supervision and input noise."""
import copy
import unittest

import numpy as np
import torch

from gnn_cb.model import ModelConfig, affine_features
from gnn_cb.train import tensor_data
from scripts.derived_objectives import (temporal_pair_mse, edge_pair_mse,
    objective, jitter_geometry, GlobalReadoutModel, label_inventory)
from scripts.train_derived_supervision import train, ARMS, partition_options
from scripts.diagnose_augmentation import train_model, load_data, split_data
from tests.test_expanded_training import synthetic_specimen


class DerivedSupervisionIntegrity(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.samples = [synthetic_specimen('S'+str(i), .03*i) for i in range(5)]

    def test_temporal_identity_matches_explicit_pairs_and_gradients(self):
        torch.manual_seed(2)
        truth = torch.randn(5,4,2, dtype=torch.float64)
        mask = torch.tensor([[1,1,0,0], [1,0,0,0], [0,1,0,0], [1,1,1,0], [1,0,0,0]], dtype=torch.bool)
        truth[~mask] = float('nan')
        for balanced in [False, True]:
            pred = torch.randn_like(truth, requires_grad=True)
            errors = []; node_means = []; counts = []
            for n in range(4):
                ids = torch.where(mask[:,n])[0]
                pairs = [((pred[a,n]-pred[b,n])-(truth[a,n]-truth[b,n])).square().mean()
                         for j,a in enumerate(ids) for b in ids[j+1:]]
                errors.extend(pairs)
                if pairs: node_means.append(torch.stack(pairs).mean()); counts.append(len(ids))
            expected = (sum(v*k for v,k in zip(node_means,counts))/sum(counts)
                        if balanced else torch.stack(errors).mean())
            actual = temporal_pair_mse(pred, truth, mask, balanced)
            torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
            ga = torch.autograd.grad(actual,pred,retain_graph=True)[0]
            gb = torch.autograd.grad(expected,pred)[0]
            torch.testing.assert_close(ga,gb,atol=1e-13,rtol=1e-13)

    def test_edge_loss_counts_one_orientation_and_excludes_missing_endpoints(self):
        d = self.samples[0]; t = tensor_data(d)
        t['mask'][2:,0] = False; t['y'][~t['mask']] = float('nan')
        pred = torch.arange(42,dtype=torch.float32).reshape(7,3,2)/20
        value = edge_pair_mse(pred,t['y'],t['mask'],t['edge_index'])
        terms = []
        for i in range(7):
            for a,b in [(0,1),(1,2)]:
                if t['mask'][i,a] and t['mask'][i,b]:
                    terms.append(((pred[i,a]-pred[i,b])-(t['y'][i,a]-t['y'][i,b])).square())
        torch.testing.assert_close(value,torch.stack(terms).mean())

    def test_no_pair_and_no_edges_have_finite_zero_auxiliary_gradient(self):
        pred = torch.randn(2,3,2,requires_grad=True)
        mask = torch.tensor([[1,0,0],[0,1,0]],dtype=torch.bool)
        truth = torch.full_like(pred,float('nan')); truth[mask] = 0
        value = temporal_pair_mse(pred,truth,mask) + edge_pair_mse(pred,truth,mask,torch.empty(2,0,dtype=torch.long))
        self.assertEqual(float(value.detach()),0.)
        value.backward(); self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(float(pred.grad.abs().sum()),0.)

    def test_replay_exactly_matches_original_training_path(self):
        a,_,ia = train_model(self.samples[:3],self.samples[3:4],[1/3]*3,{},11,epochs=10)
        b,ib = train(self.samples[:3],self.samples[3:4],[1/3]*3,{},11,epochs=10)
        self.assertEqual(ia['selected_epoch'],ib['selected_epoch'])
        for k,v in a.state_dict().items(): torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)

    def test_masked_nan_cannot_change_training_or_selection(self):
        a = copy.deepcopy(self.samples); b = copy.deepcopy(self.samples)
        for x,y in zip(a,b):
            x['mask'][2:,0] = False; y['mask'][2:,0] = False
            y['y'][~y['mask']] = np.nan
        for arm in ['joint_pairs','temporal_balanced','global_joint','mc_geometry_jitter']:
            m,mi = train(a[:3],a[3:4],[1/3]*3,ARMS[arm],11,epochs=5)
            n,ni = train(b[:3],b[3:4],[1/3]*3,ARMS[arm],11,epochs=5)
            self.assertEqual(mi['history'],ni['history'])
            for k,v in m.state_dict().items(): torch.testing.assert_close(v,n.state_dict()[k],atol=0,rtol=0)

    def test_global_readout_starts_affine_and_is_node_permutation_equivariant(self):
        d = tensor_data(self.samples[1]); model = GlobalReadoutModel(ModelConfig())
        expected = affine_features(d['x0'],d['progress'])@model.affine_coefficients
        torch.testing.assert_close(model(d['x0'],d['edge_index'],d['progress']),expected,atol=0,rtol=0)
        torch.nn.init.normal_(model.decoder[-1].weight)
        torch.nn.init.normal_(model.global_decoder[-1].weight)
        order = torch.tensor([2,0,1]); inverse = torch.argsort(order)
        a = model(d['x0'],d['edge_index'],d['progress'])
        b = model(d['x0'][order],inverse[d['edge_index']],d['progress'])
        torch.testing.assert_close(a[:,order],b,atol=1e-5,rtol=1e-5)

    def test_jitter_is_reproducible_and_does_not_mutate_source(self):
        x = tensor_data(self.samples[0])['x0']; before = x.clone()
        a = jitter_geometry(x,torch.Generator().manual_seed(4))
        b = jitter_geometry(x,torch.Generator().manual_seed(4))
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(x,before,atol=0,rtol=0)
        torch.testing.assert_close(a.mean(0),x.mean(0),atol=1e-7,rtol=1e-7)
        self.assertGreater(float((a-x).abs().max()),0.)

    def test_poisoned_test_labels_cannot_affect_any_fitting_inputs(self):
        primary,extra = load_data(); ids=[str(d['physical_specimen_id']) for d in primary]
        split={'training_physical_specimens':ids[:3],'validation_physical_specimens':ids[3:4],
               'held_out_physical_specimen':ids[4]}
        poisoned=copy.deepcopy(primary);poisoned[4]['y'][:]=1e20
        for arm in ARMS.values():
            a,av,_,aw=split_data(primary,extra,split,partition_options(arm))
            b,bv,_,bw=split_data(poisoned,extra,split,partition_options(arm))
            self.assertEqual(aw,bw)
            for x,y in zip(a+av,b+bv):
                self.assertEqual(label_inventory(x),label_inventory(y))
                for key in ['x0','y','mask','edge_index','progress']:
                    np.testing.assert_array_equal(x[key],y[key])

    def test_derived_losses_cannot_identify_constant_translation(self):
        t=tensor_data(self.samples[1]); pred=t['y']+torch.tensor([.23,-.19])
        self.assertLess(float(temporal_pair_mse(pred,t['y'],t['mask'])),1e-14)
        self.assertLess(float(edge_pair_mse(pred,t['y'],t['mask'],t['edge_index'])),1e-14)
        self.assertGreater(float(objective(pred,t['y'],t['mask'],t['edge_index'],.5,.5)),.04)


if __name__=='__main__': unittest.main()
