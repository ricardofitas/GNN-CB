"""Safeguards for the exploratory diagnosis, independent of its results."""
import copy
from pathlib import Path
import unittest
import numpy as np
import torch
from scripts.diagnose_augmentation import (train_model,load_data,split_data,ARMS,
    observation_weights,transform_extra,fit_weighted)
from gnn_cb.train_expanded import train_selected,fit_affine_balanced
from tests.test_expanded_training import synthetic_specimen


class DiagnosisIntegrity(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data=[synthetic_specimen('S'+str(i),.03*i) for i in range(5)]

    def test_original_training_path_is_exactly_reproduced(self):
        a,_,ia=train_selected(self.data[:3],self.data[3:4],'gnn',11,10,75,.002)
        b,_,ib=train_model(self.data[:3],self.data[3:4],[1/3]*3,{},11,epochs=10)
        self.assertEqual(ia['selected_epoch'],ib['selected_epoch'])
        for k,v in a.state_dict().items(): torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)

    def test_missing_values_cannot_affect_fit_or_validation_selection(self):
        source=copy.deepcopy(self.data)
        for d in source: d['mask'][2:,0]=False
        poison=copy.deepcopy(source)
        for d in poison:d['y'][~d['mask']]=np.nan
        for options in [{},{'frame_balance':True}]:
            a,_,ia=train_model(source[:3],source[3:4],[1/3]*3,options,11,epochs=5)
            b,_,ib=train_model(poison[:3],poison[3:4],[1/3]*3,options,11,epochs=5)
            self.assertEqual(ia['history'],ib['history'])
            for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)

    def test_same_heldout_specimen_poison_cannot_change_any_fit_input(self):
        primary,extra=load_data()
        groups=[str(d['physical_specimen_id']) for d in primary]
        split={'training_physical_specimens':groups[:3],'validation_physical_specimens':groups[3:4],
               'held_out_physical_specimen':groups[4]}
        changed=copy.deepcopy(primary);changed[4]['y'][:]=999999
        for options in ARMS.values():
            a,av,at,aw=split_data(primary,extra,split,options)
            b,bv,bt,bw=split_data(changed,extra,split,options)
            self.assertEqual(aw,bw);self.assertAlmostEqual(sum(aw),1.)
            for x,y in zip(a+av,b+bv):
                for key in ['x0','y','mask','edge_index','progress']:
                    np.testing.assert_array_equal(x[key],y[key])
            np.testing.assert_array_equal(at[0]['mask'],bt[0]['mask'])
            self.assertFalse(np.array_equal(at[0]['y'],bt[0]['y']))
        # Also exercise training after the poison-resistant split.
        a,av,_,aw=split_data(primary,extra,split,ARMS['both_mass14'])
        b,bv,_,bw=split_data(changed,extra,split,ARMS['both_mass14'])
        m1,_,i1=train_model(a,av,aw,{},11,epochs=0)
        m2,_,i2=train_model(b,bv,bw,{},11,epochs=0)
        self.assertEqual(i1['selected_epoch'],i2['selected_epoch'])
        for k,v in m1.state_dict().items():torch.testing.assert_close(v,m2.state_dict()[k],atol=0,rtol=0)

    def test_quality_control_preserves_phase_and_stratum_counts(self):
        _,extra=load_data();strict=transform_extra(extra,{'quality':'strict'});matched=transform_extra(extra,{'quality':'matched'})
        for d,s,m in zip(extra,strict,matched):
            bins=np.minimum((d['x0'][:,1]*3).astype(int),2)
            for band in range(3):np.testing.assert_array_equal(s['mask'][:,bins==band].sum(1),m['mask'][:,bins==band].sum(1))
            for variant in [s,m]:
                self.assertFalse((variant['mask']&~d['mask']).any())
                np.testing.assert_array_equal(variant['y'],d['y'])
                np.testing.assert_array_equal(variant['progress'],d['progress'])

    def test_phase_perturbation_does_not_revive_or_relabel_targets(self):
        _,extra=load_data()
        for coefficient in [-.4,.4]:
            for source,changed in zip(extra,transform_extra(extra,{'warp':coefficient})):
                self.assertTrue((np.diff(changed['progress'])>0).all())
                self.assertEqual(changed['progress'][0],0)
                self.assertEqual(changed['progress'][-1],1)
                self.assertLessEqual(float(np.max(np.abs(changed['progress']-source['progress']))),.100001)
                np.testing.assert_array_equal(source['y'],changed['y'])
                np.testing.assert_array_equal(source['mask'],changed['mask'])

    def test_censoring_keeps_true_full_window_phase(self):
        _,extra=load_data();transformed=transform_extra(extra,{'censor9':True})
        for a,b in zip(extra,transformed):
            np.testing.assert_array_equal(a['progress'],b['progress'])
            self.assertFalse((b['mask']&~a['mask']).any())
        self.assertFalse(transformed[1]['mask'][-1].any())

    def test_per_frame_loss_handles_empty_frames_and_total_weight(self):
        d=copy.deepcopy(self.data[0]);d['mask'][2]=False;d['mask'][3,1:]=False
        w=observation_weights(d,True)
        self.assertAlmostEqual(w.sum(),1)
        dense=np.zeros(d['mask'].shape);dense[d['mask']]=w
        np.testing.assert_allclose(dense.sum(1)[d['mask'].any(1)],1/6)
        coeff=fit_weighted([d],[1.],True)
        self.assertTrue(torch.isfinite(coeff).all())

    def test_error_decomposition_is_descriptive_and_mask_safe(self):
        from scripts.decompose_prediction_errors import decompose
        d=copy.deepcopy(self.data[0]);d['mask'][2:,0]=False
        prediction=d['y']+np.array([.02,.03],dtype=np.float32)
        d['y'][~d['mask']]=np.nan
        r=decompose(d,prediction)
        self.assertAlmostEqual(r['mse_percent_height_squared'],6.5,places=5)
        self.assertAlmostEqual(r['frame_mean_error_mse_percent_height_squared'],6.5,places=5)
        self.assertAlmostEqual(r['within_frame_centered_error_mse_percent_height_squared'],0.,places=10)

    def test_fixed_epoch_probe_does_not_reopen_validation_selection(self):
        # With zero targets, epoch zero is optimal; continue after patience only
        # to capture the predetermined probe and keep selected state unchanged.
        data=[synthetic_specimen('S'+str(i),0) for i in range(4)]
        a,_,ia=train_model(data[:3],data[3:],[1/3]*3,{},11,epochs=20,patience=5)
        b,fixed,ib=train_model(data[:3],data[3:],[1/3]*3,{},11,fixed_epoch=15,epochs=20,patience=5)
        self.assertIsNotNone(fixed);self.assertEqual(ia['history'],ib['history'])
        self.assertEqual(ia['selected_epoch'],ib['selected_epoch'])
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
