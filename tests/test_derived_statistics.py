"""Check statistical units, exact enumeration, and multiple-comparison arithmetic."""
import unittest
import numpy as np
from scripts.analyze_derived_statistics import paired_statistics, holm


class StatisticalIntegrity(unittest.TestCase):
    def test_five_equal_nonzero_effects_have_two_sided_floor(self):
        baseline=np.full((5,3),12.);candidate=baseline-1
        result=paired_statistics(candidate,baseline)
        self.assertEqual(result['nominal_two_sided_sign_flip_p'],.0625)
        self.assertEqual(result['sign_assignments'],32)
        self.assertEqual(result['case_bootstrap_ordered_resamples'],3125)
        self.assertEqual(result['conditional_case_bootstrap_95_percent_interval_pp'],[-1.,-1.])
        self.assertFalse(result['population_significance_established'])

    def test_duplicating_seeds_never_increases_statistical_n(self):
        baseline=np.full((5,1),12.);candidate=baseline+np.array([-1,2,-3,4,-2])[:,None]
        one=paired_statistics(candidate,baseline)
        repeated=paired_statistics(np.repeat(candidate,100,axis=1),np.repeat(baseline,100,axis=1))
        for key in ['nominal_two_sided_sign_flip_p','test_specimens','sign_assignments',
                    'conditional_case_bootstrap_95_percent_interval_pp']:
            self.assertEqual(one[key],repeated[key])

    def test_sign_flip_is_two_sided_and_zero_difference_is_one(self):
        baseline=np.full((5,3),12.);delta=np.array([-1,2,3,-4,5])[:,None]
        a=paired_statistics(baseline+delta,baseline)
        b=paired_statistics(baseline-delta,baseline)
        self.assertEqual(a['nominal_two_sided_sign_flip_p'],b['nominal_two_sided_sign_flip_p'])
        self.assertEqual(paired_statistics(baseline,baseline)['nominal_two_sided_sign_flip_p'],1.)

    def test_holm_sorted_and_unsorted_known_example(self):
        np.testing.assert_allclose(holm([.01,.03,.04]),[.03,.06,.06])
        np.testing.assert_allclose(holm([.04,.01,.03]),[.06,.03,.06])
        self.assertEqual(holm([.8,.9]),[1.,1.])

    def test_nonfinite_metrics_cannot_be_reported(self):
        baseline=np.ones((5,3));candidate=baseline.copy();candidate[0,0]=np.nan
        with self.assertRaises(ValueError):paired_statistics(candidate,baseline)


if __name__=='__main__':unittest.main()
