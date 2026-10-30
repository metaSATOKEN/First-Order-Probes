# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Deterministic fusion checks; no generated probe records."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('fusion', scripts / 'analyze_covariance_fusion.py')
fusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion)


class TestFusion(unittest.TestCase):
    def test_finite_sample_ranks(self):
        self.assertEqual(fusion.cutoff_order(10000, .01), 9901)
        self.assertEqual(fusion.cutoff_order(10000, .005), 9951)
        self.assertTrue(np.isinf(fusion.cutoff(np.arange(10), .01)))

    def test_reference_ranks_keep_ties_conservative(self):
        ref = np.array([[0, 3], [1, 1], [1, 1], [3, 0]], dtype=float)
        obs = np.array([[-1, -1], [0, 0], [1, 1], [2, 2], [4, 4]], dtype=float)
        expected = np.array([[sum(ref[:, m] < r[m]) for m in range(2)] for r in obs])
        np.testing.assert_array_equal(fusion.reference_ranks(ref, obs), expected)

    def test_rank_decisions_equal_raw_cutoffs(self):
        ref = np.column_stack([np.arange(400)//2, (np.arange(400)*17)%400]).astype(float)
        cal = np.column_stack([np.arange(1000)/5, ((np.arange(1000)*13)%1000)/2.5])
        raw, rank, _ = fusion.fit_rules(ref, cal)
        evals = np.r_[ref, cal, [[-1, -1], [500, 500]]]
        expect = np.any(fusion.reference_ranks(ref, evals)[:, None, :] > rank[None, :, :], axis=2)
        np.testing.assert_array_equal(fusion.decisions(evals, raw), expect)
        self.assertTrue(np.all(fusion.decisions(cal, raw).mean(axis=0) <= .01))

    def test_identical_methods_and_bonferroni_subset(self):
        ref = np.column_stack([np.arange(2000)]*2).astype(float)
        cal = np.column_stack([np.arange(1000)*2+.25]*2)
        raw, _, _ = fusion.fit_rules(ref, cal)
        flags = fusion.decisions(ref, raw)
        np.testing.assert_array_equal(flags[:, 0], flags[:, 1])
        np.testing.assert_array_equal(flags[:, 0], flags[:, 3])
        self.assertFalse(np.any(flags[:, 2] & ~flags[:, 0]))

    def test_all_pattern_accounting(self):
        flags = fusion.BITS.astype(bool)
        np.testing.assert_array_equal(fusion.patterns(flags), np.ones(16, dtype=int))
        for a, b in fusion.PAIRS:
            self.assertEqual(fusion.decomposition(flags, a, b), (4,4,4,4))

    def test_holdout_split_matches_previous_and_is_disjoint(self):
        a = fusion.prior.rng_for('holdout-split/e3_null_T2.048').permutation(20000)
        b = fusion.prior.rng_for('holdout-split/e3_null_T2.048').permutation(20000)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(np.unique(a)), 20000)
        self.assertEqual(len(np.intersect1d(a[:10000],a[10000:])), 0)

    def test_paired_multinomial_bootstrap(self):
        score = np.column_stack([np.arange(1000)]*2).astype(float)
        cuts = np.repeat(np.array([[[900, np.inf], [np.inf, 900], [950, 950], [900, 900]]]), 40, axis=0)
        power, gained, lost = fusion.bootstrap_evaluation(score, cuts, 'identical-test')
        np.testing.assert_array_equal(power[:, 0],power[:, 1])
        np.testing.assert_array_equal(power[:, 0],power[:, 3])
        self.assertTrue(np.all(power[:, 2] <= power[:, 0]))
        np.testing.assert_array_equal(gained[:, 2],np.zeros(40))
        np.testing.assert_array_equal(lost[:, 2],np.zeros(40))

    def test_tied_calibration_can_be_more_conservative(self):
        self.assertEqual(fusion.cutoff(np.ones(1000), .01), 1.)
        self.assertEqual(np.count_nonzero(np.ones(1000)>1),0)


if __name__ == '__main__':
    unittest.main()
