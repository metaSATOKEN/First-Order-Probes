# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Deterministic post-analysis tests; no experiment or simulated records."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/analyze_covariance_roc.py'
spec=importlib.util.spec_from_file_location('roc_analysis',path)
roc=importlib.util.module_from_spec(spec)
spec.loader.exec_module(roc)


class TestRocAnalysis(unittest.TestCase):
    def test_exact_tail_counts(self):
        a=np.arange(20000,dtype=float)
        for alpha in roc.ALPHAS:
            self.assertEqual(np.mean(a>roc.threshold(a,alpha)),alpha)

    def test_nonrepresentable_rate_rejected(self):
        with self.assertRaises(ValueError): roc.threshold(np.arange(7),.01)

    def test_auc_with_ties(self):
        self.assertEqual(roc.auc_rank(np.arange(5),np.arange(5)),.5)

    def test_auc_and_pauc_vs_pooled_roc(self):
        normal=np.array([0.,1.,2.,3.])
        alternative=np.array([.5,2.5,4.])
        scores=np.r_[normal,alternative]
        labels=np.r_[np.zeros(4),np.ones(3)]
        order=np.argsort(-scores)
        fp=np.r_[0,np.cumsum(1-labels[order])/4]
        tp=np.r_[0,np.cumsum(labels[order])/3]
        self.assertAlmostEqual(roc.auc_rank(normal,alternative),np.trapezoid(tp,fp))
        take=fp<=.5
        self.assertAlmostEqual(roc.normalized_pauc(normal,alternative,.5),np.trapezoid(tp[take],fp[take])/.5)

    def test_same_record_index_pairs_methods(self):
        normal=np.column_stack([np.arange(1000,dtype=float)]*2)
        q=roc.bootstrap_thresholds(normal,'identical-method-test')
        np.testing.assert_array_equal(q[:,0],q[:,1])
        alt=np.column_stack([np.arange(1000,dtype=float)]*2)
        _,diff=roc.bootstrap_power(alt,q,'identical-method-test')
        np.testing.assert_array_equal(diff,[0.,0.])


if __name__=='__main__': unittest.main()
