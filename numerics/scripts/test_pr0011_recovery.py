# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Deterministic fixtures, not scientific samples."""
import unittest,itertools
import numpy as np
from scipy.stats import binom
import analyze_pr0011_recovery as m

class TestRecovery(unittest.TestCase):
    def test_policy_states(self):
        for h,f,c,avail in itertools.product([0,1],range(4),range(4),[False,True]):
            a=m.choices(h,f,c,avail)
            if h==0:self.assertEqual(a.tolist(),[0,0,0])
            elif f==0:self.assertEqual(a.tolist(),[1,1,1])
            else:self.assertEqual(a.tolist(),[1,3,2 if c==0 and avail else 3])
        self.assertEqual(m.choices(2,0,0).tolist(),[1,4,4])
    def test_invalid_state(self):
        with self.assertRaises(ValueError):m.choices(9,0,0)
        with self.assertRaises(ValueError):m.choices(1,8,0)
    def test_cp_boundaries(self):
        for a,s,x in [(.01,211,291),(.001,12,39)]:
            self.assertEqual(m.label(m.cp(s),a),0);self.assertEqual(m.label(m.cp(s+1),a),2)
            self.assertEqual(m.label(m.cp(x),a),1);self.assertEqual(m.label(m.cp(x-1),a),2)
            self.assertLessEqual(binom.cdf(s,20000,1.25*a),m.ETA)
            self.assertLess(binom.sf(x-1,20000,1.25*a),m.ETA)
    def test_unavailable_not_support(self):
        for bound in [m.cp(0,0),m.cp(-1),m.cp(0,provenance=False),m.cp(float('nan'))]:
            self.assertEqual(m.label(bound,.01),3)
    def test_strict_ties(self):
        mod=dict(mean=[0,0],weights=[[1,0],[0,1]])
        q=np.array([[1.,0.],[-1.,0.],[0.,1.],[2.,2.]])
        b=m.decide(q,mod,np.ones((2,2,3)))
        self.assertTrue(np.array_equal(b[:3],np.zeros((3,2,2))))
        self.assertTrue(np.all(b[3]==5))
    def test_all_pair_tables_against_records(self):
        # Exhausts all 8^3 three-detector branch combinations once.
        values=np.array(list(itertools.product(range(8),repeat=3)),np.uint8)
        u,f,c=values.T
        hist=lambda a,b:np.bincount(a.astype(int)*8+b,minlength=64).reshape(8,8)
        jf,jc,jfc=hist(f,u),hist(c,u),hist(f,c)
        for l,r in itertools.product(range(5),repeat=2):
            left=np.where(values[:,l]>0,0,1) if l<3 else np.full(512,l-1)
            right=np.where(values[:,r]>0,0,1) if r<3 else np.full(512,r-1)
            expected=np.bincount(left*4+right,minlength=16).reshape(4,4)
            self.assertTrue(np.array_equal(m.pair_table(l,r,jf,jc,jfc),expected),(l,r))
    def test_marginal_corruption(self):
        j=np.ones((8,8),np.int64);bad=j.copy();bad[0,0]+=1
        with self.assertRaises(AssertionError):m.base_hist(j,bad)
    def test_inactive_and_defer_differ(self):
        h=np.zeros((3,8),np.int64);h[:,0]=10
        self.assertEqual(m.outcome(1,h).tolist(),[0,10,0,0])
        self.assertEqual(m.outcome(3,h).tolist(),[0,0,10,0])
        self.assertEqual(m.outcome(4,h).tolist(),[0,0,0,10])
    def test_candidate_axis_is_preserved(self):
        x=np.zeros((6,57,4,2,2,8,8),np.int64)
        for g in range(6):
            for k in range(57):x[g,k]=100*g+k
        y=m.select_joints(x,list(range(37)))
        self.assertEqual(y.shape,(6,37,4,2,8,8))
        self.assertEqual(y[5,36,3,1,7,7],536)
if __name__=='__main__':unittest.main()
