# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Deterministic technical fixtures only; no physical trial generation."""
import unittest
import numpy as np
import analyze_pr0010_mixture as m
import pr0006_methods as frozen
import pr0009_methods as gate

class Tests(unittest.TestCase):
    def test_layouts_aliases_and_counts(self):
        cs,ls,mp,edges=m.recipes();self.assertEqual((len(cs),len(mp),len(edges)),(57,61,24))
        for c,(f,ix) in zip(cs,ls):
            self.assertEqual(int((f>=0).sum()),c['n'])
            for k in range(3):self.assertEqual(len(set(ix[f==k])),int((f==k).sum()))
        for kind in m.PATTERNS:
            for a,b in [(80,160),(160,400),(400,800)]:
                fa,ia=ls[mp[f'{kind}_n{a}']];fb,ib=ls[mp[f'{kind}_n{b}']]
                np.testing.assert_array_equal(fa[:a],fb[:a]);np.testing.assert_array_equal(ia[:a],ib[:a])
        for e in edges:
            fp,ip=ls[e['parent']];fc,ic=ls[e['child']];keep=fc>=0
            np.testing.assert_array_equal(fp[keep],fc[keep]);np.testing.assert_array_equal(ip[keep],ic[keep])
    def test_assembly_restores_original(self):
        cs,ls,mp,edges=m.recipes();base=np.arange(15999*2).reshape(15999,2)
        faults=[np.full((800,2),-k-1) for k in range(3)];p=np.arange(15999)[::-1]
        for e in edges:
            par=m.assemble(base,faults,p,ls[e['parent']]);chi=m.assemble(base,faults,p,ls[e['child']])
            f,_=ls[e['child']];keep=np.ones(15999,bool);keep[p[np.flatnonzero(f>=0)]]=False
            np.testing.assert_array_equal(chi[keep],base[keep]);np.testing.assert_array_equal(chi[~keep],par[~keep])
    def test_calibration_against_frozen(self):
        n=15999;i=np.arange(n);q=np.column_stack([np.sin(i*.1),np.cos(i*.017)])
        v=np.zeros((n,17));v[:,[11,12]]=q
        state={'valid':True,'model':{'mean':[0.,0.],'weights':[[1.,.2],[-.1,1.]],'disabled':[]}}
        s=m.project(q,state['model']);ordered=np.sort(np.column_stack([abs(s[:9999,0]),s[:9999,1]]),axis=0)
        cut,rc,top,ties=m.calibrate_scores(s,ordered);meta,c,idx=frozen.calibrate(state,ordered,v)
        np.testing.assert_allclose(cut[:,0],c[:,0]);np.testing.assert_array_equal(rc,c[:,5,0])
        pp=m.patterns(s,cut[None])[0];packed=frozen.decide(state,ordered,meta,c,v)
        pat,valid=frozen.decode(packed);np.testing.assert_array_equal(pp,pat[:,:,[0,5]])
    def test_joint_vs_direct_masks(self):
        a=np.arange(93).reshape(3,31)%8;b=(a*3+1)%8;j=m.joint_many(a,b)
        for k in range(3):
            for x in range(8):
                for y in range(8):self.assertEqual(j[k,x,y],int(((a[k]==x)&(b[k]==y)).sum()))
    def test_strict_unknown_nonfire(self):
        np.testing.assert_array_equal(gate.gate(m.spec.GATE_CUTS[0],0),[0,0,0])
        np.testing.assert_array_equal(gate.gate(np.full(3,np.nan),0),[2,2,2])
        s=np.zeros((15999,2));ordered=np.full((9999,2),-1.)
        cut,rc,top,ties=m.calibrate_scores(s,ordered)
        np.testing.assert_array_equal(rc,[9999,9999]);self.assertFalse(m.patterns(s[:10],cut[None])[:,:,:,1].any())
    def test_role_paths_distinct(self):
        self.assertNotEqual('audit_normal','update_normal')
        # Same numerical index is not the same role-qualified record ID.
        self.assertNotEqual((0,'B_M','audit','normal',0),(0,'B_M','update','normal',0))

if __name__=='__main__':unittest.main()
