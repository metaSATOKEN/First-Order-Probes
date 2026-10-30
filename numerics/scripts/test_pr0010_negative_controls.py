# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Role/metadata and unavailable-state negative controls; artificial values only."""
import hashlib,json,unittest
import numpy as np
from analyze_pr0009_mechanisms import Archive
from analyze_pr0010_mixture import gate_score,joint_many

class NegativeControls(unittest.TestCase):
    def reader(self,role='audit_fault',start=0,hash_ok=True):
        a=Archive.__new__(Archive);a.records=0;a.chunks=0
        v=np.zeros((800,17),np.float64)
        md=dict(start=start,task=dict(case='fixture',role=role),values_sha256=hashlib.sha256(v.tobytes()).hexdigest() if hash_ok else 'bad',count=800)
        a.npz=lambda n:dict(values=v,metadata=np.array(json.dumps(md)))
        return a
    def test_role_substitution_rejected(self):
        ok=self.reader();self.assertEqual(ok.values('fixture','fixture','audit_fault').shape,(800,17))
        with self.assertRaises(AssertionError):self.reader(role='update_fault').values('fixture','fixture','audit_fault')
    def test_bad_offset_and_hash_rejected(self):
        for a in [self.reader(start=1),self.reader(hash_ok=False)]:
            with self.assertRaises(AssertionError):a.values('fixture','fixture','audit_fault')
    def test_zero_sd_unknown(self):
        q=np.zeros((8,2));s=np.zeros((8,2));score,state,T=gate_score(q,s,np.ones(3),0)
        np.testing.assert_array_equal(state,[0,2,2]);self.assertTrue(np.isnan(T).all())
    def test_one_count_corruption_rejected(self):
        a=np.array([[0,1,2,4]]);b=np.array([[0,0,2,4]]);j=joint_many(a,b)
        self.assertEqual(int(j.sum()),4);bad=j.copy();bad[0,0,0]+=1
        with self.assertRaises(AssertionError):np.testing.assert_array_equal(j,bad)

if __name__=='__main__':unittest.main()
