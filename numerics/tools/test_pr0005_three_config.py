# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Deterministic implementation fixtures; no scientific trial generation."""
from pathlib import Path
from fractions import Fraction
import copy
import json
import math
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
import analyze_pr0005_three_config as a


def test():
    cases=0
    for n in [1,4,5,10000]:
        for alpha in a.ALPHAS:
            for budget,weights in a.BUDGETS.items():
                assert sum(weights)==8
                for w in weights:
                    beta=alpha*Fraction(w,8);k=a.order(n,beta)
                    assert k==math.ceil(Fraction(n+1)*(1-beta))
                    x=np.arange(n,dtype=float)//2;cut=a.threshold(x,beta)
                    expected=float(sorted(x)[k-1]) if k<=n else np.inf
                    assert cut==expected
                    assert not bool(cut>cut);cases+=1
    assert a.threshold(np.arange(4.),Fraction(1,5))==3
    assert a.threshold(np.arange(4.),Fraction(1,6))==np.inf
    # Physical coordinates with c nonzero, and either sign of determinant.
    for D in [np.array([[2.,-1.],[.3,3.]]),np.array([[-2.,-1.],[.3,3.]])]:
        m=dict(directions=D.T.tolist(),covariance=[[4.,.1],[.1,1.]])
        ph=a.physical_model(m);assert ph['available']
        uv=np.array([[0,0],[.2,.04],[-.1,.03],[1,2.]])
        q=uv@D.T;recovered=q@np.array(ph['inverse']).T
        assert np.allclose(uv,recovered,atol=1e-14)
        for j in range(2):
            c=a.threshold(recovered[:,j],Fraction(1,3))
            assert np.array_equal(recovered[:,j]>c,7*recovered[:,j]>7*c)
        cases+=1
    for D,reason in [(np.zeros((2,2)),'zero'),(np.array([[1.,1.],[1.,1.]]),'parallel'),
                     (np.array([[1.,1.],[0.,1e-9]]),'near_parallel'),(np.array([[np.nan,1.],[1.,1.]]),'nonfinite')]:
        assert not a.physical_model(dict(directions=D.T,covariance=np.eye(2)))['available'],reason;cases+=1
    assert not a.physical_model(dict(directions=np.eye(2),covariance=np.diag([0.,1.])))['available']
    # Signed LF projection, disabling, left-insertion ties, unbounded direct tails.
    m=dict(mean=[0.,0.],weights=[[1.,0.],[0.,1.]],disabled=[])
    ph=a.physical_model(dict(directions=np.eye(2),covariance=np.eye(2)))
    o={'LF':np.array([[0.,-1.],[1.,0.],[1.,0.],[2.,1.]])}
    v=np.zeros((6,17));v[:,11:13]=[[-3,1],[-1,2],[0,-2],[1,0],[2,3],[3,-1]]
    before=a.canonical(m)
    signed,ranks,dc,pc=a.features(m,ph,o,v)
    assert np.array_equal(dc[:,0],v[:,11]) and np.array_equal(dc[:,1],-v[:,11])
    assert np.array_equal(ranks[:,0],[4,1,0,1,3,4])
    assert np.array_equal(ranks[:,1],[3,4,0,1,4,0])
    assert np.array_equal(pc,dc)
    assert np.any(dc<0),'negative corrected scores must survive'
    cuts=np.array([[3,3,3],[1,1,1],[1,1,1.]])
    br=a.branch_bits(signed,ranks,dc,pc,cuts)
    assert np.array_equal(br[:,0]>0,(ranks>3).any(1))
    assert np.array_equal(br[:,1],br[:,2])
    packed=(br.astype(np.uint16)*(1<<np.array([0,3,6]))).sum(1)+3584
    for mi in range(3):assert np.array_equal((packed>>(3*mi))&7,br[:,mi])
    for mi in range(3):assert np.all((packed>>(9+mi))&1)
    cc,tt,ss=a.calibrate(m,ph,o,v)
    v2=v.copy();v2[:,11:13]*=3
    a.calibrate(m,ph,o,v2);a.features(m,ph,o,v2)
    assert a.canonical(m)==before,'cal/eval cannot mutate representation'
    m2=copy.deepcopy(m);m2['disabled']=[0]
    _,rr,_,_=a.features(m2,ph,o,v);assert np.all(rr[:,0]==0)
    c2,_,_=a.calibrate(m2,ph,o,v)
    assert np.isinf(c2[:,:,1,:2]).all(),'disabled budget must not move'
    cases+=10
    return dict(status='PASS',deterministic_cases=cases,scientific_trials=0,tests=['integer_thresholds','ties','k_m_and_m_plus_1',
         'physical_c_nonzero_and_sign','singular_nonfinite_and_near_parallel','positive_scale_equivalence','signed_and_unclipped',
         'LF0_rank_semantics','branch_packing','cal_eval_immutability','disabled_budget_no_redistribution'])


if __name__=='__main__':
    result=test();result['test_source_sha256']=a.file_hash(Path(__file__))
    result['analysis_source_sha256']=a.file_hash(ROOT/'numerics/scripts/analyze_pr0005_three_config.py')
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
