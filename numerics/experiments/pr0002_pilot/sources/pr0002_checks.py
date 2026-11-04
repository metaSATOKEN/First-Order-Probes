# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Deterministic acceptance checks: no simulated probe records or performance fitting."""
import math
import numpy as np
import fusion_confirmation_pr0002_v1 as f


def check_trace(path):
    with np.load(path,allow_pickle=False) as z:
        oi,oj,ci,cj=[z[k] for k in ("observed_i","observed_j","control_i","control_j")]
        w,n=int(z["warm"]),int(z["n"])
        saved=z["scalars"]
    p=f.conditions()["N_M"]["params"]
    ai,aj=np.exp(-p["dt"]/np.array([p["tau_i"],p["tau_j"]]))
    lam,nu=(1+ai)*(1-aj)/(2*(1-ai*aj)),(1+aj)*(1-ai)/(2*(1-ai*aj))
    def mom(a,b):
        a,b=a[w:w+n],b[w:w+n]
        return np.array([sum(a*a)/n,sum(b*b)/n,sum(a*b)/n])
    O,N=mom(oi,oj),mom(ci,cj)
    corrected=[O,O-N*np.array([1,1,0]),O-N]
    def cr(a,b):
        u=v=0.;total=0.
        for k,(x,y) in enumerate(zip(a,b)):
            u=aj*u+(1-aj)*x;v=ai*v+(1-ai)*y
            if w<=k<w+n:total+=(u-v)**2
        return total/n
    expected=np.r_[np.concatenate(corrected),[a[2]-lam*a[0]-nu*a[1] for a in corrected],cr(oi,oj)-cr(ci,cj),0.]
    np.testing.assert_allclose(saved,expected,rtol=2e-12,atol=2e-12)
    return dict(status="PASS",max_abs_error=float(np.max(np.abs(saved-expected))),
                record="pilot/00/N_M/normal_eval/0",independent_path="explicit recurrence and scalar sums")


def deterministic_checks():
    passed=[]
    spec=f.specification("confirm")
    assert len(spec["conditions"])==16 and len(spec["tasks"])==1200
    assert sum(t["total"] for t in spec["tasks"])==10999800
    assert sum(math.ceil(t["total"]/f.CHUNK) for t in spec["tasks"])==5800
    assert len({t["id"] for t in spec["tasks"]})==1200
    assert len(f.PRIMARY)==14
    assert sum(t["total"] for t in f.tasks("pilot"))==8192
    assert [f.cutoff_order(10000,a) for a in (.01,.005)]==[9901,9951]
    passed.append("design_counts_and_exact_threshold_orders")
    labels=set()
    for mode in ("pilot","confirm"):
        for t in f.tasks(mode):
            for start in range(0,t["total"],f.MICRO):
                label=f.namespace(mode,t,start)
                assert label not in labels
                labels.add(label)
    passed.append("all_planned_rng_identifiers_disjoint")
    ref=np.array([[0,3],[1,1],[1,1],[3,0]],dtype=float)
    obs=np.array([[-1,-1],[0,0],[1,1],[2,2],[4,4]],dtype=float)
    expected=np.array([[sum(ref[:,j]<v[j]) for j in range(2)] for v in obs])
    np.testing.assert_array_equal(f.ranks(ref,obs),expected)
    ref=np.column_stack((np.arange(9999),np.arange(9999)[::-1])).astype(float)
    cal=np.column_stack((np.arange(10000),np.arange(10000)[::-1])).astype(float)
    q,_=f.fit(ref,cal); ranks=f.ranks(ref,cal)
    flags,bits=f.decide(ranks,q)
    manual=np.array([[any(row[j]>q[m,j] for j in range(2)) for m in range(4)] for row in ranks])
    np.testing.assert_array_equal(flags,manual)
    assert np.all(flags.mean(axis=0)<=.01)
    assert not f.decide(np.tile(q[3],(2,1)),q)[0][:,3].any()
    tied=np.ones((10000,2));qt,_=f.fit(tied[:9999],tied)
    assert not f.decide(f.ranks(tied[:9999],tied),qt)[0].any()
    passed.append("ranks_ties_strict_or_and_calibration_only_fit")
    allbits=np.arange(16,dtype=np.uint8)
    flags=(allbits[:,None]&(1<<np.arange(4)))!=0
    for a,b in f.PAIRS:
        d=f.decomposition(flags,a,b)
        assert (d["both"],d["gained"],d["lost"],d["neither"],d["difference"])==(4,4,4,4,0.)
    assert f.interval(np.ones(50),3.)["lower"] is None
    x=np.arange(50)/100
    ci=f.interval(x,3.)
    assert abs(ci["mcse"]-math.sqrt(sum((x-x.mean())**2)/49)/math.sqrt(50))<1e-15
    passed.append("paired_accounting_outer_intervals_zero_sd")
    p=f.conditions()["N_M"]["params"]
    ai,aj=np.exp(-p["dt"]/np.array([p["tau_i"],p["tau_j"]]))
    lam,nu=f.kernel.discrete_weights(p["tau_i"],p["tau_j"],p["dt"])
    for w in np.linspace(0,math.pi,91):
        hi=(1-ai)/(1-ai*np.exp(-1j*w));hj=(1-aj)/(1-aj*np.exp(-1j*w))
        assert abs((hi*hj.conjugate()).real-lam*abs(hi)**2-nu*abs(hj)**2)<2e-12
    A,P,_=f.kernel.digital_model(p["kappa"],p["tau_i"],p["tau_j"],p["dt"])
    phi=math.exp(-p["kappa"]*p["dt"])
    b=math.sqrt(1-phi**2)*np.array([1,1-ai,1-aj])
    np.testing.assert_allclose(P,A@P@A.T+np.outer(b,b),rtol=2e-12,atol=2e-12)
    # Explicit recurrence, independent of scipy.signal.lfilter.
    yi=np.array([[.2,-1.,.7,2.,-.4]])
    yj=np.array([[-.1,.5,1.,-.8,.2]])
    u=v=0.;rs=[]
    for i,j in zip(yi[0],yj[0]):
        u=aj*u+(1-aj)*i;v=ai*v+(1-ai)*j;rs.append((u-v)**2)
    np.testing.assert_allclose(f.kernel.cr_energy(yi,yj,p,0,5),[np.mean(rs)],rtol=2e-12,atol=2e-12)
    passed.append("independent_frequency_identity_lyapunov_cr_recurrence")
    return dict(status="PASS",checks=passed,check_count=len(passed),generated_probe_records=0,
                tolerance=f.TOLERANCE,environment=f.environment())
