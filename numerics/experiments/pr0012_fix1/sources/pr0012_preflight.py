# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Technical CPU/GPU boundary audit. NEVER used for scientific decisions.

The fixed readouts retain strict comparisons. Approximate physical statistics
cannot be required to give identical ranks at a reference order statistic.
Only flips at a numerically unresolved effective half-plane are classified as
boundary ambiguities; raw errors and away-from-boundary flips still fail.
"""
import numpy as np
import pr0012_methods as methods
from probe_backend_check_v1 import compare,TOL


def boundaries(state,ordered,cut,method):
    out=np.full((2,3),np.inf)
    if method==0:
        out[:]=cut[:,0]
    else:
        for ai in range(2):
            k=int(cut[ai,1,0])
            if k!=cut[ai,1,0] or not 0<=k<=len(ordered):raise ValueError('Invalid rank cut')
            if k<len(ordered):out[ai]=ordered[k,[0,0,1]]
        if 0 in state['model']['disabled']:out[:,:2]=np.inf
        if 1 in state['model']['disabled']:out[:,2]=np.inf
    return out


def compare_readouts(cpu,gpu,state,ordered,meta,cut):
    numeric=compare(cpu,gpu)
    if not numeric['passed']:return dict(passed=False,status='RAW_NUMERIC_FAILURE',numeric=numeric)
    if not state['valid'] or not meta['valid']:
        return dict(passed=False,status='INVALID_TECHNICAL_MODEL',numeric=numeric)
    sc,vc=methods.signed(state,cpu);sg,vg=methods.signed(state,gpu)
    if not (vc.all() and vg.all()):return dict(passed=False,status='NONFINITE_PROJECTION',numeric=numeric)
    w=np.asarray(state['model']['weights']);mu=np.asarray(state['model']['mean'])
    # Original raw comparison tolerance propagated through the SAME fixed map.
    dq=TOL['atol']+TOL['rtol']*abs(cpu[:,[11,12]])
    rounding=16*np.finfo(np.float64).eps*((abs(cpu[:,[11,12]])+abs(mu))@abs(w).T+abs(sc))
    radius=dq@abs(w).T+rounding
    if not np.isfinite(radius).all() or np.any(abs(sc-sg)>radius):
        return dict(passed=False,status='PROJECTION_NUMERIC_FAILURE',numeric=numeric)
    zc=np.column_stack((sc[:,0],-sc[:,0],sc[:,1]));zg=np.column_stack((sg[:,0],-sg[:,0],sg[:,1]))
    rad=radius[:,[0,0,1]];rows=[];bad=0;ambiguities=0
    for mi,name in enumerate(('W0','LF0')):
        th=boundaries(state,ordered,cut,mi)
        a=methods.decide(state,ordered,meta,cut,cpu,mi)
        b=methods.decide(state,ordered,meta,cut,gpu,mi)
        finite=np.isfinite(th)
        dc=abs(zc[:,None,:]-th);dg=abs(zg[:,None,:]-th)
        near=finite & (dc<=rad[:,None,:]) & (dg<=rad[:,None,:])
        flips=((a^b)[:,:,None]&np.array([1,2,4],np.uint8))!=0
        # Separate implementation check: no epsilon/tie modification here.
        halfplane_errors=0
        for z,bits in ((zc,a),(zg,b)):
            expected=((z[:,None,:]>th)*np.array([1,2,4],np.uint8)).sum(-1).astype(np.uint8)|8
            halfplane_errors+=int(np.count_nonzero(expected!=bits))
        stable=(~near).sum(0);coverage=bool(np.all(stable[finite]>0))
        unexplained=int(np.count_nonzero(flips&~near))
        availability=int(np.count_nonzero((a&8)!=(b&8)))
        bad+=unexplained+availability+halfplane_errors+int(not coverage)
        n=int(np.count_nonzero(flips&near));ambiguities+=n
        records=[]
        for i,ai,j in np.argwhere(flips):
            records.append(dict(row=int(i),alpha_index=int(ai),branch=int(j),cpu_bit=int(a[i,ai]),gpu_bit=int(b[i,ai]),
                cpu_distance=float(dc[i,ai,j]),gpu_distance=float(dg[i,ai,j]),radius=float(rad[i,j]),
                boundary=float(th[ai,j]),classified_boundary=bool(near[i,ai,j])))
        rows.append(dict(method=name,decision_pairs=int(a.size),exact_bit_mismatches=int(np.count_nonzero(a!=b)),
            boundary_branch_flips=n,unexplained_branch_flips=unexplained,halfplane_mismatches=halfplane_errors,
            availability_mismatches=availability,stable_branch_counts=stable.tolist(),
            near_boundary_counts=near.sum(0).tolist(),stable_coverage=coverage,differences=records))
    return dict(passed=bad==0,status='FAIL' if bad else ('PASS_WITH_BOUNDARY_AMBIGUITY' if ambiguities else 'PASS_EXACT'),
        numeric=numeric,projection_max_abs=float(abs(sc-sg).max()),projection_radius_max=float(radius.max()),
        original_raw_tolerance=dict(TOL),boundary_branch_flips=ambiguities,unexplained_failures=bad,methods=rows,
        scientific_thresholds_changed=False)


def audit(cpu,gpu):
    numeric=compare(cpu,gpu)
    if not numeric['passed']:return dict(passed=False,status='RAW_NUMERIC_FAILURE',numeric=numeric)
    state,ordered=methods.fit(cpu)
    if not state['valid']:return dict(passed=False,status='INVALID_TECHNICAL_MODEL',numeric=numeric)
    cal=np.tile(cpu,(16,1))[:15999].copy();cal[:,11:13]+=[.000173,.000317]
    meta,cut,_=methods.calibrate(state,ordered,cal)
    return compare_readouts(cpu,gpu,state,ordered,meta,cut)
