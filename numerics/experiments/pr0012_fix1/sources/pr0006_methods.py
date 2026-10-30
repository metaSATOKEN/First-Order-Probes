# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Fixed LF-DC budgets. This API accepts observations, never source/fault truth."""
from fractions import Fraction
import json
import math
import numpy as np
import pr0004_methods as old
import pr0006_spec as spec


def fit(values):
    try:
        model,ordered=old.fit_reference(values,spec.nominal_nu())
        json.dumps(model,allow_nan=False)
        if not np.isfinite(ordered['LF']).all():raise ValueError('Nonfinite reference projection')
        return dict(valid=True,model=model),ordered['LF']
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
        return dict(valid=False,reason=type(exc).__name__+': '+str(exc)),np.empty((0,2),np.float64)


def signed(state,values):
    a=np.asarray(values)
    if a.ndim!=2 or a.shape[1]!=17 or a.dtype!=np.float64:raise ValueError('Expected float64 Nx17')
    finite=np.isfinite(a).all(axis=1)
    s=np.zeros((len(a),2),np.float64)
    if not state['valid']:return s,np.zeros(len(a),bool)
    model=state['model']
    with np.errstate(over='ignore',invalid='ignore'):
        s[finite]=(a[finite][:,[11,12]]-np.asarray(model['mean']))@np.asarray(model['weights']).T
    finite &= np.isfinite(s).all(axis=1)
    return s,finite


def ranks(state,ordered,s):
    features=np.column_stack((np.abs(s[:,0]),s[:,1]))
    r=np.column_stack([np.searchsorted(ordered[:,j],features[:,j],side='left') for j in range(2)])
    r[:,state['model']['disabled']]=0
    return r


def order_index(m,denominator,eighths=8):
    return math.ceil((m+1)*(1-Fraction(eighths,8*denominator)))


def calibrate(state,ordered,values):
    s,finite=signed(state,values);m=len(values)
    cut=np.full((2,6,3),np.inf);indices=np.zeros((2,6,3),np.int64)
    if not state['valid'] or not finite.all():
        return dict(valid=False,reason='unavailable reference or nonfinite calibration',m=m),cut,indices
    z=np.column_stack((s[:,0],-s[:,0],s[:,1]));disabled=state['model']['disabled']
    disabled_tails=([0,1] if 0 in disabled else [])+([2] if 1 in disabled else [])
    r=ranks(state,ordered,s);maximum=r.max(axis=1)
    ties=np.zeros((2,6,3),np.int64)
    for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
        for wi,w in enumerate(spec.BUDGETS.values()):
            for b,e in enumerate(w):
                k=order_index(m,den,e);indices[ai,wi,b]=k
                if k<=m and b not in disabled_tails:cut[ai,wi,b]=np.partition(z[:,b],k-1)[k-1]
                ties[ai,wi,b]=np.count_nonzero(z[:,b]==cut[ai,wi,b])
        k=order_index(m,den);indices[ai,5,:]=k
        cut[ai,5,:]=np.partition(maximum,k-1)[k-1] if k<=m else len(ordered)
        ties[ai,5,:]=np.count_nonzero(maximum==cut[ai,5,0])
    return dict(valid=True,m=m,disabled_tails=disabled_tails,ties=ties.tolist(),
                lf0_upper_rank=len(ordered),lf0_structurally_inactive=(cut[:,5,0]>=len(ordered)).tolist()),cut,indices


def decide(state,ordered,calibration,cut,values):
    s,valid=signed(state,values)
    if not calibration['valid']:valid[:]=False
    # Each method occupies a nibble: three tail bits, then availability.
    packed=np.zeros((len(values),2),np.uint32)
    if not valid.any():return packed
    z=np.column_stack((s[:,0],-s[:,0],s[:,1]));r=ranks(state,ordered,s)
    for ai in range(2):
        for mi in range(6):
            if mi<5:branches=z>cut[ai,mi]
            else:
                g=r[:,0]>cut[ai,mi,0];p=r[:,1]>cut[ai,mi,0]
                branches=np.column_stack((g&(s[:,0]>=0),g&(s[:,0]<0),p))
            pattern=(branches.astype(np.uint32)*np.array([1,2,4],np.uint32)).sum(axis=1,dtype=np.uint32)
            nibble=np.where(valid,pattern|8,0).astype(np.uint32)
            packed[:,ai] |= nibble << (4*mi)
    return packed


def decode(packed):
    a=np.asarray(packed)
    if a.dtype!=np.uint32 or a.ndim!=2 or a.shape[1]!=2 or np.any(a>>24):raise ValueError('Invalid packed decisions')
    n=(a[:,:,None] >> (4*np.arange(6,dtype=np.uint32)))&15
    if np.any(((n&8)==0)&((n&7)!=0)):raise ValueError('Unavailable decision has alarm bits')
    return (n&7).astype(np.uint8),(n&8)!=0


def count(packed):
    patterns,valid=decode(packed);counts=np.zeros((2,6,8),np.int64);pairs=np.zeros((2,15,4),np.int64)
    available=valid.sum(axis=0,dtype=np.int64);pair_n=np.zeros((2,15),np.int64)
    for ai in range(2):
        for mi in range(6):counts[ai,mi]=np.bincount(patterns[:,ai,mi][valid[:,ai,mi]],minlength=8)
        for pi,(left,right) in enumerate(spec.PAIRS):
            ok=valid[:,ai,left]&valid[:,ai,right];a=patterns[:,ai,left][ok]>0;b=patterns[:,ai,right][ok]>0
            pairs[ai,pi]=[np.count_nonzero(a&b),np.count_nonzero(a&~b),np.count_nonzero(~a&b),np.count_nonzero(~a&~b)]
            pair_n[ai,pi]=ok.sum()
            assert pairs[ai,pi,1]-pairs[ai,pi,2]==int(a.sum())-int(b.sum())
    return counts,available,pairs,pair_n


def interval(x,family=1):
    a=np.asarray(x,float)
    if len(a)!=spec.REPEATS or not np.isfinite(a).all():
        return dict(mean=None,lower=None,upper=None,inference='unavailable outer series')
    return old.outer_interval(a,family)


def aggregate(results):
    if sorted(x['replicate'] for x in results)!=list(range(spec.REPEATS)):raise ValueError('Require all 25 fixed replicates')
    cs=np.stack([x['counts'] for x in results]);avs=np.stack([x['available'] for x in results])
    ps=np.stack([x['pairs'] for x in results]);pn=np.stack([x['pair_n'] for x in results])
    inactive=np.stack([x['inactive'] for x in results])
    totals=np.array([spec.COUNTS['null' if spec.conditions()[c]['kind']=='normal' else spec.conditions()[c]['kind']] for c,p in spec.contexts()])
    alarm=cs[:,:,:,:,1:].sum(axis=-1)
    with np.errstate(divide='ignore',invalid='ignore'):rates=alarm/avs
    output=[];comparisons=[];fpr=[]
    for ki,(case,policy) in enumerate(spec.contexts()):
        n=int(totals[ki]);kind=spec.conditions()[case]['kind']
        for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
            for mi,method in enumerate(spec.METHODS):
                x=rates[:,ki,ai,mi];available=avs[:,ki,ai,mi];a=alarm[:,ki,ai,mi]
                row=dict(case=case,policy=policy,alpha=1/den,method=method,kind=kind,
                    rate=interval(x),miss=interval(1-x),available=int(available.sum()),planned=spec.REPEATS*n,
                    detection_lower=float(a.sum()/(spec.REPEATS*n)),
                    fpr_range=[float(a.sum()/(spec.REPEATS*n)),float((a+n-available).sum()/(spec.REPEATS*n))],
                    pattern_counts=cs[:,ki,ai,mi].sum(axis=0).tolist(),
                    zero_alarm_full_replicates=int(np.count_nonzero((a==0)&(available==n))),
                    lf0_structurally_inactive_replicates=int(np.count_nonzero(inactive[:,ki,ai]==1)) if mi==5 else None,
                    per_replicate=[dict(replicate=r,alarms=int(a[r]),available=int(available[r]),planned=n,
                        wilson=list(old.wilson(int(a[r]),int(available[r]))) if available[r]>0 else None)
                        for r in range(spec.REPEATS)],
                    branches=[interval((cs[:,ki,ai,mi]*np.array([(j>>b)&1 for j in range(8)])).sum(axis=1)/np.where(available>0,available,np.nan)) for b in range(3)])
                output.append(row)
                if kind=='normal':
                    adj=interval(x,216);fpr.append(dict(case=case,policy=policy,alpha=1/den,method=method,interval=adj,
                        excess_flag=adj['lower'] is not None and adj['lower']>1/den,unavailable=int((n-available).sum())))
            for pi,(left,right) in enumerate(spec.PAIRS):
                denom=pn[:,ki,ai,pi].astype(float);denom[denom==0]=np.nan
                columns={k:interval(ps[:,ki,ai,pi,i]/denom) for i,k in enumerate(('both','gained','lost','neither'))}
                comparisons.append(dict(case=case,policy=policy,alpha=1/den,left=spec.METHODS[left],right=spec.METHODS[right],
                    difference=interval((ps[:,ki,ai,pi,1]-ps[:,ki,ai,pi,2])/denom),
                    cells=columns,joint_available=int(pn[:,ki,ai,pi].sum()),planned=spec.REPEATS*n))
    primary=[]
    for case,pol,den,left,right in spec.primary():
        ki=spec.contexts().index((case,pol));ai=spec.ALPHA_DENOMINATORS.index(den);li=spec.METHODS.index(left);ri=spec.METHODS.index(right)
        full=bool(np.all(avs[:,ki,ai,[li,ri]]==totals[ki]))
        z=interval(rates[:,ki,ai,li]-rates[:,ki,ai,ri],25) if full else dict(mean=None,lower=None,upper=None,inference='unavailable records')
        lo,hi=z['lower'],z['upper'];label='unresolved';size='unresolved'
        if lo is not None:
            label='improved' if lo>0 else 'loss' if hi<0 else 'unresolved'
            size='clear_improvement' if lo>.01 else 'clear_loss' if hi<-.01 else 'within_1pp' if lo>=-.01 and hi<=.01 else 'unresolved'
        primary.append(dict(case=case,policy=pol,alpha=1/den,left=left,right=right,interval=z,classification=label,magnitude=size))
    # Planned descriptive signed-gain and F/C contrasts, not extra primary tests.
    contrasts=[]
    for ki,(case,pol) in enumerate(spec.contexts()):
        others=[]
        if '__GP' in case:others.append(('positive-minus-negative',(case.replace('__GP','__GM'),pol)))
        if pol=='C':others.append(('C-minus-F',(case,'F')))
        for label,other in others:
            if other not in spec.contexts():continue
            kj=spec.contexts().index(other)
            for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
                for mi,method in enumerate(spec.METHODS):contrasts.append(dict(kind=label,case=case,other=other[0],policy=pol,
                    method=method,alpha=1/den,interval=interval(rates[:,ki,ai,mi]-rates[:,kj,ai,mi])))
    assert len(primary)==25 and len(fpr)==216
    return dict(primary25=primary,fpr216=fpr,rates=output,paired=comparisons,contrasts=contrasts,
                units='fractions, not percent/pp',replicates=spec.REPEATS,
                scope='fixed-budget B25; no allocation chosen from evaluation')
