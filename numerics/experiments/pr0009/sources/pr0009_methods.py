# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Two-axis bookkeeping; detector interfaces never accept scientific truth labels."""
import numpy as np
from scipy.stats import beta,t
from pr0006_methods import fit,calibrate,decide,decode
from pr0008_shadow_methods import batch_scores,paired
import pr0009_spec as spec

def gate(scores,window):
    a=np.asarray(scores,float);cut=spec.GATE_CUTS[window]
    return np.where(np.isfinite(a)&np.isfinite(cut),a>cut,2).astype(np.uint8)

def cp(k,n,eta=spec.CP_ETA):
    if not 0<=k<=n or n<=0:raise ValueError('Invalid binomial counts')
    return (0. if k==0 else float(beta.ppf(eta,k,n-k+1)),
            1. if k==n else float(beta.ppf(1-eta,k+1,n-k)))

def evidence(k,available,planned,alpha,provenance=True):
    if not provenance or available!=planned or planned<=0:
        return dict(label='UNAVAILABLE',lower=None,upper=None,alarms=int(k),available=int(available),planned=planned)
    lo,hi=cp(int(k),planned);cap=spec.CAP_MULTIPLIER*alpha
    label='FPR-SUPPORTED' if hi<=cap else 'FPR-EXCEEDED' if lo>cap else 'FPR-UNRESOLVED'
    return dict(label=label,lower=lo,upper=hi,cap=cap,alarms=int(k),available=int(available),planned=planned)

def simple_count(packed):
    p,v=decode(packed)
    return np.stack([((p==b)&v).sum(0,dtype=np.int64) for b in range(8)],axis=-1),v.sum(0,dtype=np.int64)

def shapes():
    return dict(counts=(6,4,8,2,6,8),available=(6,4,8,2,6),
        paired=(6,7,4,4,3,2,6,4),
        gate_states=(6,7,3),same_states=(6,7),v_counts=(6,5,2,6,2),inactive=(6,8,2))

def empty():
    out={k:np.zeros(s,np.int64) for k,s in shapes().items()}
    out['gate_states'][:]=2;out['same_states'][:]=2;out['inactive'][:]=-1
    return out

def selections(candidate,states,same):
    # Indices into the fixed F + seven U base bit arrays, shared by all methods/alphas.
    return [0,candidate+1]+[0 if x>0 else candidate+1 for x in states]+[0 if same>0 else candidate+1]

def accumulate(row,g,e,bits):
    if bits.dtype!=np.uint32 or bits.ndim!=3 or bits.shape[1:]!=(8,2):raise ValueError('Expected N x8x2 base bits')
    decoded=[];cache={}
    for j in range(8):
        p,v=decode(bits[:,j]);decoded.append((p>0,v))
        row['counts'][g,e,j]+=np.stack([((p==b)&v).sum(0,dtype=np.int64) for b in range(8)],axis=-1)
        row['available'][g,e,j]+=v.sum(0,dtype=np.int64)
    def pair(left,right):
        key=(min(left,right),max(left,right))
        if key not in cache:
            a,av=decoded[key[0]];b,bv=decoded[key[1]];ok=av&bv
            cache[key]=np.stack([(ok&x).sum(0,dtype=np.int64) for x in (a&b,a&~b,~a&b,~a&~b)],axis=-1)
        return cache[key] if left<=right else cache[key][...,[0,2,1,3]]
    for ci in range(7):
        sel=selections(ci,row['gate_states'][g,ci],row['same_states'][g,ci])
        for gi,idx in enumerate(sel[2:]):
            for ri,other in enumerate([ci+1,0,1]):
                row['paired'][g,ci,e,gi,ri]+=pair(idx,other)

def validate(row):
    if set(row)!=set(shapes()):raise ValueError('Carry keys differ')
    for k,shape in shapes().items():
        a=row[k]
        if a.shape!=shape or a.dtype!=np.int64 or np.any(a<(-1 if k=='inactive' else 0)):
            raise ValueError('Carry shape/type/range: '+k)
    if not np.array_equal(row['counts'].sum(-1),row['available']):raise ValueError('Count conservation')
    if np.any(row['gate_states']>2) or np.any(row['same_states']>2) or np.any(row['inactive']>1):raise ValueError('State code')
    if np.any(row['v_counts'][...,0]>row['v_counts'][...,1]) or np.any(row['v_counts'][...,1]>20000):raise ValueError('Validation counts')
    for g in range(6):
        for e in range(4):
            n=spec.COUNTS['null_eval' if e==0 else 'fault_eval']
            if np.any(row['available'][g,e]>n):raise ValueError('Evaluation denominator')
            for ci in range(7):
                sel=selections(ci,row['gate_states'][g,ci],row['same_states'][g,ci])
                for gi,idx in enumerate(sel[2:]):
                    for ri,other in enumerate([ci+1,0,1]):
                        pair=row['paired'][g,ci,e,gi,ri]
                        if np.any(pair.sum(-1)>n):raise ValueError('Pair denominator')
                        full=(row['available'][g,e,idx]==n)&(row['available'][g,e,other]==n)
                        delta=row['counts'][g,e,idx,...,1:].sum(-1)-row['counts'][g,e,other,...,1:].sum(-1)
                        if np.any((pair[...,1]-pair[...,2])[full]!=delta[full]):raise ValueError('Paired margins')

def interval(x,family=1):
    a=np.asarray(x,float);ok=np.isfinite(a);v=a[ok];n=len(v)
    out=dict(values=[float(x) if np.isfinite(x) else None for x in a],n=n,mean=float(v.mean()) if n else None,
        sd=float(v.std(ddof=1)) if n>1 else None,median=float(np.median(v)) if n else None,
        lower=None,upper=None,family=family,inference='unavailable or incomplete')
    if len(a)==50 and ok.all() and out['sd']>0:
        h=float(t.ppf(1-.05/(2*family),49))*out['sd']/np.sqrt(50)
        out.update(lower=out['mean']-h,upper=out['mean']+h,inference='approximate paired outer t')
    elif len(a)==50 and ok.all():out['inference']='zero SD; no automatic significance'
    return out

def aggregate(rows):
    if len(rows)!=50:raise ValueError('Require 50 fixed outer replicates')
    for r in rows:validate(r)
    arr={k:np.stack([r[k] for r in rows]) for k in shapes()}
    primary=[]
    for p in spec.primary():
        g=spec.groups().index(p['group']);e=('normal',)+spec.FAULTS;e=e.index(p['evaluation'])
        cells=arr['paired'][:,g,p['candidate'],e,2,0,0,0];n=spec.COUNTS['null_eval' if e==0 else 'fault_eval']
        full=cells.sum(-1)==n;delta=np.where(full,(cells[:,1]-cells[:,2])/n,np.nan)
        result=interval(delta,18)
        primary.append(dict(**p,interval=result,paired_cells=cells.tolist(),planned_each=n))
    rates=[];gate_table=[];validation=[];pairs=[]
    for g,group in enumerate(spec.groups()):
        for ci,(fault,mix) in enumerate(spec.CANDIDATES):
            gate_table.append(dict(group=group,candidate=ci,mix_fault=fault,mix_count=mix,
                states=arr['gate_states'][:,g,ci].tolist(),same_U_states=arr['same_states'][:,g,ci].tolist()))
            for e,outcome in enumerate(('normal',)+spec.FAULTS):
                n=spec.COUNTS['null_eval' if e==0 else 'fault_eval']
                for pi,policy in enumerate(spec.POLICIES):
                    idx=np.array([selections(ci,r['gate_states'][g,ci],r['same_states'][g,ci])[pi] for r in rows])
                    counts=arr['counts'][np.arange(50),g,e,idx];av=arr['available'][np.arange(50),g,e,idx]
                    alarms=counts[...,1:].sum(-1)
                    rates.append(dict(group=group,candidate=ci,evaluation=outcome,policy=policy,planned_each=n,
                        pattern_counts=counts.tolist(),available=av.tolist(),
                        rate=np.divide(alarms,av,out=np.full(av.shape,np.nan),where=av>0).tolist(),
                        inactive=arr['inactive'][np.arange(50),g,idx].tolist()))
                pairs.append(dict(group=group,candidate=ci,evaluation=outcome,
                    gate_versions=list(spec.GATES)+['G-MC-same-U'],references=['U','F','C-clean'],
                    cells=arr['paired'][:,g,ci,e].tolist(),planned_each=n))
        for r in range(50):
            for v,name in enumerate(('clean','stale','mix_GM05','mix_GP05','mix_P02')):
                for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
                    k,n=arr['v_counts'][r,g,v,ai,0];ek,en=arr['counts'][r,g,0,0,ai,0,1:].sum(),arr['available'][r,g,0,0,ai,0]
                    validation.append(dict(replicate=r,group=group,variant=name,alpha=1/den,
                        result=evidence(k,n,20000,1/den,provenance=(v!=1 or not group.startswith('B_'))),
                        independent_E=evidence(ek,en,spec.COUNTS['null_eval'],1/den),
                        caveat='MIX is a deliberately mislabelled fixed-count batch, not exact normal-binomial evidence' if v>=2 else ''))
    clean_gate=[]
    for g,group in enumerate(spec.groups()):
        for j,gate_name in enumerate(spec.GATES):
            states=arr['gate_states'][:,g,0,j];k=int((states==1).sum());unknown=int((states==2).sum())
            eta=.05/6 if gate_name=='G-MC' else .05
            clean_gate.append(dict(group=group,gate=gate_name,n=50,held_inconsistent=k,held_unknown=unknown,
                not_vetoed=int((states==0).sum()),all_policy_holds=k+unknown,
                upper=None if unknown else cp(k,50,eta)[1],
                upper_family='six groups, one-sided 95%' if gate_name=='G-MC' else 'pointwise one-sided 95% auxiliary',
                upper_unavailable_if_unknown=bool(unknown)))
    return dict(primary18=primary,policy_results=rates,paired_tradeoffs=pairs,gate_results=gate_table,clean_gate_summary=clean_gate,
        validation_evidence=validation,scope=dict(repeats=50,methods=spec.METHODS,alpha_denominators=spec.ALPHA_DENOMINATORS,
        no_policy_intervention_by_V=True,calibration_truth_not_used_for_selection=True,
        historical_development_not_pooled=True,science_is_simulation=True))
