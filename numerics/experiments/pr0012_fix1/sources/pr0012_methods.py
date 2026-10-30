# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Frozen readouts and evidence decisions; no labels/conditions enter detector API."""
import math
import numpy as np
from scipy.stats import beta,t
from pr0006_methods import fit,signed,ranks,order_index
import pr0012_spec as spec

def calibrate(state,ordered,values):
    s,valid=signed(state,values);m=len(values)
    cut=np.full((2,2,3),np.inf);indices=np.zeros((2,2,3),np.int64)
    if not state['valid'] or not valid.all():return dict(valid=False,m=m,reason='invalid reference/calibration'),cut,indices
    z=np.column_stack((s[:,0],-s[:,0],s[:,1]));dis=state['model']['disabled']
    tails=([0,1] if 0 in dis else [])+([2] if 1 in dis else [])
    maximum=ranks(state,ordered,s).max(1)
    ties=np.zeros_like(indices)
    for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
        for b,e in enumerate((2,2,4)):
            k=order_index(m,den,e);indices[ai,0,b]=k
            if k<=m and b not in tails:cut[ai,0,b]=np.partition(z[:,b],k-1)[k-1]
            ties[ai,0,b]=np.count_nonzero(z[:,b]==cut[ai,0,b])
        k=order_index(m,den);indices[ai,1,:]=k
        cut[ai,1,:]=np.partition(maximum,k-1)[k-1] if k<=m else len(ordered)
        ties[ai,1,:]=np.count_nonzero(maximum==cut[ai,1,0])
    return dict(valid=True,m=m,ties=ties.tolist(),disabled_tails=tails,
      lf0_structurally_inactive=(cut[:,1,0]>=len(ordered)).tolist()),cut,indices

def decide(state,ordered,cal,cut,values,method=0):
    if method not in (0,1):raise ValueError('Unknown readout')
    s,valid=signed(state,values);valid &= cal['valid']
    out=np.zeros((len(values),2),np.uint8)
    z=np.column_stack((s[:,0],-s[:,0],s[:,1]));r=ranks(state,ordered,s) if state['valid'] else np.zeros((len(values),2))
    for ai in range(2):
        if method==0:b=z>cut[ai,0]
        else:
            g=r[:,0]>cut[ai,1,0];p=r[:,1]>cut[ai,1,0]
            b=np.column_stack((g&(s[:,0]>=0),g&(s[:,0]<0),p))
        code=(b*np.array([1,2,4],np.uint8)).sum(1).astype(np.uint8)
        out[:,ai]=np.where(valid,code|8,0)
    return out

def decode(bits):
    a=np.asarray(bits)
    if a.dtype!=np.uint8 or np.any(a>15) or np.any(((a&8)==0)&((a&7)!=0)):raise ValueError('Invalid nibble')
    return a&7,(a&8)!=0

def histogram(bits):
    p,v=decode(bits)
    return np.stack([((p==i)&v).sum(0,dtype=np.int64) for i in range(8)]+[(~v).sum(0,dtype=np.int64)],axis=-1)

def gates(values,window):
    a=np.asarray(values);q=a[:,[11,12]];mu=q.mean(0);sd=q.std(0,ddof=1)
    ts=np.full(2,np.nan);ok=np.isfinite(a).all() & np.isfinite(sd) & (sd>0)
    ts[ok]=np.sqrt(len(a))*mu[ok]/sd[ok]
    scores=np.array([abs(ts[0]),max(abs(ts[0]),ts[1]) if np.isfinite(ts).all() else np.nan])
    states=np.where(np.isfinite(scores),scores>spec.GATE_CUTS[window],2).astype(np.int64)
    return states,dict(mean=mu.tolist(),sd=sd.tolist(),signed_T=ts.tolist(),scores=scores.tolist())

def cp(k,n,eta):
    if not 0<=k<=n or n<=0:raise ValueError('Binomial domain')
    return (0. if k==0 else float(beta.ppf(eta,k,n-k+1)),1. if k==n else float(beta.ppf(1-eta,k+1,n-k)))

def evidence(k,n,planned,alpha):
    if n!=planned:return dict(label='UNAVAILABLE',alarms=int(k),available=int(n),planned=int(planned))
    lo,hi=cp(k,n,spec.CP_ETA);cap=1.25*alpha
    return dict(label='SUPPORTED' if hi<=cap else 'EXCEEDED' if lo>cap else 'UNRESOLVED',
      lower=lo,upper=hi,alarms=int(k),available=int(n),planned=int(planned),cap=cap)

def select(gate,candidate,f_label,c_label):
    # Output context indexes; -1 DEFER, -2 INVALID. No E or truth arguments.
    if gate==2:return np.full(3,-2,np.int64)
    if gate==0:return np.full(3,candidate+1,np.int64)
    if gate!=1:raise ValueError('Gate code')
    if f_label=='SUPPORTED':return np.zeros(3,np.int64)
    return np.array([0,-1,7 if c_label=='SUPPORTED' else -1],np.int64)

def shapes():
    return dict(gates=(4,6,2),selection=(4,6,2,3),v_counts=(4,2,2,2),
      wcounts=(4,4,8,2,9),lcounts=(4,4,3,2,9),policy=(4,6,4,2,3,4),
      pairs=(4,6,4,2,3,4,4),lf_pairs=(4,4,3,2,5),inactive=(4,3,2))

def empty():
    row={k:np.zeros(s,np.int64) for k,s in shapes().items()}
    row['gates'][:]=2;row['selection'][:]=-2;row['inactive'][:]=-1
    return row

def policy_states(bits,selection):
    p,v=decode(bits);out=np.empty((len(bits),2,3),np.uint8)
    for ai in range(2):
        for pi in range(3):
            c=int(selection[ai,pi])
            if c<0:out[:,ai,pi]=2 if c==-1 else 3
            else:out[:,ai,pi]=np.where(v[:,c,ai],np.where(p[:,c,ai]>0,0,1),3)
    return out

def accumulate(row,g,e,w,l):
    if w.shape[1:]!=(8,2) or l.shape!=(len(w),3,2):raise ValueError('Decision shapes')
    row['wcounts'][g,e]+=histogram(w);row['lcounts'][g,e]+=histogram(l)
    wp,wv=decode(w);lp,lv=decode(l)
    for j,c in enumerate((0,1,7)):
        ok=wv[:,c]&lv[:,j];a=wp[:,c]>0;b=lp[:,j]>0
        row['lf_pairs'][g,e,j]+=np.stack([(ok&x).sum(0) for x in (a&b,a&~b,~a&b,~a&~b)]+[(~ok).sum(0)],axis=-1)
    for ci in range(6):
        states=policy_states(w,row['selection'][g,ci])
        row['policy'][g,ci,e]+=np.stack([(states==s).sum(0) for s in range(4)],axis=-1)
        for k,(left,right) in enumerate(((1,0),(2,0),(2,1))):
            for ai in range(2):
                row['pairs'][g,ci,e,ai,k]+=np.bincount(4*states[:,ai,left]+states[:,ai,right],minlength=16).reshape(4,4)

def validate(row,complete=True):
    if set(row)!=set(shapes()):raise ValueError('Carry fields')
    for k,shape in shapes().items():
        if row[k].shape!=shape or row[k].dtype!=np.int64:raise ValueError('Carry shape/type: '+k)
    if np.any((row['gates']<0)|(row['gates']>2)):raise ValueError('Gate states')
    if np.any((row['selection']<-2)|(row['selection']>7)):raise ValueError('Selection states')
    for k in ('wcounts','lcounts','policy','pairs','v_counts','lf_pairs'):
        if np.any(row[k]<0):raise ValueError('Negative count')
    for e,n in enumerate((20000,5000,5000,5000)):
        for k in ('wcounts','lcounts','policy','lf_pairs'):
            counts=row[k][:,:,e] if k=='policy' else row[k][:,e]
            totals=counts.sum(-1)
            if np.any(totals>n) or complete and np.any(totals!=n):raise ValueError('Denominator: '+k)
        for j,(left,right) in enumerate(((1,0),(2,0),(2,1))):
            a=row['pairs'][:,:,e,:,j]
            if not np.array_equal(a.sum(-1),row['policy'][:,:,e,:,left]):raise ValueError('Left margin')
            if not np.array_equal(a.sum(-2),row['policy'][:,:,e,:,right]):raise ValueError('Right margin')
    if np.any(row['v_counts'][...,0]>row['v_counts'][...,1]) or np.any(row['v_counts'][...,1]>20000):raise ValueError('V count')

def rate_interval(x,family=13):
    a=np.asarray(x,float)
    if len(a)!=50 or not np.isfinite(a).all():return dict(mean=None,lower=None,upper=None,status='TECHNICAL_UNESTABLISHED')
    mean=float(a.mean());sd=float(a.std(ddof=1))
    h=float(t.ppf(1-.05/(2*family),49))*sd/np.sqrt(50) if sd>0 else np.sqrt(2*np.log(2*family/.05)/50)
    return dict(mean=mean,sd=sd,lower=max(-1.,mean-h),upper=min(1.,mean+h),
      status='evaluated',inference='paired outer t approximation' if sd>0 else 'bounded Hoeffding fallback')

def binary_interval(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)!=50 or len(b)!=50 or np.any((a<0)|(a>1)|(b<0)|(b>1)):
        return dict(mean=None,lower=None,upper=None,status='TECHNICAL_UNESTABLISHED')
    n10=int(((a==1)&(b==0)).sum());n01=int(((a==0)&(b==1)).sum())
    l1,u1=cp(n10,50,.05/52);l2,u2=cp(n01,50,.05/52)
    return dict(mean=(n10-n01)/50,lower=l1-u2,upper=u1-l2,n10=n10,n01=n01,status='evaluated',inference='conservative paired CP')

def aggregate(rows):
    if len(rows)!=50:raise ValueError('Require all 50 repeats')
    for r in rows:validate(r)
    a={k:np.stack([r[k] for r in rows]) for k in shapes()}
    z=a['gates'];primary=[]
    contrasts=[binary_interval(z[:,0,3,0],z[:,0,1,0]),binary_interval(z[:,0,3,0],z[:,0,2,0]),
       binary_interval(z[:,0,3,1],z[:,0,3,0]),binary_interval(z[:,2,5,1],z[:,2,4,1])]
    for e,n in enumerate((20000,5000,5000,5000)):
        h=a['wcounts'][:,2,e];d=(h[:,6,:,1:8].sum(-1)-h[:,5,:,1:8].sum(-1))[:,0]/n
        d[(h[:,6,0,8]+h[:,5,0,8])>0]=np.nan;contrasts.append(rate_interval(d))
    s=a['selection'][:,1,5,0,2];k=int((s==7).sum())
    if np.any(s==-2):r=dict(mean=None,lower=None,upper=None,status='TECHNICAL_UNESTABLISHED')
    else:
        lo,hi=cp(k,50,.05/26);r=dict(mean=k/50,lower=lo,upper=hi,k=k,n=50,status='evaluated',inference='CP recovery probability')
    contrasts.append(r)
    for e,n in enumerate((20000,5000,5000,5000)):
        c=a['policy'][:,1,5,e,0];d=(c[:,2,0]-c[:,0,0]).astype(float)/n
        d[(c[:,2,3]+c[:,0,3])>0]=np.nan;contrasts.append(rate_interval(d))
    for p,result in zip(spec.primary(),contrasts):
        lo,hi=result['lower'],result['upper'];label='TECHNICAL_UNESTABLISHED'
        if lo is not None:
            if p['id']=='R01':label='SUPPORTED' if lo>.5 else 'OPPOSITE' if hi<.5 else 'UNRESOLVED'
            elif p['expected']>0:label='SUPPORTED' if lo>0 else 'OPPOSITE' if hi<0 else 'UNRESOLVED'
            else:label='SUPPORTED' if hi<0 else 'OPPOSITE' if lo>0 else 'UNRESOLVED'
        primary.append(dict(**p,interval=result,classification=label))
    policy_results=[];calibration_results=[];gate_results=[];v_results=[]
    for g,group in enumerate(spec.GROUPS):
        for ci,candidate in enumerate(spec.CANDIDATES):
            gate_results.append(dict(group=group,candidate=candidate,states=a['gates'][:,g,ci].tolist(),
                holds=(a['gates'][:,g,ci]==1).sum(0).tolist(),unknown=(a['gates'][:,g,ci]==2).sum(0).tolist()))
            for e,name in enumerate(('normal',)+spec.FAULTS):
                n=20000 if e==0 else 5000
                for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
                    for pi in range(3):
                        c=a['policy'][:,g,ci,e,ai,pi];total=c.sum(0);provided=total[0]+total[1]
                        policy_results.append(dict(group=group,candidate=candidate,evaluation=name,alpha=1/den,policy=f'P{pi}',
                            per_repeat=c.tolist(),planned_per_repeat=n,full_alarm=total[0]/(50*n),coverage=provided/(50*n),
                            provided_alarm=total[0]/provided if provided else None,provided_miss=total[1]/(50*n),
                            deferred=total[2]/(50*n),invalid=total[3]/(50*n)))
        for e,name in enumerate(('normal',)+spec.FAULTS):
            n=20000 if e==0 else 5000
            for key,method,contexts in (('wcounts','W0',spec.CONTEXTS),('lcounts','LF0',('F','clean','C'))):
                for c,context in enumerate(contexts):
                    for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
                        h=a[key][:,g,e,c,ai];calibration_results.append(dict(group=group,evaluation=name,method=method,
                          context=context,alpha=1/den,planned_per_repeat=n,per_repeat_histogram=h.tolist(),
                          full_alarm=h[:,1:8].sum()/(50*n),invalid=h[:,8].sum()/(50*n)))
        for rep in range(50):
            for c,context in enumerate(('F','C')):
                for ai,den in enumerate(spec.ALPHA_DENOMINATORS):v_results.append(dict(replicate=rep,group=group,context=context,
                  alpha=1/den,**evidence(*a['v_counts'][rep,g,c,ai],20000,1/den)))
    return dict(primary13=primary,outer_arrays={k:v.tolist() for k,v in a.items()},
       policy_results=policy_results,calibration_results=calibration_results,gate_results=gate_results,V_evidence=v_results,
       axes=dict(outer_replicate=list(range(50)),groups=spec.GROUPS,candidates=spec.CANDIDATES,
         evaluations=('normal',)+spec.FAULTS,W0_contexts=spec.CONTEXTS,LF0_contexts=('F','clean','C'),
         alpha_denominators=spec.ALPHA_DENOMINATORS,policy_states=('alarm','provided_quiet','defer','invalid'),
         histogram_bins=('000','001','010','011','100','101','110','111','invalid'),
         policy_pairs=('P1-P0','P2-P0','P2-P1'),LF_pair_bins=('both','W0_only','LF0_only','neither','invalid'),
         gates=('G-M','G-MC'),array_shapes=shapes(),units='fractions; outer arrays include 50 as first axis'),
       scope=dict(replicates=50,new_records=23999200,no_E_selection=True,independent_C=True,
                  fixed_known_conditions=True,all_supported_not_required=True))
