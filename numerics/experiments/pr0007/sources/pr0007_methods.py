# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Inherited fixed LF/LF-DC rules plus PR-0007 calibration and paired summaries.

No method accepts fault labels. Labels only select planned input records outside
this module and label audit tables after decisions have been frozen.
"""
import numpy as np
import pr0004_methods as old
import pr0007_spec as spec
from pr0006_methods import fit, signed, ranks, order_index, calibrate, decide, decode, count


def paired(left,right):
    a,av=decode(left);b,bv=decode(right);ok=av&bv;a=a>0;b=b>0
    cells=np.stack(((a&b&ok).sum(0),(a&~b&ok).sum(0),(~a&b&ok).sum(0),(~a&~b&ok).sum(0)),axis=-1)
    return cells.astype(np.int64),ok.sum(0,dtype=np.int64)


def shapes():
    n,k=len(spec.contexts()),len(spec.contrasts())
    return dict(counts=(n,2,6,8),available=(n,2,6),pairs=(n,2,15,4),pair_n=(n,2,15),
                inactive=(n,2),contrasts=(k,2,6,4),contrast_n=(k,2,6))


def empty():
    return {k:np.full(v,-1 if k=='inactive' else 0,np.int64) for k,v in shapes().items()}


def summary(x,family=1,complete=True):
    a=np.asarray(x,dtype=float);finite=np.isfinite(a);n=int(finite.sum())
    stats=dict(values=[float(v) if np.isfinite(v) else None for v in a],n=n,
        mean=float(a[finite].mean()) if n else None,sd=float(a[finite].std(ddof=1)) if n>1 else None,
        mcse=float(a[finite].std(ddof=1)/np.sqrt(n)) if n>1 else None,
        q10_50_90=np.quantile(a[finite],[.1,.5,.9],method='linear').tolist() if n else None,
        lower=None,upper=None,family=family,inference='unavailable records or outer series')
    if complete and len(a)==spec.REPEATS and finite.all():
        stats.update(old.outer_interval(a,family))
    return stats


def ratio(a,b):
    return np.divide(a,b,out=np.full(np.shape(a),np.nan,dtype=float),where=np.asarray(b)>0)


def aggregate(results):
    if [r['replicate'] for r in results]!=list(range(spec.REPEATS)):raise ValueError('Require ordered complete outer series')
    arrays={k:np.stack([r[k] for r in results]) for k in shapes()}
    cs,av,ps,pn=(arrays[k] for k in ('counts','available','pairs','pair_n'))
    ctx=spec.contexts();contr=spec.contrasts();cases=spec.conditions()
    totals=np.array([spec.COUNTS['null_eval' if cases[c['case']]['kind']=='normal' else 'fault_eval'] for c in ctx])
    alarms=cs[...,1:].sum(-1);rates=ratio(alarms,av)
    rates_out=[];fpr=[];within=[];policy=[];cross=[]
    for ci,c in enumerate(ctx):
        n=int(totals[ci]);normal=cases[c['case']]['kind']=='normal'
        for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
            for mi,name in enumerate(spec.METHODS):
                a=alarms[:,ci,ai,mi];v=av[:,ci,ai,mi];full=bool(np.all(v==n));x=rates[:,ci,ai,mi]
                row=dict(context=c,method=name,alpha=1/den,rate=summary(x,complete=full),
                    miss=summary(1-x,complete=full),available=int(v.sum()),planned=spec.REPEATS*n,
                    planned_rate_range=[float(a.sum()/(spec.REPEATS*n)),float((a+n-v).sum()/(spec.REPEATS*n))],
                    planned_miss_range=[float((v-a).sum()/(spec.REPEATS*n)),float((n-a).sum()/(spec.REPEATS*n))],
                    pattern_counts=cs[:,ci,ai,mi].tolist(),zero_alarm_full_replicates=int(((a==0)&(v==n)).sum()),
                    lf0_structurally_inactive=arrays['inactive'][:,ci,ai].tolist() if mi==5 else None,
                    per_replicate=[dict(replicate=r,alarms=int(a[r]),available=int(v[r]),planned=n,
                        wilson=list(old.wilson(int(a[r]),int(v[r]))) if v[r]>0 else None) for r in range(spec.REPEATS)],
                    branches=[summary(ratio((cs[:,ci,ai,mi]*np.array([(j>>b)&1 for j in range(8)])).sum(-1),v),
                                      complete=full) for b in range(3)])
                rates_out.append(row)
                if normal:
                    adjusted=summary(x,792,full)
                    fpr.append(dict(context=c,method=name,alpha=1/den,interval=adjusted,
                        excess_flag=adjusted['lower'] is not None and adjusted['lower']>1/den,
                        available=int(v.sum()),planned=spec.REPEATS*n))
            for pi,(li,ri) in enumerate(spec.PAIRS):
                cells=ps[:,ci,ai,pi];v=pn[:,ci,ai,pi];full=bool(np.all(v==n))
                within.append(dict(context=c,alpha=1/den,left=spec.METHODS[li],right=spec.METHODS[ri],
                    difference=summary(ratio(cells[:,1]-cells[:,2],v),complete=full),
                    cells={key:summary(ratio(cells[:,j],v),complete=full) for j,key in enumerate(('both','gained','lost','neither'))},
                    counts=cells.tolist(),joint_available=v.tolist(),planned_each=n))
    for k,c in enumerate(contr):
        left,right=ctx[c['left']],ctx[c['right']];n=int(totals[c['left']])
        for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
            for mi,name in enumerate(spec.METHODS):
                cells=arrays['contrasts'][:,k,ai,mi];v=arrays['contrast_n'][:,k,ai,mi];full=bool(np.all(v==n))
                delta=ratio(cells[:,1]-cells[:,2],v)
                row=dict(contrast=c,left=left,right=right,alpha=1/den,method=name,
                    difference=summary(delta,complete=full),
                    cells={key:summary(ratio(cells[:,j],v),complete=full) for j,key in enumerate(('both','gained','lost','neither'))},
                    counts=cells.tolist(),joint_available=v.tolist(),planned_each=n)
                policy.append(row)
                if c['kind']=='C-mix-minus-clean' and '__' in c['case']:
                    normal_case=left['group'];normal_index=next(q['index'] for q in ctx if q['id']==left['id'] and q['case']==normal_case)
                    clean_index=next(q['index'] for q in ctx if q['id']==right['id'] and q['case']==normal_case)
                    cross.append(dict(group=left['group'],mix_fault=left['mix_fault'],mix_count=left['mix_count'],
                        evaluation_fault=c['case'].split('__')[1],method=name,alpha=1/den,
                        difference=row['difference'],paired_counts=cells.tolist(),joint_available=v.tolist(),
                        normal_fpr_contexts=[normal_index,clean_index],
                        normal_fpr_shared_across_three_fault_columns=True,simultaneous_guarantee=False))
    primary=[]
    for p in spec.primary():
        k=next(i for i,c in enumerate(contr) if c['kind']=='C-mix-minus-clean' and c['case']==p['case'] and ctx[c['left']]['id']==p['cal_id'])
        cells=arrays['contrasts'][:,k,0,0];v=arrays['contrast_n'][:,k,0,0];n=int(totals[contr[k]['left']])
        delta=ratio(cells[:,1]-cells[:,2],v);full=bool(np.all(v==n));adj=summary(delta,8,full)
        label='unresolved'
        if adj['upper'] is not None and adj['upper']<0:label='supports_detection_loss'
        elif adj['lower'] is not None and adj['lower']>0:label='reverse_direction'
        primary.append(dict(**p,interval=adj,point95=summary(delta,complete=full),classification=label,
                            paired_counts=cells.tolist(),available=v.tolist(),planned_each=n))
    if (len(primary),len(fpr),len(cross))!=(8,792,1296):raise ValueError('Frozen analysis family size differs')
    return dict(primary8=primary,fpr792=fpr,cross1296=cross,rates=rates_out,within_methods=within,
        calibration_contrasts=policy,scope=dict(replicates=spec.REPEATS,units='fractions',
        inference='Separate families 8 and 792; cross table is exploratory; nominal not measured-equal FPR.',
        allocation_optimized_on_evaluation=False,pr0006_results_pooled=False))
