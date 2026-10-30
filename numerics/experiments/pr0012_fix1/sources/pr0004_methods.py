# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Detector API takes normal observations and nominal response, not source truth."""
from fractions import Fraction
import math
import numpy as np
from scipy import stats
from pr0004_spec import EPSILON, METHODS


def validate(values):
    a = np.asarray(values)
    if a.ndim!=2 or a.shape[1]!=17 or a.dtype!=np.float64 or not np.isfinite(a).all():
        raise ValueError('Expected finite float64 [records,17]; no silent exclusions')
    return a


def qvalues(values):
    return validate(values)[:, [11,12]]


def projections(model, q, diagonal=False):
    weights = model['weights_diagonal' if diagonal else 'weights']
    x = (q-np.asarray(model['mean'])) @ np.asarray(weights).T
    x[:,0] = np.abs(x[:,0])
    return x


def fit_reference(reference, nu):
    a = validate(reference); q = qvalues(a)
    if len(a)<3:
        raise ValueError('Insufficient reference records')
    mu, omega = q.mean(axis=0), np.cov(q, rowvar=False, ddof=1)
    if not np.isfinite(omega).all() or np.any(np.diag(omega)<=0):
        raise ValueError('Nonpositive reference variance; model failure')
    means = a.mean(axis=0)
    directions = np.array([[means[8]-2*nu*means[7],2*(means[15]-means[16])],
                           [-nu*means[7],means[15]]])
    scale = np.sqrt(np.diag(omega)); R = omega/np.outer(scale,scale)
    regular = ((1-EPSILON)*R+EPSILON*np.eye(2))*np.outer(scale,scale)
    diagonal = np.diag(np.diag(omega))
    def weights(cov):
        result=[]; disabled=[]
        for i,d in enumerate(directions):
            if np.all(d==0):
                result.append([0.,0.]); disabled.append(i); continue
            # Scale before solving so a weak but nonzero direction is not dropped.
            scaled = d/np.max(np.abs(d))
            sol = np.linalg.solve(cov,scaled)
            norm = float(scaled @ sol)
            if not math.isfinite(norm) or norm<=0:
                raise ValueError('Invalid linear direction solve')
            result.append((sol/math.sqrt(norm)).tolist())
        return result,disabled
    full, disabled = weights(regular); diag, disabled_d = weights(diagonal)
    model = dict(n_reference=len(a), mean=mu.tolist(), covariance=omega.tolist(),
        regularized_covariance=regular.tolist(), directions=directions.tolist(),
        direction_norms=[math.hypot(*d) for d in directions],
        eigenvalues=np.linalg.eigvalsh(omega).tolist(),
        regularized_eigenvalues=np.linalg.eigvalsh(regular).tolist(),
        condition_number=float(np.linalg.cond(omega)),
        regularized_condition_number=float(np.linalg.cond(regular)),
        weights=full, weights_diagonal=diag, disabled=disabled, disabled_diagonal=disabled_d,
        epsilon=EPSILON, nu=float(nu), normal_moment_means=means.tolist())
    # A singular unregularized matrix is a diagnostic, not a failure if regularization works.
    if not math.isfinite(model['condition_number']):
        model['condition_number']=None
    ordered = {'J':np.sort(np.column_stack((np.abs(q[:,0]),q[:,1])),axis=0),
               'LF':np.sort(projections(model,q),axis=0),
               'LF-D':np.sort(projections(model,q,True),axis=0)}
    return model,ordered


def component_ranks(model, ordered, values):
    q = qvalues(values)
    features = {'J':np.column_stack((np.abs(q[:,0]),q[:,1])),
                'LF':projections(model,q), 'LF-D':projections(model,q,True)}
    out = {}
    for name,x in features.items():
        out[name] = np.column_stack([np.searchsorted(ordered[name][:,j],x[:,j],side='left')
                                    for j in range(2)]).astype(np.uint16)
        disabled = model.get('disabled_diagonal' if name=='LF-D' else 'disabled',[]) if name!='J' else []
        out[name][:,disabled] = 0
    return out


def cutoff_order(n, denominator):
    x = Fraction((n+1)*(denominator-1),denominator)
    return (x.numerator+x.denominator-1)//x.denominator


def calibrate(model, ordered, calibration):
    r = component_ranks(model,ordered,calibration); n=model['n_reference']
    def cut(x, denominator):
        k=cutoff_order(len(x),denominator)
        return n if k>len(x) else int(np.partition(x,k-1)[k-1])
    j=r['J']; cuts=np.full((6,2),n,dtype=np.int64)
    cuts[0,0],cuts[1,1]=cut(j[:,0],100),cut(j[:,1],100)
    cuts[2]=[cut(j[:,i],200) for i in range(2)]
    for index,name in [(3,'J'),(4,'LF'),(5,'LF-D')]:
        cuts[index]=cut(r[name].max(axis=1),100)
    ties=[]
    for i,name in enumerate(METHODS):
        v=r['J' if i<4 else name]
        ties.append(dict(method=name, component=[int(np.sum(v[:,k]==cuts[i,k])) for k in range(2)],
                         max_ties=int(np.sum(v.max(axis=1)==cuts[i,0])) if i>=3 else None))
    return dict(cutoffs=cuts.tolist(),ties=ties,calibration_count=len(calibration),
                saturation={k:(v==n).mean(axis=0).tolist() for k,v in r.items()})


def decide(model, ordered, calibration_model, values):
    ranks=component_ranks(model,ordered,values)
    cuts=np.asarray(calibration_model['cutoffs'])
    flags=np.column_stack([np.any(ranks['J' if i<4 else name]>cuts[i],axis=1)
                           for i,name in enumerate(METHODS)])
    bits=np.sum(flags.astype(np.uint8)*(1<<np.arange(6,dtype=np.uint8)),axis=1).astype(np.uint8)
    return flags,bits,ranks


def wilson(k,n):
    z=float(stats.norm.ppf(.975)); p=k/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-half,center+half]


def overlap(a,b):
    if len(a)!=len(b): raise ValueError('Unpaired flags')
    both=int(np.sum(a&b)); gained=int(np.sum(a&~b)); lost=int(np.sum(~a&b))
    neither=len(a)-both-gained-lost
    if gained-lost!=int(a.sum())-int(b.sum()): raise AssertionError('Count conservation')
    return dict(n=len(a),both=both,gained=gained,lost=lost,neither=neither,difference=(gained-lost)/len(a))


def outer_interval(x,family=1):
    x=np.asarray(x,dtype=float)
    if len(x)<2 or not np.isfinite(x).all(): raise ValueError('Incomplete outer series')
    mean=float(x.mean()); sd=float(x.std(ddof=1)); se=sd/math.sqrt(len(x))
    t=float(stats.t.ppf(1-.05/(2*family),len(x)-1))
    lo,hi=(mean-t*se,mean+t*se) if sd>0 else (None,None)
    return dict(mean=mean,sd=sd,mcse=se,lower=lo,upper=hi,replicates=len(x),family=family,
                q10=float(np.quantile(x,.1)),q50=float(np.quantile(x,.5)),q90=float(np.quantile(x,.9)),
                inference='approximate outer-replicate t' if sd>0 else 'undefined: zero replicate SD')
