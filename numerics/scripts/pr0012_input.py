# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Inherited physical draw order; address constants supplied by PR-0012."""
import numpy as np
from pr0012_spec import conditions, core, MASTER, PREFLIGHT_MASTER, ROLES

def make_inputs(case,role,rep,start,count):
    # Same draw order as PR-0006; only frozen master/condition/role addresses differ.
    if role not in ROLES or role.endswith(('_order','_replace')) or count<=0 or start<0: raise ValueError('Invalid RNG address')
    c=conditions()[case];p=c['params'];n,w,g=core.dimensions(p);L,C=n+2*w,n+w
    _,factor=core.stationary(p['kappa'],p['tau_i'],p['tau_j'],p['dt'])
    base,_=core.stationary(20.,p['tau_i'],p['tau_j'],p['dt']);noise=np.sqrt(.1*np.diag(base)[1:])
    parts={};master=PREFLIGHT_MASTER if role=='preflight' else MASTER
    for block in range(start//64,(start+count-1)//64+1):
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([master,c['id'],ROLES[role],rep,block])))
        z={'initial':np.sqrt(p['variance'])*(rng.normal(size=(64,3))@factor.T),'source':rng.normal(size=(64,L+2*g))}
        if p['fault']=='private':
            z.update(private_initial=np.sqrt(p['variance'])*(rng.normal(size=(64,3))@factor.T),private_source=rng.normal(size=(64,L)))
        z.update(ni=noise[0]*rng.normal(size=(64,L)),nj=noise[1]*rng.normal(size=(64,L)),
                 ci=noise[0]*rng.normal(size=(64,C)),cj=noise[1]*rng.normal(size=(64,C)))
        lo,hi=max(start-block*64,0),min(start+count-block*64,64)
        for key,a in z.items(): parts.setdefault(key,[]).append(a[lo:hi])
    return {k:np.ascontiguousarray(np.concatenate(v),dtype=np.float64) for k,v in parts.items()}

