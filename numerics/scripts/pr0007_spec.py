# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Frozen PR-0007 + A01. Import is inert; no evaluation-dependent choices."""
import numpy as np
import pr0006_spec as inherited
from pr0006_spec import BUDGETS, METHODS, PAIRS, ALPHA_DENOMINATORS, LIMITS, nominal_nu, core

EXPERIMENT = 'lfdc_calibration_pr0007_v1'
VERSION = '1.0.0'
CARD_SHA = '8309b2143879fef69a5020fc2945fb2159786f3de7c1da62b787b6907d98be37'
A01_SHA = '8c87378bb87920c985f4e94477799cf382a2519c7db3678a85a63ebde05e4a7f'
MASTER, PREFLIGHT_MASTER = 2026101800, 2026101801
REPEATS, SHARD_REPEATS, CHUNK = 25, 5, 4096
SIZES, MIX_COUNTS = (7999, 15999, 31999), (160, 800)
COUNTS = dict(ref=9999, normal_cal=31999, fault_cal=800, null_eval=20000, fault_eval=5000)
ROLES = dict(ref=0, normal_cal=1, null_eval=2, fault_eval=3, fault_cal=4,
             preflight=5, normal_order=6, replacement_order=7)
FAULTS = ('GM05', 'GP05', 'P02')


def conditions():
    out = {}
    for key, k, v, T in [('B_M',20.,1.,2.048), ('B_L',20.,1.,8.192),
                        ('K2_M',2.,1.,2.048), ('K200_L',200.,1.,8.192),
                        ('V05_L',20.,.5,8.192), ('V15_L',20.,1.5,8.192)]:
        out[key] = dict(id=len(out), group=key, kind='normal', params=dict(
            dt=.002, tau_i=.04, tau_j=.01, kappa=k, variance=v, T=T, fault='none', strength=0.))
    for group in list(out):
        for tag, fault, value in [('GM05','gain',-.05), ('GP05','gain',.05), ('P02','private',.02)]:
            out[group+'__'+tag] = dict(id=len(out), group=group, kind='fault',
                params=dict(out[group]['params'], fault=fault, strength=value))
    return out


def groups(): return tuple(k for k,c in conditions().items() if c['kind']=='normal')
def baseline(group): return 'B_'+group.rsplit('_',1)[1]
def clean_policy(group): return 'B' if group.startswith('B_') else 'C-clean'
def clean_id(group,m): return f'{group}__clean_{m}'
def mix_id(group,fault,n): return f'{group}__mix_{fault}_{n}'


def tasks(rep):
    out=[]
    for role in ('ref','normal_cal','fault_cal','null_eval','fault_eval'):
        for case,c in conditions().items():
            if role=='ref' and case not in ('B_M','B_L'): continue
            if role in ('normal_cal','null_eval') and c['kind']!='normal': continue
            if role in ('fault_cal','fault_eval') and c['kind']!='fault': continue
            out.append(dict(id=f'{rep:02d}/{case}/{role}', replicate=rep, case=case, role=role, count=COUNTS[role]))
    return out


def calibrations():
    """66 contexts, 54 fitted calibrations. F reuses the baseline fit key."""
    out=[]
    for g in groups():
        for m in SIZES:
            key=clean_id(g,m)
            out.append(dict(id=key, fit_key=key, group=g, policy=clean_policy(g), m=m, mix_fault='', mix_count=0))
        if not g.startswith('B_'):
            for m in SIZES:
                out.append(dict(id=f'{g}__F_{m}', fit_key=clean_id(baseline(g),m), group=g,
                    policy='F', m=m, mix_fault='', mix_count=0))
        for fault in FAULTS:
            for n in MIX_COUNTS:
                key=mix_id(g,fault,n)
                out.append(dict(id=key, fit_key=key, group=g, policy='C-mix', m=SIZES[1], mix_fault=fault, mix_count=n))
    return out


def contexts():
    return [dict(index=i, case=case, **c) for i,(case,c) in enumerate(
        ( (case,c) for c in calibrations() for case in (c['group'],)+tuple(c['group']+'__'+f for f in FAULTS) ))]


def contrasts():
    cs=contexts();index={(c['case'],c['id']):c['index'] for c in cs};out=[]
    for c in cs:
        other=[];g=c['group']
        if c['policy']=='C-clean': other.append(('C-clean-minus-F',f'{g}__F_{c["m"]}'))
        if c['policy']=='C-mix': other.append(('C-mix-minus-clean',clean_id(g,SIZES[1])))
        if c['policy'] in ('B','C-clean','F'):
            for smaller in SIZES:
                if smaller<c['m']:
                    key=f'{g}__F_{smaller}' if c['policy']=='F' else clean_id(g,smaller)
                    other.append(('larger-minus-smaller-cal',key))
        for kind,key in other:
            out.append(dict(index=len(out),kind=kind,left=c['index'],right=index[c['case'],key],case=c['case']))
    return out


def primary():
    return [dict(id=f'H1-{i+1:02d}', group=g, fault=f, mix_count=n, cal_id=mix_id(g,f,n),
                 case=g+'__'+f, method='W0', denominator=100)
            for i,(g,f,n) in enumerate((g,f,n) for g,f in
                [('B_L','GM05'),('B_L','GP05'),('K200_L','P02'),('K2_M','P02')] for n in MIX_COUNTS)]


def orders(group,rep):
    cid=conditions()[group]['id']
    def permutation(role,n):
        seed=[MASTER,cid,ROLES[role],rep,0]
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed))).permutation(n)
    return permutation('normal_order',COUNTS['normal_cal']),permutation('replacement_order',SIZES[1])


def make_inputs(case,role,rep,start,count):
    # Same draw order as PR-0006; only frozen master/condition/role addresses differ.
    if role not in ROLES or role.endswith('_order') or count<=0 or start<0: raise ValueError('Invalid RNG address')
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


def specification():
    return dict(experiment=EXPERIMENT,version=VERSION,card_sha256=CARD_SHA,a01_sha256=A01_SHA,
        core_sha256=inherited.CORE_SHA,namespace='LFDC-calibration-contamination-v1',master=MASTER,
        preflight_master=PREFLIGHT_MASTER,repeats=REPEATS,shard_repeats=SHARD_REPEATS,chunk=CHUNK,random_block=64,
        counts=COUNTS,roles=ROLES,conditions=conditions(),calibration_sizes=SIZES,mix_counts=MIX_COUNTS,
        methods=METHODS,budgets_eighths=BUDGETS,alpha_denominators=ALPHA_DENOMINATORS,pairs=PAIRS,
        calibrations=calibrations(),contexts=contexts(),contrasts=contrasts(),primary=primary(),
        limits=LIMITS,pins=core.PINS,tolerance=core.TOL,columns=core.COLUMNS,
        expected_records=10909800,records_per_replicate=436392,unique_evaluation_records=5250000,
        policy_evaluation_records=57750000,evaluation_sets=79200,decision_count=693000000,
        primary_family=8,fpr_family=792,cross_cells=1296,direction_shrinkage=1e-6,
        transport='A01: five replicates per non-cumulative shard; delta partials; strict <1e9 bytes')
