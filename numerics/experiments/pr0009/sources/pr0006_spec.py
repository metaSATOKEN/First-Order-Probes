# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""PR-0006 frozen B25 specification; no generation on import."""
import math
import numpy as np
import probe_backend_check_v1 as core

EXPERIMENT = 'lfdc_budget_pr0006_v1'
VERSION = '1.0.0'
CARD_SHA = '6fd46a0061470fb4345403ec5f3a4ae04addfaa52054361744f97870577ec0b4'
CORE_SHA = 'aecd24fffdb04da40b9bcd630ae3cdcf68b41e9a598ad662d4f893447cf02adf'
MASTER, PREFLIGHT_MASTER = 2026101720, 2026101721
REPEATS, SHARD_REPEATS, CHUNK = 25, 5, 4096
COUNTS = dict(ref=9999, cal=15999, null=20000, fault=5000, control=5000)
ROLES = dict(ref=0, cal=1, null=2, fault=3, control=4, preflight=5)
BUDGETS = {'W0':(2,2,4), 'WP':(1,1,6), 'WG':(3,3,2), 'W+':(3,1,4), 'W−':(1,3,4)}
METHODS = tuple(BUDGETS)+('LF0',)
ALPHA_DENOMINATORS = (100,1000)
PAIRS = tuple((j,i) for i in range(6) for j in range(i+1,6))
GIB = 1024**3
LIMITS = dict(active_seconds=14400, chunk_seconds=600, rss_bytes=8*GIB,
              gpu_bytes=8*GIB, evidence_bytes=8*GIB, zip_cache_bytes=16*GIB,
              fresh_free_bytes=24*GIB, zip_bytes=1_000_000_000,
              prepack_bytes=800_000_000, periodic_zip_seconds=900)


def conditions():
    out={}
    for source,kappa,variance in [('B',20.,1.),('K2',2.,1.),('K200',200.,1.),('V05',20.,.5),('V15',20.,1.5)]:
        for win,T in [('M',2.048),('L',8.192)]:
            key=source+'_'+win
            out[key]=dict(id=len(out),group=key,kind='normal',params=dict(dt=.002,tau_i=.04,tau_j=.01,
                kappa=kappa,variance=variance,T=T,fault='none',strength=0.))
    normal=list(out)
    def add(group,tag,fault,strength,kind='fault'):
        key=group+'__'+tag
        out[key]=dict(id=len(out),group=group,kind=kind,params=dict(out[group]['params'],fault=fault,strength=strength))
    for group in normal:
        for tag,fault,value in [('GM05','gain',-.05),('GP05','gain',.05),('P02','private',.02)]:add(group,tag,fault,value)
    add('B_L','GM02','gain',-.02);add('B_L','GP02','gain',.02)
    add('B_M','P005','private',.005);add('B_M','REV','time_reverse',0.,'control')
    return out


def baseline(group): return 'B_'+group.rsplit('_',1)[1]


def tasks(rep):
    out=[];cases=conditions()
    for stage in ('ref','cal','eval'):
        for case,c in cases.items():
            if stage=='ref' and case not in ('B_M','B_L'):continue
            if stage=='cal' and c['kind']!='normal':continue
            role=stage if stage!='eval' else ('null' if c['kind']=='normal' else c['kind'])
            out.append(dict(id=f'{rep:02d}/{case}/{role}',replicate=rep,case=case,role=role,count=COUNTS[role]))
    return out


def contexts():
    return [(case,pol) for case,c in conditions().items()
            for pol in (('B',) if c['group'].startswith('B_') else ('F','C'))]


def primary():
    core_cases=[('B_L__GM05','B'),('B_L__GP05','B'),('B_M__P02','B'),
                ('K200_L__P02','C'),('V05_L__GM05','C'),('V05_L__GP05','C')]
    return [(c,p,100,w,'W0') for c,p in core_cases for w in tuple(BUDGETS)[1:]]+[
            ('K2_M__P02','C',1000,'W0','LF0')]


def specification():
    return dict(experiment=EXPERIMENT,version=VERSION,card_sha256=CARD_SHA,core_sha256=CORE_SHA,
        namespace='LFDC-budget-confirm-v1',master=MASTER,preflight_master=PREFLIGHT_MASTER,
        repeats=REPEATS,shard_repeats=SHARD_REPEATS,chunk=CHUNK,random_block=64,
        counts=COUNTS,roles=ROLES,conditions=conditions(),methods=METHODS,budgets_eighths=BUDGETS,
        alpha_denominators=ALPHA_DENOMINATORS,pairs=PAIRS,contexts=contexts(),primary=primary(),
        limits=LIMITS,pins=core.PINS,tolerance=core.TOL,columns=core.COLUMNS,
        expected_records=13749700,unique_evaluation_records=9250000,policy_evaluation_records=16250000,
        evaluation_sets=22800,decision_count=195000000,primary_family=25,fpr_family=216,
        direction_shrinkage=1e-6,descriptive_margin=.01)


def make_inputs(case,role,rep,start,count):
    if role not in ROLES or count<=0 or start<0:raise ValueError('Invalid RNG address')
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
        for key,a in z.items():parts.setdefault(key,[]).append(a[lo:hi])
    return {k:np.ascontiguousarray(np.concatenate(v),dtype=np.float64) for k,v in parts.items()}


def nominal_nu():
    ai,aj=math.exp(-.002/.04),math.exp(-.002/.01)
    return (1+aj)*(1-ai)/(2*(1-ai*aj))
