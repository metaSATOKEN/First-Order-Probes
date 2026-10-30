# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""PR-0012 B50 frozen configuration; inert imports."""
import numpy as np
from pr0007_spec import conditions as all_conditions,baseline
from pr0006_spec import core,nominal_nu
EXPERIMENT='mixture_recovery_pr0012_v1'
VERSION='1.0.0'
CARD_SHA='2a465e5e97216797e3856ccff6b1ec8c3b2a86627096797047e1bfb78a941f51'
MASTER,PREFLIGHT_MASTER=2026101820,2026101821
REPEATS,SHARD_REPEATS,CHUNK=50,5,4096
GROUPS=('B_L','K2_M','K200_L','V05_L')
FAULTS=('GM05','GP05','P02')
CANDIDATES=('clean','GM800','GP800','BAL800','BAL400','TRI800')
CONTEXTS=('F',)+CANDIDATES+('C',)
ALPHA_DENOMINATORS=(100,1000)
ROLES=dict(ref=0,old_cal=1,audit_normal=2,audit_fault=3,update_normal=4,update_fault=5,
 recovery=6,validation_normal=7,null_eval=8,fault_eval=9,preflight=10,
 audit_order=11,audit_replace=12,update_order=13,update_replace=14)
GATE_CUTS=np.array([[2.333890445586029,2.333890445586029],[2.454714455705117,3.0310874379402915]])
CP_ETA=.00625
CAP_MULTIPLIER=1.25
LIMITS=dict(active_seconds=28800,chunk_seconds=600,rss_bytes=8*1024**3,gpu_bytes=8*1024**3,
 evidence_bytes=8*1024**3,zip_cache_bytes=16*1024**3,fresh_free_bytes=24*1024**3,
 zip_bytes=1_000_000_000,prepack_bytes=800_000_000,periodic_zip_seconds=900)
COUNTS=dict(ref=9999,old_cal=15999,audit_normal=15999,update_normal=15999,recovery=15999,
 validation_normal=20000,null_eval=20000,fault_eval=5000)

def groups():return GROUPS
def conditions():
    return {k:v for k,v in all_conditions().items() if k=='B_M' or v['group'] in GROUPS}

def tasks(rep):
    if not 0<=rep<REPEATS:raise ValueError('Replicate outside frozen range')
    out=[]
    def add(case,role,n):out.append(dict(id=f'{rep:02d}/{case}/{role}',replicate=rep,case=case,role=role,count=n))
    for base in ('B_M','B_L'):
        for role in ('ref','old_cal'):add(base,role,COUNTS[role])
    for group in GROUPS:
        for prefix in ('audit','update'):
            add(group,prefix+'_normal',15999)
            for f,n in zip(FAULTS,(800,800,400)):add(group+'__'+f,prefix+'_fault',n)
        for role in ('recovery','validation_normal','null_eval'):add(group,role,COUNTS[role])
        for f in FAULTS:add(group+'__'+f,'fault_eval',5000)
    return out

def order(group,role,rep,n):
    if role not in ('audit_order','audit_replace','update_order','update_replace'):raise ValueError('Permutation role')
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([MASTER,conditions()[group]['id'],ROLES[role],rep,0])))
    return rng.permutation(n)

def recipe(name):
    if name=='clean':return []
    if name in ('GM800','GP800'):return [(name[:2]+'05',0,800,0)]
    if name=='BAL800':return [('GM05',0,400,0),('GP05',400,800,0)]
    if name=='BAL400':return [('GM05',0,200,0),('GP05',400,600,0)]
    if name=='TRI800':return recipe('BAL400')+[('P02',200,400,0),('P02',600,800,200)]
    raise ValueError('Unknown candidate')

def primary():
    out=[dict(id='M01',kind='paired_gate',group='B_L',contrast='BAL800-GM800 G-M',expected=-1),
         dict(id='M02',kind='paired_gate',group='B_L',contrast='BAL800-GP800 G-M',expected=-1),
         dict(id='M03',kind='paired_gate',group='B_L',contrast='BAL800 G-MC-G-M',expected=1),
         dict(id='M04',kind='paired_gate',group='K200_L',contrast='TRI800-BAL400 G-MC',expected=1)]
    out += [dict(id=f'M{i+5:02d}',kind='paired_rate',group='K200_L',contrast='U TRI800-BAL400',evaluation=e,expected=-1)
            for i,e in enumerate(('normal',)+FAULTS)]
    out += [dict(id='R01',kind='recovery_probability',group='K2_M',candidate='TRI800',policy='P2',threshold=.5)]
    out += [dict(id=f'R{i+2:02d}',kind='paired_rate',group='K2_M',candidate='TRI800',contrast='P2-P0',evaluation=e,expected=-1)
            for i,e in enumerate(('normal',)+FAULTS)]
    return [dict(p,method='W0',alpha=.01,family=13) for p in out]

def specification():
    return dict(experiment=EXPERIMENT,version=VERSION,card_sha256=CARD_SHA,
      namespace='closure-mixture-recovery-v1',master=MASTER,preflight_master=PREFLIGHT_MASTER,
      repeats=50,shard_repeats=5,chunk=4096,random_block=64,groups=GROUPS,conditions=conditions(),
      roles=ROLES,counts=COUNTS,task_plan=tasks(0),candidates=CANDIDATES,
      recipes={c:recipe(c) for c in CANDIDATES},methods=('W0','LF0-audit'),
      budgets_eighths=(2,2,4),alpha_denominators=ALPHA_DENOMINATORS,gate_cuts=GATE_CUTS.tolist(),
      policies=('P0','P1','P2'),cp_eta=CP_ETA,cap_multiplier=CAP_MULTIPLIER,
      primary=primary(),primary_family=13,limits=LIMITS,pins=core.PINS,tolerance=core.TOL,
      columns=core.COLUMNS,records_per_replicate=479984,expected_records=23999200,
      W0_E_decisions=112000000,LF0_E_decisions=42000000,V_decisions=16000000)

def make_inputs(*args,**kwargs):
    from pr0012_input import make_inputs as generate
    return generate(*args,**kwargs)
