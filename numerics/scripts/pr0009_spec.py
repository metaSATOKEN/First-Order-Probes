# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Frozen PR-0009 plan. No physical simulation on import."""
import numpy as np
from pr0007_spec import conditions, groups, baseline
from pr0006_spec import core, BUDGETS, METHODS, PAIRS, ALPHA_DENOMINATORS, nominal_nu

EXPERIMENT='two_axis_pr0009_v1'
VERSION='1.0.0'
CARD_SHA='bfe645d75be346a727de63f6fe4564360c2693be35fbd8e626179987e5c1ba23'
MASTER,PREFLIGHT_MASTER=2026101809,2026101810
REPEATS,SHARD_REPEATS,CHUNK=50,5,4096
FAULTS=('GM05','GP05','P02')
CANDIDATES=(('clean',0),('GM05',160),('GM05',800),('GP05',160),('GP05',800),('P02',160),('P02',800))
GATES=('G-old','G-M','G-MC')
POLICIES=('F','U')+GATES+('G-MC-same-U',)
COUNTS=dict(ref=9999,old_cal=15999,audit_normal=15999,audit_fault=800,
    update_normal=15999,update_fault=800,validation_normal=20000,validation_fault=1000,
    null_eval=20000,fault_eval=5000)
ROLES={r:i for i,r in enumerate(tuple(COUNTS)+('preflight','audit_order','audit_replace',
                                             'update_order','update_replace','validation_replace'))}
LIMITS=dict(active_seconds=43200,chunk_seconds=600,rss_bytes=8*1024**3,gpu_bytes=8*1024**3,
    evidence_bytes=16*1024**3,zip_cache_bytes=32*1024**3,fresh_free_bytes=48*1024**3,
    zip_bytes=1_000_000_000,prepack_bytes=800_000_000,periodic_zip_seconds=900)
DEVELOPMENT_SHA='93e8e891d7c47f4d3b60f4acc511fb423e23ddd1fc5dc37b825a07ec4ec2927e'
GATE_CUTS=np.array([[.04012750796924808,2.333890445586029,2.333890445586029],
                    [.029564347771735732,2.454714455705117,3.0310874379402915]])
CAP_MULTIPLIER=1.25
CP_ETA=.0125

def tasks(rep):
    if not 0<=rep<REPEATS:raise ValueError('Replicate out of range')
    out=[]
    for role,count in COUNTS.items():
        for case,c in conditions().items():
            if role in ('ref','old_cal') and case not in ('B_M','B_L'):continue
            normal=role in ('ref','old_cal','audit_normal','update_normal','validation_normal','null_eval')
            if normal!=(c['kind']=='normal'):continue
            out.append(dict(id=f'{rep:02d}/{case}/{role}',replicate=rep,case=case,role=role,count=count))
    return out

def order(group,role,rep,n):
    if role not in ('audit_order','audit_replace','update_order','update_replace','validation_replace'):
        raise ValueError('Not a permutation role')
    seed=[MASTER,conditions()[group]['id'],ROLES[role],rep,0]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed))).permutation(n)

def primary():
    out=[]
    cells=[(g,f,n,f) for g,f in [('B_L','GM05'),('B_L','GP05'),('K200_L','P02'),('K2_M','P02')]
           for n in (160,800)]+[('V05_L','GM05',800,'GP05')]
    for g,f,n,e in cells:
        for outcome in (e,'normal'):
            out.append(dict(id=f'P{len(out)+1:02d}',group=g,mix_fault=f,mix_count=n,
                evaluation=outcome,candidate=CANDIDATES.index((f,n)),policy='G-MC-minus-U',
                method='W0',alpha=.01,family=18))
    return out

def specification():
    return dict(experiment=EXPERIMENT,version=VERSION,card_sha256=CARD_SHA,
        namespace='two-axis-confirmation-pr0009-v1',master=MASTER,preflight_master=PREFLIGHT_MASTER,
        repeats=REPEATS,shard_repeats=5,chunk=CHUNK,random_block=64,roles=ROLES,counts=COUNTS,
        conditions=conditions(),methods=METHODS,budgets_eighths=BUDGETS,alpha_denominators=ALPHA_DENOMINATORS,
        candidates=CANDIDATES,gates=GATES,gate_cuts=GATE_CUTS.tolist(),development_sha256=DEVELOPMENT_SHA,
        policies=POLICIES,cap_multiplier=CAP_MULTIPLIER,cp_eta=CP_ETA,primary=primary(),
        limits=LIMITS,pins=core.PINS,tolerance=core.TOL,columns=core.COLUMNS,
        records_per_replicate=620784,expected_records=31039200,decision_count=1008000000,
        validation_decision_count=82800000,primary_family=18)

def make_inputs(*args,**kwargs):
    from pr0009_input import make_inputs as generate
    return generate(*args,**kwargs)
