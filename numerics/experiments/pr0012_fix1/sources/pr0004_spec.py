# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Frozen PR-0004 design and independent RNG namespace; import never runs trials."""
import math
import numpy as np
import probe_backend_check_v1 as core

VERSION = 'linear-fusion-pr0004-1.0.0'
EXPERIMENT = 'linear_fusion_pr0004_v1'
MASTER = 2026101710
CARD_SHA = '740001976854b195ee831bec0026c7725feb9438372c736792289d9b0963d3a9'
CORE_SHA = 'aecd24fffdb04da40b9bcd630ae3cdcf68b41e9a598ad662d4f893447cf02adf'
REPEATS, CHUNK, EPSILON = 50, 4096, 1e-6
METHODS = ('M', 'CR', 'B', 'J', 'LF', 'LF-D')
ROLES = dict(ref=0, cal=1, null=2, fault=3, control=4, preflight=5)
COUNTS = dict(ref=9999, cal=10000, null=20000, fault=5000, control=5000)
GIB = 1024**3
LIMITS = dict(active_seconds=21600, task_seconds=600, rss_bytes=8*GIB,
              gpu_reserved_bytes=8*GIB, workspace_bytes=16*GIB,
              zip_bytes=6*GIB, archive_cache_bytes=40*GIB, fresh_free_bytes=64*GIB)


def conditions():
    normal = dict(list(core.conditions().items())[:12])
    out = {k: dict(params=v, group=k, kind='normal', id=i) for i, (k,v) in enumerate(normal.items())}
    def add(group, tag, fault, strength, kind='fault'):
        key = group+'__'+tag
        out[key] = dict(params=dict(normal[group], fault=fault, strength=strength),
                        group=group, kind=kind, id=len(out))
    for group in ('B_S', 'B_M', 'B_L'):
        add(group, 'GM05', 'gain', -.05); add(group, 'GP05', 'gain', .05)
    add('B_M', 'P02', 'private', .02)
    add('B_M', 'DM2', 'delay', -2); add('B_M', 'DP2', 'delay', 2)
    add('B_R10', 'GP05', 'gain', .05)
    add('B_L', 'GM02', 'gain', -.02); add('B_L', 'GP02', 'gain', .02)
    add('B_M', 'P005', 'private', .005)
    for group in list(normal)[4:]:
        add(group, 'GM05', 'gain', -.05); add(group, 'GP05', 'gain', .05)
        add(group, 'P02', 'private', .02)
    add('B_M', 'REV', 'time_reverse', 0., 'control')
    return out


def tasks(repeats=REPEATS):
    out = []
    for rep in range(repeats):
        for case, c in conditions().items():
            roles = ('ref', 'cal', 'null') if c['kind']=='normal' else (c['kind'],)
            for role in roles:
                out.append(dict(id=f'{rep:02d}/{case}/{role}', replicate=rep, case=case,
                                role=role, count=COUNTS[role]))
    return out


def batch_size(case):
    return 512 if conditions()[case]['group']=='B_S' else 4096


def baseline(group):
    return group if group.startswith('B_') else 'B_'+group.rsplit('_', 1)[1]


def primary():
    return [(case, 'LF', other) for case in ('B_L__GM05','B_L__GP05','B_M__P02')
            for other in METHODS[:4]]


def specification():
    return dict(version=VERSION, experiment=EXPERIMENT, master_seed=MASTER,
        namespace='PR0004-LF-v1', card_sha256=CARD_SHA, core_sha256=CORE_SHA,
        repeats=REPEATS, columns=core.COLUMNS, methods=list(METHODS),
        conditions=conditions(), roles=ROLES, counts=COUNTS, chunk=CHUNK,
        random_block=64, batches={k:batch_size(k) for k in conditions()},
        epsilon=EPSILON, pins=core.PINS, tolerance=core.TOL, limits=LIMITS,
        primary=primary(), expected_records=33499400, expected_tasks=3700,
        expected_chunks=10400, main_fpr_series=140, checkpoint_replicates=[10,20,30,40])


def make_inputs(case, role, replicate, start, count):
    """PR-0003 input construction, with only the seed addressing changed.

    Logical blocks are independent of compute batch and restart boundaries.
    Source parameters live here, never in the learned detector API.
    """
    c = conditions()[case]; p = c['params']
    if count<=0 or start<0 or role not in ROLES:
        raise ValueError('Invalid logical record range')
    n,w,g = core.dimensions(p); L,C = n+2*w,n+w
    _, factor = core.stationary(p['kappa'], p['tau_i'], p['tau_j'], p['dt'])
    base, _ = core.stationary(20., p['tau_i'], p['tau_j'], p['dt'])
    noise = np.sqrt(.1*np.diag(base)[1:])
    parts = {}
    for block in range(start//64, (start+count-1)//64+1):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [MASTER,c['id'],ROLES[role],replicate,block])))
        z = {'initial': np.sqrt(p['variance'])*(rng.normal(size=(64,3)) @ factor.T),
             'source': rng.normal(size=(64,L+2*g))}
        if p['fault']=='private':
            z['private_initial'] = np.sqrt(p['variance'])*(rng.normal(size=(64,3)) @ factor.T)
            z['private_source'] = rng.normal(size=(64,L))
        z.update(ni=noise[0]*rng.normal(size=(64,L)), nj=noise[1]*rng.normal(size=(64,L)),
                 ci=noise[0]*rng.normal(size=(64,C)), cj=noise[1]*rng.normal(size=(64,C)))
        lo,hi = max(start-block*64,0), min(start+count-block*64,64)
        for key,value in z.items():
            parts.setdefault(key,[]).append(value[lo:hi])
    return {k:np.ascontiguousarray(np.concatenate(v),dtype=np.float64) for k,v in parts.items()}


def nominal_nu(dt, ti, tj):
    ai,aj = math.exp(-dt/ti),math.exp(-dt/tj)
    return (1+aj)*(1-ai)/(2*(1-ai*aj))
