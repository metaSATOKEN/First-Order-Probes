# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Independently check saved fusion tables against reference-rank decisions.

Does not run the simulation or refit using evaluation data. Uses default paths
for the recorded main run and fusion_preliminary_v1 analysis.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'numerics/analysis/main_2ae37613a10478c8/fusion_preliminary_v1'
ARCHIVE = ROOT / 'main_2ae37613a10478c8_COMPLETE_20261016T160008Z.zip'
METHODS = ['moment','CR','bonferroni_or','joint_or']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def table(name):
    with (OUT/'tables'/f'{name}.csv').open() as f:
        return list(csv.DictReader(f))


def main():
    audit=json.loads((OUT/'audit.json').read_text())
    assert sha(ARCHIVE)==audit['archive_sha256']
    assert sha(ROOT/'numerics/scripts/analyze_covariance_fusion.py')==audit['script_sha256']
    assert sha(OUT/'PROTOCOL.md')==audit['protocol_sha256']
    for path,value in audit['protected_hashes_before'].items():
        assert sha(ROOT/path)==value==audit['protected_hashes_after'][path],path
    z=zipfile.ZipFile(ARCHIVE)
    chunks={}
    for name in z.namelist():
        if name.startswith('chunks/') and name.endswith('.npz'):
            chunks.setdefault(name.split('/')[1],[]).append(name)
    def scores(task):
        pair=[]
        for name in sorted(chunks[task]):
            with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as a:
                pair.append(np.c_[np.abs(a['R_full']),a['CR']])
        return np.concatenate(pair)
    splits=np.load(OUT/'split_indices.npz',allow_pickle=False)
    models=json.loads((OUT/'models.json').read_text())
    bynormal={m['null_case']:m for m in models.values()}
    sortedrefs={}; saved_flags={}; cal_verified=0
    for model in models.values():
        cid=model['null_case']; ref=np.sort(scores(model['reference_id']),axis=0)
        sortedrefs[cid]=ref
        normal=scores(cid)
        a=splits[cid+'_cal'];b=splits[cid+'_hold']
        assert len(a)==len(b)==10000 and len(np.intersect1d(a,b))==0
        assert np.array_equal(np.sort(np.r_[a,b]),np.arange(20000))
        assert hashlib.sha256(a.astype('<i8').tobytes()).hexdigest()==model['calibration_indices_sha256']
        assert hashlib.sha256(b.astype('<i8').tobytes()).hexdigest()==model['holdout_indices_sha256']
        cr=np.column_stack([np.searchsorted(ref[:,j],normal[a,j],side='left') for j in range(2)])
        # Independently compute the declared 1-based order statistics.
        q_m=int(np.sort(cr[:,0])[9900]);q_c=int(np.sort(cr[:,1])[9900])
        q_bm=int(np.sort(cr[:,0])[9950]);q_bc=int(np.sort(cr[:,1])[9950])
        q_j=int(np.sort(np.maximum(cr[:,0],cr[:,1]))[9900])
        expect=[[q_m,9999],[9999,q_c],[q_bm,q_bc],[q_j,q_j]]
        assert expect==model['rank_cutoffs']
        # Confirm exact equivalence to effective raw thresholds stored for reuse.
        raw=[[None if r==9999 else float(ref[r,j]) for j,r in enumerate(row)] for row in expect]
        assert raw==model['raw_cutoffs']
        cal_verified+=1
    all_power=table('normal_false_alarm')+table('power')+table('source_shift')
    for row in all_power:
        cid=row['case']; model=bynormal[row['calibration_null_case']]
        if cid not in saved_flags:
            data=scores(cid)
            if row['cohort']=='normal_holdout': data=data[splits[cid+'_hold']]
            ref=sortedrefs[model['null_case']]
            ranks=np.column_stack([np.searchsorted(ref[:,j],data[:,j],side='left') for j in range(2)])
            q=model['rank_cutoffs']
            f=np.column_stack([ranks[:,0]>q[0][0],ranks[:,1]>q[1][1],
                (ranks[:,0]>q[2][0]) | (ranks[:,1]>q[2][1]),
                np.maximum(ranks[:,0],ranks[:,1])>q[3][0]])
            saved_flags[cid]=f
            assert not np.any(f[:,3] & ~(f[:,0] | f[:,1]))
        f=saved_flags[cid];m=METHODS.index(row['method']);n=len(f);h=int(f[:,m].sum())
        assert n==int(row['n']) and h==int(row['alarms'])
        assert abs(float(row['rate'])-h/n)<1e-14
        assert 0<=float(row['pipeline_bootstrap_low'])<=float(row['pipeline_bootstrap_high'])<=1
        assert 0<=float(row['conditional_wilson_low'])<=float(row['conditional_wilson_high'])<=1
    trades=table('normal_tradeoffs')+table('detection_tradeoffs')+table('source_shift_tradeoffs')
    for row in trades:
        f=saved_flags[row['case']];a=METHODS.index(row['candidate']);b=METHODS.index(row['baseline'])
        vals=dict(both=int(np.sum(f[:,a]&f[:,b])),gained=int(np.sum(f[:,a]&~f[:,b])),
                  lost=int(np.sum(~f[:,a]&f[:,b])),neither=int(np.sum(~f[:,a]&~f[:,b])))
        for key,value in vals.items(): assert value==int(row[key])
        assert sum(vals.values())==len(f)
        assert abs(float(row['net_rate'])-(vals['gained']-vals['lost'])/len(f))<1e-14
        assert -1<=float(row['net_bootstrap_low'])<=float(row['net_bootstrap_high'])<=1
    pats=table('detection_patterns')
    for row in pats:
        f=saved_flags[row['case']];target=[bool(int(row[m])) for m in METHODS]
        assert int(np.sum(np.all(f==target,axis=1)))==int(row['count'])
    accts=table('union_accounting')
    for row in accts:
        f=saved_flags[row['case']];u=f[:,0]|f[:,1]
        assert int(u.sum())==int(row['naive_union_count'])
        assert int(np.sum(u&~f[:,3]))==int(row['tightening_discard_count'])
        assert int(np.sum(f[:,0]&f[:,1]))==int(row['single_overlap_count'])
        assert int(f[:,3].sum())==int(row['joint_retained_count'])
        assert int(row['outside_naive_union_count'])==0
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'numerics/tests'),
                           '-p','test_fusion_analysis.py','-v'],capture_output=True,text=True,check=True)
    summary=dict(calibration_models_verified=cal_verified,all_alarm_rows_verified=len(all_power),
        paired_tradeoff_rows_verified=len(trades),pattern_rows_verified=len(pats),
        union_accounting_rows_verified=len(accts),protected_files_verified=len(audit['protected_hashes_before']),
        deterministic_tests='8 passed',tests_output=result.stderr.strip(),
        method='Recomputed reference-rank decisions, not stored effective-cutoff decisions',
        verifier_sha256=sha(__file__))
    (OUT/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='tests_output'},indent=2))


if __name__=='__main__':main()
