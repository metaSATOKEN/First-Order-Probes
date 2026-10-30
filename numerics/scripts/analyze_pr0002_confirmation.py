#!/usr/bin/env python3
# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Saved-data audit of PR-0002. No simulation, threshold tuning, or archive code execution."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import csv
import sys
import time
import zipfile

import numpy as np
from scipy import stats
import fusion_confirmation_pr0002_v1 as spec_source

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('M', 'CR', 'B', 'J')
PAIRS = (('B','M'), ('B','CR'), ('J','M'), ('J','CR'), ('J','B'))


def digest(b):
    return hashlib.sha256(b).hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def interval(values, critical, bounds=(-1.,1.)):
    x = list(map(float, values))
    assert len(x) == 50
    mean = math.fsum(x)/50
    sd = math.sqrt(math.fsum((v-mean)**2 for v in x)/49)
    se = sd/math.sqrt(50)
    lo, hi = (mean-critical*se, mean+critical*se) if sd else (None,None)
    return dict(mean=mean, sd=sd, mcse=se, lower=lo, upper=hi,
                display_lower=None if lo is None else max(bounds[0],lo),
                display_upper=None if hi is None else min(bounds[1],hi),
                q10=float(np.quantile(x,.1)), q50=float(np.quantile(x,.5)), q90=float(np.quantile(x,.9)),
                positive=sum(v>0 for v in x), zero=sum(v==0 for v in x), negative=sum(v<0 for v in x))


def wilson(k,n):
    z = stats.norm.ppf(.975)
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return float(center-half),float(center+half)


def compare_rows(saved, expected):
    assert len(saved) == len(expected), (len(saved),len(expected))
    for a,b in zip(saved,expected):
        assert set(a) == set(b), (set(a),set(b))
        for k,v in b.items():
            raw = a[k]
            if v is None: assert raw == '', (k,raw)
            elif isinstance(v,bool): assert raw == str(v), (k,raw,v)
            elif isinstance(v,(int,float)):
                assert math.isclose(float(raw),v,rel_tol=2e-11,abs_tol=2e-13), (k,raw,v)
            else: assert raw == v, (k,raw,v)


def analyze(path, out):
    assert not out.exists(), 'Preserve completed outputs; choose a new version'
    start_time = time.monotonic()
    expected_manifest = json.loads((ROOT/'output/runs/pr0002_pilot_acceptance_v1/accepted_confirmation_manifest.json').read_text())
    manifest_hash = digest(canonical(expected_manifest).encode())
    assert manifest_hash == '9bb38b60e74851c50625eea64e4fa8801399d8a6426992a9b2bd9a4fcc7b0a0b'
    assert expected_manifest['specification'] == spec_source.specification('confirm')
    spec = expected_manifest['specification']
    conditions = spec['conditions']
    tasks = {t['id']:t for t in spec['tasks']}
    assert len(tasks)==1200 and sum(t['total'] for t in tasks.values())==10999800
    planned_paths = {f"chunks/{t['id']}/{s:09d}.npz" for t in tasks.values() for s in range(0,t['total'],2048)}
    assert len(planned_paths)==5800
    with zipfile.ZipFile(path) as z:
        members = z.infolist(); names = [m.filename for m in members]
        assert len(names)==len(set(names)) and sum(m.file_size for m in members)<8*1024**3
        for m in members:
            p=PurePosixPath(m.filename)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in m.filename
            assert str(p)==m.filename and not m.is_dir() and (m.external_attr>>16)&0o170000 != 0o120000
        index=json.loads(z.read('_archive_index.json'))
        assert set(index)==set(names)-{'_archive_index.json'}
        for i,(name,expected) in enumerate(index.items()):
            h=hashlib.sha256(); size=0
            with z.open(name) as file:
                for b in iter(lambda:file.read(1024*1024),b''):h.update(b);size+=len(b)
            assert size==expected['size'] and h.hexdigest()==expected['sha256'],name
            if i%1500==0:print('Archive files verified',i,'/',len(index),flush=True)
        manifest=json.loads(z.read('manifest.json'))
        assert manifest==expected_manifest
        state=json.loads(z.read('state.json'))
        assert state['status']=='COMPLETE' and state['committed_records']==state['planned_records']==10999800
        assert state['failures']==[]
        for name,sha in manifest['source_hashes'].items():assert digest(z.read('sources/'+name))==sha
        assert set(n for n in names if n.startswith('chunks/'))==planned_paths
        def saved_csv(name):return list(csv.DictReader(io.StringIO(z.read('analysis/'+name+'.csv').decode())))
        checks=json.loads(z.read('checks.json'))
        assert checks['status']=='PASS' and checks['environment']==manifest['environment']
        counts=Counter(); timings=[]; seen=set(); role_hashes=set(); negative=Counter()
        max_moment_error=0.; max_reverse_error=0.
        def load(cid,rep,role):
            nonlocal max_moment_error,max_reverse_error
            t=tasks[f'{rep:02d}/{cid}/{role}']; arrays=[]
            for s in range(0,t['total'],2048):
                name=f"chunks/{t['id']}/{s:09d}.npz"
                assert name not in seen;seen.add(name)
                with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as a:
                    assert set(a.files)=={'values','metadata'}
                    values=a['values'];meta=json.loads(str(a['metadata']))
                n=min(2048,t['total']-s)
                assert meta['task']==t and meta['start']==s and meta['count']==n
                assert meta['manifest_hash']==manifest_hash and values.shape==(n,14) and values.dtype==np.dtype('<f8')
                assert np.isfinite(values).all() and digest(values.tobytes())==meta['values_sha256']
                assert meta['generation_seconds']>0 and math.isfinite(meta['generation_seconds'])
                # An identical whole-chunk score payload across roles would merit investigation.
                assert meta['values_sha256'] not in role_hashes;role_hashes.add(meta['values_sha256'])
                p=conditions[cid]['params'];ai,aj=np.exp(-p['dt']/np.array([p['tau_i'],p['tau_j']]))
                lam,nu=(1+ai)*(1-aj)/(2*(1-ai*aj)),(1+aj)*(1-ai)/(2*(1-ai*aj))
                for j in range(3):
                    delta=values[:,3*j+2]-lam*values[:,3*j]-nu*values[:,3*j+1]-values[:,9+j]
                    assert np.max(np.abs(delta))<2e-12
                    max_moment_error=max(max_moment_error,float(np.max(np.abs(delta))))
                max_reverse_error=max(max_reverse_error,float(np.max(np.abs(values[:,13]))))
                negative['CR_negative']+=int(np.sum(values[:,12]<0))
                negative['full_vi_negative']+=int(np.sum(values[:,6]<0))
                negative['full_vj_negative']+=int(np.sum(values[:,7]<0))
                counts[role]+=n;timings.append(meta)
                arrays.append(np.column_stack([abs(values[:,11]),values[:,12]]))
            return np.concatenate(arrays)
        rates=[];pairs=[];patterns=[];thresholds=[];overlap=[];models=0;decisions=0
        for rep in range(50):
            for group in ('N_S','N_M','N_L','N_R10'):
                ref=load(group,rep,'reference');cal=load(group,rep,'calibration')
                ordered=np.sort(ref,axis=0)
                def ranking(x):return np.stack([np.searchsorted(ordered[:,j],x[:,j],side='left') for j in (0,1)],axis=1).astype(np.uint16)
                rcal=ranking(cal)
                # Sorted order statistics, independent of the runner's partition implementation.
                q=np.full((4,2),9999,dtype=int)
                q[0,0]=np.sort(rcal[:,0])[9900];q[1,1]=np.sort(rcal[:,1])[9900]
                q[2]=np.sort(rcal,axis=0)[9950];q[3]=np.sort(rcal.max(axis=1))[9900]
                model=dict(rank_cutoffs=q.tolist(),component_ties=[[int(sum(rcal[:,j]==q[i,j])) for j in (0,1)] for i in range(4)],joint_max_ties=int(sum(rcal.max(axis=1)==q[3,0])))
                assert model==json.loads(z.read(f'analysis/calibration/{rep:02d}_{group}.json'))
                models+=1
                for mi,m in enumerate(METHODS):
                    for j in (0,1):thresholds.append(dict(replicate=rep,group=group,method=m,component=j,cutoff=int(q[mi,j]),ties=model['component_ties'][mi][j]))
                # Restore declared insertion order from the trusted condition definition.
                for cid in spec_source.conditions():
                    c=conditions[cid]
                    if c['group']!=group:continue
                    role={'normal':'normal_eval','fault':'fault_eval','structural':'structural_eval'}[c['kind']]
                    x=load(cid,rep,role);r=ranking(x);n=len(r)
                    flags=np.stack([r[:,0]>q[0,0],r[:,1]>q[1,1],(r[:,0]>q[2,0])|(r[:,1]>q[2,1]),r.max(axis=1)>q[3,0]],axis=1)
                    bits=flags.astype(np.uint8)@np.array([1,2,4,8],dtype=np.uint8)
                    with np.load(io.BytesIO(z.read(f'analysis/decisions/{rep:02d}_{cid}.npz')),allow_pickle=False) as a:
                        np.testing.assert_array_equal(a['ranks'],r);np.testing.assert_array_equal(a['flags'],bits)
                    decisions+=1
                    hist=np.bincount(bits,minlength=16)
                    patterns.append(dict(replicate=rep,condition=cid,n=n,**{f'pattern_{j:02d}':int(hist[j]) for j in range(16)}))
                    if c['kind']=='normal':overlap.append(dict(replicate=rep,group=group,n=n,M_CR_both=int(sum(flags[:,0]&flags[:,1])),M_CR_union=int(sum(flags[:,0]|flags[:,1])),bonferroni_component_overlap=int(sum((r[:,0]>q[2,0])&(r[:,1]>q[2,1])))))
                    for mi,m in enumerate(METHODS):
                        k=int(sum(flags[:,mi]));lo,hi=wilson(k,n)
                        rates.append(dict(replicate=rep,condition=cid,group=group,kind=c['kind'],method=m,n=n,alarms=k,rate=k/n,wilson_lower=lo,wilson_upper=hi))
                    for a,d in PAIRS:
                        fa,fd=flags[:,METHODS.index(a)],flags[:,METHODS.index(d)]
                        gain=int(sum(fa&~fd));loss=int(sum(~fa&fd));both=int(sum(fa&fd));neither=int(sum(~fa&~fd))
                        assert gain+loss+both+neither==n and gain-loss==int(sum(fa))-int(sum(fd))
                        pairs.append(dict(replicate=rep,condition=cid,group=group,kind=c['kind'],A=a,D=d,n=n,both=both,gained=gain,lost=loss,neither=neither,difference=(gain-loss)/n))
            if rep%10==9:print('Raw-score recalibration and decisions verified',rep+1,'/ 50',flush=True)
        assert seen==planned_paths and sum(counts.values())==10999800
        tables=dict(rates_by_repeat=rates,paired_by_repeat=pairs,patterns=patterns,thresholds=thresholds,normal_tail_overlap=overlap)
        critical=float(stats.t.ppf(.975,49));cp=float(stats.t.ppf(1-.05/28,49));cf=float(stats.t.ppf(1-.05/16,49))
        summary=[];diffs=[];primary=[];fpr=[]
        for cid in spec_source.conditions():
            c=conditions[cid]
            for m in METHODS:
                rows=[r for r in rates if (r['condition'],r['method'])==(cid,m)]
                rec=dict(condition=cid,group=c['group'],kind=c['kind'],method=m,**interval([r['rate'] for r in rows],critical,(0.,1.)))
                summary.append(rec)
                if c['kind']=='normal':
                    lower=None if rec['sd']==0 else rec['mean']-cf*rec['mcse']
                    fpr.append(dict(rec,audit_lower=lower,above_nominal=None if lower is None else lower>.01))
            for a,d in PAIRS:
                rows=[r for r in pairs if (r['condition'],r['A'],r['D'])==(cid,a,d)]
                diffs.append(dict(condition=cid,A=a,D=d,gained_mean=math.fsum(r['gained']/r['n'] for r in rows)/50,lost_mean=math.fsum(r['lost']/r['n'] for r in rows)/50,**interval([r['difference'] for r in rows],critical)))
        for i,(cid,a,d,direction) in enumerate(spec['primary']):
            vals=[r['difference'] for r in pairs if (r['condition'],r['A'],r['D'])==(cid,a,d)]
            ci=interval(vals,cp)
            sign=0 if ci['lower'] is None else 1 if ci['lower']>0 else -1 if ci['upper']<0 else 0
            outcome='unresolved' if not sign else ('supported' if sign==direction else 'refuted') if direction else ('A_higher' if sign>0 else 'D_higher')
            primary.append(dict(id=f'C{i+1:02d}',condition=cid,A=a,D=d,expected=direction,outcome=outcome,**ci))
        threshold_summary=[]
        for group in ('N_S','N_M','N_L','N_R10'):
            for m in METHODS:
                for j in (0,1):
                    x=[r['cutoff'] for r in thresholds if (r['group'],r['method'],r['component'])==(group,m,j)]
                    threshold_summary.append(dict(group=group,method=m,component=j,**{f'q{k}':float(np.quantile(x,k/100)) for k in (10,50,90)}))
        tables.update(rates_summary=summary,differences_summary=diffs,primary_14=primary,fpr_audit_16=fpr,threshold_summary=threshold_summary)
        for name,rows in tables.items():compare_rows(saved_csv(name),rows)
        inference=json.loads(z.read('analysis/inference.json'))
        assert inference['H1_all_six_supported']==all(primary[i-1]['outcome']=='supported' for i in (1,2,5,6,9,10))
        for key in ('scientific_exclusions','technical_failure_events','unique_records_in_failed_attempts','failed_attempt_record_fraction'):assert inference[key]==0
        assert inference['planned_records']==10999800
        for name,rows in (('primary',primary),('fpr',fpr)):
            transformed=[{k:'' if v is None else str(v) for k,v in r.items()} for r in inference[name]]
            compare_rows(transformed,rows)
        assert models==200 and decisions==800
        assert len([n for n in names if n.startswith('analysis/calibration/')])==200
        assert len([n for n in names if n.startswith('analysis/decisions/')])==800
        out.mkdir(parents=True)
        # Retain byte-identical small source tables/plots; raw scores remain in the input ZIP.
        for n in names:
            if (n.startswith('analysis/') and n.count('/')==1) or n in ('manifest.json','state.json','checks.json'):
                p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
        generation=math.fsum(t['generation_seconds'] for t in timings)
        dates=sorted(t['at'] for t in timings)
        wall=(datetime.fromisoformat(state['updated_at'])-datetime.fromisoformat(dates[0])).total_seconds()
        extra=dict(
            main_hypotheses=dict(H1=all(primary[i-1]['outcome']=='supported' for i in (1,2,5,6,9,10)),H2=all(primary[i-1]['outcome']=='supported' for i in (3,4,11,12))),
            fpr_audit_flags=sum(r['above_nominal'] is True for r in fpr),
            total_alarms={f"{c}/{m}":sum(r['alarms'] for r in rates if (r['condition'],r['method'])==(c,m)) for c in conditions for m in METHODS},
            fpr_repeat_ranges=[dict(group=c,method=m,min=min(r['rate'] for r in rates if (r['condition'],r['method'])==(c,m)),max=max(r['rate'] for r in rates if (r['condition'],r['method'])==(c,m)),above_one_percent=sum(r['rate']>.01 for r in rates if (r['condition'],r['method'])==(c,m))) for c in ('N_S','N_M','N_L','N_R10') for m in METHODS],
            paired_totals=[dict(condition=c,A=a,D=d,**{k:sum(r[k] for r in pairs if (r['condition'],r['A'],r['D'])==(c,a,d)) for k in ('n','gained','lost','both','neither')}) for c in conditions for a,d in PAIRS],
            generation_seconds_by_condition={c:math.fsum(t['generation_seconds'] for t in timings if t['task']['condition']==c) for c in conditions})
        save_json(out/'recomputed_results.json',dict(tables=tables,diagnostics=extra))
        audit=dict(status='PASS',input_zip=str(path.relative_to(ROOT)),input_sha256=spec_source.sha_file(path),
                   manifest_sha256=manifest_hash,accepted_manifest_exact_match=True,source_hashes_exact_match=True,
                   members=len(names),indexed_files=len(index),zip_bytes=path.stat().st_size,uncompressed_bytes=sum(m.file_size for m in members),
                   records=sum(counts.values()),scalar_values=sum(counts.values())*14,counts_by_role=dict(counts),chunks=len(seen),unique_chunk_payload_hashes=len(role_hashes),
                   recalibrated_models=models,recomputed_decision_sets=decisions,recomputed_decision_records=7000000,
                   all_finite=True,max_moment_error=max_moment_error,max_reversal_moment_error=max_reverse_error,
                   retained_negative_value_counts=dict(negative),matched_csv_rows={k:len(v) for k,v in tables.items()},
                   critical_values=dict(pointwise95=critical,primary14=cp,fpr16=cf),state=state,
                   generation_seconds=generation,first_chunk_at=dates[0],last_chunk_at=dates[-1],first_chunk_to_final_state_seconds=wall,
                   audit_seconds=time.monotonic()-start_time,audit_environment=spec_source.environment(),colab_environment=manifest['environment'],
                   new_simulation_records=0,thresholds_retuned=False,independent_reviewer=False,
                   limitations=['CR waveform recalculation not possible for all records: only scalar outputs are stored.',
                                'Final ZIP retains cumulative sessions/time and last pause reason, not a full per-session event log.',
                                'Main-run peak RSS and full workspace peak are not stored.'])
        save_json(out/'validation.json',audit)
        print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('zip',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.zip.resolve(),args.output.resolve())
