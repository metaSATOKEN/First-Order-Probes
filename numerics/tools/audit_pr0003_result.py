# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""PR-0003 returned-data audit. Never imports code from the result ZIP.

No source simulation, CUDA execution, or timing rerun. Existing trace arrays
are checked with an explicit recurrence and moments are recomputed from NPZ.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics as stats
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
# Trusted local, frozen sources; import does not start a worker.
import probe_backend_check_v1 as frozen
import probe_check_io as transport

ZIP = ROOT/'probe_cpu_gpu_check_v1_23538c370862_PILOT_COMPLETE_b613933e.zip'
OUT = ROOT/'numerics/analysis/pr0003_cpu_gpu_23538c370862_v1'


def compare(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==np.float64
    assert np.isfinite(a).all() and np.isfinite(b).all()
    d = np.abs(a-b)
    s = d/(2e-12+2e-12*np.abs(a))
    return dict(passed=bool(np.all(s<=1)),max_abs=float(d.max()),max_scaled=float(s.max()),
                exceed=int(np.sum(s>1)),
                max_abs_by_column=np.max(d,axis=0).tolist() if a.ndim==2 and a.shape[1]==17 else None,
                max_scaled_by_column=np.max(s,axis=0).tolist() if a.ndim==2 and a.shape[1]==17 else None,
                exceed_by_column=np.sum(s>1,axis=0).tolist() if a.ndim==2 and a.shape[1]==17 else None)


def filtered(a,pole):
    previous = np.zeros(a.shape[0],dtype=np.float64)
    out = np.empty_like(a)
    for k in range(a.shape[1]):
        previous = (1-pole)*a[:,k]+pole*previous
        out[:,k] = previous
    return out


def open_npz_bytes(raw):
    # Header checking consumes the input stream. NumPy receives a FRESH buffer.
    # The frozen worker's path-based open_values is not a BytesIO API.
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        assert len(names)==len(set(names)) and 'values.npy' in names
        assert set(names)<={'values.npy','yi.npy','yj.npy','A.npy','B.npy','Ac.npy','Bc.npy'}
        assert sum(i.file_size for i in infos)<=128*1024**2
        for item in infos:
            with archive.open(item) as stream:
                version=np.lib.format.read_magic(stream)
                reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}[version]
                shape,_,dtype=reader(stream)
                assert dtype==np.float64 and len(shape)==2 and min(shape)>0
                assert item.file_size-stream.tell()==math.prod(shape)*8
    return np.load(io.BytesIO(raw),allow_pickle=False)


def audit():
    manifest = json.loads((ROOT/'numerics/colab/probe_cpu_gpu_check_build.json').read_text())
    for path,h in manifest['sources'].items():
        assert transport.digest(ROOT/path)==h, ('changed local source',path)
    archive_index = transport.inspect_zip(ZIP,manifest['cell_id'])
    assert archive_index['run_id']=='23538c370862'
    assert archive_index['status']=='PILOT_COMPLETE'
    with zipfile.ZipFile(ZIP) as archive:
        def js(name): return json.loads(archive.read(name))
        spec = js('specification.json')
        assert spec==frozen.specification()
        m = js('manifest.json')
        for name,h in m['sources'].items():
            assert hashlib.sha256(archive.read('sources/'+name)).hexdigest()==h
        source_paths = {'probe_check_io.py':'numerics/scripts/probe_check_io.py',
                        'probe_check_launcher.py':'numerics/scripts/probe_check_launcher.py',
                        'probe_backend_check_v1.py':'numerics/scripts/probe_backend_check_v1.py',
                        'PR-0003.md':'audits/prereg/PR-0003_cpu_gpu_backend_pilot.md',
                        'LICENSE':'LICENSE','LICENSE-CC-BY-4.0.txt':'LICENSE-CC-BY-4.0.txt'}
        assert set(m['sources'])==set(source_paths)
        assert all(archive.read('sources/'+n)==(ROOT/p).read_bytes() for n,p in source_paths.items())
        expected = {t['id']:t for t in spec['tasks']}
        dones = {name.split('/')[1]:js(name) for name in archive.namelist() if name.endswith('/done.json')}
        assert set(dones)==set(expected)
        values = {}
        numeric,derived,trace_checks = [],[],[]
        expected_npzs = set()
        for name,task in expected.items():
            row = dones[name]
            assert row['task']==task and set(row['files'])=={'cpu.npz','gpu.npz'}
            prefix = 'tasks/'+name+'/'+row['attempt']+'/'
            arrays = {}
            for backend in ('cpu','gpu'):
                path = prefix+backend+'.npz'; expected_npzs.add(path)
                raw = archive.read(path)
                assert hashlib.sha256(raw).hexdigest()==row['files'][backend+'.npz']
                with open_npz_bytes(raw) as data:
                    arrays[backend] = {k:data[k] for k in data.files}
                v = arrays[backend]['values']
                assert v.shape==(task['count'],17) and v.dtype==np.float64 and np.isfinite(v).all()
                values[(name,backend)] = v
                p = spec['conditions'][task['case']]
                ai,aj = np.exp(-p['dt']/np.array([p['tau_i'],p['tau_j']]))
                lam = (1+ai)*(1-aj)/(2*(1-ai*aj)); nu = (1+aj)*(1-ai)/(2*(1-ai*aj))
                for k in range(3):
                    calc = (v[:,k*3+2]-lam*v[:,k*3]-nu*v[:,k*3+1])[:,None]
                    derived.append(dict(task=name,backend=backend,test='M-'+str(k),**compare(v[:,9+k:10+k],calc)))
                cr = compare(v[:,12:13],(v[:,14]+v[:,15]-2*v[:,16])[:,None])
                assert cr==row['comparison']['cr_identity'][backend]
                derived.append(dict(task=name,backend=backend,test='CR-expansion',**cr))
                derived.append(dict(task=name,backend=backend,test='diag-full-auto',**compare(v[:,3:5],v[:,6:8])))
                derived.append(dict(task=name,backend=backend,test='diag-none-cross',**compare(v[:,5:6],v[:,2:3])))
                if task['phase']=='check':
                    t = arrays[backend]
                    n = round(p['T']/p['dt']); w = math.ceil(20*max(p['tau_i'],p['tau_j'])/p['dt'])
                    assert set(t)=={'values','yi','yj','A','B','Ac','Bc'}
                    for field,pole,source in [('A',aj,'yi'),('B',ai,'yj')]:
                        trace_checks.append(dict(task=name,backend=backend,test=field+'-recurrence',
                                                 **compare(t[field],filtered(t[source],pole))))
                    def moments(a,b):
                        a,b=a[:,w:w+n],b[:,w:w+n]
                        return np.column_stack([(a*a).mean(axis=1),(b*b).mean(axis=1),(a*b).mean(axis=1)])
                    for label,actual,computed in [
                        ('O',v[:,:3],moments(t['yi'],t['yj'])),
                        ('P',v[:,14:],moments(t['A'],t['B'])-moments(t['Ac'],t['Bc'])),
                        ('CR-trace',v[:,12:13],(((t['A'][:,w:w+n]-t['B'][:,w:w+n])**2).mean(axis=1)
                            -((t['Ac'][:,w:w+n]-t['Bc'][:,w:w+n])**2).mean(axis=1))[:,None])]:
                        trace_checks.append(dict(task=name,backend=backend,test=label,**compare(actual,computed)))
            assert set(arrays['cpu'])==set(arrays['gpu'])
            for key,a in arrays['cpu'].items():
                c = compare(a,arrays['gpu'][key])
                assert c==row['comparison']['arrays'][key], (name,key)
                numeric.append(dict(task=name,array=key,**c))
        assert {n for n in archive.namelist() if n.endswith('.npz')}==expected_npzs
        batch_checks = []
        for task in expected.values():
            if task['phase']=='time' and task['batch']!=64:
                ref = f"time-{task['case']}-r{task['repeat']}-b64"
                for backend in ('cpu','gpu'):
                    batch_checks.append(dict(task=task['id'],backend=backend,
                        comparison=compare(values[(ref,backend)],values[(task['id'],backend)])))
        original = js('summary.json')
        assert sorted(batch_checks,key=lambda x:(x['task'],x['backend']))==sorted(original['batch_comparisons'],key=lambda x:(x['task'],x['backend']))
        assert all(x['passed'] for x in numeric+derived+trace_checks)
        assert all(x['comparison']['passed'] for x in batch_checks)
        environments = {n:js(n) for n in archive.namelist() if n.endswith('/environment.json')}
        env_ids = {transport.sha(json.dumps(e,sort_keys=True).encode())[:12] for e in environments.values()}
        assert all(r['environment_id'] in env_ids for r in dones.values())
        groups = {}
        for t in expected.values():
            if t['phase']=='time':
                row=dones[t['id']]
                key=f"{row['environment_id']} / {t['case']} / batch={t['batch']}"
                groups.setdefault(key,[]).append(row)
        timing = {}
        for key,rows in groups.items():
            saved = original['timing_by_environment'][key]
            assert len(rows)==saved['rounds']==3 and saved['full_three_rounds']
            cpu64 = [dones[f"time-{r['task']['case']}-r{r['task']['repeat']}-b64"] for r in rows]
            ratios = {}
            for metric in ('compute_seconds','end_to_end_seconds'):
                r=[x['timing']['gpu'][metric]/x['timing']['cpu'][metric] for x in rows]
                assert r==saved[metric]['gpu_over_cpu'] and stats.median(r)==saved[metric]['median_ratio']
                ratios[metric]=dict(median_gpu_over_cpu=stats.median(r),speedup=1/stats.median(r),min=min(r),max=max(r))
            r=[x['timing']['gpu']['end_to_end_seconds']/c['timing']['cpu']['end_to_end_seconds'] for x,c in zip(rows,cpu64)]
            assert r==saved['end_to_end_vs_cpu64']['gpu_over_cpu64']
            assert stats.median(r)==saved['end_to_end_vs_cpu64']['median_ratio']
            ratios['versus_cpu64']=dict(median_gpu_over_cpu=stats.median(r),speedup=1/stats.median(r),min=min(r),max=max(r))
            stages = {b:{k:stats.median(r['timing'][b][k] for r in rows) for k in (
                'input_seconds','h2d_seconds','compute_seconds','gpu_event_seconds','d2h_seconds','save_seconds','end_to_end_seconds')}
                for b in ('cpu','gpu')}
            share = {k:stats.median(r['timing']['gpu'][k]/r['timing']['gpu']['end_to_end_seconds'] for r in rows)
                     for k in ('input_seconds','h2d_seconds','compute_seconds','d2h_seconds','save_seconds')}
            timing[key]=dict(case=rows[0]['task']['case'],batch=rows[0]['task']['batch'],ratios=ratios,median_seconds=stages,gpu_stage_fraction=share)
        assert set(timing)==set(original['timing_by_environment'])
        errors=[n for n in archive.namelist() if n.endswith(('error.json','audit_error.json'))]
        sessions={n:js(n) for n in archive.namelist() if n.endswith('/session.json')}
        launches={n:js(n) for n in archive.namelist() if n.endswith('/launch.json')}
        assert not errors and len(sessions)==len(launches)==1
        assert all(v['status']=='PILOT_COMPLETE' and v['returncode']==0 for v in launches.values())
        assert sum(t['count'] for t in expected.values())==295286
        assert original['completed_tasks']==original['planned_tasks']==94 and original['numeric_pass']
        results = dict(status='PASS',raw_zip=dict(path=ZIP.name,bytes=ZIP.stat().st_size,sha256=transport.digest(ZIP),
            members=len(archive.namelist()),hashed_members=len(archive_index['entries']),expanded_bytes=sum(x.file_size for x in archive.infolist())),
            run_id=m['run_id'],cell_id=m['cell_id'],tasks=94,check_tasks=22,time_tasks=72,
            records_per_backend=295286,distinct_seeded_records=98678,scalars_per_backend=295286*17,
            numerical=dict(comparisons=len(numeric),max_abs=max(x['max_abs'] for x in numeric),
                           max_scaled=max(x['max_scaled'] for x in numeric),
                           array_exact_value_equal=sum(x['max_abs']==0 for x in numeric),
                           max_trace_difference=max(x['max_abs'] for x in numeric if x['array']!='values'),
                           derived_checks=len(derived),trace_checks=len(trace_checks),batch_checks=len(batch_checks),
                           max_trace_recompute=max(x['max_abs'] for x in trace_checks)),
            timing=timing,environments=environments,sessions=sessions,launches=launches,
            bootstrap={n:js(n) for n in archive.namelist() if n.endswith('/bootstrap.json')},
            warmup={n:js(n) for n in archive.namelist() if n.endswith('/warmup.json')},
            max_gpu_reserved_bytes=max(r['timing']['gpu']['gpu_pool_reserved_bytes_after'] for r in dones.values()),
            task_total_seconds={b:sum(r['timing'][b]['end_to_end_seconds'] for r in dones.values()) for b in ('cpu','gpu')},
            errors=errors,new_simulation=False,new_gpu_execution=False,
            limitations=['Raw RNG/control arrays not saved; no independent source regeneration.',
                         'Three technical repeats in one runtime, not independent hardware trials.',
                         'Fresh launch only; actual Colab resume/restore not exercised.',
                         'Archive creation/final browser download duration not measured.'],
            detailed_numeric=numeric,derived_checks=derived,trace_checks=trace_checks,batch_checks=batch_checks)
        assert results['numerical']['max_abs']==original['numerical_summary']['max_abs']
        assert results['numerical']['max_scaled']==original['numerical_summary']['max_scaled']
        OUT.mkdir(parents=True,exist_ok=True)
        transport.write(OUT/'audit_results.json',results)
        print(json.dumps({k:v for k,v in results.items() if k in ('status','raw_zip','run_id','cell_id','tasks','records_per_backend','distinct_seeded_records','scalars_per_backend','numerical','max_gpu_reserved_bytes','task_total_seconds')},indent=2))
        print('GPU 4096 input fraction range:',min(v['gpu_stage_fraction']['input_seconds'] for v in timing.values() if v['batch']==4096),max(v['gpu_stage_fraction']['input_seconds'] for v in timing.values() if v['batch']==4096))


def delivery():
    """Preservation, report transcription and links; no prior report overwritten."""
    result=json.loads((OUT/'audit_results.json').read_text())
    checked=[]
    def check(name,condition):
        checked.append(dict(name=name,passed=bool(condition)))
    baseline=(ROOT/'audits/verification/2026-10-17_setup_baseline.sha256').read_text().splitlines()
    for line in baseline:
        h,p=line.split('  ',1)
        check('protected:'+p,transport.digest(ROOT/p)==h)
    handoff=(ROOT/'HANDOFF/2026-10-17_session-01.md').read_text()
    anchors=re.findall(r'\| `([^`]+)` \| `([a-f0-9]{64})` \|',handoff)
    for p,h in anchors: check('handoff:'+p,transport.digest(ROOT/p)==h)
    frozen_manifest=json.loads((ROOT/'numerics/colab/probe_cpu_gpu_check_build.json').read_text())
    for p,h in {**frozen_manifest['sources'],**frozen_manifest['artifacts']}.items():
        check('frozen:'+p,transport.digest(ROOT/p)==h)
    check('raw_zip_unchanged',transport.digest(ZIP)==result['raw_zip']['sha256'])
    check('ledger_prefix',transport.sha((ROOT/'audits/ledger_2026-10.md').read_bytes()[:64384])==
          'c88df93d6327245162e3c13bf683838fba8f015e424ac37b444e52dcbf79eb53')
    check('corpus_prefix',transport.sha((ROOT/'audits/fp_corpus.md').read_bytes()[:4746])==
          '84443ae31857393622e505212ab3ee06bbb86ad39bb36e9b9670ba360e602b5e')
    labels=[('B_S','基準・短窓0.512秒'),('B_M','基準・中窓2.048秒'),('B_L','基準・長窓8.192秒'),
            ('B_R10','時定数比r=10・中窓'),('K2_L','κ=2・長窓'),('K200_L','κ=200・長窓'),
            ('PRIVATE_M','私的入力0.02・中窓'),('DELAY_PLUS','遅延+2標本・中窓')]
    timings={(v['case'],v['batch']):v for v in result['timing'].values()}
    text=(OUT/'AUDIT_REPORT_JA.md').read_text()
    for c,label in labels:
        a,b,d=timings[c,64],timings[c,512],timings[c,4096]
        expected=f"| {label} | {a['median_seconds']['cpu']['end_to_end_seconds']:.4f} | {b['median_seconds']['gpu']['end_to_end_seconds']:.4f} | {d['median_seconds']['gpu']['end_to_end_seconds']:.4f} | {d['ratios']['versus_cpu64']['speedup']:.3f} | {d['ratios']['end_to_end_seconds']['speedup']:.3f} | {d['ratios']['compute_seconds']['speedup']:.1f} |"
        check('timing_table:'+c,expected in text)
    paths=[OUT/'AUDIT_REPORT_JA.md',ROOT/'audits/README.md',ROOT/'audits/prereg/README.md',
           ROOT/'HANDOFF/LATEST.md',ROOT/'audits/fp_corpus.md',ROOT/'audits/ledger_2026-10.md']
    links=[]
    for p in paths:
        for ref in re.findall(r'\[[^\]]*\]\(([^)\n]+)\)',p.read_text()):
            ref=ref.split('#',1)[0].strip('<>')
            if not ref or re.match(r'^\w+://',ref): continue
            links.append((str(p.relative_to(ROOT)),ref,(p.parent/ref).exists()))
    check('local_links',all(x[2] for x in links))
    check('asset_counts',len(baseline)==124 and len(anchors)==8)
    report=dict(status='PASS' if all(x['passed'] for x in checked) else 'FAIL',checks=checked,
                protected_assets=len(baseline),anchors=len(anchors),links=len(links),
                archive_sha256=result['raw_zip']['sha256'],new_simulation=False,gpu_executed_locally=False,
                sources={str(p.relative_to(ROOT)):transport.digest(p) for p in [
                    Path(__file__),OUT/'AUDIT_REPORT_JA.md',OUT/'audit_results.json']})
    transport.write(OUT/'delivery_validation.json',report)
    print('Delivery:',report['status'],'links',len(links))
    assert report['status']=='PASS',[x for x in checked if not x['passed']]


if __name__=='__main__':
    if '--delivery-only' not in sys.argv:
        audit()
    delivery()
