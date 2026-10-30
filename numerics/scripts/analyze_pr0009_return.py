# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Read-only PR-0009 archive audit and saved-statistic reproduction. No simulation."""
from pathlib import Path, PurePosixPath
import hashlib, io, json, re, sys, time, zipfile
import numpy as np
from scipy.stats import beta, t
import pr0009_spec as spec
import pr0009_methods as m
from pr0009_storage import canonical
from pr0009_runner import verify_values
from pr0008_shadow_methods import independent_decide

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_two_axis_v1'
def sha(b): return hashlib.sha256(b).hexdigest()
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def equal(a,b,where):
    if isinstance(a,dict):
        assert set(a)==set(b),where
        for k in a:equal(a[k],b[k],where+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),where
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,where+'/'+str(i))
    elif isinstance(a,(float,np.floating)):
        assert np.isclose(a,b,rtol=2e-11,atol=2e-13,equal_nan=True), (where,a,b)
    else: assert a==b,(where,a,b)

class Store:
    def __init__(self):
        self.zips={};self.map={};self.hashes={};self.indices=[]
        for p in sorted((ROOT/'Pr0009').glob('*.zip')):
            z=zipfile.ZipFile(p);idx=json.loads(z.read('ARCHIVE_INDEX.json'))
            self.zips[p.name]=z;self.indices.append((idx['sequence'],p,idx))
        self.indices.sort(key=lambda a:a[0])
    def read(self,name):return self.zips[self.map[name]].read(name)
    def js(self,name):return json.loads(self.read(name))
    def npz(self,name):
        with np.load(io.BytesIO(self.read(name)),allow_pickle=False) as z:return {k:z[k] for k in z.files}
    def audit(self):
        archives=[];coverage=[];duplicates=0;members=0
        for seq,p,idx in self.indices:
            z=self.zips[p.name];names=z.namelist()
            assert len(names)==len(set(names))
            assert set(names)==set(idx['entries'])|{'ARCHIVE_INDEX.json'}
            assert p.stat().st_size<1_000_000_000
            assert idx['card_sha256']==spec.CARD_SHA and idx['cell_id']=='51187a0316767e6a'
            assert idx['run_id']=='9adfa7a8c6e0' and idx['experiment']==spec.EXPERIMENT
            for name,e in idx['entries'].items():
                q=PurePosixPath(name)
                assert not q.is_absolute() and '..' not in q.parts and '\\' not in name
                h=hashlib.sha256();size=0
                with z.open(name) as f:
                    for b in iter(lambda:f.read(1024**2),b''):h.update(b);size+=len(b)
                assert size==e['size'] and h.hexdigest()==e['sha256'],(p.name,name)
                if name in self.hashes and name.startswith(('sources/','carry/','shards/','analysis/')):
                    assert self.hashes[name]==h.hexdigest(),('immutable mismatch',name)
                    duplicates+=1
                self.hashes[name]=h.hexdigest();self.map[name]=p.name;members+=1
            with p.open('rb') as f:ph=hashlib.file_digest(f,'sha256').hexdigest()
            archives.append(dict(file=p.name,bytes=p.stat().st_size,sha256=ph,sequence=seq,status=idx['status'],members=len(idx['entries']),coverage=idx['covered_replicates']))
            if idx['status'].startswith('SHARD_'):coverage+=idx['covered_replicates']
            print('ARCHIVE',seq,p.name,len(idx['entries']),flush=True)
        assert sorted(coverage)==list(range(50)) and len(coverage)==50
        manifest=self.js('manifest.json')
        build=json.loads((ROOT/'numerics/colab/pr0009_two_axis_a50_build.json').read_text())
        # Every capsule source is matched to the frozen local build, not executed from ZIP.
        for name,h in manifest['sources'].items():assert self.hashes['sources/'+name]==h
        sys.path.insert(0,str(ROOT/'numerics/tools'))
        from build_pr0009_colab import PATHS
        assert manifest['sources']=={n:build['sources'][p] for n,p in PATHS.items()}
        for n,p in PATHS.items():assert sha((ROOT/p).read_bytes())==manifest['sources'][n]
        equal(spec.specification(),self.js('specification.json'),'specification')
        for a in self.js('catalog.json')['archives']:
            if a['name'] in self.zips:
                found=next(x for x in archives if x['file']==a['name'])
                assert found['sha256']==a['sha256'] and found['bytes']==a['bytes']
        dump(OUT/'receipt.json',dict(status='PASS_RECEIPT',archives=archives,indexed_member_occurrences=members,
            unique_files=len(self.map),immutable_duplicate_occurrences=duplicates,coverage=sorted(coverage),
            summary=self.js('summary.json'),sources_checked=len(manifest['sources'])))

def independent_counts(row,g,e,bits):
    # Histogram path, independent of worker's boolean reductions and pair cache.
    patterns=[];valid=[]
    for j in range(8):
        p,v=m.decode(bits[:,j]);patterns.append(p);valid.append(v)
        for ai in range(2):
            for mi in range(6):
                row['counts'][g,e,j,ai,mi]+=np.bincount(p[:,ai,mi][v[:,ai,mi]],minlength=8)
        row['available'][g,e,j]+=v.sum(0)
    for ci in range(7):
        choices=[0 if s else ci+1 for s in list(row['gate_states'][g,ci])+[row['same_states'][g,ci]]]
        for gi,j in enumerate(choices):
            for ri,k in enumerate((ci+1,0,1)):
                a=patterns[j]>0;b=patterns[k]>0;ok=valid[j]&valid[k]
                code=np.where(a,np.where(b,0,1),np.where(b,2,3))
                for ai in range(2):
                    for mi in range(6):row['paired'][g,ci,e,gi,ri,ai,mi]+=np.bincount(code[:,ai,mi][ok[:,ai,mi]],minlength=4)

def reproduce(s,rep):
    folder=f'shards/{rep//5:02d}/rep_{rep:02d}'
    read=lambda p:s.npz(folder+'/'+p)
    meta=s.js(f'carry/rep_{rep:02d}.json');expected=spec.tasks(rep)
    assert meta['replicate']==rep and meta['records']==620784
    assert meta['statistics_sha256']==s.hashes[f'carry/rep_{rep:02d}.npz']
    assert set(meta['files'])=={p[len(folder)+1:] for p in s.map if p.startswith(folder+'/')}
    for name,h in meta['files'].items():assert s.hashes[folder+'/'+name]==h
    sh=sha(canonical(spec.specification()).encode());assert meta['specification_sha256']==sh
    raw={};stats={'raw_records':0,'raw_chunks':0,'nonfinite_rows':0,'E_decisions':0,'V_decisions':0}
    for task in expected:
        parts=[]
        for start in range(0,task['count'],spec.CHUNK):
            a=read(f"chunks/{task['case']}/{task['role']}/{start:09d}.npz")
            v=a['values'];d=json.loads(str(a['metadata']));assert d['task']==task and d['start']==start
            assert d['count']==len(v)==min(spec.CHUNK,task['count']-start)
            assert d['specification_sha256']==sh and d['values_sha256']==sha(v.tobytes())
            bad=verify_values(v,task['case']);assert bad==d['unavailable_rows']
            stats['raw_records']+=len(v);stats['raw_chunks']+=1;stats['nonfinite_rows']+=bad;parts.append(v)
        raw[task['case'],task['role']]=np.concatenate(parts)
    assert len(raw)==100 and stats['raw_records']==620784
    models={};cals={};row=m.empty()
    def check_cal(path,model,data):
        stored=read(path);state,ordered=model
        md,cut,ids=m.calibrate(state,ordered,data)
        equal(md,json.loads(str(stored['metadata'])),path)
        assert np.allclose(cut,stored['cut'],rtol=2e-11,atol=2e-13,equal_nan=True)
        assert np.array_equal(ids,stored['indices']) and str(stored['input_sha256'])==sha(data.tobytes())
        return json.loads(str(stored['metadata'])),stored['cut']
    for base in ('B_M','B_L'):
        data=raw[base,'ref'];state,ordered=m.fit(data);saved=read('models/'+base+'.npz')
        equal(state,json.loads(str(saved['state'])),'model')
        assert np.allclose(ordered,saved['ordered'],rtol=2e-11,atol=2e-13)
        assert str(saved['input_sha256'])==sha(data.tobytes())
        models[base]=(json.loads(str(saved['state'])),saved['ordered'])
        cals[base]=check_cal('old_cal/'+base+'.npz',models[base],raw[base,'old_cal'])
    for g,group in enumerate(spec.groups()):
        model=models[spec.baseline(group)];old=cals[spec.baseline(group)]
        for role in ('audit','update'):
            normal=raw[group,role+'_normal'];orders=read('orders/'+group+'_'+role+'.npz')
            order=orders['normal_order'];pos=orders['replacement_order']
            assert np.array_equal(order,spec.order(group,role+'_order',rep,len(normal)))
            assert np.array_equal(pos,spec.order(group,role+'_replace',rep,len(normal)))
            for ci,(f,n) in enumerate(spec.CANDIDATES):
                data=normal[order].copy();ids=order.copy();origin=np.zeros(len(data),np.uint8)
                if n:data[pos[:n]]=raw[group+'__'+f,role+'_fault'][:n];ids[pos[:n]]=np.arange(n);origin[pos[:n]]=1
                a=read(f'{role}/{group}/candidate_{ci}.npz');p=json.loads(str(a['provenance']))
                assert np.array_equal(ids,a['ids']) and np.array_equal(origin,a['origin'])
                assert p['input_sha256']==sha(data.tobytes()) and p['replacement_positions']==pos[:n].tolist()
                assert p['normal_task']==f'{rep:02d}/{group}/{role}_normal'
                assert p['fault_task']==(f'{rep:02d}/{group}__{f}/{role}_fault' if n else None)
                bit=independent_decide(data,*model,*old);pat,val=m.decode(bit)
                q=data[:,[11,12]];T=np.sqrt(len(q))*np.mean(q,axis=0)/np.std(q,axis=0,ddof=1)
                scores=np.array([(pat[:,0,0]>0).mean(),abs(T[0]),max(abs(T[0]),T[1])])
                assert val.all() and np.allclose(scores,a['scores'],rtol=2e-11,atol=2e-13)
                states=(scores>spec.GATE_CUTS[0 if group.endswith('_M') else 1]).astype(int)
                assert np.array_equal(states,a['states'])
                if role=='audit':row['gate_states'][g,ci]=states
                else:
                    row['same_states'][g,ci]=states[2]
                    cals[group,ci]=check_cal(f'calibration/{group}/candidate_{ci}.npz',model,data)
        for j,cal in enumerate([old]+[cals[group,c] for c in range(7)]):
            if cal[0]['valid']:row['inactive'][g,j]=cal[0]['lf0_structurally_inactive']
    frozen=s.js(folder+'/CALIBRATION_FROZEN.json')
    assert set(frozen)=={p for p in meta['files'] if p.split('/')[0] in ('models','old_cal','orders','audit','update','calibration')}
    for p,h in frozen.items():assert s.hashes[folder+'/'+p]==h
    freeze_hash=sha(canonical(frozen).encode());vbits={}
    for g,group in enumerate(spec.groups()):
        model=models[spec.baseline(group)];old=cals[spec.baseline(group)]
        for kind in ('V','E'):
            for ei,f in enumerate(('normal',)+spec.FAULTS):
                case=group if not ei else group+'__'+f
                role=('validation_normal' if not ei else 'validation_fault') if kind=='V' else ('null_eval' if not ei else 'fault_eval')
                values=raw[case,role];cs=[old] if kind=='V' else [old]+[cals[group,c] for c in range(7)];parts=[]
                task=next(t for t in expected if t['case']==case and t['role']==role)
                for start in range(0,len(values),spec.CHUNK):
                    v=values[start:start+spec.CHUNK];path=('validation' if kind=='V' else 'decisions')+f'/{case}/{start:09d}.npz'
                    a=read(path);bits=a['packed'];d=json.loads(str(a['metadata']))
                    assert d==dict(task=task,start=start,count=len(v),calibration_sha256=freeze_hash,
                        raw_sha256=sha(v.tobytes()),packed_sha256=sha(bits.tobytes()))
                    new=np.stack([independent_decide(v,*model,*cal) for cal in cs],axis=1)
                    assert np.array_equal(new,bits),('bits',rep,path)
                    stats[kind+'_decisions']+=bits.shape[0]*bits.shape[1]*12
                    if kind=='E':independent_counts(row,g,ei,bits)
                    else:parts.append(bits[:,0])
                if kind=='V':vbits[group,f]=np.concatenate(parts)
        pos=read('validation/'+group+'_positions.npz')['positions']
        assert np.array_equal(pos,spec.order(group,'validation_replace',rep,20000)[:1000])
        for vi,f in enumerate(('normal',)+spec.FAULTS):
            bits=vbits[group,'normal'].copy()
            if vi:bits[pos]=vbits[group,f]
            p,v=m.decode(bits);j=0 if not vi else vi+1
            row['v_counts'][g,j,...,0]=((p>0)&v).sum(0);row['v_counts'][g,j,...,1]=v.sum(0)
    for g,group in enumerate(spec.groups()):
        if not group.startswith('B_'):
            p,v=m.decode(vbits[spec.baseline(group),'normal'])
            row['v_counts'][g,1,...,0]=((p>0)&v).sum(0);row['v_counts'][g,1,...,1]=v.sum(0)
    saved=s.npz(f'carry/rep_{rep:02d}.npz')
    for k in row:assert np.array_equal(row[k],saved[k]),('carry',rep,k)
    m.validate(row);np.savez_compressed(OUT/'recomputed'/f'rep_{rep:02d}.npz',**row)
    return row,stats

def main():
    assert not (OUT/'validation.json').exists(),'Use a new version for changed completed analyses'
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'recomputed').mkdir(exist_ok=True)
    t0=time.monotonic();anchor=json.loads((ROOT/'audits/verification/2026-10-18_PR-0009_implementation.json').read_text())
    anchors={**anchor['outputs'],**anchor['current_navigation_hashes']}
    for p,h in re.findall(r'\| `([^`]+)` \| `([a-f0-9]{64})` \|',(ROOT/'HANDOFF/2026-10-18_session-02.md').read_text()):anchors.setdefault(p,h)
    for p,h in anchors.items():assert sha((ROOT/p).read_bytes())==h,p
    dump(OUT/'start_snapshot.json',dict(anchors=anchors,ledger_bytes=(ROOT/'audits/ledger_2026-10.md').stat().st_size))
    s=Store();s.audit();rows=[];checks=[]
    for rep in range(50):
        row,st=reproduce(s,rep);rows.append(row);checks.append(dict(replicate=rep,**st))
        print('REPRODUCED',rep+1,'/50',round(time.monotonic()-t0,1),'sec',flush=True)
    # Aggregate the now independently reconstructed integer cells.
    agg=m.aggregate(rows)
    for key in ('primary18','gate_results','clean_gate_summary','validation_evidence','scope'):
        equal(agg[key],s.js('analysis/'+key+'.json'),key);dump(OUT/(key+'.json'),agg[key])
    # Large serialized tables checked one at a time; never retain physical waveforms.
    for key in ('policy_results','paired_tradeoffs'):
        equal(agg[key],s.js('analysis/'+key+'.json'),key)
    sessions={n:s.js(n) for n in s.map if n.startswith('sessions/') and n.endswith('.json') and '/preflight/' not in n}
    preflight=[s.js(n) for n in s.map if n.endswith('/preflight/result.json')]
    assert all(p['passed'] and p['common_calibration_decisions_equal'] for p in preflight)
    assert not any(n.endswith('worker_error.json') for n in s.map)
    dump(OUT/'runtime_evidence.json',dict(sessions=sessions,preflight=preflight,active=s.js('active_time.json'),
        delivery=s.js('delivery_status.json') if 'delivery_status.json' in s.map else None))
    dump(OUT/'validation.json',dict(status='PASS_SAVED_DATA_REPRODUCTION',per_replicate=checks,
        totals={k:sum(c[k] for c in checks) for k in checks[0] if k!='replicate'},
        elapsed_seconds=time.monotonic()-t0,anchors_checked=len(anchors),
        source_sha256=sha(Path(__file__).read_bytes()),independent_reviewer=False,
        verification='Same agent; component-wise projection, histogram/pair path, frozen reference/calibration reproduction',
        science_records_generated_locally=0,manuscript_edits=0,
        scope_gaps=['Original time-series not stored or regenerated','No real VM-loss restoration test','No physical experiment or external normal-provenance validation']))
    print('PASS',flush=True)
if __name__=='__main__':main()
