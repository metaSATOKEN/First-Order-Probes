# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""PR-0012 read-only receipt and saved-statistic reconstruction; no simulation."""
from pathlib import Path,PurePosixPath
import hashlib,io,json,re,stat,sys,time,zipfile
import numpy as np
from scipy.stats import binom
import pr0012_spec as spec
import pr0012_methods as m
from pr0012_storage import canonical
from pr0012_runner import verify_values
from analyze_pr0009_return import equal,dump
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0012_64bf0f2c7697_confirmation_v1'
RUN='64bf0f2c7697';CELL='d41a11833edd6d21'
def sha(b):return hashlib.sha256(b).hexdigest()

class Store:
    def __init__(self):
        self.zips={};self.map={};self.hashes={};self.indices=[]
        for p in sorted((ROOT/'pr0012').glob('*.zip')):
            z=zipfile.ZipFile(p);idx=json.loads(z.read('ARCHIVE_INDEX.json'))
            self.zips[p.name]=z;self.indices.append((idx['sequence'],p,idx))
        self.indices.sort(key=lambda a:a[0])
    def read(self,name):return self.zips[self.map[name]].read(name)
    def js(self,name):return json.loads(self.read(name))
    def npz(self,name):
        with np.load(io.BytesIO(self.read(name)),allow_pickle=False) as z:return {k:z[k] for k in z.files}
    def audit(self):
        records=[];coverage=[];duplicates=0;members=0
        assert [x[0] for x in self.indices]==list(range(11))
        for seq,p,idx in self.indices:
            z=self.zips[p.name];names=z.namelist();assert len(names)==len(set(names))
            assert set(names)==set(idx['entries'])|{'ARCHIVE_INDEX.json'}
            assert p.stat().st_size<1_000_000_000
            assert idx['cell_id']==CELL and idx['run_id']==RUN and idx['experiment']==spec.EXPERIMENT
            assert idx['card_sha256']==spec.CARD_SHA
            for name,e in idx['entries'].items():
                q=PurePosixPath(name);assert not q.is_absolute() and '..' not in q.parts and '\\' not in name
                assert not stat.S_ISLNK(z.getinfo(name).external_attr>>16)
                h=hashlib.sha256();size=0
                with z.open(name) as f:
                    for b in iter(lambda:f.read(1024**2),b''):h.update(b);size+=len(b)
                assert size==e['size'] and h.hexdigest()==e['sha256'],(p.name,name)
                if name in self.hashes and name.startswith(('sources/','shards/','carry/','analysis/')):
                    assert self.hashes[name]==h.hexdigest(),('immutable changed',name);duplicates+=1
                self.hashes[name]=h.hexdigest();self.map[name]=p.name;members+=1
            with p.open('rb') as f:ph=hashlib.file_digest(f,'sha256').hexdigest()
            records.append(dict(file=p.name,bytes=p.stat().st_size,sha256=ph,sequence=seq,status=idx['status'],
                members=len(idx['entries']),uncompressed_bytes=idx['uncompressed_bytes'],coverage=idx['covered_replicates']))
            if seq<10:
                assert idx['status']==f'SHARD_{seq+1:02d}_COMPLETE'
                assert idx['covered_replicates']==list(range(5*seq,5*seq+5));coverage+=idx['covered_replicates']
            else:assert idx['status']=='COMPLETE_INDEX'
            print('ARCHIVE',seq+1,'/11 verified',p.stat().st_size,flush=True)
        assert coverage==list(range(50))
        sys.path.insert(0,str(ROOT/'numerics/tools'));from build_pr0012_fix1_colab import PATHS
        manifest=self.js('manifest.json');assert manifest['run_id']==RUN and manifest['cell_id']==CELL
        assert manifest['sources']=={n:sha((ROOT/p).read_bytes()) for n,p in PATHS.items()}
        for n,h in manifest['sources'].items():assert self.hashes['sources/'+n]==h
        equal(spec.specification(),self.js('specification.json'),'spec')
        state=self.js('state.json');assert state['status']=='COMPLETE' and sorted(state['completed'])==[f'{i:02d}' for i in range(50)]
        for i in range(50):assert state['completed'][f'{i:02d}']['metadata_sha256']==self.hashes[f'carry/rep_{i:02d}.json']
        for a in self.js('catalog.json')['archives']:
            found=next(x for x in records if x['file']==a['name']);assert found['sha256']==a['sha256'] and found['bytes']==a['bytes']
        sessions={n:self.js(n) for n in self.map if n.startswith('sessions/') and n.endswith('.json') and '/preflight/' not in n}
        preflight=[self.js(n) for n in self.map if n.endswith('/preflight/result.json')]
        assert len(preflight)==1 and preflight[0]['passed'] and preflight[0]['readout']['passed']
        for row in preflight[0]['rows']:assert all(v['passed'] for v in row['checks'].values())
        assert not any(n.endswith(('worker_error.json','launcher_error.json','archive_error.json')) for n in self.map)
        dump(OUT/'runtime_evidence.json',dict(sessions=sessions,preflight=preflight,active=self.js('active_time.json'),delivery=self.js('delivery_status.json')))
        dump(OUT/'receipt.json',dict(status='PASS_RECEIPT',archives=records,indexed_member_occurrences=members,
            unique_files=len(self.map),immutable_duplicate_occurrences=duplicates,coverage=coverage,
            summary=self.js('summary.json'),sources_checked=len(manifest['sources'])))

def independent_decide(values,model,cal,method):
    state,ordered=model;meta,cut=cal
    assert state['valid'] and meta['valid'] and np.isfinite(values).all()
    q=values[:,[11,12]]-state['model']['mean'];w=np.asarray(state['model']['weights'])
    s=np.column_stack((q[:,0]*w[0,0]+q[:,1]*w[0,1],q[:,0]*w[1,0]+q[:,1]*w[1,1]))
    out=np.full((len(values),2),8,np.uint8)
    for ai in range(2):
        if method==0:tests=(s[:,0]>cut[ai,0,0],-s[:,0]>cut[ai,0,1],s[:,1]>cut[ai,0,2])
        else:
            k=int(cut[ai,1,0]);tests=[np.zeros(len(values),bool) for _ in range(3)]
            if k<len(ordered):
                if 0 not in state['model']['disabled']:tests[0]=s[:,0]>ordered[k,0];tests[1]=-s[:,0]>ordered[k,0]
                if 1 not in state['model']['disabled']:tests[2]=s[:,1]>ordered[k,1]
        for b,test in enumerate(tests):out[:,ai]|=test.astype(np.uint8)*(1<<b)
    return out

def counts(row,g,e,w,l):
    # np.bincount joint codes, not the worker's repeated boolean reductions.
    for key,bits in [('wcounts',w),('lcounts',l)]:
        for c in range(bits.shape[1]):
            for ai in range(2):
                b=bits[:,c,ai];code=np.where(b&8,b&7,8)
                row[key][g,e,c,ai]+=np.bincount(code,minlength=9)
    for j,c in enumerate((0,1,7)):
        for ai in range(2):
            a,b=w[:,c,ai],l[:,j,ai];valid=(a&8>0)&(b&8>0)
            code=np.where(valid,np.where(a&7>0,np.where(b&7>0,0,1),np.where(b&7>0,2,3)),4)
            row['lf_pairs'][g,e,j,ai]+=np.bincount(code,minlength=5)
    for ci in range(6):
        for ai in range(2):
            states=[]
            for pi,context in enumerate(row['selection'][g,ci,ai]):
                if context<0:x=np.full(len(w),2 if context==-1 else 3,np.int64)
                else:
                    b=w[:,context,ai];x=np.where(b&8>0,np.where(b&7>0,0,1),3)
                states.append(x);row['policy'][g,ci,e,ai,pi]+=np.bincount(x,minlength=4)
            for j,(a,b) in enumerate(((1,0),(2,0),(2,1))):
                row['pairs'][g,ci,e,ai,j]+=np.bincount(states[a]*4+states[b],minlength=16).reshape(4,4)

def reproduce(store,rep):
    folder=f'shards/{rep//5:02d}/rep_{rep:02d}';read=lambda n:store.npz(folder+'/'+n)
    meta=store.js(f'carry/rep_{rep:02d}.json');tasks=spec.tasks(rep);lookup={(t['case'],t['role']):t for t in tasks}
    assert meta['replicate']==rep and meta['records']==479984
    assert meta['statistics_sha256']==store.hashes[f'carry/rep_{rep:02d}.npz']
    assert set(meta['files'])=={n[len(folder)+1:] for n in store.map if n.startswith(folder+'/')}
    for n,h in meta['files'].items():assert store.hashes[folder+'/'+n]==h
    sh=sha(canonical(spec.specification()).encode());assert meta['specification_sha256']==sh
    raw={};st=dict(raw_records=0,raw_chunks=0,nonfinite_rows=0,V_decisions=0,E_W0_decisions=0,E_LF0_decisions=0,
                 reference_models=0,calibrations=0,candidates=0,CP_tail_checks=0)
    raw_names=set();times={}
    for task in tasks:
        parts=[]
        for start in range(0,task['count'],spec.CHUNK):
            name=f"chunks/{task['case']}/{task['role']}/{start:09d}.npz";raw_names.add(name)
            z=read(name);a=z['values'];md=json.loads(str(z['metadata']))
            assert md['task']==task and md['start']==start and md['count']==len(a)==min(spec.CHUNK,task['count']-start)
            assert md['specification_sha256']==sh and md['values_sha256']==sha(a.tobytes())
            bad=verify_values(a,task['case']);assert bad==md['unavailable_rows'];assert bad==0,'Do not silently exclude invalid records'
            st['raw_records']+=len(a);st['raw_chunks']+=1;st['nonfinite_rows']+=bad;parts.append(a)
            times.setdefault(task['role'],[]).append(md['at'])
        raw[task['case'],task['role']]=np.concatenate(parts)
    assert raw_names=={n for n in meta['files'] if n.startswith('chunks/')};assert st['raw_records']==479984
    models={};cal={};row=m.empty();details=[]
    def check_cal(path,model,values):
        z=read(path);md,cut,ids=m.calibrate(*model,values)
        equal(md,json.loads(str(z['metadata'])),path);assert np.allclose(cut,z['cut'],rtol=2e-11,atol=2e-13)
        assert np.array_equal(ids,z['indices']) and str(z['input_sha256'])==sha(values.tobytes());assert md['valid']
        st['calibrations']+=1;return json.loads(str(z['metadata'])),z['cut']
    for base in ('B_M','B_L'):
        a=raw[base,'ref'];state,ordered=m.fit(a);z=read('models/'+base+'.npz');assert state['valid']
        equal(state,json.loads(str(z['state'])),'model');assert np.allclose(ordered,z['ordered'],rtol=2e-11,atol=2e-13)
        assert str(z['input_sha256'])==sha(a.tobytes());st['reference_models']+=1
        models[base]=(json.loads(str(z['state'])),z['ordered'])
        cal[base]=check_cal('old_cal/'+base+'.npz',models[base],raw[base,'old_cal'])
    for gi,group in enumerate(spec.GROUPS):
        model=models[spec.baseline(group)]
        assert sha(raw[group,'audit_normal'].tobytes())!=sha(raw[group,'update_normal'].tobytes())
        assert sha(raw[group,'recovery'].tobytes())!=sha(raw[group,'update_normal'].tobytes())
        for role in ('audit','update'):
            normal=raw[group,role+'_normal'];o=read('orders/'+group+'_'+role+'.npz');order=o['normal_order'];pos=o['replacement_order']
            assert np.array_equal(order,spec.order(group,role+'_order',rep,15999))
            assert np.array_equal(pos,spec.order(group,role+'_replace',rep,15999))
            for ci,name in enumerate(spec.CANDIDATES):
                data=normal[order].copy();ids=order.copy();origin=np.zeros(15999,np.uint8)
                for f,lo,hi,start in spec.recipe(name):
                    data[pos[lo:hi]]=raw[group+'__'+f,role+'_fault'][start:start+hi-lo]
                    ids[pos[lo:hi]]=np.arange(start,start+hi-lo);origin[pos[lo:hi]]=spec.FAULTS.index(f)+1
                z=read(f'{role}/{group}/{name}.npz');pr=json.loads(str(z['provenance']))
                assert np.array_equal(ids,z['ids']) and np.array_equal(origin,z['origin'])
                assert pr==dict(normal_task=lookup[group,role+'_normal']['id'],fault_tasks={f:lookup[group+'__'+f,role+'_fault']['id'] for f in spec.FAULTS},input_sha256=sha(data.tobytes()))
                q=data[:,[11,12]];mu=q.mean(0);sd=q.std(0,ddof=1);T=np.sqrt(len(q))*mu/sd
                scores=np.array([abs(T[0]),max(abs(T[0]),T[1])]);states=(scores>spec.GATE_CUTS[0 if group.endswith('_M') else 1]).astype(np.int64)
                assert np.array_equal(states,z['states']);st['candidates']+=1
                contrib=[]
                for j in (1,2,3):
                    mask=origin==j;contrib.append(((data[mask][:,[11,12]]-normal[order[mask]][:,[11,12]]).sum(0)/15999).tolist())
                details.append(dict(group=group,role=role,candidate=name,mean=mu.tolist(),sd=sd.tolist(),signed_T=T.tolist(),scores=scores.tolist(),component_contributions=contrib))
                if role=='audit':row['gates'][gi,ci]=states
                else:cal[group,ci]=check_cal(f'calibration/{group}/{name}.npz',model,data)
                if name=='BAL400':parent=(data.copy(),ids.copy(),origin.copy())
                if name=='TRI800':
                    mask=parent[2]>0;assert np.array_equal(data[mask],parent[0][mask]) and np.array_equal(ids[mask],parent[1][mask])
        cal[group,'C']=check_cal(f'calibration/{group}/C.npz',model,raw[group,'recovery'])
        for j,c in enumerate((cal[spec.baseline(group)],cal[group,0],cal[group,'C'])):row['inactive'][gi,j]=c[0]['lf0_structurally_inactive']
    equal(details,store.js(folder+'/gate_details.json'),'gate details')
    frozen=store.js(folder+'/CALIBRATION_FROZEN.json')
    assert set(frozen)=={n for n in meta['files'] if n.split('/')[0] in ('models','old_cal','orders','audit','update','calibration')}
    for n,h in frozen.items():assert store.hashes[folder+'/'+n]==h
    fh=sha(canonical(frozen).encode());bit_names=set();labels=[]
    def check_bits(path,task,start,values,model,cals,method,frozen_hash):
        z=read(path);bits=z['bits'];md=json.loads(str(z['metadata']));bit_names.add(path)
        assert md==dict(task=task,start=start,count=len(values),frozen_sha256=frozen_hash,method=method,raw_sha256=sha(values.tobytes()),bits_sha256=sha(bits.tobytes()))
        new=np.stack([independent_decide(values,model,c,method) for c in cals],axis=1)
        assert np.array_equal(new,bits),('readout mismatch',rep,path,np.count_nonzero(new!=bits))
        return new
    for gi,group in enumerate(spec.GROUPS):
        cals=[cal[spec.baseline(group)],cal[group,'C']];task=lookup[group,'validation_normal'];values=raw[group,'validation_normal']
        for start in range(0,len(values),spec.CHUNK):
            v=values[start:start+spec.CHUNK];b=check_bits(f'validation/{group}/{start:09d}.npz',task,start,v,models[spec.baseline(group)],cals,0,fh)
            row['v_counts'][gi,...,0]+=(b&7>0).sum(0);row['v_counts'][gi,...,1]+=(b&8>0).sum(0);st['V_decisions']+=b.size
        ev=[]
        for c in range(2):
            es=[]
            for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
                k,n=row['v_counts'][gi,c,ai];e=m.evidence(k,n,20000,1/den)
                label='SUPPORTED' if binom.cdf(k,n,1.25/den)<=spec.CP_ETA else 'EXCEEDED' if binom.sf(k-1,n,1.25/den)<spec.CP_ETA else 'UNRESOLVED'
                assert e['label']==label;st['CP_tail_checks']+=1;es.append(e)
            ev.append(es)
        labels.append(ev)
        for ci in range(6):
            for ai in range(2):
                gate=row['gates'][gi,ci,1]
                row['selection'][gi,ci,ai]=[ci+1]*3 if gate==0 else ([0]*3 if ev[0][ai]['label']=='SUPPORTED' else [0,-1,7 if ev[1][ai]['label']=='SUPPORTED' else -1])
    receipt=store.js(folder+'/SELECTION_BEFORE_E.json')
    expected=dict(calibration_sha256=fh,validation={n:store.hashes[folder+'/'+n] for n in bit_names},evidence=labels,selection=row['selection'].tolist(),gate_states=row['gates'].tolist(),E_used=False)
    equal(receipt,expected,'V/E receipt');selection_hash=store.hashes[folder+'/SELECTION_BEFORE_E.json']
    for gi,group in enumerate(spec.GROUPS):
        model=models[spec.baseline(group)];cals=[cal[spec.baseline(group)]]+[cal[group,ci] for ci in range(6)]+[cal[group,'C']]
        for ei,tag in enumerate(('normal',)+spec.FAULTS):
            case=group if ei==0 else group+'__'+tag;role='null_eval' if ei==0 else 'fault_eval';task=lookup[case,role];values=raw[case,role]
            for start in range(0,len(values),spec.CHUNK):
                v=values[start:start+spec.CHUNK]
                w=check_bits(f'decisions/{case}/{start:09d}_W0.npz',task,start,v,model,cals,0,selection_hash)
                l=check_bits(f'decisions/{case}/{start:09d}_LF0.npz',task,start,v,model,[cals[j] for j in (0,1,7)],1,selection_hash)
                counts(row,gi,ei,w,l);st['E_W0_decisions']+=w.size;st['E_LF0_decisions']+=l.size
    assert bit_names=={n for n in meta['files'] if n.startswith(('validation/','decisions/'))}
    for role in ('null_eval','fault_eval'):assert min(times[role])>=max(times['validation_normal'])
    saved=store.npz(f'carry/rep_{rep:02d}.npz');assert set(saved)==set(row)
    for k in row:assert np.array_equal(saved[k],row[k]),('carry',rep,k)
    m.validate(row);np.savez_compressed(OUT/'recomputed'/f'rep_{rep:02d}.npz',**row)
    dump(OUT/'details'/f'rep_{rep:02d}.json',details)
    return row,st

def main():
    assert not OUT.exists(),'Use new version for changed analysis';OUT.mkdir(parents=True)
    (OUT/'recomputed').mkdir();(OUT/'details').mkdir();began=time.monotonic()
    previous=ROOT/'audits/verification/2026-10-18_PR-0012_fix1_closeout.json';baseline=json.loads(previous.read_text())
    anchors={**baseline['protected_sources'],**baseline['outputs'],**baseline['current_navigation'],str(previous.relative_to(ROOT)):sha(previous.read_bytes())}
    for n,h in anchors.items():assert sha((ROOT/n).read_bytes())==h,n
    dump(OUT/'start_snapshot.json',dict(anchors=anchors,ledger=baseline['ledger'],corpus=baseline['corpus'],
        source_sha256=sha(Path(__file__).read_bytes()),new_scientific_generation=False,manuscript_edits=False))
    store=Store();store.audit();rows=[];checks=[]
    for rep in range(50):
        row,st=reproduce(store,rep);rows.append(row);checks.append(dict(replicate=rep,**st))
        print('REPRODUCED',rep+1,'/50',round(time.monotonic()-began,1),'s',flush=True)
    agg=m.aggregate(rows)
    for key,value in agg.items():
        equal(value,store.js('analysis/'+key+'.json'),key)
        if key!='outer_arrays':dump(OUT/(key+'.json'),value)
    np.savez_compressed(OUT/'outer_arrays.npz',**{k:np.stack([r[k] for r in rows]) for k in m.shapes()})
    dump(OUT/'validation.json',dict(status='PASS_SAVED_DATA_REPRODUCTION',per_replicate=checks,
        totals={k:sum(c[k] for c in checks) for k in checks[0] if k!='replicate'},elapsed_seconds=time.monotonic()-began,
        source_sha256=sha(Path(__file__).read_bytes()),independent_reviewer=False,science_records_generated_locally=0,manuscript_edits=0,
        verification='same agent; scalar projection/effective-halfplane, joint histogram path, fixed model/calibration replay, binomial-tail CP labels',
        gaps=['raw waveforms not saved or regenerated','no physical experiment','no actual VM-loss restore test']))
    print('PASS_SAVED_DATA_REPRODUCTION',flush=True)
if __name__=='__main__':main()
