# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""PR-0006 GPU worker. Import is inert; scientific execution is Colab-only."""
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import zipfile
import numpy as np
import probe_backend_check_v1 as core
import pr0006_io as disk
import pr0006_methods as methods
import pr0006_spec as spec


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)


def open_npz(path):
    """Validate sizes/types before allocation. No pickle or executable archive data."""
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(names)!=len(set(names)) or len(names)>64 or sum(i.file_size for i in infos)>32*1024**2:
            raise ValueError('Invalid/oversized NPZ')
        for item in infos:
            if '/' in item.filename or not item.filename.endswith('.npy'):raise ValueError('Invalid NPZ member')
            with z.open(item) as f:
                version=np.lib.format.read_magic(f)
                reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:raise ValueError('Unsupported NPY')
                shape,_,dtype=reader(f)
                if dtype.hasobject or dtype.kind not in 'fiuU' or len(shape)>5 or any(n<0 for n in shape):
                    raise ValueError('Invalid array type/shape')
                if math.prod(shape)*dtype.itemsize!=item.file_size-f.tell():raise ValueError('NPY size mismatch')
    return np.load(path,allow_pickle=False)


def save_npz(path,**arrays):
    """Immutable logical file: retries may confirm identical contents, never replace."""
    path=Path(path);arrays={k:np.asarray(v) for k,v in arrays.items()}
    if path.exists():
        with open_npz(path) as old:
            if set(old.files)!=set(arrays):raise ValueError('Existing NPZ fields differ')
            for k,a in arrays.items():
                b=old[k]
                equal=np.array_equal(a,b,equal_nan=True) if a.dtype.kind=='f' else np.array_equal(a,b)
                if a.dtype!=b.dtype or not equal:raise ValueError('Existing evidence differs: '+str(path))
        return
    b=io.BytesIO();np.savez(b,**arrays);disk.atomic(path,b.getvalue())


def save_json(path,data):
    path=Path(path)
    if path.exists():
        if canonical(disk.read(path))!=canonical(data):raise ValueError('Existing JSON differs: '+str(path))
    else:disk.write(path,data)


def chunk_path(root,task,start):
    return disk.rep_dir(root,task['replicate'])/'chunks'/task['case']/task['role']/f'{start:09d}.npz'


def verify_values(a,case):
    if a.dtype!=np.float64 or a.ndim!=2 or a.shape[1]!=17:raise ValueError('Expected float64 Nx17')
    finite=np.isfinite(a).all(axis=1);b=a[finite]
    if len(b):
        _,_,_,lam,nu=core.coefficients(spec.conditions()[case]['params'])
        for k,idx in ((0,9),(3,10),(6,11)):
            want=b[:,k+2]-lam*b[:,k]-nu*b[:,k+1]
            if not core.compare(b[:,idx:idx+1],want[:,None])['passed']:raise ValueError('Moment identity mismatch')
        if not core.identity(b)['passed']:raise ValueError('CR expansion mismatch')
    return int((~finite).sum())


def read_chunk(root,task,start,spec_hash):
    with open_npz(chunk_path(root,task,start)) as z:
        if set(z.files)!={'values','metadata'}:raise ValueError('Unexpected chunk fields')
        a=z['values'];meta=json.loads(str(z['metadata']))
    n=min(spec.CHUNK,task['count']-start)
    if (a.shape!=(n,17) or meta['task']!=task or meta['start']!=start or
        meta['specification_sha256']!=spec_hash or meta['values_sha256']!=disk.sha(a.tobytes())):
        raise ValueError('Chunk identity/checksum mismatch')
    if verify_values(a,task['case'])!=meta['unavailable_rows']:raise ValueError('Nonfinite count mismatch')
    return a


def memory(gpu=None):
    return dict(rss_high_water_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
                gpu_reserved_bytes=0 if gpu is None else int(gpu.cp.get_default_memory_pool().total_bytes()))


def gpu_values(gpu,p,inputs,traces=False):
    cp=gpu.cp;t=time.perf_counter();device={k:cp.asarray(v) for k,v in inputs.items()}
    cp.cuda.get_current_stream().synchronize();h2d=time.perf_counter()-t
    a,b=cp.cuda.Event(),cp.cuda.Event();t=time.perf_counter();a.record()
    values,trace=core.calculate(p,device,cp,gpu.filt,traces=traces);b.record();b.synchronize()
    compute=time.perf_counter()-t;event=cp.cuda.get_elapsed_time(a,b)/1000;t=time.perf_counter()
    values=cp.asnumpy(values);trace={k:cp.asnumpy(v[:3]) for k,v in trace.items()}
    return values,trace,dict(h2d_seconds=h2d,compute_seconds=compute,gpu_event_seconds=event,d2h_seconds=time.perf_counter()-t)


def preflight(root,gpu,session,clock):
    rows=[];folder=Path(root)/'sessions'/session/'preflight'
    for case,c in spec.conditions().items():
        clock.check();z=spec.make_inputs(case,'preflight',0,0,3)
        cpu,tr_cpu=core.calculate(c['params'],z,traces=True)
        a,tr_gpu,tm=gpu_values(gpu,c['params'],z,traces=True)
        checks=dict(values=core.compare(cpu,a),CR=core.identity(a))
        checks.update({k:core.compare(tr_cpu[k],tr_gpu[k]) for k in tr_cpu})
        save_npz(folder/(case+'.npz'),cpu=cpu,gpu=a)
        rows.append(dict(case=case,checks=checks,timing=tm,input_sha256={k:disk.sha(v.tobytes()) for k,v in z.items()}))
        if not all(x['passed'] for x in checks.values()):
            disk.write(folder/'result.json',dict(passed=False,rows=rows));raise ValueError('GPU preflight mismatch')
    disk.write(folder/'result.json',dict(passed=True,rows=rows,records=132,scientific_records=0,**memory(gpu)))
    print('Technical float64 preflight passed: 44 conditions / 132 records; not scientific evidence.',flush=True)


def generate_chunk(root,task,start,spec_hash,gpu):
    began=time.perf_counter();n=min(spec.CHUNK,task['count']-start)
    z=spec.make_inputs(task['case'],task['role'],task['replicate'],start,n);input_seconds=time.perf_counter()-began
    values,_,tm=gpu_values(gpu,spec.conditions()[task['case']]['params'],z)
    values=np.ascontiguousarray(values,dtype=np.float64);bad=verify_values(values,task['case'])
    meta=dict(task=task,start=start,count=n,specification_sha256=spec_hash,values_sha256=disk.sha(values.tobytes()),
              unavailable_rows=bad,timing=dict(input_seconds=input_seconds,**tm),memory=memory(gpu),at=disk.stamp())
    save_npz(chunk_path(root,task,start),values=values,metadata=np.array(canonical(meta)))
    if time.perf_counter()-began>spec.LIMITS['chunk_seconds']:raise TimeoutError('Chunk time budget exceeded after atomic save')


class Worker:
    def __init__(self,root,archives,gpu,clock):
        self.root,self.archives,self.gpu,self.clock=Path(root),Path(archives),gpu,clock
        self.spec_hash=disk.sha(canonical(spec.specification()).encode());self.last_zip=time.monotonic()
        self.rep=0;self.new_chunks=0

    def check(self,checkpoint=True):
        self.clock.check();m=memory(self.gpu)
        if m['rss_high_water_bytes']>spec.LIMITS['rss_bytes']:raise MemoryError('Fixed RSS budget exceeded')
        if m['gpu_reserved_bytes']>spec.LIMITS['gpu_bytes']:raise MemoryError('Fixed GPU pool budget exceeded')
        if self.new_chunks%50==0 and sum(p.stat().st_size for p in disk.evidence(self.root))>spec.LIMITS['evidence_bytes']:
            raise RuntimeError('Fixed evidence budget exceeded')
        if checkpoint and time.monotonic()-self.last_zip>=spec.LIMITS['periodic_zip_seconds']:
            self.pack('PARTIAL',range(self.rep//5*5,self.rep+1))

    def pack(self,status,reps):
        for p in disk.snapshot(self.root,self.archives,status,list(reps)):
            print('CHECKPOINT_ZIP='+str(p),flush=True)
        self.last_zip=time.monotonic();self.clock.commit()

    def task(self,t):
        arrays=[]
        for start in range(0,t['count'],spec.CHUNK):
            self.check()
            if not chunk_path(self.root,t,start).exists():
                def expired(signum,frame):raise TimeoutError('Chunk deadline reached')
                old=signal.signal(signal.SIGALRM,expired);signal.alarm(spec.LIMITS['chunk_seconds'])
                try:generate_chunk(self.root,t,start,self.spec_hash,self.gpu)
                finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
                self.new_chunks+=1;self.clock.commit()
            arrays.append(read_chunk(self.root,t,start,self.spec_hash))
        print(f"rep {self.rep+1:02d}/25 | {t['case']} {t['role']} | {t['count']} saved | active {self.clock.seconds()/60:.1f} min",flush=True)
        return np.concatenate(arrays)

    def replicate(self,rep):
        self.rep=rep;folder=disk.rep_dir(self.root,rep);tasks=spec.tasks(rep)
        expected={chunk_path(self.root,t,s) for t in tasks for s in range(0,t['count'],spec.CHUNK)}
        if set((folder/'chunks').rglob('*.npz'))-expected:raise ValueError('Unknown scientific chunk IDs')
        lookup={(t['case'],t['role']):t for t in tasks};models={};cal={}
        for base in ('B_M','B_L'):
            values=self.task(lookup[base,'ref']);state,ordered=methods.fit(values)
            save_npz(folder/'models'/(base+'.npz'),state=np.array(canonical(state)),ordered=ordered,
                     input_sha256=np.array(disk.sha(values.tobytes())))
            models[base]=(state,ordered)
        # All calibration is complete and immutable before any evaluation records.
        for group,c in spec.conditions().items():
            if c['kind']!='normal':continue
            values=self.task(lookup[group,'cal']);state,ordered=models[spec.baseline(group)]
            meta,cut,idx=methods.calibrate(state,ordered,values);cal[group]=(meta,cut)
            save_npz(folder/'calibration'/(group+'.npz'),metadata=np.array(canonical(meta)),cut=cut,indices=idx,
                     input_sha256=np.array(disk.sha(values.tobytes())))
        frozen={str(p.relative_to(folder)):disk.digest(p) for d in ('models','calibration') for p in disk.evidence(folder/d)}
        nctx=len(spec.contexts());counts=np.zeros((nctx,2,6,8),np.int64);available=np.zeros((nctx,2,6),np.int64)
        pairs=np.zeros((nctx,2,15,4),np.int64);pair_n=np.zeros((nctx,2,15),np.int64)
        inactive=np.full((nctx,2),-1,np.int64)
        for case,c in spec.conditions().items():
            role='null' if c['kind']=='normal' else c['kind'];values=self.task(lookup[case,role])
            base=spec.baseline(c['group']);state,ordered=models[base]
            for policy in (('B',) if c['group'].startswith('B_') else ('F','C')):
                meta,cut=cal[base if policy in ('B','F') else c['group']]
                packed=methods.decide(state,ordered,meta,cut,values)
                save_npz(folder/'decisions'/(case+'_'+policy+'.npz'),packed=packed)
                k=spec.contexts().index((case,policy));counts[k],available[k],pairs[k],pair_n[k]=methods.count(packed)
                if meta['valid']:inactive[k]=meta['lf0_structurally_inactive']
        if any(disk.digest(folder/p)!=h for p,h in frozen.items()):raise ValueError('Calibration changed during evaluation')
        # Recount every chunk and check inventory, including resumes, before certification.
        actual=set((folder/'chunks').rglob('*.npz'))
        if actual!=expected:raise ValueError('Incomplete scientific inventory')
        records=sum(len(read_chunk(self.root,t,s,self.spec_hash)) for t in tasks for s in range(0,t['count'],spec.CHUNK))
        if records!=sum(t['count'] for t in tasks):raise ValueError('Per-replicate record count mismatch')
        carry=self.root/'carry'/f'rep_{rep:02d}.npz'
        save_npz(carry,counts=counts,available=available,pairs=pairs,pair_n=pair_n,inactive=inactive)
        metadata=dict(replicate=rep,records=records,specification_sha256=self.spec_hash,
                      statistics_sha256=disk.digest(carry),calibration_unchanged=True,
                      files={str(p.relative_to(folder)):disk.digest(p) for p in disk.evidence(folder)})
        save_json(carry.with_suffix('.json'),metadata)
        state=disk.read(self.root/'state.json');state['completed'][f'{rep:02d}']=dict(metadata_sha256=disk.digest(carry.with_suffix('.json')))
        state['status']='PARTIAL';disk.write(self.root/'state.json',state);self.clock.commit()
        print(f'Replicate {rep+1:02d}/25 certified: {records} unique records. No performance-based stopping.',flush=True)


def carry_results(root):
    out=[]
    for rep in range(25):
        with open_npz(Path(root)/'carry'/f'rep_{rep:02d}.npz') as z:
            expected=dict(counts=(76,2,6,8),available=(76,2,6),pairs=(76,2,15,4),pair_n=(76,2,15),inactive=(76,2))
            if set(z.files)!=set(expected):raise ValueError('Invalid carry arrays')
            row={k:z[k] for k in z.files}
            for k,shape in expected.items():
                if row[k].shape!=shape or row[k].dtype!=np.int64 or np.any(row[k]<(-1 if k=='inactive' else 0)):raise ValueError('Invalid carry shape/type/count')
            if np.any(row['inactive']>1):raise ValueError('Invalid inactivity code')
            if not np.array_equal(row['counts'].sum(axis=-1),row['available']) or not np.array_equal(row['pairs'].sum(axis=-1),row['pair_n']):
                raise ValueError('Carry conservation failed')
            for ki,(case,_) in enumerate(spec.contexts()):
                kind=spec.conditions()[case]['kind'];n=spec.COUNTS['null' if kind=='normal' else kind]
                if np.any(row['available'][ki]>n) or np.any(row['pair_n'][ki]>n):raise ValueError('Carry exceeds planned denominator')
            out.append(dict(replicate=rep,**row))
    return out


def main(root,archives,session):
    if sys.platform!='linux' or not Path('/content').is_dir():raise RuntimeError('Scientific worker is Colab GPU only')
    root=Path(root);clock=disk.ActiveClock(root/'active_time.json',spec.LIMITS['active_seconds'])
    try:
        if sum(t['count'] for t in spec.tasks(0))!=549988:raise ValueError('Frozen per-replicate plan differs')
        save_json(root/'specification.json',spec.specification());remote=disk.verify_carry(root)
        done=sorted(int(x) for x in disk.read(root/'state.json')['completed'])
        if done!=list(range(len(done))):raise ValueError('Completed replicates must be a contiguous prefix')
        # A crash just after the fifth certification must not skip its archive.
        receipts=disk.catalog(root)['archives']
        covered={int(k) for x in receipts for k in x.get('carry_hashes',{})}
        for start in range(0,len(done)//5*5,5):
            if not set(range(start,start+5))<=covered:
                for p in disk.snapshot(root,archives,f'SHARD_{start//5+1:02d}_COMPLETE',range(start,start+5)):
                    print('CHECKPOINT_ZIP='+str(p),flush=True)
        if len(done)<25:
            clock.check();gpu=core.GPU();disk.write(root/'sessions'/session/'environment.json',core.environment(gpu))
            preflight(root,gpu,session,clock);worker=Worker(root,archives,gpu,clock)
            for rep in range(len(done),25):
                worker.replicate(rep)
                if (rep+1)%5==0 and rep<24:worker.pack(f'SHARD_{rep//5+1:02d}_COMPLETE',range(rep-4,rep+1))
        results=carry_results(root);report=methods.aggregate(results)
        save_json(root/'analysis'/'aggregate.json',report)
        summary=dict(status='COMPLETE',replicates=25,unique_records=13749700,
                     unique_evaluation_records=9250000,policy_evaluation_records=16250000,
                     decision_count=195000000,raw_not_present_this_runtime=remote,
                     statement='Prior raw absent here is archive-certified, not revalidated here. Keep all data shards.',
                     code_only_technical_preflight_excluded=True)
        # Runtime provenance can change on restoration; settled aggregate is immutable.
        disk.write(root/'summary.json',summary);state=disk.read(root/'state.json');state['status']='COMPLETE';disk.write(root/'state.json',state)
        if len(done)<25:worker.pack('SHARD_05_COMPLETE',range(20,25))
        clock.commit();print('COMPLETE: 25 fixed replicates. Preparing small final index.',flush=True)
        return 0
    except BaseException as exc:
        clock.commit();disk.write(root/'sessions'/session/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        state=disk.read(root/'state.json');state['status']='PARTIAL' if isinstance(exc,TimeoutError) else 'TECHNICAL_FAILURE'
        disk.write(root/'state.json',state)
        print(traceback.format_exc(),flush=True);return 3 if isinstance(exc,TimeoutError) else 2


if __name__=='__main__':sys.exit(main(*sys.argv[1:]))
