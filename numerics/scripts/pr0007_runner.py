# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Streaming PR-0007 worker. Scientific execution is restricted to Colab GPU."""
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid
import numpy as np
import pr0007_io as disk
import pr0007_spec as spec
import pr0007_methods as methods
from pr0006_runner import canonical,open_npz,save_npz,save_json,memory,gpu_values
core=spec.core


def chunk_path(root,t,start):
    return disk.rep_dir(root,t['replicate'])/'chunks'/t['case']/t['role']/f'{start:09d}.npz'


def verify_values(a,case):
    if a.dtype!=np.float64 or a.ndim!=2 or a.shape[1]!=17:raise ValueError('Expected float64 Nx17')
    finite=np.isfinite(a).all(1);b=a[finite]
    if len(b):
        _,_,_,lam,nu=core.coefficients(spec.conditions()[case]['params'])
        for k,idx in ((0,9),(3,10),(6,11)):
            if not core.compare(b[:,idx,None],(b[:,k+2]-lam*b[:,k]-nu*b[:,k+1])[:,None])['passed']:
                raise ValueError('Moment expansion mismatch')
        if not core.identity(b)['passed']:raise ValueError('CR expansion mismatch')
    return int((~finite).sum())


def read_chunk(root,t,start,spec_hash):
    with open_npz(chunk_path(root,t,start)) as z:
        if set(z.files)!={'values','metadata'}:raise ValueError('Unexpected chunk fields')
        a=z['values'];meta=json.loads(str(z['metadata']))
    n=min(spec.CHUNK,t['count']-start)
    if (a.shape!=(n,17) or meta['task']!=t or meta['start']!=start or meta['specification_sha256']!=spec_hash or
        meta['values_sha256']!=disk.sha(a.tobytes())):raise ValueError('Raw chunk identity/checksum mismatch')
    if verify_values(a,t['case'])!=meta['unavailable_rows']:raise ValueError('Unavailable raw count differs')
    return a


def generate_chunk(root,t,start,spec_hash,gpu):
    began=time.perf_counter();n=min(spec.CHUNK,t['count']-start)
    z=spec.make_inputs(t['case'],t['role'],t['replicate'],start,n);input_seconds=time.perf_counter()-began
    values,_,timing=gpu_values(gpu,spec.conditions()[t['case']]['params'],z)
    values=np.ascontiguousarray(values,dtype=np.float64);bad=verify_values(values,t['case'])
    meta=dict(task=t,start=start,count=n,specification_sha256=spec_hash,values_sha256=disk.sha(values.tobytes()),
        unavailable_rows=bad,timing=dict(input_seconds=input_seconds,**timing),memory=memory(gpu),at=disk.stamp())
    save_npz(chunk_path(root,t,start),values=values,metadata=np.array(canonical(meta)))
    if time.perf_counter()-began>spec.LIMITS['chunk_seconds']:raise TimeoutError('Chunk budget exceeded after save')


def preflight(root,gpu,session,clock):
    rows=[];cpus=[];gpus=[];folder=Path(root)/'sessions'/session/'preflight'
    for case,c in spec.conditions().items():
        clock.check();z=spec.make_inputs(case,'preflight',0,0,3)
        cpu,tc=core.calculate(c['params'],z,traces=True);a,tg,tm=gpu_values(gpu,c['params'],z,traces=True)
        checks=dict(values=core.compare(cpu,a),CR=core.identity(a))
        checks.update({k:core.compare(tc[k],tg[k]) for k in tc})
        save_npz(folder/(case+'.npz'),cpu=cpu,gpu=a);cpus.append(cpu);gpus.append(a)
        rows.append(dict(case=case,checks=checks,timing=tm,input_sha256={k:disk.sha(v.tobytes()) for k,v in z.items()}))
        if not all(c['passed'] for c in checks.values()):raise ValueError('CPU/GPU preflight numeric mismatch')
    # Decisions under a common frozen calibration; these reused 72 technical
    # rows are NOT calibration/evaluation records in the scientific experiment.
    cpu=np.concatenate(cpus);gpu_a=np.concatenate(gpus);state,ordered=methods.fit(cpu)
    if not state['valid']:raise ValueError('Technical projection unavailable')
    cal=np.tile(cpu,(112,1))[:7999].copy();cal[:,11:13]+=np.array([.000173,.000317])
    meta,cut,_=methods.calibrate(state,ordered,cal)
    same=np.array_equal(methods.decide(state,ordered,meta,cut,cpu),methods.decide(state,ordered,meta,cut,gpu_a))
    if not same:raise ValueError('CPU/GPU preflight final decision bits differ')
    disk.write(folder/'result.json',dict(passed=True,rows=rows,records=72,scientific_records=0,
        common_calibration_decisions_equal=same,**memory(gpu)))
    print('Technical float64 preflight passed: 24 conditions / 72 records; excluded from science.',flush=True)


class Worker:
    def __init__(self,root,archives,gpu,clock,session=None):
        self.root,self.archives,self.gpu,self.clock=Path(root),Path(archives),gpu,clock
        self.spec_hash=disk.sha(canonical(spec.specification()).encode());self.last_zip=time.monotonic()
        self.rep=0;self.new_chunks=0
        self.stage_path=self.root/'sessions'/(session or ('technical_'+uuid.uuid4().hex[:12]))/'stages.json'
        self.stages=[]

    def stage(self,name,started,**details):
        self.stages.append(dict(stage=name,seconds=time.monotonic()-started,at=disk.stamp(),replicate=self.rep,**details))
        disk.write(self.stage_path,self.stages)

    def check(self,checkpoint=True):
        self.clock.check();m=memory(self.gpu)
        if m['rss_high_water_bytes']>spec.LIMITS['rss_bytes']:raise MemoryError('RSS budget exceeded')
        if m['gpu_reserved_bytes']>spec.LIMITS['gpu_bytes']:raise MemoryError('GPU memory budget exceeded')
        if self.new_chunks%50==0 and sum(p.stat().st_size for p in disk.evidence(self.root))>spec.LIMITS['evidence_bytes']:
            raise MemoryError('Evidence disk budget exceeded')
        if checkpoint and time.monotonic()-self.last_zip>=spec.LIMITS['periodic_zip_seconds']:
            self.pack('PARTIAL',range(self.rep//5*5,self.rep+1))

    def pack(self,status,reps):
        began=time.monotonic()
        def expired(signum,frame):raise TimeoutError('Archive stage exceeded 30 minutes')
        old=signal.signal(signal.SIGALRM,expired);signal.alarm(1800)
        try:
            for p in disk.snapshot(self.root,self.archives,status,list(reps)):print('CHECKPOINT_ZIP='+str(p),flush=True)
        finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
        self.last_zip=time.monotonic();self.clock.commit()
        self.stage('archive',began,status=status)
        print(f'Archive stage {time.monotonic()-began:.1f}s; original files retained.',flush=True)

    def chunks(self,t):
        for start in range(0,t['count'],spec.CHUNK):
            self.check()
            if not chunk_path(self.root,t,start).exists():
                def expired(signum,frame):raise TimeoutError('Chunk deadline reached')
                old=signal.signal(signal.SIGALRM,expired);signal.alarm(spec.LIMITS['chunk_seconds'])
                try:generate_chunk(self.root,t,start,self.spec_hash,self.gpu)
                finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
                self.new_chunks+=1;self.clock.commit()
            yield start,read_chunk(self.root,t,start,self.spec_hash)

    def task(self,t):
        # Only ref/cal arrays (<=31,999 x17) are concatenated. Eval stays chunked.
        if t['role'].endswith('_eval'):raise ValueError('Evaluation must stream')
        return np.concatenate([a for _,a in self.chunks(t)])

    def replicate(self,rep):
        began=time.monotonic();self.rep=rep;folder=disk.rep_dir(self.root,rep);tasks=spec.tasks(rep)
        lookup={(t['case'],t['role']):t for t in tasks}
        expected={chunk_path(self.root,t,s) for t in tasks for s in range(0,t['count'],spec.CHUNK)}
        if set((folder/'chunks').rglob('*.npz'))-expected:raise ValueError('Unknown scientific chunk IDs')
        models={};cal={}
        for base in ('B_M','B_L'):
            stage_start=time.monotonic()
            values=self.task(lookup[base,'ref']);state,ordered=methods.fit(values);models[base]=(state,ordered)
            save_npz(folder/'models'/(base+'.npz'),state=np.array(canonical(state)),ordered=ordered,
                     input_sha256=np.array(disk.sha(values.tobytes())))
            self.stage('reference',stage_start,case=base)
        for group in spec.groups():
            stage_start=time.monotonic()
            values=self.task(lookup[group,'normal_cal']);order,positions=spec.orders(group,rep)
            save_npz(folder/'orders'/(group+'.npz'),normal_order=order,replacement_order=positions)
            state,ordered=models[spec.baseline(group)]
            def freeze(key,data,ids,origin,extra):
                meta,cut,idx=methods.calibrate(state,ordered,data);cal[key]=(meta,cut)
                save_npz(folder/'calibration'/(key+'.npz'),metadata=np.array(canonical(meta)),cut=cut,indices=idx,
                    record_ids=ids,origin_role=origin,provenance=np.array(canonical(extra)),
                    input_sha256=np.array(disk.sha(data.tobytes())))
            for m in spec.SIZES:
                ids=order[:m]
                freeze(spec.clean_id(group,m),values[ids],ids,np.zeros(m,np.uint8),dict(normal_case=group))
            for fault in spec.FAULTS:
                fault_values=self.task(lookup[group+'__'+fault,'fault_cal'])
                for n in spec.MIX_COUNTS:
                    m=spec.SIZES[1];ids=order[:m].copy();data=values[ids].copy();origin=np.zeros(m,np.uint8)
                    replace=positions[:n];removed=ids[replace].copy();data[replace]=fault_values[:n]
                    ids[replace]=np.arange(n);origin[replace]=1
                    freeze(spec.mix_id(group,fault,n),data,ids,origin,dict(normal_case=group,
                        fault_case=group+'__'+fault,replaced_positions=replace.tolist(),removed_normal_ids=removed.tolist(),
                        fault_record_ids=list(range(n)),normal_role='normal_cal',fault_role='fault_cal'))
            self.stage('calibration',stage_start,group=group)
            print(f'rep {rep+1:02d}/25 | {group}: nested and mixed calibration frozen',flush=True)
        if len(cal)!=54:raise ValueError('Expected 54 unique fits')
        frozen={str(p.relative_to(folder)):disk.digest(p) for d in ('models','orders','calibration') for p in disk.evidence(folder/d)}
        save_json(folder/'CALIBRATION_FROZEN.json',frozen);frozen_hash=disk.sha(canonical(frozen).encode())
        # No eval draws were requested before this immutable freeze record.
        result=methods.empty();ctx=spec.contexts();contr=spec.contrasts();decision_paths=set()
        for case,c in spec.conditions().items():
            stage_start=time.monotonic()
            role='null_eval' if c['kind']=='normal' else 'fault_eval';t=lookup[case,role]
            relevant=[x for x in ctx if x['case']==case];indices=[x['index'] for x in relevant]
            relevant_contr=[x for x in contr if x['case']==case]
            state,ordered=models[spec.baseline(c['group'])]
            for start,values in self.chunks(t):
                p=folder/'decisions'/case/f'{start:09d}.npz';decision_paths.add(p)
                identity=dict(task=t,start=start,count=len(values),context_indices=indices,
                    raw_sha256=disk.digest(chunk_path(self.root,t,start)),calibration_sha256=frozen_hash)
                if p.exists():
                    with open_npz(p) as z:
                        if set(z.files)!={'packed','metadata'}:raise ValueError('Unexpected decision fields')
                        packed=z['packed'];meta=json.loads(str(z['metadata']))
                    packed_sha=meta.pop('packed_sha256',None)
                    if meta!=identity or packed_sha!=disk.sha(packed.tobytes()):raise ValueError('Decision input/provenance/checksum differs')
                else:
                    packed=np.stack([methods.decide(state,ordered,*cal[x['fit_key']],values) for x in relevant],axis=1)
                    save_npz(p,packed=packed,metadata=np.array(canonical(dict(identity,packed_sha256=disk.sha(packed.tobytes())))))
                if packed.dtype!=np.uint32 or packed.shape!=(len(values),len(indices),2):raise ValueError('Packed shape/type differs')
                for j,x in enumerate(relevant):
                    k=x['index'];cc,aa,pp,nn=methods.count(packed[:,j])
                    result['counts'][k]+=cc;result['available'][k]+=aa;result['pairs'][k]+=pp;result['pair_n'][k]+=nn
                    meta,_=cal[x['fit_key']]
                    if meta['valid']:result['inactive'][k]=meta['lf0_structurally_inactive']
                local={index:j for j,index in enumerate(indices)}
                for x in relevant_contr:
                    cc,nn=methods.paired(packed[:,local[x['left']]],packed[:,local[x['right']]])
                    result['contrasts'][x['index']]+=cc;result['contrast_n'][x['index']]+=nn
                self.clock.commit()
            self.stage('evaluation',stage_start,case=case)
            print(f'rep {rep+1:02d}/25 | {case} {role}: {t["count"]} records / {len(indices)} contexts saved',flush=True)
        if any(disk.digest(folder/p)!=h for p,h in frozen.items()):raise ValueError('Calibration changed during evaluation')
        if set((folder/'chunks').rglob('*.npz'))!=expected or set((folder/'decisions').rglob('*.npz'))!=decision_paths:
            raise ValueError('Incomplete/excess committed inventory')
        records=sum(len(read_chunk(self.root,t,s,self.spec_hash)) for t in tasks for s in range(0,t['count'],spec.CHUNK))
        if records!=sum(t['count'] for t in tasks):raise ValueError('Record-count mismatch')
        validate_result(result)
        carry=self.root/'carry'/f'rep_{rep:02d}.npz';save_npz(carry,**result)
        save_json(carry.with_suffix('.json'),dict(replicate=rep,records=records,specification_sha256=self.spec_hash,
            statistics_sha256=disk.digest(carry),calibration_unchanged=True,
            files={str(p.relative_to(folder)):disk.digest(p) for p in disk.evidence(folder)}))
        state=disk.read(self.root/'state.json');state['completed'][f'{rep:02d}']=dict(metadata_sha256=disk.digest(carry.with_suffix('.json')))
        state['status']='PARTIAL';disk.write(self.root/'state.json',state);disk.update_inventory(self.root);self.clock.commit()
        self.stage('replicate_total',began,unique_records=records)
        print(f'Rep {rep+1:02d}/25 certified: {records} unique records, {time.monotonic()-began:.1f}s. No result-based stopping.',flush=True)


def validate_result(row):
    if set(row)!=set(methods.shapes()):raise ValueError('Unexpected carry fields')
    for k,shape in methods.shapes().items():
        if row[k].shape!=shape or row[k].dtype!=np.int64 or np.any(row[k]<(-1 if k=='inactive' else 0)):
            raise ValueError('Invalid carry shape/type/count: '+k)
    if np.any(row['inactive']>1):raise ValueError('Invalid inactivity')
    for a,b in [('counts','available'),('pairs','pair_n'),('contrasts','contrast_n')]:
        if not np.array_equal(row[a].sum(-1),row[b]):raise ValueError('Integer count conservation failed')
    for c in spec.contexts():
        role='null_eval' if spec.conditions()[c['case']]['kind']=='normal' else 'fault_eval';n=spec.COUNTS[role];k=c['index']
        if np.any(row['available'][k]>n) or np.any(row['pair_n'][k]>n):raise ValueError('Count exceeds planned denominator')
        for pi,(left,right) in enumerate(spec.PAIRS):
            # When availability differs, paired-only margins differ from marginal totals.
            full=(row['available'][k,:,left]==n)&(row['available'][k,:,right]==n)
            diff=row['counts'][k,:,left,1:].sum(-1)-row['counts'][k,:,right,1:].sum(-1)
            if np.any((row['pairs'][k,:,pi,1]-row['pairs'][k,:,pi,2])[full]!=diff[full]):raise ValueError('Paired margin mismatch')
    for c in spec.contrasts():
        k=c['index'];role='null_eval' if spec.conditions()[c['case']]['kind']=='normal' else 'fault_eval';n=spec.COUNTS[role]
        if np.any(row['contrast_n'][k]>n):raise ValueError('Contrast count exceeds planned denominator')
        left,right=c['left'],c['right'];full=(row['available'][left]==n)&(row['available'][right]==n)
        diff=row['counts'][left,...,1:].sum(-1)-row['counts'][right,...,1:].sum(-1)
        if np.any((row['contrasts'][k,...,1]-row['contrasts'][k,...,2])[full]!=diff[full]):raise ValueError('Contrast margin mismatch')


def carry_results(root):
    out=[]
    for rep in range(spec.REPEATS):
        with open_npz(Path(root)/'carry'/f'rep_{rep:02d}.npz') as z:row={k:z[k] for k in z.files}
        validate_result(row);out.append(dict(replicate=rep,**row))
    return out


def main(root,archives,session):
    if sys.platform!='linux' or not Path('/content').is_dir():raise RuntimeError('Scientific main is Colab GPU only')
    root=Path(root);clock=disk.ActiveClock(root/'active_time.json',spec.LIMITS['active_seconds'])
    try:
        if sum(t['count'] for t in spec.tasks(0))!=436392:raise ValueError('Frozen generation plan differs')
        save_json(root/'specification.json',spec.specification());remote=disk.verify_carry(root)
        done=sorted(int(k) for k in disk.read(root/'state.json')['completed']);worker=None
        # Recover a crash after certification but before issuing the shard archive.
        receipts=disk.catalog(root)['archives']
        for start in range(0,len(done)//5*5,5):
            label=f'SHARD_{start//5+1:02d}_COMPLETE'
            if not any(a['status'].startswith(label) for a in receipts):
                for p in disk.snapshot(root,archives,label,range(start,start+5)):print('CHECKPOINT_ZIP='+str(p),flush=True)
        if len(done)<25:
            clock.check();gpu=core.GPU();disk.write(root/'sessions'/session/'environment.json',core.environment(gpu))
            preflight(root,gpu,session,clock);worker=Worker(root,archives,gpu,clock,session)
            for rep in range(len(done),25):
                worker.replicate(rep)
                if (rep+1)%5==0:worker.pack(f'SHARD_{rep//5+1:02d}_COMPLETE',range(rep-4,rep+1))
        clock.check();aggregate_start=time.monotonic();result=methods.aggregate(carry_results(root))
        for name,body in result.items():save_json(root/'analysis'/(name+'.json'),body)
        disk.write(root/'sessions'/session/'aggregate_timing.json',dict(seconds=time.monotonic()-aggregate_start,at=disk.stamp()))
        clock.check()
        disk.write(root/'summary.json',dict(status='COMPLETE',replicates=25,unique_records=10909800,
            unique_evaluation_records=5250000,policy_evaluation_records=57750000,decision_count=693000000,
            raw_not_present_this_runtime=remote,all_scientific_records_certified=True,
            note='Remote completed evidence is receipt-certified, not re-read here. Keep all data parts.'))
        state=disk.read(root/'state.json');state['status']='COMPLETE';disk.write(root/'state.json',state)
        clock.commit();print('COMPLETE: fixed 25 replicates; preparing small final index.',flush=True);return 0
    except BaseException as exc:
        clock.commit();disk.write(root/'sessions'/session/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        state=disk.read(root/'state.json');state['status']='PARTIAL' if isinstance(exc,(TimeoutError,MemoryError,OverflowError)) else 'TECHNICAL_FAILURE'
        disk.write(root/'state.json',state);print(traceback.format_exc(),flush=True)
        return 3 if state['status']=='PARTIAL' else 2


if __name__=='__main__':sys.exit(main(*sys.argv[1:]))
