# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Streaming PR-0009 worker. Scientific execution is restricted to Colab GPU."""
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid
import numpy as np
import pr0009_io as disk
import pr0009_spec as spec
import pr0009_methods as methods
from pr0006_runner import memory,gpu_values
from pr0009_storage import canonical,open_npz,save_npz,save_json
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
        if t["role"].endswith("_eval"):raise ValueError("Evaluation must stream")
        return np.concatenate([a for _,a in self.chunks(t)])

    def saved_bits(self,path,values,task,start,frozen_hash,models,cals):
        identity=dict(task=task,start=start,count=len(values),calibration_sha256=frozen_hash,
            raw_sha256=disk.sha(values.tobytes()))
        if path.exists():
            with open_npz(path) as z:
                if set(z.files)!={'packed','metadata'}:raise ValueError('Decision fields')
                bits=z['packed'];meta=json.loads(str(z['metadata']))
            h=meta.pop('packed_sha256',None)
            if meta!=identity or h!=disk.sha(bits.tobytes()):raise ValueError('Decision provenance changed')
        else:
            state,ordered=models
            bits=np.stack([methods.decide(state,ordered,*c,values) for c in cals],axis=1)
            save_npz(path,packed=bits,metadata=np.array(canonical(dict(identity,packed_sha256=disk.sha(bits.tobytes())))))
        if bits.dtype!=np.uint32 or bits.shape!=(len(values),len(cals),2):raise ValueError('Decision shape')
        return bits

    def replicate(self,rep):
        began=time.monotonic();self.rep=rep;folder=disk.rep_dir(self.root,rep);tasks=spec.tasks(rep)
        lookup={(t['case'],t['role']):t for t in tasks};row=methods.empty();models={};cal={};gate_details=[]
        expected={chunk_path(self.root,t,s) for t in tasks for s in range(0,t['count'],spec.CHUNK)}
        if set((folder/'chunks').rglob('*.npz'))-expected:raise ValueError('Unknown raw IDs')
        for base in ('B_M','B_L'):
            values=self.task(lookup[base,'ref']);state,ordered=methods.fit(values);models[base]=(state,ordered)
            save_npz(folder/'models'/(base+'.npz'),state=np.array(canonical(state)),ordered=ordered,
                input_sha256=np.array(disk.sha(values.tobytes())))
            old=self.task(lookup[base,'old_cal']);meta,cut,idx=methods.calibrate(state,ordered,old);cal[base]=(meta,cut)
            save_npz(folder/'old_cal'/(base+'.npz'),metadata=np.array(canonical(meta)),cut=cut,indices=idx,
                input_sha256=np.array(disk.sha(old.tobytes())))
        for gi,group in enumerate(spec.groups()):
            stage=time.monotonic();state,ordered=models[spec.baseline(group)];old=cal[spec.baseline(group)]
            for role in ('audit','update'):
                normal=self.task(lookup[group,role+'_normal'])
                faults={f:self.task(lookup[group+'__'+f,role+'_fault']) for f in spec.FAULTS}
                order=spec.order(group,role+'_order',rep,len(normal));pos=spec.order(group,role+'_replace',rep,len(normal))
                save_npz(folder/'orders'/(group+'_'+role+'.npz'),normal_order=order,replacement_order=pos)
                for ci,(fault,n) in enumerate(spec.CANDIDATES):
                    data=normal[order].copy();ids=order.copy();origin=np.zeros(len(data),np.uint8)
                    if n:
                        data[pos[:n]]=faults[fault][:n];ids[pos[:n]]=np.arange(n);origin[pos[:n]]=1
                    oldbits=methods.decide(state,ordered,*old,data);patterns,valid=methods.decode(oldbits)
                    scores,desc=methods.batch_scores(data,patterns[:,0,0]>0,valid[:,0,0])
                    states=methods.gate(scores,0 if group.endswith('_M') else 1)
                    path=folder/role/group/f'candidate_{ci}.npz'
                    provenance=dict(normal_task=lookup[group,role+'_normal']['id'],
                        fault_task=lookup[group+'__'+fault,role+'_fault']['id'] if n else None,
                        replacement_positions=pos[:n].tolist(),input_sha256=disk.sha(data.tobytes()))
                    save_npz(path,scores=scores,states=states,ids=ids,origin=origin,
                        provenance=np.array(canonical(provenance)))
                    gate_details.append(dict(group=group,role=role,candidate=ci,scores=scores.tolist(),details=desc))
                    if role=='audit':row['gate_states'][gi,ci]=states
                    else:
                        row['same_states'][gi,ci]=states[2]
                        meta,cut,idx=methods.calibrate(state,ordered,data);cal[group,ci]=(meta,cut)
                        save_npz(folder/'calibration'/group/f'candidate_{ci}.npz',metadata=np.array(canonical(meta)),
                            cut=cut,indices=idx,input_sha256=np.array(provenance['input_sha256']))
                del normal,faults,data
            for j,c in enumerate([old]+[cal[group,ci] for ci in range(7)]):
                if c[0]['valid']:row['inactive'][gi,j]=c[0]['lf0_structurally_inactive']
            self.stage('candidate_audit_and_calibration',stage,group=group)
            print(f'rep {rep+1:02d}/50 | {group}: A and U independently frozen',flush=True)
        frozen={str(p.relative_to(folder)):disk.digest(p) for d in ('models','old_cal','orders','audit','update','calibration')
                for p in disk.evidence(folder/d)}
        save_json(folder/'CALIBRATION_FROZEN.json',frozen);freeze_hash=disk.sha(canonical(frozen).encode())
        save_json(folder/'gate_details.json',json_safe(gate_details))
        # Neither V nor E has been requested before the immutable freeze.
        vbits={};decision_paths=set()
        for gi,group in enumerate(spec.groups()):
            stage=time.monotonic();model=models[spec.baseline(group)];old=cal[spec.baseline(group)]
            pos=spec.order(group,'validation_replace',rep,spec.COUNTS['validation_normal'])[:spec.COUNTS['validation_fault']]
            save_npz(folder/'validation'/(group+'_positions.npz'),positions=pos)
            for tag in ('normal',)+spec.FAULTS:
                case=group if tag=='normal' else group+'__'+tag
                role='validation_normal' if tag=='normal' else 'validation_fault';t=lookup[case,role];parts=[]
                for start,values in self.chunks(t):
                    p=folder/'validation'/case/f'{start:09d}.npz';decision_paths.add(p)
                    parts.append(self.saved_bits(p,values,t,start,freeze_hash,model,[old])[:,0])
                vbits[group,tag]=np.concatenate(parts)
            for vi,tag in enumerate(('normal',)+spec.FAULTS):
                bits=vbits[group,'normal'].copy()
                if vi:bits[pos]=vbits[group,tag]
                pat,valid=methods.decode(bits);vindex=0 if vi==0 else vi+1
                row['v_counts'][gi,vindex,...,0]=((pat>0)&valid).sum(0)
                row['v_counts'][gi,vindex,...,1]=valid.sum(0)
            self.stage('validation',stage,group=group)
        for gi,group in enumerate(spec.groups()):
            if not group.startswith('B_'):
                bits=vbits[spec.baseline(group),'normal'];pat,valid=methods.decode(bits)
                row['v_counts'][gi,1,...,0]=((pat>0)&valid).sum(0);row['v_counts'][gi,1,...,1]=valid.sum(0)
        del vbits
        for gi,group in enumerate(spec.groups()):
            model=models[spec.baseline(group)];cals=[cal[spec.baseline(group)]]+[cal[group,ci] for ci in range(7)]
            for ei,tag in enumerate(('normal',)+spec.FAULTS):
                stage=time.monotonic();case=group if ei==0 else group+'__'+tag
                t=lookup[case,'null_eval' if ei==0 else 'fault_eval']
                for start,values in self.chunks(t):
                    p=folder/'decisions'/case/f'{start:09d}.npz';decision_paths.add(p)
                    bits=self.saved_bits(p,values,t,start,freeze_hash,model,cals)
                    methods.accumulate(row,gi,ei,bits);self.clock.commit()
                self.stage('evaluation',stage,case=case)
                print(f'rep {rep+1:02d}/50 | {case}: E saved ({t["count"]} records)',flush=True)
        if any(disk.digest(folder/p)!=h for p,h in frozen.items()):raise ValueError('Calibration changed during V/E')
        found=set((folder/'decisions').rglob('*.npz'))|{p for p in (folder/'validation').rglob('*.npz') if not p.name.endswith('_positions.npz')}
        if found!=decision_paths or set((folder/'chunks').rglob('*.npz'))!=expected:raise ValueError('Evidence inventory differs')
        records=sum(len(read_chunk(self.root,t,s,self.spec_hash)) for t in tasks for s in range(0,t['count'],spec.CHUNK))
        if records!=sum(t['count'] for t in tasks):raise ValueError('Record total differs')
        methods.validate(row);carry=self.root/'carry'/f'rep_{rep:02d}.npz';save_npz(carry,**row)
        save_json(carry.with_suffix('.json'),dict(replicate=rep,records=records,specification_sha256=self.spec_hash,
            statistics_sha256=disk.digest(carry),calibration_unchanged=True,
            files={str(p.relative_to(folder)):disk.digest(p) for p in disk.evidence(folder)}))
        state=disk.read(self.root/'state.json');state['completed'][f'{rep:02d}']=dict(metadata_sha256=disk.digest(carry.with_suffix('.json')))
        state['status']='PARTIAL';disk.write(self.root/'state.json',state);disk.update_inventory(self.root);self.clock.commit()
        self.stage('replicate_total',began,unique_records=records)
        print(f'Rep {rep+1:02d}/50 certified: {records} records. No outcome-based stopping.',flush=True)

def json_safe(x):
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,dict):return {k:json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [json_safe(v) for v in x]
    return x

def main(root,archives,session):
    if sys.platform!='linux' or not Path('/content').is_dir():raise RuntimeError('Scientific main is Colab GPU only')
    root=Path(root);clock=disk.ActiveClock(root/'active_time.json',spec.LIMITS['active_seconds'])
    try:
        if sum(t['count'] for t in spec.tasks(0))!=620784:raise ValueError('Frozen generation plan differs')
        save_json(root/'specification.json',spec.specification());remote=disk.verify_carry(root)
        done=sorted(int(k) for k in disk.read(root/'state.json')['completed'])
        receipts=disk.catalog(root)['archives']
        for start in range(0,len(done)//5*5,5):
            label=f'SHARD_{start//5+1:02d}_COMPLETE'
            if not any(a['status'].startswith(label) for a in receipts):
                for p in disk.snapshot(root,archives,label,range(start,start+5)):print('CHECKPOINT_ZIP='+str(p),flush=True)
        if len(done)<50:
            clock.check();gpu=core.GPU();disk.write(root/'sessions'/session/'environment.json',core.environment(gpu))
            preflight(root,gpu,session,clock);worker=Worker(root,archives,gpu,clock,session)
            for rep in range(len(done),50):
                worker.replicate(rep)
                if (rep+1)%5==0:worker.pack(f'SHARD_{rep//5+1:02d}_COMPLETE',range(rep-4,rep+1))
        rows=[]
        for rep in range(50):
            with open_npz(root/'carry'/f'rep_{rep:02d}.npz') as z:rows.append({k:z[k] for k in z.files})
        started=time.monotonic();result=methods.aggregate(rows)
        for name,body in result.items():save_json(root/'analysis'/(name+'.json'),json_safe(body))
        disk.write(root/'sessions'/session/'aggregate_timing.json',dict(seconds=time.monotonic()-started))
        clock.check();disk.write(root/'summary.json',dict(status='COMPLETE',replicates=50,unique_records=31039200,
            decision_count=1008000000,validation_decision_count=82800000,all_scientific_records_certified=True,
            raw_not_present_this_runtime=remote,note='Keep every data shard; receipt-only evidence is not re-read here.'))
        state=disk.read(root/'state.json');state['status']='COMPLETE';disk.write(root/'state.json',state);clock.commit()
        print('COMPLETE: all 50 replicates. Preparing final index; browser queue may still be active.',flush=True);return 0
    except BaseException as exc:
        clock.commit();disk.write(root/'sessions'/session/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        state=disk.read(root/'state.json');state['status']='PARTIAL' if isinstance(exc,(TimeoutError,MemoryError,OverflowError)) else 'TECHNICAL_FAILURE'
        disk.write(root/'state.json',state);print(traceback.format_exc(),flush=True);return 3 if state['status']=='PARTIAL' else 2

if __name__=='__main__':sys.exit(main(*sys.argv[1:]))
