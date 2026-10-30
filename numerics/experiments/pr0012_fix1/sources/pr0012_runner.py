# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Bounded, resumable B50 worker. Scientific main is Colab GPU only."""
import json,signal,sys,time,traceback,uuid
from pathlib import Path
import numpy as np
import pr0012_spec as spec
import pr0012_methods as methods
import pr0012_io as disk
from pr0012_preflight import audit as audit_preflight_readout
from pr0012_storage import canonical,open_npz,save_npz,save_json
from pr0006_runner import memory,gpu_values
core=spec.core

def safe(x):
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    return x

def chunk_path(root,t,start):return disk.rep_dir(root,t['replicate'])/'chunks'/t['case']/t['role']/f'{start:09d}.npz'

def verify_values(a,case):
    if a.dtype!=np.float64 or a.ndim!=2 or a.shape[1]!=17:raise ValueError('raw17 required')
    valid=np.isfinite(a).all(1);b=a[valid]
    if len(b):
        *_,lam,nu=core.coefficients(spec.conditions()[case]['params'])
        for k,idx in ((0,9),(3,10),(6,11)):
            if not core.compare(b[:,idx,None],(b[:,k+2]-lam*b[:,k]-nu*b[:,k+1])[:,None])['passed']:raise ValueError('Moment identity')
        if not core.identity(b)['passed']:raise ValueError('CR identity')
    return int((~valid).sum())

def read_chunk(root,t,start,spec_hash):
    with open_npz(chunk_path(root,t,start)) as z:
        if set(z.files)!={'values','metadata'}:raise ValueError('Raw keys')
        a=z['values'];m=json.loads(str(z['metadata']))
    if a.shape!=(min(spec.CHUNK,t['count']-start),17) or m['task']!=t or m['start']!=start or m['specification_sha256']!=spec_hash or m['values_sha256']!=disk.sha(a.tobytes()):raise ValueError('Raw identity')
    if verify_values(a,t['case'])!=m['unavailable_rows']:raise ValueError('Raw unavailable count')
    return a

def generate_chunk(root,t,start,spec_hash,gpu):
    if sys.platform!='linux' or not Path('/content').is_dir():raise RuntimeError('No local scientific generation')
    began=time.perf_counter();n=min(spec.CHUNK,t['count']-start)
    z=spec.make_inputs(t['case'],t['role'],t['replicate'],start,n);input_seconds=time.perf_counter()-began
    a,_,tm=gpu_values(gpu,spec.conditions()[t['case']]['params'],z);a=np.ascontiguousarray(a,dtype=np.float64)
    meta=dict(task=t,start=start,count=n,specification_sha256=spec_hash,values_sha256=disk.sha(a.tobytes()),
      unavailable_rows=verify_values(a,t['case']),timing=dict(input_seconds=input_seconds,**tm),at=disk.stamp())
    save_npz(chunk_path(root,t,start),values=a,metadata=np.array(canonical(meta)))

def preflight(root,gpu,session,clock):
    folder=Path(root)/'sessions'/session/'preflight';rows=[];cs=[];gs=[]
    for case,c in spec.conditions().items():
        clock.check();z=spec.make_inputs(case,'preflight',0,0,64)
        cpu,tc=core.calculate(c['params'],z,traces=True);a,tg,tm=gpu_values(gpu,c['params'],z,traces=True)
        checks=dict(values=core.compare(cpu,a),CR=core.identity(a))
        checks.update({k:core.compare(tc[k][:3],tg[k]) for k in tc})
        save_npz(folder/(case+'.npz'),cpu=cpu,gpu=a);cs.append(cpu);gs.append(a)
        rows.append(dict(case=case,checks=checks,timing=tm,input_sha256={k:disk.sha(v.tobytes()) for k,v in z.items()}))
        passed=all(v['passed'] for v in checks.values())
        save_json(folder/'checks'/(case+'.json'),dict(passed=passed,**rows[-1],scientific_records=0))
        if not passed:
            save_json(folder/'result.json',dict(passed=False,status='NUMERIC_FAILURE',rows=rows,scientific_records=0))
            raise ValueError('CPU/GPU preflight mismatch')
    cpu,g=np.concatenate(cs),np.concatenate(gs)
    audit=audit_preflight_readout(cpu,g)
    save_json(folder/'result.json',dict(passed=audit['passed'],rows=rows,records=len(cpu),scientific_records=0,readout=audit))
    if not audit['passed']:raise ValueError('Technical readout mismatch: '+audit['status'])
    print('Float64 preflight '+audit['status']+': 17 x64 technical records; '+str(audit['boundary_branch_flips'])+
          ' boundary branch flips logged, no scientific thresholds changed.',flush=True)

def mixture(normal,faults,order,pos,name):
    if len(normal)!=15999 or not np.array_equal(np.sort(order),np.arange(15999)) or not np.array_equal(np.sort(pos),np.arange(15999)):raise ValueError('Mixture slots')
    data=normal[order].copy();ids=order.copy();origin=np.zeros(len(data),np.uint8)
    for fault,lo,hi,start in spec.recipe(name):
        data[pos[lo:hi]]=faults[fault][start:start+hi-lo];ids[pos[lo:hi]]=np.arange(start,start+hi-lo)
        origin[pos[lo:hi]]=spec.FAULTS.index(fault)+1
    contributions=[]
    for f in range(1,4):
        mask=origin==f
        contributions.append(((data[mask][:,[11,12]]-normal[order[mask]][:,[11,12]]).sum(0)/len(data)).tolist())
    return data,ids,origin,contributions

class Worker:
    def __init__(self,root,archives,gpu,clock,session):
        self.root,self.archives,self.gpu,self.clock=Path(root),Path(archives),gpu,clock
        self.spec_hash=disk.sha(canonical(spec.specification()).encode());self.rep=0;self.new_chunks=0
        self.last_zip=time.monotonic();self.stage_path=self.root/'sessions'/session/'stages.json';self.stages=[]
    def check(self):
        self.clock.check();m=memory(self.gpu)
        if m['rss_high_water_bytes']>spec.LIMITS['rss_bytes'] or m['gpu_reserved_bytes']>spec.LIMITS['gpu_bytes']:raise MemoryError('Memory budget')
        if self.new_chunks%50==0 and sum(p.stat().st_size for p in disk.evidence(self.root))>spec.LIMITS['evidence_bytes']:raise MemoryError('Evidence budget')
        if time.monotonic()-self.last_zip>=900:self.pack('PARTIAL',range(self.rep//5*5,self.rep+1))
    def pack(self,status,reps):
        def expired(*_):raise TimeoutError('Packing exceeded 30 minutes')
        old=signal.signal(signal.SIGALRM,expired);signal.alarm(1800)
        try:
            for p in disk.snapshot(self.root,self.archives,status,list(reps)):print('CHECKPOINT_ZIP='+str(p),flush=True)
        finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
        self.last_zip=time.monotonic();self.clock.commit()
    def chunks(self,t):
        for start in range(0,t['count'],spec.CHUNK):
            self.check()
            if not chunk_path(self.root,t,start).exists():
                def expired(*_):raise TimeoutError('Chunk exceeded 10 minutes')
                old=signal.signal(signal.SIGALRM,expired);signal.alarm(600)
                try:generate_chunk(self.root,t,start,self.spec_hash,self.gpu)
                finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
                self.new_chunks+=1;self.clock.commit()
            yield start,read_chunk(self.root,t,start,self.spec_hash)
    def task(self,t):
        if t['role'].endswith('_eval'):raise ValueError('E must stream')
        return np.concatenate([a for _,a in self.chunks(t)])
    def bits(self,path,values,t,start,frozen,model,cals,method):
        identity=dict(task=t,start=start,count=len(values),frozen_sha256=frozen,method=method,raw_sha256=disk.sha(values.tobytes()))
        if path.exists():
            with open_npz(path) as z:b=z['bits'];meta=json.loads(str(z['metadata']))
            h=meta.pop('bits_sha256')
            if meta!=identity or h!=disk.sha(b.tobytes()):raise ValueError('Saved bit identity')
        else:
            b=np.stack([methods.decide(*model,*c,values,method) for c in cals],axis=1)
            save_npz(path,bits=b,metadata=np.array(canonical(dict(identity,bits_sha256=disk.sha(b.tobytes())))))
        if b.shape!=(len(values),len(cals),2):raise ValueError('Saved bit shape')
        methods.decode(b);return b
    def replicate(self,rep):
        began=time.monotonic();self.rep=rep;folder=disk.rep_dir(self.root,rep);tasks=spec.tasks(rep)
        lookup={(t['case'],t['role']):t for t in tasks};row=methods.empty();models={};cal={};details=[]
        expected={chunk_path(self.root,t,s) for t in tasks for s in range(0,t['count'],spec.CHUNK)}
        if set((folder/'chunks').rglob('*.npz'))-expected:raise ValueError('Unknown raw IDs')
        for base in ('B_M','B_L'):
            values=self.task(lookup[base,'ref']);state,ordered=methods.fit(values);models[base]=(state,ordered)
            audit=projection_audit(state,values)
            save_npz(folder/'models'/(base+'.npz'),state=np.array(canonical(state)),ordered=ordered,
              input_sha256=np.array(disk.sha(values.tobytes())),audit=np.array(canonical(safe(audit))))
            old=self.task(lookup[base,'old_cal']);meta,cut,idx=methods.calibrate(state,ordered,old);cal[base]=(meta,cut)
            save_npz(folder/'old_cal'/(base+'.npz'),metadata=np.array(canonical(meta)),cut=cut,indices=idx,input_sha256=np.array(disk.sha(old.tobytes())))
        for gi,group in enumerate(spec.GROUPS):
            model=models[spec.baseline(group)]
            for role in ('audit','update'):
                normal=self.task(lookup[group,role+'_normal'])
                faults={f:self.task(lookup[group+'__'+f,role+'_fault']) for f in spec.FAULTS}
                order=spec.order(group,role+'_order',rep,len(normal));pos=spec.order(group,role+'_replace',rep,len(normal))
                save_npz(folder/'orders'/(group+'_'+role+'.npz'),normal_order=order,replacement_order=pos)
                for ci,name in enumerate(spec.CANDIDATES):
                    data,ids,origin,contrib=mixture(normal,faults,order,pos,name)
                    states,desc=methods.gates(data,0 if group.endswith('_M') else 1)
                    save_npz(folder/role/group/(name+'.npz'),ids=ids,origin=origin,states=states,
                      provenance=np.array(canonical(dict(normal_task=lookup[group,role+'_normal']['id'],
                       fault_tasks={f:lookup[group+'__'+f,role+'_fault']['id'] for f in spec.FAULTS},
                       input_sha256=disk.sha(data.tobytes())))))
                    details.append(safe(dict(group=group,role=role,candidate=name,**desc,component_contributions=contrib)))
                    if role=='audit':row['gates'][gi,ci]=states
                    else:
                        meta,cut,idx=methods.calibrate(*model,data);cal[group,ci]=(meta,cut)
                        save_npz(folder/'calibration'/group/(name+'.npz'),metadata=np.array(canonical(meta)),cut=cut,indices=idx,
                          input_sha256=np.array(disk.sha(data.tobytes())))
                del normal,faults,data
            data=self.task(lookup[group,'recovery']);meta,cut,idx=methods.calibrate(*model,data);cal[group,'C']=(meta,cut)
            save_npz(folder/'calibration'/group/'C.npz',metadata=np.array(canonical(meta)),cut=cut,indices=idx,input_sha256=np.array(disk.sha(data.tobytes())))
            for j,c in enumerate((cal[spec.baseline(group)],cal[group,0],cal[group,'C'])):
                if c[0]['valid']:row['inactive'][gi,j]=c[0]['lf0_structurally_inactive']
            print(f'rep {rep+1:02d}/50 {group}: independent A/U/C frozen',flush=True)
        frozen={str(p.relative_to(folder)):disk.digest(p) for d in ('models','old_cal','orders','audit','update','calibration') for p in disk.evidence(folder/d)}
        save_json(folder/'CALIBRATION_FROZEN.json',frozen);fh=disk.sha(canonical(frozen).encode());save_json(folder/'gate_details.json',details)
        paths=set();labels=[]
        for gi,group in enumerate(spec.GROUPS):
            cals=[cal[spec.baseline(group)],cal[group,'C']];t=lookup[group,'validation_normal']
            for start,values in self.chunks(t):
                path=folder/'validation'/group/f'{start:09d}.npz';paths.add(path)
                b=self.bits(path,values,t,start,fh,models[spec.baseline(group)],cals,0);p,v=methods.decode(b)
                row['v_counts'][gi,...,0]+=((p>0)&v).sum(0);row['v_counts'][gi,...,1]+=v.sum(0)
            ev=[[methods.evidence(*row['v_counts'][gi,c,ai],20000,1/den) for ai,den in enumerate(spec.ALPHA_DENOMINATORS)] for c in range(2)]
            labels.append(ev)
            for ci in range(6):
                for ai in range(2):row['selection'][gi,ci,ai]=methods.select(row['gates'][gi,ci,1],ci,ev[0][ai]['label'],ev[1][ai]['label'])
        # Immutable causal boundary: persisted BEFORE the first E request.
        receipt=dict(calibration_sha256=fh,validation={str(p.relative_to(folder)):disk.digest(p) for p in sorted(paths)},
          evidence=labels,selection=row['selection'].tolist(),gate_states=row['gates'].tolist(),E_used=False)
        save_json(folder/'SELECTION_BEFORE_E.json',receipt);sh=disk.digest(folder/'SELECTION_BEFORE_E.json')
        readout=[]
        for gi,group in enumerate(spec.GROUPS):
            model=models[spec.baseline(group)];cals=[cal[spec.baseline(group)]]+[cal[group,ci] for ci in range(6)]+[cal[group,'C']]
            for ei,tag in enumerate(('normal',)+spec.FAULTS):
                case=group if ei==0 else group+'__'+tag;t=lookup[case,'null_eval' if ei==0 else 'fault_eval']
                for start,values in self.chunks(t):
                    wp=folder/'decisions'/case/f'{start:09d}_W0.npz';lp=folder/'decisions'/case/f'{start:09d}_LF0.npz';paths.update((wp,lp))
                    w=self.bits(wp,values,t,start,sh,model,cals,0)
                    l=self.bits(lp,values,t,start,sh,model,[cals[i] for i in (0,1,7)],1)
                    methods.accumulate(row,gi,ei,w,l)
                    readout.append(dict(case=case,start=start,**readout_audit(model,[cals[i] for i in (0,1,7)],values,l)))
                print(f'rep {rep+1:02d}/50 {case}: E committed ({t["count"]})',flush=True)
        save_json(folder/'readout_audit.json',safe(readout))
        if any(disk.digest(folder/p)!=h for p,h in frozen.items()) or disk.digest(folder/'SELECTION_BEFORE_E.json')!=sh:raise ValueError('Frozen evidence changed')
        found=set((folder/'decisions').rglob('*.npz'))|set((folder/'validation').rglob('*.npz'))
        if found!=paths or set((folder/'chunks').rglob('*.npz'))!=expected:raise ValueError('Evidence inventory')
        records=sum(len(read_chunk(self.root,t,s,self.spec_hash)) for t in tasks for s in range(0,t['count'],spec.CHUNK))
        if records!=479984:raise ValueError('Unique record total')
        methods.validate(row);carry=self.root/'carry'/f'rep_{rep:02d}.npz';save_npz(carry,**row)
        save_json(carry.with_suffix('.json'),dict(replicate=rep,records=records,specification_sha256=self.spec_hash,
          statistics_sha256=disk.digest(carry),files={str(p.relative_to(folder)):disk.digest(p) for p in disk.evidence(folder)}))
        state=disk.read(self.root/'state.json');state['completed'][f'{rep:02d}']=dict(metadata_sha256=disk.digest(carry.with_suffix('.json')))
        state['status']='PARTIAL';disk.write(self.root/'state.json',state);disk.update_inventory(self.root);self.clock.commit()
        self.stages.append(dict(replicate=rep,seconds=time.monotonic()-began,records=records));disk.write(self.stage_path,self.stages)
        average=sum(x['seconds'] for x in self.stages)/len(self.stages)
        print(f'Rep {rep+1:02d}/50 certified. Session mean {average/60:.1f} min/rep; remaining compute estimate {(49-rep)*average/60:.1f} min (excludes transfers). No outcome-dependent stopping.',flush=True)

def projection_audit(state,values):
    if not state['valid']:return dict(valid=False,reason='reference invalid')
    w=np.array(state['model']['weights']);scale=np.sqrt(np.diag(state['model']['covariance']));ws=w*scale
    rank=int(np.linalg.matrix_rank(ws));out=dict(valid=True,rank=rank,scaled_condition=float(np.linalg.cond(ws)))
    if rank==2 and len(values):
        s,ok=methods.signed(state,values);q=(values[:,[11,12]]-state['model']['mean'])/scale
        out['max_reconstruction_error']=float(np.max(abs(s@np.linalg.inv(ws).T-q)))
    return out

def readout_audit(model,cals,values,bits):
    state,ordered=model;s,valid=methods.signed(state,values);mismatches=0
    for j,(meta,cut) in enumerate(cals):
        if not state['valid'] or not meta['valid']:continue
        for ai in range(2):
            k=int(cut[ai,1,0]);expected=np.zeros(len(values),np.uint8)
            if k<len(ordered):
                g=abs(s[:,0])>ordered[k,0];p=s[:,1]>ordered[k,1]
                if 0 in state['model']['disabled']:g[:]=False
                if 1 in state['model']['disabled']:p[:]=False
                expected=((g&(s[:,0]>=0)).astype(np.uint8)+2*(g&(s[:,0]<0))+4*p).astype(np.uint8)
            expected=np.where(valid,expected|8,0).astype(np.uint8)
            mismatches+=int(np.count_nonzero(expected!=bits[:,j,ai]))
    if mismatches:raise ValueError('Rank/half-plane equivalence failed')
    return dict(records=len(values),rank_halfplane_mismatches=mismatches,**projection_audit(state,values[np.isfinite(values).all(1)]))

def main(root,archives,session):
    if sys.platform!='linux' or not Path('/content').is_dir():raise RuntimeError('Scientific main is Colab GPU only')
    root=Path(root);clock=disk.ActiveClock(root/'active_time.json',spec.LIMITS['active_seconds'])
    try:
        if sum(t['count'] for t in spec.tasks(0))!=479984:raise ValueError('Frozen task plan')
        save_json(root/'specification.json',spec.specification());remote=disk.verify_carry(root)
        done=sorted(int(k) for k in disk.read(root/'state.json')['completed'])
        for start in range(0,len(done)//5*5,5):
            label=f'SHARD_{start//5+1:02d}_COMPLETE'
            if not any(a['status'].startswith(label) for a in disk.catalog(root)['archives']):
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
        result=methods.aggregate(rows)
        for name,body in result.items():save_json(root/'analysis'/(name+'.json'),safe(body))
        clock.check();disk.write(root/'summary.json',dict(status='COMPLETE',replicates=50,unique_records=23999200,
          all_scientific_records_certified=True,raw_not_present_this_runtime=remote,note='Keep ten data shards plus index; receipt-only raw was not re-read.'))
        state=disk.read(root/'state.json');state['status']='COMPLETE';disk.write(root/'state.json',state);clock.commit()
        print('COMPLETE: science finished; browser delivery may still be pending.',flush=True);return 0
    except BaseException as exc:
        clock.commit();disk.write(root/'sessions'/session/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        state=disk.read(root/'state.json');state['status']='PARTIAL' if isinstance(exc,(TimeoutError,MemoryError,OverflowError,KeyboardInterrupt)) else 'TECHNICAL_FAILURE'
        disk.write(root/'state.json',state);print(traceback.format_exc(),flush=True);return 3 if state['status']=='PARTIAL' else 2

if __name__=='__main__':sys.exit(main(*sys.argv[1:]))
