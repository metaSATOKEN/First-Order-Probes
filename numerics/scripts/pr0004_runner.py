# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""GPU scientific worker. Main is Colab-only; no work on import."""
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
import uuid
import zipfile
import numpy as np
import probe_backend_check_v1 as core
import pr0004_io as disk
import pr0004_methods as methods
import pr0004_spec as design


def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)


def save_npz(path,**arrays):
    if Path(path).exists(): raise FileExistsError('Completed evidence already exists: '+str(path))
    b=io.BytesIO(); np.savez(b,**arrays); disk.atomic(path,b.getvalue())


def open_npz(path):
    """Reject malformed headers before allocation, also on same-runtime resumes."""
    with zipfile.ZipFile(path) as z:
        infos=z.infolist(); names=[i.filename for i in infos]
        if len(names)!=len(set(names)) or len(names)>64 or sum(i.file_size for i in infos)>32*1024**2:
            raise ValueError('Invalid/oversized NPZ')
        for item in infos:
            if '/' in item.filename or not item.filename.endswith('.npy'): raise ValueError('Invalid NPZ member')
            with z.open(item) as f:
                version=np.lib.format.read_magic(f)
                reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                if reader is None: raise ValueError('Unsupported NPY')
                shape,_,dtype=reader(f)
                if dtype.hasobject or dtype.kind not in 'fiuU' or len(shape)>3 or any(n<0 for n in shape):
                    raise ValueError('Invalid array type/shape')
                if math.prod(shape)*dtype.itemsize!=item.file_size-f.tell(): raise ValueError('NPY size mismatch')
    return np.load(path,allow_pickle=False)


def chunk_path(root,task,start): return Path(root)/'chunks'/task['id']/f'{start:09d}.npz'


def verify_values(values,case):
    a=methods.validate(values)
    _,_,_,lam,nu=core.coefficients(design.conditions()[case]['params'])
    for k,idx in [(0,9),(3,10),(6,11)]:
        want=a[:,k+2]-lam*a[:,k]-nu*a[:,k+1]
        if not core.compare(a[:,idx:idx+1],want[:,None])['passed']:
            raise ValueError('Saved moment identity mismatch')
    if not core.identity(a)['passed']: raise ValueError('CR expansion mismatch')


def read_chunk(root,task,start,spec_hash):
    with open_npz(chunk_path(root,task,start)) as z:
        if set(z.files)!={'values','metadata'}: raise ValueError('Unexpected chunk fields')
        a=z['values']; meta=json.loads(str(z['metadata']))
    count=min(design.CHUNK,task['count']-start)
    if (a.shape!=(count,17) or meta['task']!=task or meta['start']!=start or
        meta['specification_sha256']!=spec_hash or meta['values_sha256']!=disk.sha(a.tobytes())):
        raise ValueError('Chunk identity/hash mismatch')
    verify_values(a,task['case'])
    return a,meta


def load_task(root,task,spec_hash):
    return np.concatenate([read_chunk(root,task,s,spec_hash)[0] for s in range(0,task['count'],design.CHUNK)])


def inventory(root,tasks,spec_hash):
    expected={chunk_path(root,t,s):(t,s) for t in tasks for s in range(0,t['count'],design.CHUNK)}
    actual=set((Path(root)/'chunks').rglob('*.npz'))
    if actual-set(expected): raise ValueError('Unknown/duplicate task or chunk identifiers')
    count=0
    for p in sorted(actual):
        a,_=read_chunk(root,*expected[p],spec_hash); count+=len(a)
    return dict(records=count,chunks=len(actual),expected_chunks=len(expected))


def memory(gpu=None):
    rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    return dict(rss_high_water_bytes=rss,gpu_reserved_bytes=0 if gpu is None else int(gpu.cp.get_default_memory_pool().total_bytes()))


def limits(root,gpu,clock,check_disk=False):
    clock.check(); m=memory(gpu)
    if m['rss_high_water_bytes']>design.LIMITS['rss_bytes']: raise MemoryError('RSS exceeds fixed budget')
    if m['gpu_reserved_bytes']>design.LIMITS['gpu_reserved_bytes']: raise MemoryError('GPU reservation exceeds fixed budget')
    if check_disk and sum(p.stat().st_size for p in disk.evidence(root))>design.LIMITS['workspace_bytes']:
        raise RuntimeError('Evidence workspace exceeds fixed budget')
    return m


def gpu_values(gpu,p,inputs,traces=False):
    cp=gpu.cp; timings={}; t=time.perf_counter()
    device={k:cp.asarray(v) for k,v in inputs.items()}; cp.cuda.get_current_stream().synchronize()
    timings['h2d_seconds']=time.perf_counter()-t
    a,b=cp.cuda.Event(),cp.cuda.Event(); t=time.perf_counter(); a.record()
    values,trace=core.calculate(p,device,cp,gpu.filt,traces=traces)
    b.record(); b.synchronize()
    timings['compute_seconds']=time.perf_counter()-t
    timings['gpu_event_seconds']=cp.cuda.get_elapsed_time(a,b)/1000
    t=time.perf_counter(); values=cp.asnumpy(values)
    trace={k:cp.asnumpy(v[:3]) for k,v in trace.items()}
    timings['d2h_seconds']=time.perf_counter()-t
    return values,trace,timings


def preflight(root,gpu,launch,clock):
    folder=Path(root)/'preflight'/launch; rows=[]
    for case,c in design.conditions().items():
        clock.check(); z=design.make_inputs(case,'preflight',1000,0,3)
        cpu,tr_cpu=core.calculate(c['params'],z,traces=True)
        value,tr_gpu,timing=gpu_values(gpu,c['params'],z,traces=True)
        checks={'values':core.compare(cpu,value)}
        checks.update({k:core.compare(tr_cpu[k],tr_gpu[k]) for k in tr_cpu})
        checks['CR']=core.identity(value)
        save_npz(folder/(case+'.npz'),cpu=cpu,gpu=value,
                 **{'input_'+k:v for k,v in z.items()},
                 **{'cpu_'+k:v for k,v in tr_cpu.items()},**{'gpu_'+k:v for k,v in tr_gpu.items()})
        row=dict(case=case,checks=checks,timing=timing,passed=all(x['passed'] for x in checks.values()))
        rows.append(row)
        if not row['passed']:
            disk.write(folder/'result.json',dict(passed=False,rows=rows)); raise ValueError('GPU preflight mismatch')
    disk.write(folder/'result.json',dict(passed=True,rows=rows,records=150,scientific_records=0,**memory(gpu)))
    print('CPU/GPU float64 preflight: 50 conditions passed.',flush=True)


def generate_chunk(root,task,start,spec_hash,gpu):
    case=task['case']; p=design.conditions()[case]['params']; count=min(design.CHUNK,task['count']-start)
    batch=design.batch_size(case); total={k:0. for k in ('input_seconds','h2d_seconds','compute_seconds','gpu_event_seconds','d2h_seconds')}
    chunks=[]; began=time.perf_counter()
    sample=(task['replicate']==0 and task['role'] in ('ref','fault','control') and start==0)
    sample_path=Path(root)/'waveforms'/('r00_'+case+'.npz')
    for offset in range(0,count,batch):
        t=time.perf_counter(); z=design.make_inputs(case,task['role'],task['replicate'],start+offset,min(batch,count-offset))
        total['input_seconds']+=time.perf_counter()-t
        a,tr,tm=gpu_values(gpu,p,z,traces=sample and offset==0)
        for k,v in tm.items(): total[k]+=v
        chunks.append(a)
        if sample and offset==0 and not sample_path.exists():
            save_npz(sample_path,values=a[:3],**{'input_'+k:v[:3] for k,v in z.items()},
                     **{'trace_'+k:v for k,v in tr.items()})
        del z,tr,a
    a=np.ascontiguousarray(np.concatenate(chunks),dtype=np.float64); verify_values(a,case)
    total['generation_scoring_seconds']=time.perf_counter()-began
    meta=dict(task=task,start=start,count=count,specification_sha256=spec_hash,
              values_sha256=disk.sha(a.tobytes()),timing=total,memory=memory(gpu),at=disk.stamp())
    save_start=time.perf_counter()
    save_npz(chunk_path(root,task,start),values=a,metadata=np.array(canonical(meta)))
    timing=dict(**total,save_seconds=time.perf_counter()-save_start)
    disk.write(chunk_path(root,task,start).with_suffix('.timing.json'),timing)
    return timing


def save_reference(folder,model,ordered):
    disk.write(folder/'model.json',model)
    save_npz(folder/'reference_ranks.npz',**ordered)


def evaluate_rep(root,rep,spec_hash,check=lambda:None):
    """No fault/normal evaluation input enters either fit_reference or calibrate."""
    done=Path(root)/'analysis'/'repeats'/f'{rep:02d}'/'done.json'
    if done.exists():
        record=disk.read(done); folder=done.parent/record['attempt']
        if record['replicate']!=rep or record['specification_sha256']!=spec_hash:
            raise ValueError('Analysis checkpoint identity mismatch')
        disk.safe_name(record['attempt'])
        if set(record['files'])!={str(p.relative_to(folder)) for p in disk.evidence(folder)}:
            raise ValueError('Analysis checkpoint inventory differs')
        for name,h in record['files'].items():
            if disk.digest(folder/disk.safe_name(name))!=h: raise ValueError('Analysis checkpoint corrupt')
        result=disk.read(folder/'results.json')
        if result['replicate']!=rep: raise ValueError('Analysis replicate mismatch')
        return result
    folder=done.parent/('attempt_'+uuid.uuid4().hex[:12]); folder.mkdir(parents=True)
    lookup={(t['case'],t['role']):t for t in design.tasks(rep+1) if t['replicate']==rep}
    def load(case,role): return load_task(root,lookup[case,role],spec_hash)
    models={}; times=dict(reference_learning_seconds=0.,calibration_seconds=0.,evaluation_seconds=0.,save_seconds=0.)
    for group,c in design.conditions().items():
        if c['kind']!='normal': continue
        check(); ref=load(group,'ref'); cal=load(group,'cal'); p=c['params']
        t=time.perf_counter()
        model,ordered=methods.fit_reference(ref,design.nominal_nu(p['dt'],p['tau_i'],p['tau_j']))
        times['reference_learning_seconds']+=time.perf_counter()-t
        t=time.perf_counter(); threshold=methods.calibrate(model,ordered,cal)
        times['calibration_seconds']+=time.perf_counter()-t
        own=(model,ordered,threshold)
        if group.startswith('B_'): variants={'B':own}
        else:
            base=models[design.baseline(group)]['B']; t=time.perf_counter()
            updated=methods.calibrate(base[0],base[1],cal)
            times['calibration_seconds']+=time.perf_counter()-t
            variants={'F':base,'C':(base[0],base[1],updated),'R':own}
        models[group]=variants
        t=time.perf_counter(); save_reference(folder/'models'/group,model,ordered)
        for policy,(_,_,threshold) in variants.items():
            disk.write(folder/'models'/group/(policy+'_calibration.json'),threshold)
        times['save_seconds']+=time.perf_counter()-t
    rates=[]; paired=[]; updates=[]; patterns=[]; saturation=[]
    for case,c in design.conditions().items():
        check(); role='null' if c['kind']=='normal' else c['kind']; data=load(case,role); flags_by={}
        for policy,(model,ordered,threshold) in models[c['group']].items():
            t=time.perf_counter(); flags,bits,ranks=methods.decide(model,ordered,threshold,data)
            times['evaluation_seconds']+=time.perf_counter()-t
            flags_by[policy]=flags
            t=time.perf_counter(); save_npz(folder/'decisions'/(case+'_'+policy+'.npz'),bits=bits,**ranks)
            times['save_seconds']+=time.perf_counter()-t
            base=dict(replicate=rep,case=case,group=c['group'],kind=c['kind'],policy=policy,n=len(data))
            for i,name in enumerate(design.METHODS):
                k=int(flags[:,i].sum()); lo,hi=methods.wilson(k,len(flags))
                rates.append(dict(**base,method=name,count=k,rate=k/len(flags),misses=len(flags)-k,
                                  wilson_lower=lo,wilson_upper=hi))
            for i,name in enumerate(design.METHODS):
                if name=='LF': continue
                d=methods.overlap(flags[:,4],flags[:,i]); d.pop('n')
                paired.append(dict(**base,left='LF',right=name,**d))
            patterns.append(dict(**base,main32=np.bincount(bits&31,minlength=32).tolist(),
                                 auxiliary64=np.bincount(bits,minlength=64).tolist()))
            saturation.append(dict(**base,upper_rank=model['n_reference'],
                by_coordinates={k:(v==model['n_reference']).sum(axis=0).tolist() for k,v in ranks.items()}))
        if 'R' in flags_by:
            for other in ('F','C'):
                for i,name in enumerate(design.METHODS):
                    d=methods.overlap(flags_by['R'][:,i],flags_by[other][:,i]); d.pop('n')
                    updates.append(dict(replicate=rep,case=case,kind=c['kind'],n=len(data),method=name,
                                        left='R',right=other,**d))
    result=dict(replicate=rep,rates=rates,paired=paired,updates=updates,patterns=patterns,saturation=saturation,timing=times)
    disk.write(folder/'results.json',result)
    files={str(p.relative_to(folder)):disk.digest(p) for p in disk.evidence(folder)}
    disk.write(done,dict(replicate=rep,specification_sha256=spec_hash,attempt=folder.name,files=files,at=disk.stamp()))
    return result


def aggregate(results):
    if len(results)!=design.REPEATS or sorted(r['replicate'] for r in results)!=list(range(design.REPEATS)):
        raise ValueError('All 50 outer replicates required')
    def table(key,fields,value,family=1):
        grouped={}
        for result in results:
            for row in result[key]:
                group=tuple(row[f] for f in fields); grouped.setdefault(group,[]).append(row[value])
        output=[]
        for group,x in sorted(grouped.items()):
            if len(x)!=design.REPEATS: raise ValueError('Missing outer-repeat cell')
            output.append(dict(zip(fields,group),**methods.outer_interval(x,family)))
        return output
    rates=table('rates',['case','kind','policy','method'],'rate')
    paired=table('paired',['case','kind','policy','left','right'],'difference')
    updates=table('updates',['case','kind','method','left','right'],'difference')
    overlap_summary=[]
    paired_groups={}
    for result in results:
        for row in result['paired']:
            key=tuple(row[k] for k in ('case','kind','policy','left','right'))
            paired_groups.setdefault(key,[]).append(row)
    for key,rows in sorted(paired_groups.items()):
        if len(rows)!=design.REPEATS: raise ValueError('Incomplete overlap series')
        cells={k:dict(total_count=sum(row[k] for row in rows),
                      **methods.outer_interval([row[k]/row['n'] for row in rows]))
               for k in ('both','gained','lost','neither')}
        overlap_summary.append(dict(zip(('case','kind','policy','left','right'),key),
                                    total_records=sum(row['n'] for row in rows),cells=cells))
    # Paired normal-source FPR changes use the same outer-replicate baseline fit.
    fpr_changes=[]
    for group,c in design.conditions().items():
        if c['kind']!='normal' or group.startswith('B_'): continue
        for policy in ('F','C','R'):
            for name in design.METHODS:
                x=[]
                for result in results:
                    shifted=next(v['rate'] for v in result['rates'] if v['case']==group and v['policy']==policy and v['method']==name)
                    base=next(v['rate'] for v in result['rates'] if v['case']==design.baseline(group) and v['policy']=='B' and v['method']==name)
                    x.append(shifted-base)
                fpr_changes.append(dict(case=group,policy=policy,method=name,baseline=design.baseline(group),
                                        **methods.outer_interval(x)))
    primary=[]
    for case,left,right in design.primary():
        x=[next(v['difference'] for v in r['paired'] if v['case']==case and v['policy']=='B' and v['right']==right) for r in results]
        row=dict(case=case,left=left,right=right,**methods.outer_interval(x,12))
        row['classification']='improvement' if row['lower'] is not None and row['lower']>0 else 'loss' if row['upper'] is not None and row['upper']<0 else 'unresolved'
        primary.append(row)
    fpr=[]
    for row in rates:
        if row['kind']!='normal' or row['method']=='LF-D': continue
        x=[next(v['rate'] for v in r['rates'] if all(v[k]==row[k] for k in ('case','policy','method'))) for r in results]
        entry=dict(case=row['case'],policy=row['policy'],method=row['method'],**methods.outer_interval(x,140))
        entry['excess_flag']=entry['lower'] is not None and entry['lower']>.01
        entry['pointwise95']={k:row[k] for k in ('lower','upper')}
        entry['ratio_to_nominal']=entry['mean']/.01; entry['difference_from_nominal']=entry['mean']-.01
        fpr.append(entry)
    if len(primary)!=12 or len(fpr)!=140: raise AssertionError('Inferential family size mismatch')
    return dict(primary12=primary,fpr140=fpr,rates=rates,paired_differences=paired,policy_differences=updates,
                fpr_changes_from_baseline=fpr_changes,overlap=overlap_summary,
                units='Rates/proportions and differences are fractions, not percent/pp.',
                caveats=['Nominal 1% calibration, not exactly matched empirical FPR.',
                         'Normal source changes are not faults.', 'LF-D is supplementary.',
                         'Zero-SD t intervals are undefined.','Two separate multiplicity families.'])


def main(root):
    if sys.platform!='linux' or not Path('/content').is_dir():
        raise RuntimeError('Main simulation disabled locally; use Colab GPU')
    root=Path(root); launch=uuid.uuid4().hex[:12]; clock=disk.ActiveClock(root/'active_time.json',design.LIMITS['active_seconds'])
    def interrupted(signum,frame): raise InterruptedError('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    status='TECHNICAL_FAILURE'; gpu=None; started=time.perf_counter()
    try:
        if disk.digest(Path(core.__file__))!=design.CORE_SHA: raise ValueError('Accepted GPU core changed')
        spec=design.specification(); spec_hash=disk.sha(canonical(spec).encode()); path=root/'specification.json'
        if path.exists():
            if canonical(disk.read(path))!=canonical(spec): raise ValueError('Frozen specification mismatch')
        else: disk.write(path,spec)
        clock.check(); gpu=core.GPU(); env=core.environment(gpu)
        disk.write(root/'sessions'/launch/'environment.json',env)
        preflight(root,gpu,launch,clock)
        tasklist=design.tasks(); before=inventory(root,tasklist,spec_hash)
        print('Verified saved records:',before['records'],'/ 33499400',flush=True)
        done_records=before['records']; new_chunks=0
        for rep in range(design.REPEATS):
            limits(root,gpu,clock,True)
            for task in (t for t in tasklist if t['replicate']==rep):
                limits(root,gpu,clock)
                for start in range(0,task['count'],design.CHUNK):
                    if chunk_path(root,task,start).exists(): continue  # already verified by inventory
                    limits(root,gpu,clock); begin=time.perf_counter()
                    tm=generate_chunk(root,task,start,spec_hash,gpu)
                    new_chunks+=1; done_records+=min(design.CHUNK,task['count']-start)
                    disk.write(root/'progress.json',dict(records=done_records,expected=33499400,
                               latest_task=task,latest_start=start,latest_timing=tm,at=disk.stamp()))
                    clock.commit()
                    if time.perf_counter()-begin>design.LIMITS['task_seconds']: raise TimeoutError('Chunk time budget exceeded')
            print(f'Generation {rep+1}/50 complete; records={done_records:,}; active={clock.seconds()/60:.1f} min',flush=True)
            checkpoint=root/'checkpoints'/f'{rep+1:02d}.json'
            if rep+1 in (10,20,30,40) and not checkpoint.exists():
                t=time.perf_counter(); archive=disk.snapshot(root,root.parents[1],'PARTIAL')
                disk.write(checkpoint,dict(path=str(archive),seconds=time.perf_counter()-t,at=disk.stamp()))
                clock.commit(); print('CHECKPOINT_ZIP='+str(archive),flush=True)
        counts=inventory(root,tasklist,spec_hash)
        if counts['records']!=33499400 or counts['chunks']!=10400: raise ValueError('Main record inventory incomplete')
        print('All scientific records saved. Learning/calibration/evaluation begins.',flush=True)
        results=[]
        for rep in range(design.REPEATS):
            results.append(evaluate_rep(root,rep,spec_hash,lambda:limits(root,gpu,clock)))
            clock.commit(); print(f'Analysis {rep+1}/50 saved',flush=True)
        t=time.perf_counter(); summary=aggregate(results)
        disk.write(root/'analysis'/'summary.json',summary)
        aggregation_seconds=time.perf_counter()-t
        # Timing sidecars are committed after the corresponding raw values. An
        # abrupt loss between them does not invalidate the raw chunk: report the gap.
        timing_files=sorted((root/'chunks').rglob('*.timing.json'))
        totals={}
        for p in timing_files:
            for key,value in disk.read(p).items(): totals[key]=totals.get(key,0.)+value
        disk.write(root/'analysis'/'costs.json',dict(chunk_timing_totals=totals,
          timing_chunks=len(timing_files),missing_chunk_save_timings=10400-len(timing_files),
          analysis_by_replicate=[dict(replicate=r['replicate'],**r['timing']) for r in results],
          aggregation_seconds=aggregation_seconds,additional_three_moments='included in GPU calculation; not separately timed',
          final_archive='archive_receipt sidecar is outside ZIP to avoid recursive self-inclusion',
          gpu_memory='pool snapshots, not a guaranteed global peak'))
        limits(root,gpu,clock,True)
        status='COMPLETE'
        disk.write(root/'summary.json',dict(status=status,records=counts['records'],chunks=counts['chunks'],
                   repeats=50,primary_comparisons=12,fpr_series=140,scientific_exclusions=0,
                   previous_error_records=len(list(root.rglob('error.json'))),
                   aggregation_seconds=aggregation_seconds,active_seconds=clock.seconds(),**memory(gpu)))
    except BaseException as exc:
        status='PARTIAL' if isinstance(exc,(TimeoutError,InterruptedError,KeyboardInterrupt)) else 'TECHNICAL_FAILURE'
        disk.write(root/'sessions'/launch/'error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        disk.write(root/'summary.json',dict(status=status,active_seconds=clock.seconds(),planned_records=33499400,
                   error=str(exc),completed_counts_not_yet_accepted=True))
        print(traceback.format_exc(),flush=True)
    finally:
        clock.commit(); disk.write(root/'sessions'/launch/'session.json',dict(status=status,
           session_seconds=time.perf_counter()-started,cumulative_active_seconds=clock.seconds(),at=disk.stamp(),**memory(gpu)))
    return 0 if status=='COMPLETE' else 2


if __name__=='__main__':
    sys.exit(main(Path(sys.argv[1])))
