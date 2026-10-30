# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""PR-0005: frozen, role-separated analysis of saved 17-column observations only."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import resource
import sys
import time
import zipfile
import numpy as np
from scipy.stats import t
from analyze_pr0004_lowfpr import ROOT, load_arrays, canonical, file_hash, put
import pr0004_spec as spec
import pr0004_methods as old

METHODS=['LF0','LF-DC','PH-DC']
ALPHAS=[Fraction(1,100),Fraction(1,1000)]
BUDGETS={'W0':(2,2,4),'WP':(1,1,6),'WG':(3,3,2),'W+':(3,1,4),'W−':(1,3,4)}
PAIRS=[(1,0),(2,0),(2,1)]
PRIMARY=[('B_L__GM05','B'),('B_L__GP05','B'),('B_M__P02','B'),
         ('K200_L__P02','C'),('V05_L__GM05','C'),('V05_L__GP05','C')]
FREEZE=ROOT/'audits/verification/2026-10-17_PR-0005_design_freeze.json'
SOURCE_SHA='7e07909140a0d9080ccdf60837fe1c84ec15d52004c227d66e6f7472feb7a589'


def check_protected(freeze):
    for p,h in freeze['protected'].items():
        if p=='audits/fp_corpus.md':
            # QA-007 is an explicit append-only correction; retain the frozen prefix.
            assert hashlib.sha256((ROOT/p).read_bytes()[:9157]).hexdigest()==h,p
        else:assert file_hash(ROOT/p)==h,p


def order(n, alpha):
    x=(n+1)*(1-alpha)
    return (x.numerator+x.denominator-1)//x.denominator


def threshold(x, alpha):
    k=order(len(x),alpha)
    return np.inf if k>len(x) else float(np.partition(x,k-1)[k-1])


def physical_model(model):
    D=np.asarray(model['directions'],float).T
    sd=np.sqrt(np.diag(model['covariance']))
    result=dict(D=D.tolist(),available=False,reason=None)
    if not np.isfinite(D).all() or not np.isfinite(sd).all() or np.any(sd<=0):
        result['reason']='nonfinite D or invalid scale';return result
    A=D/sd[:,None];length=np.linalg.norm(A,axis=0)
    if not np.isfinite(length).all() or np.any(length<=0):
        result['reason']='zero or nonfinite direction';return result
    B=A/length;sv=np.linalg.svd(B,compute_uv=False)
    cond=float(sv[0]/sv[-1]) if sv[-1]>0 else np.inf
    result.update(sd=sd.tolist(),column_lengths=length.tolist(),singular_values=sv.tolist(),
                  condition=cond if np.isfinite(cond) else None,determinant=float(np.linalg.det(D)))
    if not np.isfinite(cond) or cond>1e6:
        result['reason']='condition gate';return result
    try:
        inv=np.linalg.solve(D,np.eye(2))
    except np.linalg.LinAlgError:
        result['reason']='singular solve';return result
    if not np.isfinite(inv).all():
        result['reason']='nonfinite solve';return result
    result.update(available=True,inverse=inv.tolist())
    return result


def features(model, ph, ordered, values):
    q=old.qvalues(values);centered=q-np.array(model['mean'])
    signed=centered@np.array(model['weights']).T
    ranks=np.column_stack([np.searchsorted(ordered['LF'][:,j],abs(signed[:,j]) if j==0 else signed[:,j],side='left') for j in range(2)])
    ranks[:,model['disabled']]=0
    direct=np.column_stack((signed[:,0],-signed[:,0],signed[:,1]))
    physical=None
    if ph['available']:
        uv=centered@np.array(ph['inverse']).T
        if not np.isfinite(uv).all():raise FloatingPointError('nonfinite evaluation projection; preserve and audit')
        physical=np.column_stack((uv[:,0],-uv[:,0],uv[:,1]))
    return signed,ranks,direct,physical


def calibrate(model,ph,ordered,values):
    signed,ranks,dc,pc=features(model,ph,ordered,values)
    cuts=np.full((2,5,3,3),np.inf)
    ties=np.zeros(cuts.shape,np.uint32)
    for ai,alpha in enumerate(ALPHAS):
        cut=threshold(ranks.max(1),alpha)
        for bi,weight in enumerate(BUDGETS.values()):
            cuts[ai,bi,0]=cut
            ties[ai,bi,0]=[(ranks[:,0]==cut).sum()]*2+[(ranks[:,1]==cut).sum()]
            for mi,scores in [(1,dc),(2,pc)]:
                if scores is None:continue
                for j in range(3):
                    if mi==1 and (0 if j<2 else 1) in model['disabled']:continue
                    c=threshold(scores[:,j],alpha*Fraction(weight[j],8))
                    cuts[ai,bi,mi,j]=c;ties[ai,bi,mi,j]=(scores[:,j]==c).sum()
    return cuts,ties,((ranks==9999).mean(0)).tolist()


def branch_bits(signed,ranks,dc,pc,cuts):
    n=len(dc);out=np.zeros((n,3),np.uint8)
    g=ranks[:,0]>cuts[0,0];p=ranks[:,1]>cuts[0,2]
    out[:,0]=(g&(signed[:,0]>0)).astype(np.uint8)+2*(g&(signed[:,0]<0)).astype(np.uint8)+4*p.astype(np.uint8)
    assert np.array_equal(out[:,0]>0,g|p)
    for mi,x in [(1,dc),(2,pc)]:
        if x is not None:out[:,mi]=((x>cuts[mi]).astype(np.uint8)*(1<<np.arange(3,dtype=np.uint8))).sum(1)
    return out


def interval(x,family=1):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    if not len(a):return dict(mean=None,replicates=0,lower=None,upper=None,sd=None,mcse=None,q10=None,q50=None,q90=None)
    mean=float(a.mean());sd=float(a.std(ddof=1)) if len(a)>1 else None
    se=sd/np.sqrt(len(a)) if sd is not None else None
    w=float(t.ppf(1-.05/(2*family),len(a)-1)*se) if sd and len(a)>1 else None
    return dict(mean=mean,sd=sd,mcse=se,lower=mean-w if w else None,upper=mean+w if w else None,
                replicates=len(a),q10=float(np.quantile(a,.1)),q50=float(np.quantile(a,.5)),q90=float(np.quantile(a,.9)))


def summarize(counts,pairs,available,meta):
    rates=[];comparisons=[];asymmetry=[]
    keys=[tuple(k) for k in meta['keys']];conditions=meta['conditions']
    for ki,(case,policy) in enumerate(keys):
        c=conditions[case];n=meta['n'][ki]
        for ai,alpha in enumerate(ALPHAS):
            for bi,budget in enumerate(BUDGETS):
                for mi,method in enumerate(METHODS):
                    valid=available[:,ki,mi];h=counts[:,ki,ai,bi,mi];al=h[:,1:].sum(1)
                    rr=interval(al[valid]/n);planned=int(50*n);unknown=int((~valid).sum()*n)
                    branches=[interval(h[valid][:,[j for j in range(8) if j&(1<<b)]].sum(1)/n) for b in range(3)]
                    rec=dict(case=case,policy=policy,kind=c['kind'],group=c['group'],alpha=float(alpha),budget=budget,method=method,
                             rate=rr,branches=branches,miss=interval((n-al[valid])/n),pattern_counts=h.sum(0).tolist(),
                             valid_replicates=int(valid.sum()),planned=planned,unavailable=unknown,
                             detection_lower=float(al.sum()/planned),fpr_range=[float(al.sum()/planned),float((al.sum()+unknown)/planned)])
                    if c['kind']=='normal' and bi==0:
                        adj=interval(al[valid]/n,120);rec['adjusted']=adj
                        rec['excess']=bool(valid.all() and adj['lower'] is not None and adj['lower']>float(alpha))
                    rates.append(rec)
                for pi,(new,ref) in enumerate(PAIRS):
                    valid=available[:,ki,new]&available[:,ki,ref];h=pairs[:,ki,ai,bi,pi]
                    diff=(h[valid,1].astype(float)-h[valid,2])/n
                    ismain=ai==0 and bi==0 and (case,policy) in PRIMARY
                    rec=dict(case=case,policy=policy,kind=c['kind'],alpha=float(alpha),budget=budget,new=METHODS[new],reference=METHODS[ref],
                             counts=h.sum(0).tolist(),valid_replicates=int(valid.sum()),n_joint=int(valid.sum()*n),
                             difference=interval(diff),gained=interval(h[valid,1]/n),lost=interval(h[valid,2]/n),primary=ismain)
                    if ismain:
                        adj=interval(diff,18);rec['adjusted']=adj
                        lo,hi=adj['lower'],adj['upper']
                        rec['classification']=('pending' if not valid.all() or lo is None else 'improved' if lo>0 else
                                               'within_1pp' if lo>-.01 else 'loss_over_1pp' if hi<-.01 else 'uncertain')
                    comparisons.append(rec)
    for ki,(case,policy) in enumerate(keys):
        if '__GP' not in case:continue
        negative=case.replace('__GP','__GM')
        if (negative,policy) not in keys:continue
        kj=keys.index((negative,policy));n=meta['n'][ki]
        for ai,alpha in enumerate(ALPHAS):
            for bi,budget in enumerate(BUDGETS):
                for mi,method in enumerate(METHODS):
                    valid=available[:,ki,mi]&available[:,kj,mi]
                    x=counts[valid,ki,ai,bi,mi,1:].sum(1).astype(float)/n
                    y=counts[valid,kj,ai,bi,mi,1:].sum(1)/n
                    asymmetry.append(dict(positive=case,negative=negative,policy=policy,alpha=float(alpha),budget=budget,method=method,
                                          positive_rate=interval(x),negative_rate=interval(y),difference=interval(x-y)))
    shifts=[]
    for ki,(case,policy) in enumerate(keys):
        if policy!='C' or conditions[case]['kind']!='normal':continue
        kf=keys.index((case,'F'));kb=keys.index((spec.baseline(case),'B'));n=meta['n'][ki]
        for ai,alpha in enumerate(ALPHAS):
            for bi,budget in enumerate(BUDGETS):
                for mi,method in enumerate(METHODS):
                    valid=available[:,ki,mi]&available[:,kf,mi]&available[:,kb,mi]
                    ar=[counts[valid,k,ai,bi,mi,1:].sum(1).astype(float)/n for k in (ki,kf,kb)]
                    shifts.append(dict(group=case,alpha=float(alpha),budget=budget,method=method,C_minus_F=interval(ar[0]-ar[1]),F_minus_B=interval(ar[1]-ar[2])))
    return dict(rates=rates,comparisons=comparisons,asymmetry=asymmetry,source_shifts=shifts)


def run(archive,out):
    archive,out=Path(archive),Path(out);out.mkdir(exist_ok=False)
    start=time.monotonic();freeze=json.loads(FREEZE.read_text())
    check_protected(freeze)
    val=dict(status='IN_PROGRESS',new_scientific_records=0,post_hoc=True,freeze_sha256=file_hash(FREEZE),
             source_sha256=file_hash(Path(__file__)),archive=str(archive),role_records=dict(ref=0,cal=0,null=0,fault=0,control=0),
             checks=dict(models=0,calibrations=0,saved_rank_sets=0,saved_main_decisions=0,chunks=0),timings={})
    put(out/'validation.json',val)
    try:
        t0=time.monotonic();assert file_hash(archive)==SOURCE_SHA
        val['archive_sha256']=SOURCE_SHA;val['timings']['archive_hash_before']=time.monotonic()-t0
        conditions=spec.conditions();groups=[k for k,c in conditions.items() if c['kind']=='normal'];bases=groups[:4]
        keys=[(k,p) for k,c in conditions.items() for p in (['B'] if c['group'].startswith('B_') else ['F','C'])]
        nullkeys=[k for k in keys if conditions[k[0]]['kind']=='normal']
        ns=[spec.COUNTS['null' if conditions[k]['kind']=='normal' else conditions[k]['kind']] for k,p in keys]
        assert len(keys)==82 and len(nullkeys)==20
        meta=dict(methods=METHODS,alphas=list(map(float,ALPHAS)),budgets=BUDGETS,pairs=PAIRS,keys=keys,null_keys=nullkeys,n=ns,
                  conditions=conditions,pattern_bits=['positive','negative','private'],pair_cells=['both','new_only','reference_only','neither'],
                  packed='bits 0:2 LF0,3:5 LF-DC,6:8 PH-DC,9:11 availability; columns alpha .01,.001')
        put(out/'metadata.json',meta)
        counts=np.zeros((50,82,2,5,3,8),np.uint32);pairs=np.zeros((50,82,2,5,3,4),np.uint32)
        available=np.zeros((50,82,3),bool);wilson=np.full((50,82,2,5,3,2),np.nan)
        cutall=np.full((50,20,2,5,3,3),np.inf);tiesall=np.zeros(cutall.shape,np.uint32)
        saturation=np.zeros((50,82,2));manifest={};diagnostics=[]
        for name in ['models','calibration','decisions']:(out/name).mkdir()
        with zipfile.ZipFile(archive) as z:
            index=json.loads(z.read('ARCHIVE_INDEX.json'))['entries']
            def read(p):
                b=z.read(p);h=hashlib.sha256(b).hexdigest();assert h==index[p]['sha256'],p
                manifest[p]=h;return b
            def js(p):return json.loads(read(p))
            def arr(p):return load_arrays(read(p))
            assert canonical(js('specification.json'))==canonical(spec.specification())
            sh=hashlib.sha256(canonical(spec.specification()).encode()).hexdigest();tasks=spec.tasks()
            for rep in range(50):
                rstart=time.monotonic();done=js(f'analysis/repeats/{rep:02d}/done.json')
                pre=f'analysis/repeats/{rep:02d}/'+done['attempt']+'/'
                taskmap={(t['case'],t['role']):t for t in tasks if t['replicate']==rep}
                inputs={}
                def values(case,role):
                    t0=time.monotonic();task=taskmap[case,role];parts=[];paths=[]
                    for s in range(0,task['count'],spec.CHUNK):
                        path='chunks/'+task['id']+f'/{s:09d}.npz';a=arr(path);v=a['values'];m=json.loads(str(a['metadata']))
                        assert m['task']==task and m['start']==s and m['specification_sha256']==sh
                        assert v.shape==(min(spec.CHUNK,task['count']-s),17) and v.dtype==np.float64 and np.isfinite(v).all()
                        assert hashlib.sha256(v.tobytes()).hexdigest()==m['values_sha256']
                        parts.append(v);paths.append(dict(path=path,sha256=manifest[path]));val['checks']['chunks']+=1
                    val['role_records'][role]+=task['count'];inputs[case+'/'+role]=paths
                    val['timings']['read_values']=val['timings'].get('read_values',0)+time.monotonic()-t0
                    return np.concatenate(parts)
                models={};ph={};ordered={};cal={};calties={};calsat={};rankout={}
                for group in bases:
                    ref=values(group,'ref');p=conditions[group]['params'];nu=spec.nominal_nu(p['dt'],p['tau_i'],p['tau_j'])
                    t0=time.monotonic();m,o=old.fit_reference(ref,nu);saved=js(pre+f'models/{group}/model.json')
                    for name in ['mean','weights','directions','normal_moment_means']:
                        assert np.allclose(m[name],saved[name],rtol=1e-12,atol=1e-14),(rep,group,name)
                    oo=arr(pre+f'models/{group}/reference_ranks.npz')
                    assert np.allclose(o['LF'],oo['LF'],rtol=1e-12,atol=1e-13)
                    val['max_reference_table_refit_difference']=max(val.get('max_reference_table_refit_difference',0.),float(np.max(abs(o['LF']-oo['LF']))))
                    # Preserve the actual deployed baseline, not portable-BLAS refit roundoff.
                    m,o=saved,oo
                    models[group]=m;ordered[group]=o;ph[group]=physical_model(m)
                    # Direct inverse and physical solve must agree on all reference q.
                    if ph[group]['available']:
                        q=old.qvalues(ref)-m['mean'];u=q@np.array(ph[group]['inverse']).T
                        assert np.allclose(u,np.linalg.solve(np.array(ph[group]['D']),q.T).T,rtol=1e-10,atol=1e-11)
                    diagnostics.append(dict(replicate=rep,group=group,physical=ph[group],disabled=m['disabled']))
                    rankout[group]=o['LF'];val['checks']['models']+=1
                    val['timings']['fit']=val['timings'].get('fit',0)+time.monotonic()-t0
                put(out/f'models/rep_{rep:02d}.json',dict(models=models,physical=ph,inputs=inputs.copy()))
                np.savez_compressed(out/f'models/rep_{rep:02d}_ranks.npz',**rankout)
                for group in groups:
                    cv=values(group,'cal');base=spec.baseline(group);t0=time.monotonic()
                    for policy in (['B'] if group in bases else ['F','C']):
                        if policy=='F':cc,tt,ss=cal[base,'B'],calties[base,'B'],calsat[base,'B']
                        else:cc,tt,ss=calibrate(models[base],ph[base],ordered[base],cv)
                        saved=js(pre+f'models/{group}/{policy}_calibration.json')
                        assert cc[0,0,0,0]==saved['cutoffs'][4][0]
                        cal[group,policy]=cc;calties[group,policy]=tt;calsat[group,policy]=ss
                        ni=nullkeys.index((group,policy));cutall[rep,ni]=cc;tiesall[rep,ni]=tt
                        val['checks']['calibrations']+=1
                    val['timings']['calibrate']=val['timings'].get('calibrate',0)+time.monotonic()-t0
                # Persist all thresholds before the first evaluation read of this replicate.
                np.savez_compressed(out/f'calibration/rep_{rep:02d}.npz',cuts=cutall[rep],ties=tiesall[rep])
                put(out/f'calibration/rep_{rep:02d}.json',dict(inputs={k:v for k,v in inputs.items() if k.endswith('/cal')},
                    models_sha256=file_hash(out/f'models/rep_{rep:02d}.json'),
                    thresholds_sha256=file_hash(out/f'calibration/rep_{rep:02d}.npz'),
                    calibration_rank_saturation={g+'_'+p:s for (g,p),s in calsat.items()}))
                model_hash=file_hash(out/f'models/rep_{rep:02d}.json');cal_hash=file_hash(out/f'calibration/rep_{rep:02d}.npz')
                packed={}
                for case,c in conditions.items():
                    role='null' if c['kind']=='normal' else c['kind'];v=values(case,role);base=spec.baseline(c['group']);n=len(v)
                    t0=time.monotonic();signed,ranks,dc,pc=features(models[base],ph[base],ordered[base],v)
                    for policy in (['B'] if c['group'] in bases else ['F','C']):
                        ki=keys.index((case,policy));av=np.array([True,True,ph[base]['available']]);available[rep,ki]=av
                        cc=cal[c['group'],policy];sv=arr(pre+f'decisions/{case}_{policy}.npz')
                        assert np.array_equal(ranks,sv['LF']);val['checks']['saved_rank_sets']+=1
                        saturation[rep,ki]=(ranks==9999).mean(0);bits=np.zeros((n,2),np.uint16)
                        for ai in range(2):
                            for bi in range(5):
                                br=branch_bits(signed,ranks,dc,pc,cc[ai,bi]);flags=br>0
                                if ai==0 and bi==0:
                                    assert np.array_equal(flags[:,0],(sv['bits']&16)!=0);val['checks']['saved_main_decisions']+=n
                                for mi in range(3):
                                    if av[mi]:
                                        counts[rep,ki,ai,bi,mi]=np.bincount(br[:,mi],minlength=8)
                                        wilson[rep,ki,ai,bi,mi]=old.wilson(int(flags[:,mi].sum()),n)
                                for pi,(new,ref) in enumerate(PAIRS):
                                    if av[new] and av[ref]:
                                        x,y=flags[:,new],flags[:,ref]
                                        h=np.array([(x&y).sum(),(x&~y).sum(),(~x&y).sum(),(~x&~y).sum()])
                                        assert h.sum()==n and h[1]-h[2]==int(x.sum())-int(y.sum())
                                        pairs[rep,ki,ai,bi,pi]=h
                                if bi==0:
                                    bits[:,ai]=(br.astype(np.uint16)*(1<<np.array([0,3,6]))).sum(1)+int(np.sum(av*(1<<np.array([9,10,11]))))
                        packed[case+'_'+policy]=bits
                    val['timings']['evaluate']=val['timings'].get('evaluate',0)+time.monotonic()-t0
                assert file_hash(out/f'models/rep_{rep:02d}.json')==model_hash and file_hash(out/f'calibration/rep_{rep:02d}.npz')==cal_hash
                t0=time.monotonic();np.savez_compressed(out/f'decisions/rep_{rep:02d}.npz',**packed)
                val['timings']['save_decisions']=val['timings'].get('save_decisions',0)+time.monotonic()-t0
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform!='darwin':rss*=1024
                val['max_rss_bytes']=int(rss)
                print(f'PR0005 {rep+1}/50; {time.monotonic()-rstart:.1f}s; RSS {rss/1e6:.0f} MB',flush=True)
                assert time.monotonic()-start<3600,'wall time limit';assert rss<2_000_000_000,'RSS limit'
        assert val['role_records']==dict(ref=1999800,cal=6000000,null=12000000,fault=9250000,control=250000)
        assert val['checks']['models']==200 and val['checks']['calibrations']==1000 and val['checks']['saved_rank_sets']==4100
        assert val['checks']['saved_main_decisions']==35500000
        np.savez_compressed(out/'replicate_statistics.npz',counts=counts,pairs=pairs,available=available,wilson=wilson,
                            cuts=cutall,ties=tiesall,saturation=saturation)
        put(out/'input_manifest.json',manifest);put(out/'diagnostics.json',diagnostics)
        t0=time.monotonic();summary=summarize(counts,pairs,available,meta);put(out/'summary.json',summary)
        val['timings']['summary']=time.monotonic()-t0
        # Independent-calibration legacy rates, not evaluation-selected ROC points.
        legacy=json.loads((ROOT/'numerics/analysis/pr0004_e0af6fcba596_lowfpr_roc_v1/summary.json').read_text())
        reference={(r['case'],r['policy'],r['beta']):r['nominal_tpr']['mean'] for r in legacy['rates'] if r['method']=='LF'}
        reference.update({(r['group'],r['policy'],r['beta']):r['nominal_fpr']['mean'] for r in legacy['normal'] if r['method']=='LF'})
        ncheck=0
        for r in summary['rates']:
            if r['method']=='LF0' and r['budget']=='W0':
                assert abs(r['rate']['mean']-reference[r['case'],r['policy'],r['alpha']])<2e-14;ncheck+=1
        val['legacy_low_and_main_rate_checks']=ncheck
        t0=time.monotonic();assert file_hash(archive)==SOURCE_SHA
        check_protected(freeze)
        val['timings']['archive_and_protected_after']=time.monotonic()-t0
        val.update(status='PASS',elapsed_seconds=time.monotonic()-start,old_evidence_unchanged=True,
                   unavailable_models=sum(not d['physical']['available'] for d in diagnostics))
        put(out/'validation.json',val);print(json.dumps(val,indent=2),flush=True)
    except Exception as e:
        val.update(status='FAILED',error=repr(e),elapsed_seconds=time.monotonic()-start);put(out/'validation.json',val);raise


if __name__=='__main__':run(sys.argv[1],sys.argv[2])
