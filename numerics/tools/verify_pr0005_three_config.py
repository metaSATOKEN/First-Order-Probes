# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Separate scalar-inverse/sorted-threshold path for PR-0005 saved-data audit."""
from pathlib import Path
import sys,json,zipfile,hashlib,time,math
from fractions import Fraction
import numpy as np
from scipy.stats import t,norm
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
from analyze_pr0004_lowfpr import load_arrays,file_hash,put
import pr0004_spec as spec


def verify(out):
    start=time.monotonic();out=Path(out);val=json.loads((out/'validation.json').read_text());assert val['status']=='PASS'
    meta=json.loads((out/'metadata.json').read_text());summary=json.loads((out/'summary.json').read_text())
    with np.load(out/'replicate_statistics.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    counts,pairs,available=data['counts'],data['pairs'],data['available'];keys=[tuple(x) for x in meta['keys']]
    nullkeys=[tuple(x) for x in meta['null_keys']];nvec=np.array(meta['n']);checks=dict(raw_contexts=0,calibration_contexts=0,
        count_tables=0,pair_tables=0,intervals=0,threshold_entries=0,packed_records=0)
    manifest=json.loads((out/'input_manifest.json').read_text());alphas=[Fraction(1,100),Fraction(1,1000)]
    budgets=list(meta['budgets'].values());tasks=spec.tasks();conds=meta['conditions']
    with zipfile.ZipFile(val['archive']) as z:
        def read(p):
            b=z.read(p);assert hashlib.sha256(b).hexdigest()==manifest[p];return load_arrays(b)
        for rep in range(50):
            modeldoc=json.loads((out/f'models/rep_{rep:02d}.json').read_text());models=modeldoc['models'];ph=modeldoc['physical']
            with np.load(out/f'models/rep_{rep:02d}_ranks.npz',allow_pickle=False) as f:tables={k:f[k] for k in f.files}
            with np.load(out/f'decisions/rep_{rep:02d}.npz',allow_pickle=False) as f:packed={k:f[k] for k in f.files}
            taskmap={(q['case'],q['role']):q for q in tasks if q['replicate']==rep}
            def values(case,role):
                task=taskmap[case,role]
                return np.concatenate([read('chunks/'+task['id']+f'/{s:09d}.npz')['values'] for s in range(0,task['count'],4096)])
            def scores(case,v):
                base=spec.baseline(conds[case]['group']);m=models[base];q=v[:,11:13]-np.array(m['mean']);w=np.array(m['weights'])
                g=q[:,0]*w[0,0]+q[:,1]*w[0,1];p=q[:,0]*w[1,0]+q[:,1]*w[1,1]
                r=np.column_stack([np.searchsorted(tables[base][:,0],abs(g),side='left'),np.searchsorted(tables[base][:,1],p,side='left')])
                r[:,m['disabled']]=0
                xs=[None,np.column_stack([g,-g,p]),None]
                if ph[base]['available']:
                    D=np.array(ph[base]['D']);a,b=D[0];c,d=D[1];det=a*d-b*c
                    u=(d*q[:,0]-b*q[:,1])/det;v=(-c*q[:,0]+a*q[:,1])/det
                    xs[2]=np.column_stack([u,-u,v])
                return g,r,xs
            for ni,(case,policy) in enumerate(nullkeys):
                if policy=='F':
                    nj=nullkeys.index((spec.baseline(case),'B'));assert np.array_equal(data['cuts'][rep,ni],data['cuts'][rep,nj]);continue
                v=values(case,'cal');g,r,xs=scores(case,v)
                for ai,alpha in enumerate(alphas):
                    for bi,weights in enumerate(budgets):
                        for mi in range(3):
                            if mi==2 and xs[mi] is None:continue
                            for j in range(3):
                                if mi==1 and (0 if j<2 else 1) in models[spec.baseline(case)]['disabled']:continue
                                x=r.max(1) if mi==0 else xs[mi][:,j]
                                beta=alpha if mi==0 else alpha*Fraction(weights[j],8)
                                k=math.ceil((len(x)+1)*(1-beta));cut=np.inf if k>len(x) else np.sort(x)[k-1]
                                actual=data['cuts'][rep,ni,ai,bi,mi,j]
                                assert np.isclose(cut,actual,atol=1e-10,rtol=1e-11),(rep,case,mi,cut,actual)
                                checks['threshold_entries']+=1
                checks['calibration_contexts']+=1
            for case,c in conds.items():
                role='null' if c['kind']=='normal' else c['kind'];v=values(case,role);g,r,xs=scores(case,v)
                for policy in (['B'] if c['group'].startswith('B_') else ['F','C']):
                    ki=keys.index((case,policy));ni=nullkeys.index((c['group'],policy));n=len(v)
                    bits=packed[case+'_'+policy];assert bits.shape==(n,2) and bits.dtype==np.uint16 and np.all(bits<4096)
                    for ai in range(2):
                        for mi in range(3):assert np.all(((bits[:,ai]>>(9+mi))&1)==available[rep,ki,mi])
                        for bi in range(5):
                            cuts=data['cuts'][rep,ni,ai,bi];bb=np.zeros((n,3),np.uint8)
                            ag=r[:,0]>cuts[0,0];ap=r[:,1]>cuts[0,2]
                            bb[:,0]=(ag&(g>0)).astype(np.uint8)+2*(ag&(g<0)).astype(np.uint8)+4*ap.astype(np.uint8)
                            for mi in (1,2):
                                if xs[mi] is not None:
                                    bb[:,mi]=(xs[mi][:,0]>cuts[mi,0]).astype(np.uint8)+2*(xs[mi][:,1]>cuts[mi,1]).astype(np.uint8)+4*(xs[mi][:,2]>cuts[mi,2]).astype(np.uint8)
                            for mi in range(3):
                                h=np.bincount(bb[:,mi],minlength=8) if available[rep,ki,mi] else np.zeros(8,dtype=int)
                                assert np.array_equal(h,counts[rep,ki,ai,bi,mi]);checks['count_tables']+=1
                                if bi==0:assert np.array_equal(bb[:,mi],(bits[:,ai]>>(3*mi))&7)
                            for pi,(new,ref) in enumerate(meta['pairs']):
                                if available[rep,ki,new] and available[rep,ki,ref]:
                                    a=bb[:,new]>0;b=bb[:,ref]>0
                                    h=np.array([(a&b).sum(),(a&~b).sum(),(~a&b).sum(),(~a&~b).sum()])
                                else:h=np.zeros(4,dtype=int)
                                assert np.array_equal(h,pairs[rep,ki,ai,bi,pi]);checks['pair_tables']+=1
                    checks['packed_records']+=n*2;checks['raw_contexts']+=1
            print(f'Independent scalar audit {rep+1}/50',flush=True)
    def ci_check(x,d,family=1):
        x=np.asarray(x,float);assert len(x)==d['replicates']
        if not len(x):assert d['mean'] is None;return
        assert abs(x.mean()-d['mean'])<1e-14
        sd=x.std(ddof=1) if len(x)>1 else None
        if not sd:assert d['lower'] is None and d['upper'] is None
        else:
            w=t.ppf(1-.05/(2*family),len(x)-1)*sd/np.sqrt(len(x))
            assert np.allclose([x.mean()-w,x.mean()+w],[d['lower'],d['upper']],rtol=1e-12,atol=1e-14)
        checks['intervals']+=1
    for rr in summary['rates']:
        ki=keys.index((rr['case'],rr['policy']));ai=meta['alphas'].index(rr['alpha']);bi=list(meta['budgets']).index(rr['budget']);mi=meta['methods'].index(rr['method'])
        valid=available[:,ki,mi];h=counts[valid,ki,ai,bi,mi];n=nvec[ki];rate=h[:,1:].sum(1)/n
        ci_check(rate,rr['rate']);ci_check(1-rate,rr['miss'])
        for j in range(3):ci_check(h[:,[k for k in range(8) if k&(1<<j)]].sum(1)/n,rr['branches'][j])
        assert np.array_equal(h.sum(0),rr['pattern_counts'])
        if 'adjusted' in rr:ci_check(rate,rr['adjusted'],120)
        # Independently verify per-replicate Wilson intervals.
        zz=norm.ppf(.975);den=1+zz**2/n;center=(rate+zz**2/(2*n))/den
        half=zz*np.sqrt(rate*(1-rate)/n+zz**2/(4*n*n))/den
        assert np.allclose(data['wilson'][valid,ki,ai,bi,mi],np.stack([center-half,center+half],axis=1),atol=1e-14)
    for rr in summary['comparisons']:
        ki=keys.index((rr['case'],rr['policy']));ai=meta['alphas'].index(rr['alpha']);bi=list(meta['budgets']).index(rr['budget'])
        mi=meta['methods'].index(rr['new']);mj=meta['methods'].index(rr['reference']);pi=meta['pairs'].index([mi,mj])
        valid=available[:,ki,mi]&available[:,ki,mj];h=pairs[valid,ki,ai,bi,pi].astype(float);n=nvec[ki]
        diff=(h[:,1]-h[:,2])/n;ci_check(diff,rr['difference']);ci_check(h[:,1]/n,rr['gained']);ci_check(h[:,2]/n,rr['lost'])
        assert np.array_equal(h.sum(0),rr['counts'])
        if rr['primary']:ci_check(diff,rr['adjusted'],18)
    freeze=json.loads((ROOT/'audits/verification/2026-10-17_PR-0005_design_freeze.json').read_text())
    for p,h in freeze['protected'].items():
        if p=='audits/fp_corpus.md':assert hashlib.sha256((ROOT/p).read_bytes()[:9157]).hexdigest()==h
        else:assert file_hash(ROOT/p)==h
    ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes();assert hashlib.sha256(ledger[:freeze['ledger_bytes']]).hexdigest()==freeze['ledger_sha256']
    assert sum(nvec)*50==35500000 and available.shape==(50,82,3)
    result=dict(status='PASS',same_agent=True,independent_researcher=False,checks=checks,elapsed_seconds=time.monotonic()-start,
                source_sha256=file_hash(Path(__file__)),analysis_source_sha256=val['source_sha256'],new_scientific_records=0)
    put(out/'independent_path_validation.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':verify(sys.argv[1])
