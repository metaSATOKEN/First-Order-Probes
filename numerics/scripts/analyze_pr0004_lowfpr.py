# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Saved-data, tie-preserving calibration-supported ROC; never generates trials."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import re
import sys
import time
import zipfile
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'numerics/tools'))
from audit_pr0004_receipt import load_arrays, file_hash, canonical
import pr0004_spec as spec

METHODS = ['M', 'CR', 'B', 'J', 'LF']
ALPHA = np.array([.001, .005, .01, .02])
GRID = np.arange(401)/20000
ANCHORS = np.array([20, 100, 200, 400])
NCAL = 10000
PRIOR = ROOT/'numerics/analysis/pr0004_e0af6fcba596_performance_v2'
BRANCH = ROOT/'numerics/analysis/pr0004_e0af6fcba596_branches_v1'


def put(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def ci(values):
    a = np.asarray(values, float)
    sd = float(a.std(ddof=1)); mean = float(a.mean())
    w = float(t.ppf(.975, len(a)-1)*sd/np.sqrt(len(a)))
    return dict(mean=mean, lower=mean-w if sd else None, upper=mean+w if sd else None,
                sd=sd, replicates=len(a))


def order(alpha, bonferroni=False, n=NCAL):
    a = Fraction(str(alpha)) / (2 if bonferroni else 1)
    x = (n+1)*(1-a)
    return (x.numerator+x.denominator-1)//x.denominator


def cal_arrays(ranks):
    j, lf = ranks['J'], ranks['LF']
    return [np.sort(j[:,0]), np.sort(j[:,1]), np.sort(j.max(axis=1)), np.sort(lf.max(axis=1))]


def scores(ranks, ordered_cal):
    j, lf = ranks['J'], ranks['LF']
    m = np.searchsorted(ordered_cal[0], j[:,0], side='left')
    cr = np.searchsorted(ordered_cal[1], j[:,1], side='left')
    jj = np.searchsorted(ordered_cal[2], j.max(axis=1), side='left')
    ll = np.searchsorted(ordered_cal[3], lf.max(axis=1), side='left')
    return np.column_stack((m, cr, np.maximum(m, cr), jj, ll)).astype(np.uint16)


def ranks_from_cal(model, ordered, values):
    q = values[:,[11,12]]
    features = {'J':np.column_stack((abs(q[:,0]), q[:,1]))}
    lf = (q-np.array(model['mean'])) @ np.array(model['weights']).T
    lf[:,0] = abs(lf[:,0]); features['LF'] = lf
    out = {}
    for name, x in features.items():
        out[name] = np.column_stack([np.searchsorted(ordered[name][:,i], x[:,i], side='left') for i in range(2)])
    out['LF'][:,model['disabled']] = 0
    return out


def tail(hist):
    return np.concatenate((np.cumsum(hist[:,::-1], axis=1)[:,::-1], np.zeros((len(hist),1), np.int64)), axis=1)


def threshold_grid(null_tail):
    # Smallest k = most permissive nested set. Only normal evaluation chooses k.
    return np.stack([np.maximum(1, np.searchsorted(-v, -np.arange(401), side='left')) for v in null_tail])


def check_state():
    manifest = json.loads((ROOT/'numerics/colab/pr0004_gpu_main_build.json').read_text())
    frozen = manifest['sources'] | manifest['artifacts']
    for p,h in frozen.items(): assert file_hash(ROOT/p)==h, p
    anchors = dict(re.findall(r'\| `([^`]+)` \| `([0-9a-f]{64})` \|', (ROOT/'HANDOFF/2026-10-17_session-01.md').read_text()))
    for p,h in anchors.items(): assert file_hash(ROOT/p)==h, p
    previous = {}
    for folder in (PRIOR, BRANCH):
        delivery = json.loads((folder/'delivery_validation.json').read_text())
        for p,h in delivery['files'].items():
            assert file_hash(folder/p)==h, p
            previous[str((folder/p).relative_to(ROOT))] = h
    return dict(frozen=frozen, anchors=anchors, previous=previous)


def self_test():
    # Deliberate ties, unequal marginal calibrations, and upper cap.
    cal = {'J':np.array([[0,0],[1,0],[1,2],[3,2],[3,4]]),
           'LF':np.array([[0,0],[0,1],[1,1],[2,3],[4,4]])}
    ev = {'J':np.array([[0,0],[1,1],[2,2],[3,4],[4,4],[4,5]]),
          'LF':np.array([[0,0],[1,1],[2,2],[3,4],[4,4],[5,5]])}
    cc = cal_arrays(cal); u = scores(ev, cc)
    for k in range(1,7):
        if k==6: direct=np.zeros((6,5), bool)
        else:
            m=ev['J'][:,0]>cc[0][k-1]; cr=ev['J'][:,1]>cc[1][k-1]
            direct=np.column_stack((m,cr,m|cr,ev['J'].max(1)>cc[2][k-1],ev['LF'].max(1)>cc[3][k-1]))
        assert np.array_equal(u>=k, direct)
    hist=np.stack([np.bincount(u[:,j], minlength=NCAL+1) for j in range(5)])
    tt=tail(hist)
    for k in range(1,7): assert np.array_equal(tt[:,k],(u>=k).sum(0))
    assert order(.01)==9901 and order(.01,True)==9951
    assert not np.array_equal(u[:,2],u[:,3]), 'B and J must not be conflated'
    return 4


def run(archive, out):
    archive, out = Path(archive), Path(out)
    assert (out/'PROTOCOL.md').is_file() and not (out/'validation.json').exists()
    start=time.monotonic(); state=check_state()
    val=dict(status='IN_PROGRESS',post_hoc=True,new_scientific_records=0,changed_fusion_rule=False,
        changed_manuscript=False,same_agent=True,protocol_sha256=file_hash(out/'PROTOCOL.md'),
        ledger_before_bytes=(ROOT/'audits/ledger_2026-10.md').stat().st_size,
        ledger_before_sha256=file_hash(ROOT/'audits/ledger_2026-10.md'),
        corpus_sha256=file_hash(ROOT/'audits/fp_corpus.md'),preserved=state)
    put(out/'validation.json',val)
    try:
        val['small_tests']=self_test()
        receipt=json.loads((PRIOR/'validation.json').read_text())
        assert file_hash(archive)==receipt['archive_sha256']
        val.update(archive=str(archive),archive_sha256=receipt['archive_sha256'])
        conditions=spec.conditions(); tasks=spec.tasks()
        keys=[(case,p) for case,c in conditions.items() for p in (['B'] if c['group'].startswith('B_') else ['F','C','R'])]
        null_keys=[k for k in keys if conditions[k[0]]['kind']=='normal']
        fault_keys=[k for k in keys if conditions[k[0]]['kind']!='normal']
        nk={k:i for i,k in enumerate(null_keys)}; fk={k:i for i,k in enumerate(fault_keys)}
        power=np.zeros((50,len(fault_keys),5,401)); actual=np.zeros((50,len(null_keys),5,401))
        thresholds=np.zeros(actual.shape, np.uint16)
        adjacent=np.zeros((50,len(null_keys),5,4))
        nominal_fp=np.zeros((50,len(null_keys),5,4)); nominal_tp=np.zeros((50,len(fault_keys),5,4))
        overlap=np.zeros((50,len(fault_keys),4,4,4),np.uint32) # target, other, both/gain/loss/neither
        norders=np.array([[order(float(a),method=='B') for a in ALPHA] for method in METHODS])
        checks=dict(models=0,calibrations=0,calibration_chunks=0,calibration_records=0,
            decisions=0,unique_evaluation_records=0,policy_evaluation_records=0,nominal_bit_checks=0)
        (out/'counts').mkdir()
        with zipfile.ZipFile(archive) as z:
            index=json.loads(z.read('ARCHIVE_INDEX.json'))['entries']
            def read(path):
                b=z.read(path); assert hashlib.sha256(b).hexdigest()==index[path]['sha256'],path
                return b
            def js(path):return json.loads(read(path))
            def arr(path):return load_arrays(read(path))
            assert canonical(js('specification.json'))==canonical(spec.specification())
            sh=hashlib.sha256(canonical(spec.specification()).encode()).hexdigest()
            for rep in range(50):
                done=js(f'analysis/repeats/{rep:02d}/done.json')
                pre=f'analysis/repeats/{rep:02d}/'+done['attempt']+'/'
                taskmap={(x['case'],x['role']):x for x in tasks if x['replicate']==rep}
                models={}; null_tails={}; all_hist=[]
                for group,c in conditions.items():
                    if c['kind']!='normal':continue
                    m=js(pre+f'models/{group}/model.json'); o=arr(pre+f'models/{group}/reference_ranks.npz')
                    checks['models']+=1
                    task=taskmap[group,'cal']; chunks=[]
                    for s in range(0,task['count'],spec.CHUNK):
                        a=arr('chunks/'+task['id']+f'/{s:09d}.npz'); v=a['values']; meta=json.loads(str(a['metadata']))
                        assert meta['task']==task and meta['start']==s and meta['specification_sha256']==sh
                        assert hashlib.sha256(v.tobytes()).hexdigest()==meta['values_sha256']
                        assert v.dtype==np.float64 and v.shape==(min(spec.CHUNK,task['count']-s),17) and np.isfinite(v).all()
                        chunks.append(v);checks['calibration_chunks']+=1
                    values=np.concatenate(chunks);checks['calibration_records']+=len(values)
                    models[group]={}
                    for p in (['B'] if group.startswith('B_') else ['F','C','R']):
                        saved=js(pre+f'models/{group}/{p}_calibration.json');cuts=np.array(saved['cutoffs'])
                        mm,oo=(m,o) if p in ('B','R') else models[spec.baseline(group)]['B'][:2]
                        cc=models[spec.baseline(group)]['B'][2] if p=='F' else cal_arrays(ranks_from_cal(mm,oo,values))
                        assert saved['calibration_count']==NCAL
                        expected=np.array([[cc[0][9900],9999],[9999,cc[1][9900]],
                            [cc[0][9950],cc[1][9950]],[cc[2][9900]]*2,[cc[3][9900]]*2])
                        assert np.array_equal(expected,cuts[:5]),(rep,group,p)
                        models[group][p]=(mm,oo,cc,cuts)
                        checks['calibrations']+=1
                for case,p in keys:
                    c=conditions[case]; saved=arr(pre+f'decisions/{case}_{p}.npz')
                    n=spec.COUNTS['null' if c['kind']=='normal' else c['kind']]
                    assert saved['bits'].shape==(n,)
                    for name in ('J','LF'):
                        assert saved[name].shape==(n,2) and saved[name].dtype==np.uint16
                        assert np.max(saved[name])<=9999
                    mm,oo,cc,cuts=models[c['group']][p]
                    u=scores(saved,cc)
                    flags=u>=norders[:,2]
                    for j in range(5):
                        assert np.array_equal(flags[:,j],(saved['bits']&(1<<j))!=0),(rep,case,p,j)
                        direct=(saved['J' if j<4 else 'LF']>cuts[j]).any(1)
                        assert np.array_equal(flags[:,j],direct)
                    checks['decisions']+=1;checks['nominal_bit_checks']+=n*5;checks['policy_evaluation_records']+=n
                    if p in ('B','F'):checks['unique_evaluation_records']+=n
                    hist=np.stack([np.bincount(u[:,j],minlength=NCAL+1) for j in range(5)]).astype(np.uint32)
                    assert np.all(hist.sum(1)==n)
                    tt=tail(hist);all_hist.append(hist)
                    op=np.take_along_axis(tt,norders,axis=1)/n
                    assert np.all(np.diff(tt,axis=1)<=0) and np.all(tt[:,-1]==0)
                    if c['kind']=='normal':
                        ni=nk[case,p];null_tails[case,p]=tt
                        kg=threshold_grid(tt)
                        thresholds[rep,ni]=kg
                        achieved=np.take_along_axis(tt,kg,axis=1)
                        assert np.all(achieved<=np.arange(401))
                        assert np.all(np.take_along_axis(tt,np.maximum(1,kg-1),axis=1)>np.arange(401))
                        actual[rep,ni]=achieved/n
                        adjacent[rep,ni]=np.take_along_axis(tt,kg[:,ANCHORS]-1,axis=1)/n
                        nominal_fp[rep,ni]=op
                    else:
                        fi=fk[case,p];ni=nk[c['group'],p];kg=thresholds[rep,ni]
                        power[rep,fi]=np.take_along_axis(tt,kg,axis=1)/n
                        nominal_tp[rep,fi]=op
                        for ai,gg in enumerate(ANCHORS):
                            fl=u>=kg[:,gg]
                            for j in range(4):
                                a,b=fl[:,4],fl[:,j]
                                tab=np.array([(a&b).sum(),(a&~b).sum(),(~a&b).sum(),(~a&~b).sum()])
                                assert tab.sum()==n and tab[1]-tab[2]==a.sum()-b.sum()
                                overlap[rep,fi,ai,j]=tab
                np.savez_compressed(out/f'counts/rep_{rep:02d}.npz',counts=np.stack(all_hist))
                print(f'ROC {rep+1}/50; nominal decision sets {checks["decisions"]}',flush=True)
        assert checks==dict(models=600,calibrations=1400,calibration_chunks=1800,calibration_records=6000000,
            decisions=5700,unique_evaluation_records=21500000,policy_evaluation_records=49500000,nominal_bit_checks=247500000),checks
        assert np.all(np.diff(power,axis=-1)>=0)
        # Every jump is on this one-normal-record grid; left rectangles are exact step integration.
        pauc=power[:,:,:,:-1].mean(axis=-1)
        old_rates={(r['case'],r['policy'],r['method']):r for r in json.loads((PRIOR/'recomputed_summary.json').read_text())['rates']}
        for i,(case,p) in enumerate(keys):
            x=nominal_fp[:,nk[case,p]] if (case,p) in nk else nominal_tp[:,fk[case,p]]
            for j,method in enumerate(METHODS):
                assert np.isclose(x[:,j,2].mean(),old_rates[case,p,method]['mean'],atol=2e-14,rtol=0)
        metadata=dict(methods=METHODS,alphas=ALPHA.tolist(),grid=GRID.tolist(),anchor_indices=ANCHORS.tolist(),
            count_keys=keys,null_keys=null_keys,fault_keys=fault_keys,conditions=conditions,
            overlap_cells=['both','LF_only','other_only','neither'],orders=norders.tolist())
        put(out/'metadata.json',metadata)
        np.savez_compressed(out/'replicate_metrics.npz',power=power,actual_fpr=actual,thresholds=thresholds,
            next_fpr=adjacent,nominal_fpr=nominal_fp,nominal_tpr=nominal_tp,overlap=overlap,npauc=pauc)
        rows=[];diff=[];areas=[];normal=[]
        for ni,(group,p) in enumerate(null_keys):
            for j,meth in enumerate(METHODS):
                for ai,beta in enumerate(ALPHA):
                    x=actual[:,ni,j,ANCHORS[ai]]
                    normal.append(dict(group=group,policy=p,method=meth,beta=float(beta),
                        actual_fpr=ci(x),next_fpr=ci(adjacent[:,ni,j,ai]),
                        min_fpr=float(x.min()),max_fpr=float(x.max()),exact_replicates=int(np.sum(x==beta)),
                        nominal_fpr=ci(nominal_fp[:,ni,j,ai])))
        for fi,(case,p) in enumerate(fault_keys):
            ni=nk[conditions[case]['group'],p]
            for j,meth in enumerate(METHODS):
                areas.append(dict(case=case,policy=p,method=meth,npauc=ci(pauc[:,fi,j])))
                for ai,beta in enumerate(ALPHA):
                    rows.append(dict(case=case,policy=p,method=meth,beta=float(beta),kind=conditions[case]['kind'],
                        tpr=ci(power[:,fi,j,ANCHORS[ai]]),actual_fpr=ci(actual[:,ni,j,ANCHORS[ai]]),
                        nominal_tpr=ci(nominal_tp[:,fi,j,ai]),nominal_fpr=ci(nominal_fp[:,ni,j,ai])))
            for j,meth in enumerate(METHODS[:4]):
                for ai,beta in enumerate(ALPHA):
                    tab=overlap[:,fi,ai,j]/5000
                    d=power[:,fi,4,ANCHORS[ai]]-power[:,fi,j,ANCHORS[ai]]
                    assert np.allclose(d,tab[:,1]-tab[:,2],atol=1e-14,rtol=0)
                    diff.append(dict(case=case,policy=p,other=meth,beta=float(beta),difference=ci(d),
                        npauc_difference=ci(pauc[:,fi,4]-pauc[:,fi,j]),
                        **{k:ci(tab[:,ii]) for ii,k in enumerate(metadata['overlap_cells'])}))
        put(out/'summary.json',dict(rates=rows,paired=diff,areas=areas,normal=normal,
            interval='exploratory pointwise 95% outer-replicate t, no multiplicity correction'))
        assert check_state()==state and file_hash(archive)==val['archive_sha256']
        val.update(status='PASS',checks=checks,rate_rows=len(rows),paired_rows=len(diff),area_rows=len(areas),
            normal_rows=len(normal),archive_unchanged=True,previous_outputs_unchanged=True,
            source_sha256=file_hash(Path(__file__)),elapsed_seconds=time.monotonic()-start)
        put(out/'validation.json',val)
        print(json.dumps({k:v for k,v in val.items() if k!='preserved'},ensure_ascii=False,indent=2),flush=True)
    except Exception as e:
        val.update(status='FAILED',error=repr(e),elapsed_seconds=time.monotonic()-start)
        put(out/'validation.json',val);raise


if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='--self-test': print('PASS',self_test())
    else:run(sys.argv[1],sys.argv[2])
