# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Post-hoc branch decomposition of saved PR-0004 data; no generation or fitting."""
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import zipfile
from collections import defaultdict
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'numerics/tools'))
from audit_pr0004_receipt import load_arrays, file_hash, canonical
import pr0004_spec as spec

STATES = ['neither', 'g_only', 'p_only', 'both']
PRIOR = ROOT/'numerics/analysis/pr0004_e0af6fcba596_performance_v2'


def put(path, x):
    path.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def ci(x):
    a = np.asarray(x, float); mean = float(a.mean()); sd = float(a.std(ddof=1))
    width = float(t.ppf(.975, len(a)-1)*sd/np.sqrt(len(a)))
    return dict(mean=mean, sd=sd, lower=mean-width if sd else None,
                upper=mean+width if sd else None, replicates=len(a))


def branch_state(ranks, cutoff):
    b = ranks > cutoff
    code = b[:, 0].astype(np.uint8)+2*b[:, 1].astype(np.uint8)
    counts = np.bincount(code, minlength=4)
    assert int(counts.sum()) == len(ranks)
    return b, code, counts


def self_test():
    # Ties and upper-rank saturation must respect strict exceedance.
    ref = np.array([0., 1., 1., 3.]); x = np.array([-.1, 0., 1., 1.01, 3., 4.])
    ranks = np.searchsorted(ref, x, side='left')
    for cut in range(5):
        direct = x > ref[cut] if cut < len(ref) else np.zeros(len(x), bool)
        assert np.array_equal(ranks > cut, direct)
    b, code, counts = branch_state(np.array([[0,0],[2,0],[0,2],[2,2]]), np.array([1,1]))
    assert counts.tolist() == [1,1,1,1] and np.array_equal(b.any(axis=1), code>0)
    assert np.allclose(ci([1.,2.,3.])['mean'], 2.)
    return 3


def run(archive, out):
    archive, out = Path(archive), Path(out)
    assert (out/'PROTOCOL.md').is_file()
    assert not (out/'validation.json').exists(), 'New version required, do not overwrite'
    start = time.monotonic()
    val = dict(status='IN_PROGRESS', post_hoc=True, new_scientific_records=0,
               fusion_rule_changed=False, manuscript_changed=False, same_agent=True)
    put(out/'validation.json', val)
    try:
        val['small_tests'] = self_test()
        old = json.loads((PRIOR/'validation.json').read_text()); assert old['status']=='PASS'
        assert file_hash(archive)==old['archive_sha256']
        val.update(archive=str(archive), archive_sha256=old['archive_sha256'],
                   protocol_sha256=file_hash(out/'PROTOCOL.md'),
                   ledger_before_bytes=(ROOT/'audits/ledger_2026-10.md').stat().st_size,
                   ledger_before_sha256=file_hash(ROOT/'audits/ledger_2026-10.md'))
        manifest = json.loads((ROOT/'numerics/colab/pr0004_gpu_main_build.json').read_text())
        for p,h in (manifest['sources']|manifest['artifacts']).items(): assert file_hash(ROOT/p)==h
        delivery = json.loads((PRIOR/'delivery_validation.json').read_text())
        for p,h in delivery['files'].items(): assert file_hash(PRIOR/p)==h
        prior_rates = {(x['case'],x['policy'],x['method']):x for x in json.loads((PRIOR/'recomputed_summary.json').read_text())['rates']}
        conditions = spec.conditions(); tasks = spec.tasks()
        rows = []; geom = []; model_rows = []; asym = []; plot = {}
        checks = dict(decisions=0, branch_sets=0, chunks=0, unique_evaluation_records=0,
                      policy_evaluation_records=0, branch_record_checks=0)
        with zipfile.ZipFile(archive) as z:
            def js(path): return json.loads(z.read(path))
            def arr(path): return load_arrays(z.read(path))
            assert canonical(js('specification.json'))==canonical(spec.specification())
            spec_hash=hashlib.sha256(canonical(spec.specification()).encode()).hexdigest()
            index=js('ARCHIVE_INDEX.json')['entries']
            def checked_bytes(path):
                data=z.read(path)
                assert hashlib.sha256(data).hexdigest()==index[path]['sha256']
                return data
            for rep in range(spec.REPEATS):
                done=js(f'analysis/repeats/{rep:02d}/done.json')
                pre=f'analysis/repeats/{rep:02d}/'+done['attempt']+'/'
                models={}; own={}; normal={}; rep_rows={}
                taskmap={(x['case'],x['role']):x for x in tasks if x['replicate']==rep}
                for group,c in conditions.items():
                    if c['kind']!='normal': continue
                    m=json.loads(checked_bytes(pre+f'models/{group}/model.json'))
                    o=load_arrays(checked_bytes(pre+f'models/{group}/reference_ranks.npz'))
                    own[group]=m
                    policies=['B'] if group.startswith('B_') else ['F','C','R']
                    models[group]={}
                    for policy in policies:
                        cal=json.loads(checked_bytes(pre+f'models/{group}/{policy}_calibration.json'))
                        mm,oo=(m,o) if policy in ('B','R') else models[spec.baseline(group)]['B'][:2]
                        models[group][policy]=(mm,oo,np.array(cal['cutoffs']))
                    model_rows.append(dict(replicate=rep,group=group,**{k:m[k] for k in
                        ['mean','covariance','directions','weights','weights_diagonal','disabled','disabled_diagonal']}))
                for case,c in conditions.items():
                    role='null' if c['kind']=='normal' else c['kind']; task=taskmap[case,role]
                    chunks=[]
                    for s in range(0,task['count'],spec.CHUNK):
                        a=load_arrays(checked_bytes('chunks/'+task['id']+f'/{s:09d}.npz'))
                        v=a['values']; meta=json.loads(str(a['metadata']))
                        assert meta['task']==task and meta['start']==s and meta['specification_sha256']==spec_hash
                        assert hashlib.sha256(v.tobytes()).hexdigest()==meta['values_sha256']
                        assert v.dtype==np.float64 and v.shape==(min(spec.CHUNK,task['count']-s),17) and np.isfinite(v).all()
                        chunks.append(v); checks['chunks']+=1
                    data=np.concatenate(chunks); q=data[:,[11,12]]; n=len(q)
                    checks['unique_evaluation_records']+=n
                    if c['kind']=='normal': normal[case]=(q.mean(axis=0),np.cov(q,rowvar=False))
                    for policy,(m,o,cuts) in models[c['group']].items():
                        saved=load_arrays(checked_bytes(pre+f'decisions/{case}_{policy}.npz'))
                        flags=np.column_stack([(saved['bits']&(1<<i))!=0 for i in range(6)])
                        base=dict(replicate=rep,case=case,group=c['group'],kind=c['kind'],policy=policy,n=n)
                        jfeat=np.column_stack((abs(q[:,0]),q[:,1]))
                        jr=np.column_stack([np.searchsorted(o['J'][:,j],jfeat[:,j],side='left') for j in range(2)])
                        assert np.array_equal(jr,saved['J'])
                        for i in range(4): assert np.array_equal((jr>cuts[i]).any(axis=1),flags[:,i])
                        checks['decisions']+=1;checks['policy_evaluation_records']+=n
                        centered=q-np.array(m['mean']); cov=np.array(m['covariance']); scale=np.sqrt(np.diag(cov))
                        for method,mi,key,disabled in [('LF',4,'weights','disabled'),('LF-D',5,'weights_diagonal','disabled_diagonal')]:
                            w=np.array(m[key]); signed=centered@w.T
                            feat=signed.copy();feat[:,0]=abs(feat[:,0])
                            rank=np.column_stack([np.searchsorted(o[method][:,j],feat[:,j],side='left') for j in range(2)])
                            rank[:,m[disabled]]=0
                            assert np.array_equal(rank,saved[method]),(rep,case,policy,method)
                            b,code,counts=branch_state(rank,cuts[mi])
                            direct=np.zeros_like(b); thresholds=[]
                            for j in range(2):
                                th=None if cuts[mi,j]>=m['n_reference'] or j in m[disabled] else float(o[method][cuts[mi,j],j])
                                thresholds.append(th)
                                if th is not None: direct[:,j]=feat[:,j]>th
                            assert np.array_equal(b,direct)
                            assert np.array_equal(code>0,flags[:,mi])
                            positive=int(np.sum(b[:,0]&(signed[:,0]>0)));negative=int(np.sum(b[:,0]&(signed[:,0]<0)))
                            assert positive+negative==int(b[:,0].sum())
                            metrics={k:float(v/n) for k,v in zip(STATES,counts)}
                            metrics.update(total=float((code>0).mean()),g_fire=float(b[:,0].mean()),p_fire=float(b[:,1].mean()),
                                           g_miss=float((~b[:,0]).mean()),p_miss=float((~b[:,1]).mean()),
                                           g_positive=positive/n,g_negative=negative/n)
                            assert abs(metrics['total']-(metrics['g_only']+metrics['p_only']+metrics['both']))<1e-14
                            cross={}
                            for i,other in enumerate(spec.METHODS):
                                if i==mi:continue
                                table=np.bincount(code.astype(int)*2+flags[:,i].astype(int),minlength=8).reshape(4,2)
                                assert np.array_equal(table.sum(axis=1),counts)
                                assert int(table[:,1].sum())==int(flags[:,i].sum())
                                cross[other]=table.tolist()
                            row=dict(**base,method=method,counts=counts.tolist(),metrics=metrics,
                                     gain_tail_counts=[negative,positive],thresholds=thresholds,
                                     signed_mean=signed.mean(axis=0).tolist(),signed_sd=signed.std(axis=0,ddof=1).tolist(),
                                     cross=cross)
                            rows.append(row);rep_rows[case,policy,method]=row
                            checks['branch_sets']+=1;checks['branch_record_checks']+=n
                            # Mean shifts use the true source group's NORMAL reference, not fault-fitted directions.
                            if c['params']['fault']=='gain':
                                delta=c['params']['strength']; dg,dp=np.array(own[c['group']]['directions'])
                                linear=delta*dg;quadratic=delta*delta*dp;observed=q.mean(axis=0)-normal[c['group']][0]
                                fcov=np.cov(q,rowvar=False)
                                geom.append(dict(**base,method=method,delta=delta,
                                    observed_q_shift=(observed/scale).tolist(),linear_q_shift=(linear/scale).tolist(),
                                    quadratic_q_shift=(quadratic/scale).tolist(),predicted_q_shift=((linear+quadratic)/scale).tolist(),
                                    prediction_error=((observed-linear-quadratic)/scale).tolist(),
                                    observed_projection_shift=(w@observed).tolist(),linear_projection_shift=(w@linear).tolist(),
                                    quadratic_projection_shift=(w@quadratic).tolist(),
                                    fault_covariance_standardized=(fcov/np.outer(scale,scale)).tolist(),
                                    projected_sd_ratio=np.sqrt(np.diag(w@fcov@w.T)/np.diag(w@normal[c['group']][1]@w.T)).tolist()))
                        if rep==0 and policy=='B' and (case=='B_L' or case in ['B_L__GM05','B_L__GP05','B_L__GM02','B_L__GP02']):
                            plot[case+'_q']=q;plot[case+'_bits']=saved['bits'];plot[case+'_LF']=saved['LF'];plot[case+'_LFD']=saved['LF-D']
                            if case=='B_L':
                                put(out/'geometry_model_rep00.json',dict(model=m,cutoffs=cuts.tolist(),
                                    reference_thresholds={name:[[None if cuts[i,j]>=m['n_reference'] else float(o[name][cuts[i,j],j]) for j in range(2)] for i in range(6)] for name in o}))
                                for name,v in o.items():plot['reference_'+name]=v
                for case,c in conditions.items():
                    if c['params']['fault']!='gain' or c['params']['strength']<=0:continue
                    neg=case.replace('__GP','__GM')
                    if neg not in conditions:continue
                    for policy in models[c['group']]:
                        for method in ('LF','LF-D'):
                            posrow=rep_rows[case,policy,method];negrow=rep_rows[neg,policy,method]
                            diff={k:posrow['metrics'][k]-negrow['metrics'][k] for k in posrow['metrics']}
                            asym.append(dict(replicate=rep,group=c['group'],policy=policy,method=method,
                                             amplitude=c['params']['strength'],positive=case,negative=neg,differences=diff))
                print(f'Branches {rep+1}/50; decisions {checks["decisions"]}; rows {len(rows)}',flush=True)
        expected_unique=spec.REPEATS*sum(spec.COUNTS['null' if c['kind']=='normal' else c['kind']] for c in conditions.values())
        expected_policy=spec.REPEATS*sum(spec.COUNTS['null' if c['kind']=='normal' else c['kind']]*(1 if c['group'].startswith('B_') else 3) for c in conditions.values())
        assert checks['unique_evaluation_records']==expected_unique==21500000
        assert checks['policy_evaluation_records']==expected_policy==49500000
        assert checks['decisions']==5700 and checks['branch_sets']==11400 and checks['branch_record_checks']==2*expected_policy
        groups=defaultdict(list)
        for r in rows:groups[r['case'],r['policy'],r['method']].append(r)
        summaries=[]
        for key,rr in sorted(groups.items()):
            metrics={k:ci([r['metrics'][k] for r in rr]) for k in rr[0]['metrics']}
            assert np.isclose(metrics['total']['mean'],prior_rates[key]['mean'],rtol=0,atol=2e-14),key
            cross={m:np.sum([r['cross'][m] for r in rr],axis=0).tolist() for m in rr[0]['cross']}
            summaries.append(dict(case=key[0],policy=key[1],method=key[2],kind=rr[0]['kind'],
                n_total=sum(r['n'] for r in rr),metrics=metrics,cross=cross,
                thresholds=[ci([r['thresholds'][j] for r in rr]) if all(r['thresholds'][j] is not None for r in rr) else None for j in range(2)],
                signed_mean=[ci([r['signed_mean'][j] for r in rr]) for j in range(2)],
                signed_sd=[ci([r['signed_sd'][j] for r in rr]) for j in range(2)]))
        ag=defaultdict(list)
        for r in asym:ag[r['group'],r['policy'],r['method'],r['amplitude']].append(r)
        asum=[dict(group=k[0],policy=k[1],method=k[2],amplitude=k[3],
            differences={s:ci([r['differences'][s] for r in rr]) for s in rr[0]['differences']}) for k,rr in sorted(ag.items())]
        gg=defaultdict(list)
        for r in geom:gg[r['case'],r['policy'],r['method']].append(r)
        gsum=[]
        vector_keys=['observed_q_shift','linear_q_shift','quadratic_q_shift','predicted_q_shift','prediction_error',
                     'observed_projection_shift','linear_projection_shift','quadratic_projection_shift','projected_sd_ratio']
        for k,rr in sorted(gg.items()):
            gsum.append(dict(case=k[0],policy=k[1],method=k[2],
                **{v:[ci([r[v][j] for r in rr]) for j in range(2)] for v in vector_keys}))
        put(out/'replicate_branches.json',rows);put(out/'replicate_geometry.json',geom)
        put(out/'models.json',model_rows);put(out/'replicate_asymmetry.json',asym)
        put(out/'summary.json',dict(states=STATES,branches=summaries,asymmetry=asum,geometry=gsum,
             interval='pointwise 95% t over 50 outer replicates; exploratory; no multiplicity adjustment'))
        np.savez_compressed(out/'geometry_rep00.npz',**plot)
        assert file_hash(archive)==val['archive_sha256']
        for p,h in (manifest['sources']|manifest['artifacts']).items():assert file_hash(ROOT/p)==h
        for p,h in delivery['files'].items():assert file_hash(PRIOR/p)==h
        val.update(status='PASS',checks=checks,branch_summary_rows=len(summaries),
                   asymmetry_rows=len(asum),mean_shift_rows=len(gsum),old_outputs_unchanged=True,
                   archive_unchanged=True,elapsed_seconds=time.monotonic()-start,source_sha256=file_hash(Path(__file__)),
                   limitations=['Saved fixed reference/calibration; no refitting or threshold change.',
                     'Branch firing is not causal fault classification.',
                     'Positive/negative gain records have different random inputs.',
                     'Post-hoc observational decomposition, not independent confirmation.'])
        put(out/'validation.json',val)
        print(json.dumps(val,ensure_ascii=False,indent=2),flush=True)
    except Exception as e:
        val.update(status='FAILED',error=repr(e),elapsed_seconds=time.monotonic()-start)
        put(out/'validation.json',val);raise


if __name__=='__main__':run(sys.argv[1],sys.argv[2])
