# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Saved-data-only PR-0004 performance audit. No RNG, simulation, or archive execution.

Reconstructs learning statistics with explicit 2x2 inverses, then verifies saved
reference ranks, calibrations, and every evaluation decision. Recounts paired
outcomes and independently computes outer-replicate intervals. New output only.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'numerics/tools'))
from audit_pr0004_receipt import load_arrays, file_hash, canonical
import pr0004_spec as spec

METHODS = list(spec.METHODS)


def put(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def close(a, b, label):
    if not np.allclose(a, b, atol=2e-12, rtol=2e-12):
        raise AssertionError('Numerical disagreement: ' + label)


def learned(reference, nu):
    q = reference[:, [11, 12]]
    mu = q.mean(axis=0); centered = q - mu
    cov = centered.T @ centered / (len(q) - 1)
    means = reference.mean(axis=0)
    directions = np.array([[means[8] - 2*nu*means[7], 2*(means[15]-means[16])],
                           [-nu*means[7], means[15]]])
    reg = cov.copy(); reg[0, 1] *= 1-spec.EPSILON; reg[1, 0] *= 1-spec.EPSILON
    weights = []
    for c in [reg, np.diag(np.diag(cov))]:
        inverse = np.array([[c[1, 1], -c[0, 1]], [-c[1, 0], c[0, 0]]]) / np.linalg.det(c)
        rows = []
        for d in directions:
            if not np.any(d):
                rows.append([0., 0.]); continue
            v = d / max(abs(d)); sol = inverse @ v
            rows.append(sol / math.sqrt(v @ sol))
        weights.append(np.array(rows))
    return dict(mean=mu, covariance=cov, regularized_covariance=reg, directions=directions,
                normal_moment_means=means, weights=weights[0], weights_diagonal=weights[1])


def features(a, model):
    q = a[:, [11, 12]]; c = q - np.array(model['mean'])
    out = {'J': np.column_stack([abs(q[:, 0]), q[:, 1]])}
    for name, key in [('LF', 'weights'), ('LF-D', 'weights_diagonal')]:
        w = np.array(model[key])
        # Explicit scalar linear combinations, separate from production matrix multiply.
        out[name] = np.column_stack([abs(c[:, 0]*w[0, 0] + c[:, 1]*w[0, 1]),
                                    c[:, 0]*w[1, 0] + c[:, 1]*w[1, 1]])
    return out


def ranks(a, model, reference):
    out = {}
    for name, x in features(a, model).items():
        out[name] = np.column_stack([np.searchsorted(reference[name][:, j], x[:, j], 'left') for j in range(2)])
        disabled = [] if name == 'J' else model['disabled_diagonal' if name == 'LF-D' else 'disabled']
        out[name][:, disabled] = 0
    return out


def thresholds(rank, n):
    def cut(x, denominator):
        k = ((len(x)+1)*(denominator-1) + denominator-1) // denominator
        return n if k > len(x) else int(np.sort(x)[k-1])
    j = rank['J']; c = np.full((6, 2), n)
    c[0, 0] = cut(j[:, 0], 100); c[1, 1] = cut(j[:, 1], 100)
    c[2] = [cut(j[:, k], 200) for k in range(2)]
    for i, name in enumerate(METHODS[3:], 3):
        c[i] = cut(np.maximum(rank[name][:, 0], rank[name][:, 1]), 100)
    return c


def flags(rank, cuts):
    j = rank['J']
    cols = [j[:, 0] > cuts[0, 0], j[:, 1] > cuts[1, 1],
            (j[:, 0] > cuts[2, 0]) | (j[:, 1] > cuts[2, 1])]
    cols += [rank[name].max(axis=1) > cuts[i, 0] for i, name in enumerate(METHODS[3:], 3)]
    return np.column_stack(cols)


def interval(x, family=1):
    x = np.asarray(x, float); n = len(x); mean = float(sum(x)/n)
    sd = float(np.sqrt(np.dot(x-x.mean(), x-x.mean())/(n-1)))
    se = sd/math.sqrt(n); width = stats.t.ppf(1-.05/(2*family), n-1)*se
    return dict(mean=mean, sd=sd, mcse=se, lower=float(mean-width) if sd else None,
                upper=float(mean+width) if sd else None, replicates=n, family=family,
                q10=float(np.quantile(x, .1)), q50=float(np.quantile(x, .5)), q90=float(np.quantile(x, .9)))


def paired(a, b):
    cells = dict(both=int((a & b).sum()), gained=int((a & ~b).sum()),
                 lost=int((~a & b).sum()), neither=int((~a & ~b).sum()))
    return dict(**cells, difference=(cells['gained']-cells['lost'])/len(a))


def compare_interval(new, old, label):
    for key, v in new.items():
        if v is None:
            assert old[key] is None, (label, key)
        else:
            close(v, old[key], label + ':' + key)


def run(archive, output):
    start = time.monotonic(); archive, output = Path(archive), Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    validation = dict(status='IN_PROGRESS', new_scientific_records=0, archive=str(archive),
                      same_agent=True, method='separate learning/projection/count/interval calculations')
    put(output/'validation.json', validation)
    try:
        receipt = json.loads((ROOT/'numerics/analysis/pr0004_e0af6fcba596_receipt_v1/validation.json').read_text())
        assert receipt['status'] == 'PASS_RECEIPT'
        print('Rechecking archive SHA-256...', flush=True)
        assert file_hash(archive) == receipt['archive_sha256']
        frozen = json.loads((ROOT/'numerics/colab/pr0004_gpu_main_build.json').read_text())
        for p, h in (frozen['sources'] | frozen['artifacts']).items(): assert file_hash(ROOT/p) == h
        validation.update(archive_sha256=receipt['archive_sha256'], archive_bytes=archive.stat().st_size,
                          sources_checked=15, ledger_before_bytes=(ROOT/'audits/ledger_2026-10.md').stat().st_size,
                          ledger_before_sha256=file_hash(ROOT/'audits/ledger_2026-10.md'))
        series = defaultdict(list); overlaps = defaultdict(list); updates = defaultdict(list)
        model_diagnostics = []; cal_diagnostics = []; sat_diagnostics = []; repeat_rows = []
        checked = dict(models=0, calibrations=0, decisions=0, chunks=0, raw_records=0, evaluated_records=0)
        conditions = spec.conditions(); tasks = spec.tasks()
        with zipfile.ZipFile(archive) as z:
            def js(n): return json.loads(z.read(n))
            def arr(n): return load_arrays(z.read(n))
            assert canonical(js('specification.json')) == canonical(spec.specification())
            expected_spec_hash = hashlib.sha256(canonical(spec.specification()).encode()).hexdigest()
            for rep in range(50):
                done = js(f'analysis/repeats/{rep:02d}/done.json')
                prefix = f'analysis/repeats/{rep:02d}/'+done['attempt']+'/'
                old = js(prefix+'results.json')
                taskmap = {(t['case'], t['role']): t for t in tasks if t['replicate'] == rep}
                def load(case, role):
                    t = taskmap[case, role]; chunks = []
                    for s in range(0, t['count'], spec.CHUNK):
                        a = arr('chunks/'+t['id']+f'/{s:09d}.npz')
                        meta = json.loads(str(a['metadata'])); v = a['values']
                        assert meta['task'] == t and meta['start'] == s
                        assert meta['specification_sha256'] == expected_spec_hash
                        assert hashlib.sha256(v.tobytes()).hexdigest() == meta['values_sha256']
                        assert v.shape == (min(spec.CHUNK, t['count']-s), 17)
                        checked['chunks'] += 1; checked['raw_records'] += len(v); chunks.append(v)
                    return np.concatenate(chunks)
                models = {}
                for group, c in conditions.items():
                    if c['kind'] != 'normal': continue
                    ref = load(group, 'ref'); cal = load(group, 'cal')
                    saved = js(prefix+'models/'+group+'/model.json')
                    order = arr(prefix+'models/'+group+'/reference_ranks.npz')
                    p = c['params']; nu = spec.nominal_nu(p['dt'], p['tau_i'], p['tau_j'])
                    ours = learned(ref, nu)
                    for k, v in ours.items(): close(v, saved[k], f'{rep}/{group}/{k}')
                    for k, v in features(ref, saved).items(): close(np.sort(v, axis=0), order[k], f'{rep}/{group}/ranks/{k}')
                    checked['models'] += 1
                    source = models[spec.baseline(group)]['B'] if not group.startswith('B_') else None
                    variants = {'B':(saved, order, cal)} if source is None else {
                        'F':source, 'C':(source[0], source[1], cal), 'R':(saved, order, cal)}
                    actual = {}
                    for policy, (m, o, calibration) in variants.items():
                        stored = js(prefix+'models/'+group+'/'+policy+'_calibration.json')
                        if policy == 'F':
                            cuts = np.array(source[2])  # inherited baseline cutoff, never shifted calibration
                        else:
                            r = ranks(calibration, m, o); cuts = thresholds(r, m['n_reference'])
                            for key, value in r.items(): close((value == 9999).mean(axis=0), stored['saturation'][key], 'cal saturation')
                        assert np.array_equal(cuts, stored['cutoffs']), (rep, group, policy)
                        checked['calibrations'] += 1
                        actual[policy] = (m, o, cuts)
                        cal_diagnostics.append(dict(replicate=rep, group=group, policy=policy, **stored))
                    models[group] = actual
                    cov = np.array(saved['covariance']); dg, dp = np.array(saved['directions'])
                    model_diagnostics.append(dict(replicate=rep, group=group, correlation=float(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])),
                        condition_number=saved['condition_number'], regularized_condition_number=saved['regularized_condition_number'],
                        direction_norms=saved['direction_norms'], directions=saved['directions'], weights=saved['weights'],
                        disabled=saved['disabled'], disabled_diagonal=saved['disabled_diagonal'],
                        curvature_to_linear_norm_5pct=float(.05*np.linalg.norm(dp)/np.linalg.norm(dg)) if np.linalg.norm(dg) else None))
                rows = []; pair_rows = []; update_rows = []
                for case, c in conditions.items():
                    data = load(case, 'null' if c['kind'] == 'normal' else c['kind']); by_policy = {}
                    for policy, (m, o, cut) in models[c['group']].items():
                        r = ranks(data, m, o); f = flags(r, cut)
                        saved = arr(prefix+'decisions/'+case+'_'+policy+'.npz')
                        for name in r: assert np.array_equal(r[name], saved[name]), (rep, case, policy, name)
                        packed = sum(f[:, i].astype(np.uint8)*(1 << i) for i in range(6))
                        assert np.array_equal(packed, saved['bits']), (rep, case, policy)
                        checked['decisions'] += 1; checked['evaluated_records'] += len(data)
                        by_policy[policy] = f; base = dict(replicate=rep, case=case, kind=c['kind'], policy=policy, n=len(data))
                        old_pat = next(v for v in old['patterns'] if v['case']==case and v['policy']==policy)
                        assert np.bincount(packed, minlength=64).tolist() == old_pat['auxiliary64']
                        assert np.bincount(packed & 31, minlength=32).tolist() == old_pat['main32']
                        for i, method in enumerate(METHODS):
                            count = int(f[:, i].sum()); rate = count/len(data)
                            old_r = next(v for v in old['rates'] if v['case']==case and v['policy']==policy and v['method']==method)
                            assert old_r['count']==count and old_r['misses']==len(data)-count and old_r['rate']==rate
                            rows.append(dict(**base, method=method, count=count, misses=len(data)-count, rate=rate))
                            series[case, c['kind'], policy, method].append(rate)
                            if method != 'LF':
                                pair = paired(f[:, 4], f[:, i]); key=(case, c['kind'], policy, 'LF', method)
                                old_p=next(v for v in old['paired'] if v['case']==case and v['policy']==policy and v['right']==method)
                                for k,v in pair.items(): assert v==old_p[k]
                                overlaps[key].append(dict(n=len(data), **pair))
                                pair_rows.append(dict(**base, left='LF', right=method, **pair))
                        sat_diagnostics.append(dict(**base, coordinates={k:(v==9999).mean(axis=0).tolist() for k,v in r.items()}))
                    if 'R' in by_policy:
                        for other in ['F','C']:
                            for i, method in enumerate(METHODS):
                                pair = paired(by_policy['R'][:,i], by_policy[other][:,i]); key=(case,c['kind'],method,'R',other)
                                old_u=next(v for v in old['updates'] if v['case']==case and v['right']==other and v['method']==method)
                                for k,v in pair.items(): assert v==old_u[k]
                                updates[key].append(dict(n=len(data), **pair))
                                update_rows.append(dict(replicate=rep, case=case, kind=c['kind'], method=method, left='R', right=other, n=len(data), **pair))
                repeat_rows.append(dict(replicate=rep,rates=rows,paired=pair_rows,updates=update_rows))
                print(f'Recomputed {rep+1}/50: models={checked["models"]}, decisions={checked["decisions"]}', flush=True)
            saved_summary = js('analysis/summary.json'); costs = js('analysis/costs.json')
        expected_evaluations = 50 * sum(spec.COUNTS['null' if c['kind']=='normal' else c['kind']]
            * (1 if c['group'].startswith('B_') else 3) for c in conditions.values())
        assert checked == dict(models=600, calibrations=1400, decisions=5700, chunks=10400,
                               raw_records=33499400, evaluated_records=expected_evaluations), checked
        rates=[]; pairs=[]; policy_diff=[]; primary=[]; fpr=[]
        for key,x in sorted(series.items()):
            row=dict(zip(('case','kind','policy','method'),key), **interval(x)); rates.append(row)
            old=next(v for v in saved_summary['rates'] if all(v[k]==row[k] for k in ('case','policy','method')))
            compare_interval(interval(x),old,'rate')
            if key[1]=='normal' and key[3]!='LF-D':
                r=dict(case=key[0],policy=key[2],method=key[3],**interval(x,140))
                r['excess_flag']=r['lower'] is not None and r['lower']>.01
                r['pointwise95']={k:row[k] for k in ('lower','upper')}; fpr.append(r)
                old=next(v for v in saved_summary['fpr140'] if all(v[k]==r[k] for k in ('case','policy','method')))
                compare_interval(interval(x,140),old,'fpr');assert r['excess_flag']==old['excess_flag']
        for key,rows in sorted(overlaps.items()):
            row=dict(zip(('case','kind','policy','left','right'),key), **interval([r['difference'] for r in rows]))
            row['total_records']=sum(r['n'] for r in rows)
            row['cells']={k:dict(total_count=sum(r[k] for r in rows),**interval([r[k]/r['n'] for r in rows])) for k in ('both','gained','lost','neither')}
            pairs.append(row)
            old=next(v for v in saved_summary['paired_differences'] if all(v[k]==row[k] for k in ('case','policy','right')))
            compare_interval(interval([r['difference'] for r in rows]),old,'pair')
        for case,left,right in spec.primary():
            x=[r['difference'] for r in overlaps[case,'fault','B',left,right]]
            row=dict(case=case,left=left,right=right,**interval(x,12))
            row['classification']='improvement' if row['lower'] is not None and row['lower']>0 else 'loss' if row['upper'] is not None and row['upper']<0 else 'unresolved'
            primary.append(row)
            old=next(v for v in saved_summary['primary12'] if v['case']==case and v['right']==right)
            compare_interval(interval(x,12),old,'primary');assert row['classification']==old['classification']
        for key,rows in sorted(updates.items()):
            row=dict(zip(('case','kind','method','left','right'),key),**interval([r['difference'] for r in rows]))
            row['cells']={k:dict(total_count=sum(r[k] for r in rows),**interval([r[k]/r['n'] for r in rows])) for k in ('both','gained','lost','neither')}
            policy_diff.append(row)
            old=next(v for v in saved_summary['policy_differences'] if all(v[k]==row[k] for k in ('case','method','right')))
            compare_interval(interval([r['difference'] for r in rows]),old,'policy')
        changes=[]
        for key,x in series.items():
            case,kind,policy,method=key
            if kind!='normal' or policy=='B':continue
            base=series[spec.baseline(case),'normal','B',method]
            row=dict(case=case,policy=policy,method=method,**interval(np.array(x)-np.array(base)))
            changes.append(row)
            old=next(v for v in saved_summary['fpr_changes_from_baseline'] if all(v[k]==row[k] for k in ('case','policy','method')))
            compare_interval(interval(np.array(x)-np.array(base)),old,'baseline difference')
        output_summary=dict(rates=rates,paired=pairs,primary12=primary,fpr140=fpr,policy_differences=policy_diff,
                            fpr_changes=changes,conditions=conditions,methods=METHODS,units='fractions, not percentages')
        put(output/'recomputed_summary.json',output_summary)
        put(output/'replicate_results.json',repeat_rows)
        put(output/'diagnostics.json',dict(models=model_diagnostics,calibrations=cal_diagnostics,evaluation_saturation=sat_diagnostics,costs=costs))
        assert file_hash(archive) == receipt['archive_sha256']
        validation.update(status='PASS',checks=checked,primary_classes={k:sum(r['classification']==k for r in primary) for k in ['improvement','loss','unresolved']},
                          fpr_flags_by_policy={p:sum(r['excess_flag'] for r in fpr if r['policy']==p) for p in ['B','F','C','R']},
                          raw_archive_unchanged=True, elapsed_seconds=time.monotonic()-start,
                          analysis_source_sha256=file_hash(__file__),
                          scope='All normal reference fits, cutoffs, evaluation ranks/bits, paired counts and aggregate intervals verified.',
                          limitations=['Saved, numerically verified model/rank values used for evaluation to preserve frozen detector.',
                                       'Same agent, separate arithmetic path; not independent scientific replication.',
                                       'No fresh trajectories, no threshold adjustment, no empirical-FPR matching.',
                                       'Per-repeat Wilson bounds not independently recomputed in this analysis; outer t intervals recomputed.'])
        put(output/'validation.json',validation)
        print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        validation.update(status='FAILED',error=type(exc).__name__+': '+str(exc),elapsed_seconds=time.monotonic()-start)
        put(output/'validation.json',validation);raise


if __name__=='__main__':run(sys.argv[1],sys.argv[2])
