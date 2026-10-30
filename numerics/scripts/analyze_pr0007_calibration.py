# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Read-only archive audit and saved-statistic recomputation; no simulation calls."""
from pathlib import Path
import hashlib
import json
import math
import platform
import resource
import time
import zipfile
from collections import Counter
from functools import lru_cache

import numpy as np
import scipy
from scipy import stats
import mpmath as mp
import analyze_pr0006_budget as helper
import pr0007_spec as spec
import pr0007_io as disk
import pr0007_methods as frozen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'numerics/analysis/pr0007_60acab7af20d_performance_v1'
RUN, CELL = '60acab7af20d', 'af385f6b95739a08'
C = helper.COUNTERS
sha, digest, canonical, jread, npz, close = helper.sha, helper.digest, helper.canonical, helper.jread, helper.npz, helper.close


def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def tree(a, b, path='aggregate'):
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a)^set(b))
        for k in a: tree(a[k], b[k], path+'/'+k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for x, y in zip(a, b): tree(x, y, path+'[]')
    elif isinstance(a, float):
        endpoint = path.startswith('aggregate/') and path.rsplit('/', 1)[-1] in ('lower', 'upper')
        assert math.isclose(a, b, abs_tol=1e-10 if endpoint else 2e-12, rel_tol=2e-12), (path, a, b)
        if endpoint:
            helper.DIFFS['cross_library_endpoint'] = max(helper.DIFFS.get('cross_library_endpoint', 0.), abs(a-b))
    else: assert a == b, (path, a, b)


@lru_cache(None)
def critical(family):
    mp.mp.dps = 60
    tail = mp.mpf('.05')/(2*family)
    root = mp.findroot(lambda t: mp.betainc(12, mp.mpf('.5'), 0, 24/(24+t*t), regularized=True)/2-tail,
                       float(stats.t.isf(float(tail), 24)))
    return float(root)


def intervals(obj):
    if isinstance(obj, dict):
        if {'values', 'mean', 'sd', 'mcse', 'family', 'lower', 'upper'} <= set(obj):
            a = np.array(obj['values'], float)
            assert a.shape == (25,) and np.isfinite(a).all()
            m = math.fsum(a)/25
            sd = math.sqrt(math.fsum((x-m)**2 for x in a)/24)
            for key, val in [('mean', m), ('sd', sd), ('mcse', sd/5)]: close(obj[key], val, 'independent_'+key)
            if obj['lower'] is not None:
                for key, sign in [('lower', -1), ('upper', 1)]:
                    assert abs(obj[key]-(m+sign*critical(obj['family'])*sd/5)) <= 1e-10
            else: assert obj['sd'] == 0, obj
            C['independent_intervals'] += 1
        for x in obj.values(): intervals(x)
    elif isinstance(obj, list):
        for x in obj: intervals(x)


def paired(left, right):
    # Independent bitwise decoding, not the frozen paired() implementation.
    cells = np.zeros((2, 6, 4), np.int64)
    denom = np.zeros((2, 6), np.int64)
    for mi in range(6):
        a, b = (left >> (4*mi)) & 15, (right >> (4*mi)) & 15
        ok = ((a & 8)>0) & ((b & 8)>0)
        aa, bb = (a & 7)>0, (b & 7)>0
        for j, mask in enumerate((aa & bb, aa & ~bb, ~aa & bb, ~aa & ~bb)):
            cells[:, mi, j] = (ok & mask).sum(0)
        denom[:, mi] = ok.sum(0)
    return cells, denom


def receipt():
    prior = json.loads((ROOT/'audits/verification/2026-10-18_PR-0007_implementation.json').read_text())
    protected = {**prior['protected_anchors'], **prior['source_hashes'], **prior['outputs']}
    protected['audits/fp_corpus.md'] = prior['current_corpus_sha256']
    for p, h in protected.items(): assert digest(ROOT/p) == h, p
    ledger = ROOT/'audits/ledger_2026-10.md'
    save('start_snapshot.json', dict(protected=protected, ledger_bytes=ledger.stat().st_size,
        ledger_sha256=digest(ledger), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        protocol_sha256=digest(OUT/'PROTOCOL.md'), source_sha256=digest(__file__)))
    build = json.loads((ROOT/'numerics/colab/pr0007_lfdc_calibration_a25_build.json').read_text())
    paths = sorted(ROOT.glob('lfdc_calibration_pr0007_v1_'+RUN+'_*.zip'))
    assert len(paths) == 7
    inventory, idxs, by_rep = [], {}, {}
    for p in paths:
        idx = disk.inspect_zip(p, CELL)
        assert idx['run_id'] == RUN and idx['card_sha256'] == spec.CARD_SHA and idx['a01_sha256'] == spec.A01_SHA
        with zipfile.ZipFile(p) as z:
            manifest = jread(z, 'manifest.json')
            assert set(manifest['sources'].values()) == set(build['sources'].values())
            for n, h in manifest['sources'].items(): assert sha(z.read('sources/'+n)) == h
            assert canonical(jread(z, 'specification.json')) == canonical(spec.specification())
        for rep in idx['covered_replicates']:
            assert rep not in by_rep
            by_rep[rep] = p
        idxs[p.name] = idx
        inventory.append(dict(path=p.name, sha256=digest(p), bytes=p.stat().st_size,
            members=len(idx['entries'])+1, sequence=idx['sequence'], coverage=idx['covered_replicates'], created=idx['created']))
        print('Integrity PASS:', p.name, flush=True)
    assert sorted(by_rep) == list(range(25))
    indices = sorted([p for p in paths if 'COMPLETE_INDEX' in p.name], key=lambda p: idxs[p.name]['sequence'])
    assert len(indices) == 2
    old, new = [idxs[p.name]['entries'] for p in indices]
    changed = sorted(n for n in old.keys() & new.keys() if old[n] != new[n])
    added = sorted(new.keys()-old.keys())
    assert not old.keys()-new.keys()
    assert changed == ['active_time.json', 'catalog.json', 'checkpoint_inventory.json', 'delivery_status.json']
    assert all(n.startswith('sessions/') for n in added)
    with zipfile.ZipFile(indices[-1]) as z:
        cat = {a['name']: a for a in jread(z, 'catalog.json')['archives']}
        for row in inventory:
            if row['path'] in cat:
                assert row['sha256'] == cat[row['path']]['sha256'] and row['bytes'] == cat[row['path']]['bytes']
        sessions = {n: jread(z, n) for n in z.namelist() if n.startswith('sessions/') and
                    (n.endswith('/launch.json') or n.endswith('/bootstrap.json') or n.endswith('/environment.json') or n.endswith('/aggregate_timing.json'))}
        pre_name = next(n for n in z.namelist() if n.endswith('preflight/result.json'))
        pre = jread(z, pre_name)
        assert pre['passed'] and pre['records'] == 72 and pre['common_calibration_decisions_equal']
        for r in pre['rows']:
            assert all(v['passed'] for v in r['checks'].values())
            vals = npz(z, pre_name.rsplit('/', 1)[0]+'/'+r['case']+'.npz')
            assert spec.core.compare(vals['cpu'], vals['gpu'])['passed']
        save('runtime_evidence.json', dict(sessions=sessions, preflight=pre,
            summary=jread(z, 'summary.json'), delivery=jread(z, 'delivery_status.json'),
            active_time=jread(z, 'active_time.json')))
    save('archive_inventory.json', inventory)
    save('receipt.json', dict(status='PASS_RECEIPT', run_id=RUN, cell_id=CELL, archives=7,
        member_occurrences=sum(r['members'] for r in inventory), changed_index_members=changed,
        added_session_members=added, scientific_index_content_identical=True,
        selected_index=indices[-1].name, coverage=list(by_rep), source_files=len(manifest['sources']),
        delivery_note='All seven local files checked. Archived browser acknowledgement remains queued; not proof of browser memory/transfer behavior.'))
    return protected, inventory, idxs, by_rep, indices[-1]


def audit_rep(rep, path, idx, iz, spec_hash):
    prefix = f'shards/{rep//5:02d}/rep_{rep:02d}/'
    with zipfile.ZipFile(path) as z:
        cm = jread(iz, f'carry/rep_{rep:02d}.json')
        assert cm == jread(z, f'carry/rep_{rep:02d}.json')
        assert cm['specification_sha256'] == spec_hash and cm['calibration_unchanged']
        assert {n[len(prefix):] for n in z.namelist() if n.startswith(prefix)} == set(cm['files'])
        for n, h in cm['files'].items(): assert idx['entries'][prefix+n]['sha256'] == h
        carry_name = f'carry/rep_{rep:02d}.npz'
        assert sha(z.read(carry_name)) == cm['statistics_sha256'] and z.read(carry_name) == iz.read(carry_name)
        state = jread(iz, 'state.json')
        assert state['completed'][f'{rep:02d}']['metadata_sha256'] == sha(iz.read(f'carry/rep_{rep:02d}.json'))
        arrays, expected, raw_meta = {}, set(), []
        for task in spec.tasks(rep):
            chunks = []
            for start in range(0, task['count'], spec.CHUNK):
                name = prefix+f'chunks/{task["case"]}/{task["role"]}/{start:09d}.npz'
                expected.add(name); f = npz(z, name); a = f['values']; m = json.loads(str(f['metadata']))
                n = min(spec.CHUNK, task['count']-start)
                assert a.dtype == np.float64 and a.shape == (n, 17)
                assert m['task'] == task and m['start'] == start and m['count'] == n
                assert m['specification_sha256'] == spec_hash and m['values_sha256'] == sha(a.tobytes())
                valid = np.isfinite(a).all(1)
                assert m['unavailable_rows'] == int((~valid).sum())
                C['nonfinite_rows'] += int((~valid).sum())
                b = a[valid]; ai, aj = math.exp(-.002/.04), math.exp(-.002/.01)
                lam = (1+ai)*(1-aj)/(2*(1-ai*aj)); nu = (1+aj)*(1-ai)/(2*(1-ai*aj))
                for k, j in ((0, 9), (3, 10), (6, 11)):
                    close(b[:, j], b[:, k+2]-lam*b[:, k]-nu*b[:, k+1], 'moment_expansion')
                close(b[:, 12], b[:, 14]+b[:, 15]-2*b[:, 16], 'CR_expansion')
                chunks.append(a); raw_meta.append(m)
                C['chunks'] += 1; C['records'] += n; C['records_'+task['role']] += n
            arrays[task['case'], task['role']] = np.concatenate(chunks)
            C['tasks'] += 1
        assert expected == {n for n in z.namelist() if n.startswith(prefix+'chunks/')}
        models, cals, cal_rows = {}, {}, []
        for base in ('B_M', 'B_L'):
            models[base] = helper.model_check(arrays[base, 'ref'], npz(z, prefix+'models/'+base+'.npz'))
        for g in spec.groups():
            order_file = npz(z, prefix+'orders/'+g+'.npz')
            order, positions = spec.orders(g, rep)  # permutations only; no scientific random draws
            assert np.array_equal(order, order_file['normal_order']) and np.array_equal(positions, order_file['replacement_order'])
            for c in [x for x in spec.calibrations() if x['group'] == g and x['policy'] != 'F']:
                m, n, key = c['m'], c['mix_count'], c['fit_key']
                ids = order[:m].copy(); origin = np.zeros(m, np.uint8)
                data = arrays[g, 'normal_cal'][ids].copy(); prov = dict(normal_case=g)
                if n:
                    replace = positions[:n]; removed = ids[replace].copy()
                    data[replace] = arrays[g+'__'+c['mix_fault'], 'fault_cal'][:n]
                    ids[replace] = np.arange(n); origin[replace] = 1
                    prov.update(fault_case=g+'__'+c['mix_fault'], replaced_positions=replace.tolist(),
                        removed_normal_ids=removed.tolist(), fault_record_ids=list(range(n)),
                        normal_role='normal_cal', fault_role='fault_cal')
                saved = npz(z, prefix+'calibration/'+key+'.npz')
                assert np.array_equal(saved['record_ids'], ids) and np.array_equal(saved['origin_role'], origin)
                assert json.loads(str(saved['provenance'])) == prov
                meta, cut = helper.cal_check(data, *models[spec.baseline(g)], saved)
                cals[key] = meta, cut
                cal_rows.append(dict(replicate=rep, calibration=c, cut=cut.tolist(), indices=saved['indices'].tolist(), metadata=meta))
        freeze = jread(z, prefix+'CALIBRATION_FROZEN.json')
        assert set(freeze) == {n for n in cm['files'] if n.split('/')[0] in ('models', 'orders', 'calibration')}
        for n, h in freeze.items(): assert h == cm['files'][n]
        frozen_hash = sha(canonical(freeze).encode())
        result = frozen.empty(); expected_decisions = set(); ctx = spec.contexts()
        for case, c in spec.conditions().items():
            role = 'null_eval' if c['kind'] == 'normal' else 'fault_eval'
            task = next(t for t in spec.tasks(rep) if t['case'] == case and t['role'] == role)
            rel = [x for x in ctx if x['case'] == case]; indices = [x['index'] for x in rel]
            local = {ix: j for j, ix in enumerate(indices)}
            for start in range(0, task['count'], spec.CHUNK):
                values = arrays[case, role][start:start+spec.CHUNK]
                name = prefix+f'decisions/{case}/{start:09d}.npz'; expected_decisions.add(name)
                saved = npz(z, name); meta = json.loads(str(saved['metadata']))
                raw_name = prefix+f'chunks/{case}/{role}/{start:09d}.npz'
                identity = dict(task=task, start=start, count=len(values), context_indices=indices,
                    raw_sha256=idx['entries'][raw_name]['sha256'], calibration_sha256=frozen_hash,
                    packed_sha256=sha(saved['packed'].tobytes()))
                assert meta == identity
                bits = []
                for x in rel:
                    k = x['index']; cal = cals[x['fit_key']]
                    packed, cc, aa, pp, nn = helper.decisions(values, *models[spec.baseline(c['group'])], *cal)
                    bits.append(packed)
                    for key, val in [('counts', cc), ('available', aa), ('pairs', pp), ('pair_n', nn)]: result[key][k] += val
                    result['inactive'][k] = cal[0]['lf0_structurally_inactive'] if cal[0]['valid'] else [-1, -1]
                    C['policy_evaluation_records'] += len(values)
                packed = np.stack(bits, axis=1)
                assert saved['packed'].dtype == np.uint32 and np.array_equal(saved['packed'], packed), (rep, case, start)
                for x in spec.contrasts():
                    if x['case'] != case: continue
                    cc, nn = paired(packed[:, local[x['left']]], packed[:, local[x['right']]])
                    result['contrasts'][x['index']] += cc; result['contrast_n'][x['index']] += nn
                C['decision_chunks'] += 1
            C['evaluation_contexts'] += len(rel)
        assert expected_decisions == {n for n in z.namelist() if n.startswith(prefix+'decisions/')}
        saved = npz(iz, carry_name)
        for k in result:
            assert saved[k].dtype == np.int64 and np.array_equal(result[k], saved[k]), (rep, k)
        assert cm['records'] == sum(len(a) for a in arrays.values()) == 436392
        C['carry_arrays_exact'] += len(result)
        save(f'replicate_{rep:02d}_audit.json', dict(replicate=rep, status='PASS', records=cm['records'],
            raw_chunks=len(expected), decision_chunks=len(expected_decisions),
            max_rss_bytes=max(m['memory']['rss_high_water_bytes'] for m in raw_meta),
            max_gpu_reserved_bytes=max(m['memory']['gpu_reserved_bytes'] for m in raw_meta),
            timing_sums={k:sum(m['timing'][k] for m in raw_meta) for k in raw_meta[0]['timing']}))
        return dict(replicate=rep, **result), cal_rows


def main():
    started = time.monotonic()
    assert (OUT/'PROTOCOL.md').exists() and not (OUT/'validation.json').exists()
    helper.tree_close = tree
    protected, inventory, idxs, by_rep, index_path = receipt()
    results, calibrations = [], []
    spec_hash = sha(canonical(spec.specification()).encode())
    with zipfile.ZipFile(index_path) as iz:
        for rep in range(25):
            r, c = audit_rep(rep, by_rep[rep], idxs[by_rep[rep].name], iz, spec_hash)
            results.append(r); calibrations.extend(c)
            print(f'Raw/ref/cal/packed/carry PASS: {rep+1:02d}/25', flush=True)
        assert C['records'] == 10909800 and C['policy_evaluation_records'] == 57750000
        assert C['records_null_eval']+C['records_fault_eval'] == 5250000
        assert C['models'] == 50 and C['calibrations'] == 1350 and C['evaluation_contexts'] == 6600
        assert C['nonfinite_rows'] == 0, 'Dedicated missingness analysis required'
        agg = frozen.aggregate(results)
        for key, val in agg.items(): tree(val, jread(iz, 'analysis/'+key+'.json'), 'aggregate/'+key)
        intervals(agg)
        for p in agg['primary8']:
            d=p['interval']; expected='supports_detection_loss' if d['upper'] is not None and d['upper']<0 else 'reverse_direction' if d['lower'] is not None and d['lower']>0 else 'unresolved'
            assert p['classification'] == expected
        for p in agg['fpr792']: assert p['excess_flag'] == (p['interval']['lower'] is not None and p['interval']['lower']>p['alpha'])
        for key, val in agg.items(): save('recomputed_'+key+'.json', val)
    save('calibration_audit.json', calibrations)
    np.savez_compressed(OUT/'recomputed_counts.npz', **{k:np.stack([r[k] for r in results]) for k in frozen.shapes()})
    for p, h in protected.items(): assert digest(ROOT/p) == h, p
    for row in inventory: assert digest(ROOT/row['path']) == row['sha256']
    save('validation.json', dict(status='PASS_SAVED_DATA_RECOMPUTATION', run_id=RUN, cell_id=CELL,
        counters=dict(C), max_absolute_differences=helper.DIFFS, raw_and_projection_atol=2e-12,
        raw_and_projection_rtol=2e-12, interval_endpoint_atol=1e-10, final_decisions_exact=693000000,
        protected_unchanged=True, protected_count=len(protected), source_zip_unchanged=True,
        independent_t_critical={str(f):critical(f) for f in (1,8,792)},
        scientific_records_generated=0, elapsed_seconds=time.monotonic()-started,
        local_rss_high_water_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitations='Same-agent audit of saved 17-column statistics; not waveform regeneration or independent review.',
        source_sha256=digest(__file__)))
    print('PASS_SAVED_DATA_RECOMPUTATION', dict(C), flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        save('validation_failure.json', dict(status='STOP', error=repr(exc), counters=dict(C), diffs=helper.DIFFS))
        raise
