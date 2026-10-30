# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Saved-data-only audit of PR-0006. No simulator/seed generation is called."""
from pathlib import Path
import hashlib
import io
import json
import math
import platform
import re
import sys
import time
import zipfile
from collections import Counter
from functools import lru_cache

import numpy as np
import scipy
from scipy import stats
import pr0006_spec as spec
import pr0006_io as disk
import pr0006_methods as frozen
import pr0004_methods as old

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'numerics/analysis/pr0006_6f23fc7946f8_performance_v1'
RUN, CELL = '6f23fc7946f8', 'f98f8d0167e97f69'
TOL = dict(atol=2e-12, rtol=2e-12)
COUNTERS = Counter()
DIFFS = {}


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def sha(b): return hashlib.sha256(b).hexdigest()
def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)
def jread(z, name): return json.loads(z.read(name))
def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def close(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert np.allclose(a, b, equal_nan=True, **TOL), label
    ok = np.isfinite(a) & np.isfinite(b)
    d = float(np.max(np.abs(a[ok]-b[ok]))) if ok.any() else 0.
    DIFFS[label] = max(DIFFS.get(label, 0.), d)


def tree_close(a, b, path='aggregate'):
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a)^set(b))
        for k in a: tree_close(a[k], b[k], path+'/'+k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for x, y in zip(a, b): tree_close(x, y, path+'[]')
    elif isinstance(a, float):
        assert math.isclose(a, b, **dict(abs_tol=2e-12, rel_tol=2e-12)), (path, a, b)
    else: assert a == b, (path, a, b)


def npz(z, name):
    raw = z.read(name)
    with zipfile.ZipFile(io.BytesIO(raw)) as nested:
        items = nested.infolist()
        assert len(items) <= 64 and len({x.filename for x in items}) == len(items)
        assert sum(x.file_size for x in items) <= 32*1024**2
        for item in items:
            assert '/' not in item.filename and item.filename.endswith('.npy')
            with nested.open(item) as f:
                version = np.lib.format.read_magic(f)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}[version]
                shape, _, dtype = reader(f)
                assert not dtype.hasobject and dtype.kind in 'fiuU'
                assert len(shape) <= 5 and all(n >= 0 for n in shape)
                assert math.prod(shape)*dtype.itemsize == item.file_size-f.tell()
    # A new buffer deliberately avoids the QA-004 exhausted-stream mistake.
    with np.load(io.BytesIO(raw), allow_pickle=False) as f:
        result = {k: f[k] for k in f.files}
    COUNTERS['npz_checked'] += 1
    return result


def signed(state, values):
    assert values.dtype == np.float64 and values.shape[1] == 17
    valid = np.isfinite(values).all(axis=1)
    result = np.zeros((len(values), 2))
    if not state['valid']: return result, np.zeros(len(values), bool)
    m = state['model']
    d0 = values[valid, 11]-m['mean'][0]
    d1 = values[valid, 12]-m['mean'][1]
    for k, w in enumerate(m['weights']):
        result[valid, k] = d0*w[0]+d1*w[1]
    valid &= np.isfinite(result).all(axis=1)
    return result, valid


def ranks(state, ordered, s):
    r = np.column_stack((np.searchsorted(ordered[:, 0], np.abs(s[:, 0]), side='left'),
                         np.searchsorted(ordered[:, 1], s[:, 1], side='left')))
    r[:, state['model']['disabled']] = 0
    return r


def model_check(values, saved):
    assert str(saved['input_sha256']) == sha(values.tobytes())
    state = json.loads(str(saved['state']))
    refit, order = frozen.fit(values)
    tree_close(refit, state, 'reference')
    close(order, saved['ordered'], 'reference_sorted_refit')
    if not state['valid']:
        COUNTERS['invalid_models'] += 1
        return state, saved['ordered']
    m = state['model']
    q = values[:, [11, 12]]
    mean = np.sum(q, axis=0)/len(q)
    centered = q-mean
    a = float(np.sum(centered[:, 0]**2)/(len(q)-1))
    b = float(np.sum(centered[:, 0]*centered[:, 1])/(len(q)-1))*(1-1e-6)
    d = float(np.sum(centered[:, 1]**2)/(len(q)-1))
    inverse = np.array([[d, -b], [-b, a]])/(a*d-b*b)
    independent_weights = []
    means = np.sum(values, axis=0)/len(values)
    nu = spec.nominal_nu()
    directions = [[means[8]-2*nu*means[7], 2*(means[15]-means[16])], [-nu*means[7], means[15]]]
    for direction in directions:
        direction = np.array(direction)
        if not direction.any(): independent_weights.append([0., 0.]); continue
        direction /= max(abs(direction))
        w = inverse @ direction
        independent_weights.append(w / math.sqrt(sum(direction*w)))
    close(mean, m['mean'], 'independent_reference_mean')
    close(independent_weights, m['weights'], 'independent_2x2_weights')
    COUNTERS['disabled_directions'] += len(m['disabled'])
    COUNTERS['models'] += 1
    return state, saved['ordered']


def cal_check(values, state, ordered, saved):
    assert str(saved['input_sha256']) == sha(values.tobytes())
    meta = json.loads(str(saved['metadata']))
    s, valid = signed(state, values)
    assert meta['valid'] == bool(state['valid'] and valid.all())
    if not meta['valid']:
        COUNTERS['invalid_calibrations'] += 1
        return meta, saved['cut']
    tails = np.sort(np.column_stack((s[:, 0], -s[:, 0], s[:, 1])), axis=0)
    maximum = np.sort(ranks(state, ordered, s).max(axis=1))
    cut = np.full((2, 6, 3), np.inf)
    indices = np.zeros((2, 6, 3), np.int64)
    disabled = ([0, 1] if 0 in state['model']['disabled'] else []) + ([2] if 1 in state['model']['disabled'] else [])
    assert meta['disabled_tails'] == disabled and meta['m'] == len(values)
    for ai, den in enumerate(spec.ALPHA_DENOMINATORS):
        for mi, budget in enumerate(spec.BUDGETS.values()):
            for j, w in enumerate(budget):
                numerator = (len(values)+1)*(8*den-w)
                k = (numerator+8*den-1)//(8*den)
                indices[ai, mi, j] = k
                if k <= len(values) and j not in disabled: cut[ai, mi, j] = tails[k-1, j]
        k = ((len(values)+1)*(den-1)+den-1)//den
        indices[ai, 5] = k
        cut[ai, 5] = maximum[k-1] if k <= len(values) else len(ordered)
    assert np.array_equal(indices, saved['indices'])
    close(cut, saved['cut'], 'calibration_cut')
    assert meta['lf0_upper_rank'] == len(ordered)
    assert meta['lf0_structurally_inactive'] == (cut[:, 5, 0] >= len(ordered)).tolist()
    z = np.column_stack((s[:, 0], -s[:, 0], s[:, 1]))
    ties = np.zeros((2, 6, 3), int)
    for ai in range(2):
        for mi in range(5): ties[ai, mi] = (z == cut[ai, mi]).sum(axis=0)
        ties[ai, 5] = (maximum == cut[ai, 5, 0]).sum()
    assert np.array_equal(ties, meta['ties'])
    COUNTERS['calibrations'] += 1
    return meta, saved['cut']


def decisions(values, state, ordered, meta, cut):
    s, valid = signed(state, values)
    if not meta['valid']: valid[:] = False
    bits = np.zeros((len(values), 2, 6), np.uint8)
    if valid.any():
        z = np.column_stack((s[:, 0], -s[:, 0], s[:, 1]))
        r = ranks(state, ordered, s)
        for ai in range(2):
            for mi in range(6):
                if mi < 5:
                    b = z > cut[ai, mi]
                else:
                    g = r[:, 0] > cut[ai, mi, 0]
                    b = np.column_stack((g & (s[:, 0] >= 0), g & (s[:, 0] < 0), r[:, 1] > cut[ai, mi, 0]))
                bits[:, ai, mi] = b[:, 0]+2*b[:, 1]+4*b[:, 2]
        bits[~valid] = 0
    packed = np.zeros((len(values), 2), np.uint32)
    for mi in range(6): packed += ((bits[:, :, mi].astype(np.uint32)+8*valid[:, None]) << (4*mi)).astype(np.uint32)
    counts = np.zeros((2, 6, 8), np.int64)
    pairs = np.zeros((2, 15, 4), np.int64)
    for ai in range(2):
        for mi in range(6):
            for pattern in range(8): counts[ai, mi, pattern] = np.sum(valid & (bits[:, ai, mi] == pattern))
        for pi, (left, right) in enumerate(spec.PAIRS):
            a, b = bits[:, ai, left] > 0, bits[:, ai, right] > 0
            for j, mask in enumerate((a & b, a & ~b, ~a & b, ~a & ~b)): pairs[ai, pi, j] = np.sum(mask & valid)
    return packed, counts, np.full((2, 6), valid.sum(), np.int64), pairs, np.full((2, 15), valid.sum(), np.int64)


@lru_cache(None)
def critical(family): return float(stats.t.isf(.05/(2*family), 24))


def interval_check(x, got, family=1):
    x = np.asarray(x, float)
    assert x.shape == (25,) and np.isfinite(x).all()
    mean = math.fsum(x)/25
    sd = math.sqrt(math.fsum((v-mean)**2 for v in x)/24)
    se = sd/5
    assert got['family'] == family and got['replicates'] == 25
    for name, value in [('mean', mean), ('sd', sd), ('mcse', se),
                        ('q10', np.quantile(x, .1)), ('q50', np.median(x)), ('q90', np.quantile(x, .9))]:
        close(value, got[name], 'outer_'+name)
    if np.ptp(x) == 0:
        # The frozen routine uses its numeric SD==0 criterion, not ptp==0.
        if got['sd'] == 0: assert got['lower'] is None and got['upper'] is None
        else:
            close(got['lower'], mean-critical(family)*se, 'outer_lower')
            close(got['upper'], mean+critical(family)*se, 'outer_upper')
    else:
        close(got['lower'], mean-critical(family)*se, 'outer_lower')
        close(got['upper'], mean+critical(family)*se, 'outer_upper')
    COUNTERS['independent_intervals'] += 1


def aggregate_check(results, agg):
    cs = np.stack([r['counts'] for r in results])
    av = np.stack([r['available'] for r in results])
    pair = np.stack([r['pairs'] for r in results])
    pn = np.stack([r['pair_n'] for r in results])
    assert np.all(av > 0), 'Unavailable cases require a dedicated report, not silent dropping'
    rates = (av-cs[..., 0])/av
    def ix(row):
        return (spec.contexts().index((row['case'], row['policy'])),
                spec.ALPHA_DENOMINATORS.index(round(1/row['alpha'])), spec.METHODS.index(row['method']))
    for r in agg['rates']:
        ki, ai, mi = ix(r)
        interval_check(rates[:, ki, ai, mi], r['rate'])
        interval_check(1-rates[:, ki, ai, mi], r['miss'])
        for b in range(3):
            branch = sum(cs[:, ki, ai, mi, p] for p in range(8) if p & (1 << b))
            interval_check(branch/av[:, ki, ai, mi], r['branches'][b])
    for r in agg['fpr216']:
        ki, ai, mi = ix(r)
        interval_check(rates[:, ki, ai, mi], r['interval'], 216)
        assert r['excess_flag'] == (r['interval']['lower'] is not None and r['interval']['lower'] > r['alpha'])
    for r in agg['primary25']:
        ki = spec.contexts().index((r['case'], r['policy']))
        ai = spec.ALPHA_DENOMINATORS.index(round(1/r['alpha']))
        li, ri = spec.METHODS.index(r['left']), spec.METHODS.index(r['right'])
        interval_check(rates[:, ki, ai, li]-rates[:, ki, ai, ri], r['interval'], 25)
    for r in agg['paired']:
        ki = spec.contexts().index((r['case'], r['policy']))
        ai = spec.ALPHA_DENOMINATORS.index(round(1/r['alpha']))
        pi = spec.PAIRS.index((spec.METHODS.index(r['left']), spec.METHODS.index(r['right'])))
        for j, name in enumerate(('both', 'gained', 'lost', 'neither')):
            interval_check(pair[:, ki, ai, pi, j]/pn[:, ki, ai, pi], r['cells'][name])
        interval_check((pair[:, ki, ai, pi, 1]-pair[:, ki, ai, pi, 2])/pn[:, ki, ai, pi], r['difference'])
        assert np.array_equal(pair[:, ki, ai, pi].sum(axis=1), pn[:, ki, ai, pi])
    for r in agg['contrasts']:
        ki, ai, mi = ix(r)
        other = (r['other'], r['policy'] if r['kind'] == 'positive-minus-negative' else 'F')
        kj = spec.contexts().index(other)
        interval_check(rates[:, ki, ai, mi]-rates[:, kj, ai, mi], r['interval'])


def main():
    started = time.monotonic()
    assert (OUT/'PROTOCOL.md').exists() and not (OUT/'validation.json').exists()
    baseline = {}
    handoff = (ROOT/'HANDOFF/2026-10-17_session-01.md').read_text()
    anchors = re.findall(r'\| `([^`]+)` \| `([a-f0-9]{64})` \|', handoff)
    for p, h in anchors:
        assert digest(ROOT/p) == h
        baseline[p] = h
    build = json.loads((ROOT/'numerics/colab/pr0006_lfdc_budget_b25_build.json').read_text())
    for section in ('sources', 'artifacts'):
        for p, h in build[section].items():
            assert digest(ROOT/p) == h
            baseline[p] = h
    freeze = json.loads((ROOT/'audits/verification/2026-10-17_PR-0006_design_freeze.json').read_text())
    for p, h in freeze['protected'].items():
        assert digest(ROOT/p) == h, p
    save('start_snapshot.json', dict(baseline=baseline, protected_count=len(freeze['protected']),
        ledger_bytes=(ROOT/'audits/ledger_2026-10.md').stat().st_size,
        ledger_sha256=digest(ROOT/'audits/ledger_2026-10.md'),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        protocol_sha256=digest(OUT/'PROTOCOL.md'), source_sha256=digest(__file__)))
    index_path = next(ROOT.glob('*pr0006*COMPLETE_INDEX*.zip'))
    archives = [index_path] + sorted(ROOT.glob('*pr0006*SHARD*COMPLETE*.zip'))
    assert len(archives) == 6
    index_zip = zipfile.ZipFile(index_path)
    catalog = {r['name']: r for r in jread(index_zip, 'catalog.json')['archives']}
    sources = set(build['sources'].values())
    records = []
    by_rep = {}
    indexes = {}
    for path in archives:
        idx = disk.inspect_zip(path, CELL)
        indexes[path.name] = idx
        assert idx['run_id'] == RUN
        h = digest(path)
        with zipfile.ZipFile(path) as z:
            m = jread(z, 'manifest.json')
            assert set(m['sources'].values()) == sources
            assert canonical(jread(z, 'specification.json')) == canonical(spec.specification())
        if path != index_path:
            c = catalog[path.name]
            assert c['sha256'] == h and c['bytes'] == path.stat().st_size
            assert c['covered_replicates'] == idx['covered_replicates']
        for rep in idx['covered_replicates']:
            assert rep not in by_rep
            by_rep[rep] = path
        records.append(dict(path=path.name, sha256=h, bytes=path.stat().st_size,
                            members=len(idx['entries']), coverage=idx['covered_replicates']))
        print('Archive integrity:', path.name, flush=True)
    assert sorted(by_rep) == list(range(25))
    save('archive_inventory.json', records)
    spec_hash = sha(canonical(spec.specification()).encode())
    results = []
    for rep in range(25):
        z = zipfile.ZipFile(by_rep[rep]); prefix = f'shards/{rep//5:02d}/rep_{rep:02d}/'
        idx = indexes[by_rep[rep].name]
        carry_meta = jread(index_zip, f'carry/rep_{rep:02d}.json')
        assert carry_meta == jread(z, f'carry/rep_{rep:02d}.json')
        assert carry_meta['specification_sha256'] == spec_hash and carry_meta['calibration_unchanged']
        assert {n[len(prefix):] for n in z.namelist() if n.startswith(prefix)} == set(carry_meta['files'])
        for name, h in carry_meta['files'].items(): assert idx['entries'][prefix+name]['sha256'] == h
        assert sha(z.read(f'carry/rep_{rep:02d}.npz')) == carry_meta['statistics_sha256']
        assert z.read(f'carry/rep_{rep:02d}.npz') == index_zip.read(f'carry/rep_{rep:02d}.npz')
        arrays = {}
        expected = set()
        for task in spec.tasks(rep):
            chunks = []
            for start in range(0, task['count'], spec.CHUNK):
                name = prefix+f'chunks/{task["case"]}/{task["role"]}/{start:09d}.npz'
                expected.add(name); f = npz(z, name); a = f['values']; meta = json.loads(str(f['metadata']))
                n = min(spec.CHUNK, task['count']-start)
                assert a.dtype == np.float64 and a.shape == (n, 17)
                assert meta['task'] == task and meta['start'] == start and meta['count'] == n
                assert meta['specification_sha256'] == spec_hash and meta['values_sha256'] == sha(a.tobytes())
                finite = np.isfinite(a).all(axis=1)
                assert meta['unavailable_rows'] == int((~finite).sum())
                COUNTERS['nonfinite_rows'] += int((~finite).sum())
                b = a[finite]
                ai, aj = math.exp(-.002/.04), math.exp(-.002/.01)
                lam = (1+ai)*(1-aj)/(2*(1-ai*aj)); nu = (1+aj)*(1-ai)/(2*(1-ai*aj))
                for k, j in ((0, 9), (3, 10), (6, 11)):
                    close(b[:, j], b[:, k+2]-lam*b[:, k]-nu*b[:, k+1], 'moment_identity')
                close(b[:, 12], b[:, 14]+b[:, 15]-2*b[:, 16], 'CR_identity')
                chunks.append(a); COUNTERS['chunks'] += 1; COUNTERS['records'] += n
                COUNTERS['records_'+task['role']] += n
            arrays[task['case'], task['role']] = np.concatenate(chunks)
            COUNTERS['tasks'] += 1
        assert expected == {n for n in z.namelist() if n.startswith(prefix+'chunks/')}
        models = {}; calibration = {}
        for base in ('B_M', 'B_L'):
            models[base] = model_check(arrays[base, 'ref'], npz(z, prefix+'models/'+base+'.npz'))
        for group, c in spec.conditions().items():
            if c['kind'] == 'normal':
                calibration[group] = cal_check(arrays[group, 'cal'], *models[spec.baseline(group)], npz(z, prefix+'calibration/'+group+'.npz'))
        recomputed = dict(counts=[], available=[], pairs=[], pair_n=[], inactive=[])
        for case, policy in spec.contexts():
            c = spec.conditions()[case]; role = 'null' if c['kind'] == 'normal' else c['kind']
            base = spec.baseline(c['group']); state, ordered = models[base]
            meta, cut = calibration[base if policy in ('B', 'F') else c['group']]
            a = arrays[case, role]
            packed, counts, available, pairs, pair_n = decisions(a, state, ordered, meta, cut)
            saved = npz(z, prefix+f'decisions/{case}_{policy}.npz')['packed']
            assert saved.dtype == np.uint32 and np.array_equal(saved, packed), (rep, case, policy, 'decision bits')
            for key, value in [('counts', counts), ('available', available), ('pairs', pairs), ('pair_n', pair_n)]: recomputed[key].append(value)
            recomputed['inactive'].append(meta['lf0_structurally_inactive'] if meta['valid'] else [-1, -1])
            COUNTERS['evaluation_contexts'] += 1; COUNTERS['policy_evaluation_records'] += len(a)
        saved_carry = npz(index_zip, f'carry/rep_{rep:02d}.npz')
        for key in recomputed:
            recomputed[key] = np.asarray(recomputed[key], np.int64)
            assert np.array_equal(recomputed[key], saved_carry[key]), (rep, key)
        assert carry_meta['records'] == sum(len(a) for a in arrays.values()) == 549988
        results.append(dict(replicate=rep, **recomputed))
        z.close()
        print(f'Raw/ref/cal/decisions/carry matched: {rep+1:02d}/25', flush=True)
    assert COUNTERS['records'] == 13749700 and COUNTERS['policy_evaluation_records'] == 16250000
    assert COUNTERS['records_null']+COUNTERS['records_fault']+COUNTERS['records_control'] == 9250000
    assert COUNTERS['models'] == 50 and COUNTERS['calibrations'] == 250 and COUNTERS['evaluation_contexts'] == 1900
    agg = frozen.aggregate(results)
    tree_close(agg, jread(index_zip, 'analysis/aggregate.json'))
    aggregate_check(results, agg)
    save('recomputed_results.json', agg)
    np.savez_compressed(OUT/'recomputed_counts.npz', **{k:np.stack([r[k] for r in results]) for k in results[0] if k != 'replicate'})
    for p, h in baseline.items(): assert digest(ROOT/p) == h
    for p, h in freeze['protected'].items(): assert digest(ROOT/p) == h
    summary = jread(index_zip, 'summary.json')
    save('run_summary.json', summary)
    save('validation.json', dict(status='PASS_SAVED_DATA_RECOMPUTATION', run_id=RUN, cell_id=CELL,
        counts=dict(COUNTERS), max_absolute_differences=DIFFS, tolerance=TOL,
        archives=len(archives), archive_member_occurrences=sum(r['members'] for r in records),
        protected_files=len(freeze['protected']), protected_unchanged=True,
        exact_final_decisions=True, calibration_roles_separate=True,
        inference_check='fsum variance + inverse-survival t; all rate/branch/pair/primary/FPR/contrast intervals',
        limitation='same agent self-audit; stored 17-column statistics, not regenerated waveforms; no simulator called',
        elapsed_seconds=time.monotonic()-started, source_sha256=digest(__file__)))
    print('PASS_SAVED_DATA_RECOMPUTATION', dict(COUNTERS), flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        save('validation_failure.json', dict(status='STOP', error=repr(exc), counts=dict(COUNTERS), max_absolute_differences=DIFFS))
        raise
