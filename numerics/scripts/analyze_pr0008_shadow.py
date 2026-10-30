# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Saved PR-0007 statistics only: independent old calibration and shadow update gates."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import math
import platform
import resource
import time
import zipfile
import numpy as np
import analyze_pr0006_budget as h
import pr0006_methods as frozen
import pr0007_spec as spec
import pr0008_shadow_methods as m

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT/'numerics/analysis/pr0007_60acab7af20d_performance_v1'
DEFAULT = ROOT/'numerics/analysis/pr0008_60acab7af20d_shadow_gates_v1'
CARD = ROOT/'audits/prereg/PR-0008_update_gate_saved_data.md'


def clean_json(x):
    if isinstance(x, dict): return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray): return clean_json(x.tolist())
    if isinstance(x, np.generic): return clean_json(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def write(out, name, obj):
    (out/name).write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def snapshot(out):
    prev = json.loads((PRIOR/'delivery_validation.json').read_text())
    orig = json.loads((PRIOR/'start_snapshot.json').read_text())
    protected = {**orig['protected'], **prev['outputs'], **prev['sources']}
    for k in ['numerics/design/POST_PR0007_NORMALITY_AND_UPDATE_GATE_OPTIONS_v1_JA.md',
              'audits/verification/2026-10-18_post_pr0007_design.json',
              'audits/prereg/PR-0008_update_gate_saved_data.md']:
        protected[k] = h.digest(ROOT/k)
    for k, v in protected.items(): assert h.digest(ROOT/k) == v, k
    assert h.digest(CARD) == m.CARD_SHA
    nav = json.loads((ROOT/'audits/verification/2026-10-18_post_pr0007_design.json').read_text())
    for k, v in nav['navigation_hashes'].items(): assert h.digest(ROOT/k) == v, k
    ledger = ROOT/'audits/ledger_2026-10.md'
    result = dict(protected=protected, ledger_bytes=ledger.stat().st_size, ledger_sha256=h.digest(ledger),
                  card_sha256=m.CARD_SHA, python=platform.python_version(), numpy=np.__version__,
                  sources={str(p.relative_to(ROOT)):h.digest(p) for p in
                           (Path(__file__), Path(m.__file__), ROOT/'numerics/scripts/test_pr0008_shadow.py')})
    write(out, 'start_snapshot.json', result)
    return result


def raw_task(z, prefix, task, counters):
    chunks = []
    for start in range(0, task['count'], spec.CHUNK):
        a = h.npz(z, prefix+f'chunks/{task["case"]}/{task["role"]}/{start:09d}.npz')
        v = a['values']; meta = json.loads(str(a['metadata']))
        assert v.dtype == np.float64 and v.shape == (min(spec.CHUNK, task['count']-start), 17)
        assert meta['task'] == task and meta['start'] == start and meta['values_sha256'] == h.sha(v.tobytes())
        assert np.isfinite(v).all(), 'Nonfinite saved data: dedicated failure report required'
        chunks.append(v); counters['raw_chunks_checked'] += 1; counters['raw_records_checked'] += len(v)
    return np.concatenate(chunks)


def independent_score(a, alarm, valid):
    means, sds = [], []
    for col in (11, 12):
        avg = math.fsum(a[:, col])/len(a)
        sd = math.sqrt(math.fsum((float(v)-avg)**2 for v in a[:, col])/(len(a)-1))
        means.append(avg); sds.append(sd)
    g0 = np.count_nonzero(alarm)/len(a) if valid.all() else np.nan
    tm = math.sqrt(len(a))*abs(means[0])/sds[0] if sds[0] > 0 else np.nan
    tc = math.sqrt(len(a))*means[1]/sds[1] if sds[1] > 0 else np.nan
    return np.array([g0, tm, max(tm, tc) if np.isfinite([tm, tc]).all() else np.nan])


def run(out):
    start_time = time.perf_counter(); counters = Counter(); snap = snapshot(out)
    inventory = json.loads((PRIOR/'archive_inventory.json').read_text()); by_rep = {}
    for row in inventory:
        path = ROOT/row['path']
        assert path.stat().st_size == row['bytes'] and h.digest(path) == row['sha256'], path
        for r in row['coverage']: assert r not in by_rep; by_rep[r] = path
    assert sorted(by_rep) == list(range(25))
    write(out, 'input_archives.json', inventory)
    (out/'old_calibration').mkdir(); (out/'replicates').mkdir(); (out/'splits').mkdir()
    score = np.full((25, 6, 7, 3), np.nan); details = []
    uc = np.zeros((25, 6, 7, 4, 2, 6), np.int64)
    fc = np.zeros((25, 6, 4, 2, 6), np.int64)
    uf = np.zeros(uc.shape+(4,), np.int64)
    ui = np.zeros((25, 6, 7, 2), np.int64); fi = np.zeros((25, 6, 2), np.int64)
    split_audit = []; crows = spec.contexts()
    old_bit_files = []
    for r in range(25):
        prefix = f'shards/{r//5:02d}/rep_{r:02d}/'
        with zipfile.ZipFile(by_rep[r]) as z:
            arrays = {(t['case'], t['role']): raw_task(z, prefix, t, counters) for t in spec.tasks(r)}
            models = {b:h.model_check(arrays[b, 'ref'], h.npz(z, prefix+'models/'+b+'.npz')) for b in ('B_M','B_L')}
            orders = {g:h.npz(z, prefix+'orders/'+g+'.npz') for g in m.GROUPS}
            old = {}; split_save = {}
            for g in m.GROUPS:
                order = orders[g]['normal_order']; pos = orders[g]['replacement_order']
                assert np.array_equal(np.sort(order), np.arange(31999))
                assert np.array_equal(np.sort(pos), np.arange(15999))
                split_save[g+'_U_normal_ids'] = order[:15999]
                split_save[g+'_replacement_positions'] = pos
                if g in models:
                    old_ids = order[15999:31998]
                    assert len(old_ids) == 15999 and not np.intersect1d(order[:15999], old_ids).size
                    split_save[g+'_Ostar_ids'] = old_ids
                    a = arrays[g, 'normal_cal'][old_ids]; state, ordered = models[g]
                    meta, cut, indices = frozen.calibrate(state, ordered, a)
                    assert meta['valid']
                    saved = dict(metadata=np.array(json.dumps(meta)), cut=cut, indices=indices,
                                 input_sha256=np.array(h.sha(a.tobytes())))
                    h.cal_check(a, state, ordered, saved)
                    np.savez_compressed(out/'old_calibration'/f'{r:02d}_{g}.npz', **saved, ids=old_ids)
                    old[g] = meta, cut
                    split_audit.append(dict(replicate=r, baseline=g, u_count=15999, old_count=15999,
                        intersection=0, omitted=1, old_input_sha256=str(saved['input_sha256']),
                        ids_sha256=h.sha(old_ids.tobytes()), source_role='normal_cal'))
                    counters['old_calibrations'] += 1
            np.savez_compressed(out/'splits'/f'rep_{r:02d}.npz', **split_save)
            cals = {}
            for gi, g in enumerate(m.GROUPS):
                state, ordered = models[spec.baseline(g)]; order = orders[g]['normal_order']; pos = orders[g]['replacement_order']
                fi[r, gi] = old[spec.baseline(g)][0]['lf0_structurally_inactive']
                for ci, (fault, n) in enumerate(m.CANDIDATES):
                    key = spec.clean_id(g, 15999) if not n else spec.mix_id(g, fault, n)
                    data = arrays[g, 'normal_cal'][order[:15999]].copy()
                    ids = order[:15999].copy(); origin = np.zeros(15999, np.uint8)
                    if n:
                        data[pos[:n]] = arrays[g+'__'+fault, 'fault_cal'][:n]
                        ids[pos[:n]] = np.arange(n); origin[pos[:n]] = 1
                    saved = h.npz(z, prefix+'calibration/'+key+'.npz')
                    assert np.array_equal(ids, saved['record_ids']) and np.array_equal(origin, saved['origin_role'])
                    meta, cut = h.cal_check(data, state, ordered, saved); assert meta['valid']
                    cals[gi, ci] = meta, cut
                    ui[r, gi, ci] = meta['lf0_structurally_inactive']
                    om, oc = old[spec.baseline(g)]
                    signed, finite = frozen.signed(state, data)
                    alarm = (np.column_stack((signed[:,0], -signed[:,0], signed[:,1])) > oc[0,0]).any(1)
                    s, desc = m.batch_scores(data, alarm, finite)
                    s2 = independent_score(data, alarm, finite)
                    assert np.allclose(s, s2, atol=2e-11, rtol=2e-12, equal_nan=True), (r,g,ci,s,s2)
                    score[r, gi, ci] = s
                    details.append(dict(replicate=r, group=g, candidate=ci, fault=fault, count=n,
                                        score=s, **desc))
                    counters['candidate_batches'] += 1
            oldbits = {}
            for gi, g in enumerate(m.GROUPS):
                state, ordered = models[spec.baseline(g)]; om, oc = old[spec.baseline(g)]
                for ei, fault in enumerate(('normal',)+spec.FAULTS):
                    case = g if ei == 0 else g+'__'+fault
                    role = 'null_eval' if ei == 0 else 'fault_eval'
                    all_old = []
                    rel = [c for c in crows if c['case'] == case]
                    for start in range(0, len(arrays[case, role]), spec.CHUNK):
                        values = arrays[case, role][start:start+spec.CHUNK]
                        packed = frozen.decide(state, ordered, om, oc, values)
                        second = m.independent_decide(values, state, ordered, om, oc)
                        assert np.array_equal(packed, second), (r,case,start,'Fstar')
                        fa, fv = m.decode(packed); assert fv.all()
                        fc[r,gi,ei] += fa.sum(0)
                        all_old.append(packed); counters['Fstar_binary_decisions'] += len(values)*12
                        saved = h.npz(z, prefix+f'decisions/{case}/{start:09d}.npz')
                        smeta = json.loads(str(saved['metadata']))
                        assert smeta['context_indices'] == [c['index'] for c in rel]
                        assert smeta['packed_sha256'] == h.sha(saved['packed'].tobytes())
                        for ci,(tag,n) in enumerate(m.CANDIDATES):
                            key = spec.clean_id(g,15999) if not n else spec.mix_id(g,tag,n)
                            pos = next(i for i,c in enumerate(rel) if c['id'] == key)
                            ubits = saved['packed'][:,pos]
                            meta, cut = cals[gi,ci]
                            check = frozen.decide(state, ordered, meta, cut, values)
                            assert np.array_equal(ubits,check), (r,case,key,start,'U')
                            ua, uv = m.decode(ubits); assert uv.all()
                            uc[r,gi,ci,ei] += ua.sum(0)
                            uf[r,gi,ci,ei] += m.paired(ubits,packed)
                            counters['U_binary_decisions_rechecked'] += len(values)*12
                    oldbits[case] = np.concatenate(all_old)
            fname = out/'replicates'/f'Fstar_{r:02d}.npz'
            np.savez_compressed(fname, **oldbits); old_bit_files.append(str(fname.relative_to(out)))
        print(f'Replicate {r:02d} PASS: independent split, 42 candidates, Fstar and U bits', flush=True)
    thresholds = m.fold_thresholds(score[:,:,0]); states = m.gate_states(score, thresholds)
    # Independent explicit fold enumeration; neither evaluation counts nor mixed scores used.
    for fold in range(5):
        for wi in range(2):
            for gate in range(3):
                rows = [score[r,gi,0,gate] for r in range(25) if r//5 != fold
                        for gi in range(6) if m.WINDOWS[gi] == wi]
                got = sorted(rows)[-1] if all(math.isfinite(x) for x in rows) else np.nan
                assert np.isclose(got,thresholds[fold,wi,gate],rtol=0,atol=0,equal_nan=True)
                for r in range(fold*5,fold*5+5):
                    for gi in range(6):
                        if m.WINDOWS[gi] != wi: continue
                        for ci in range(7):
                            s = score[r,gi,ci,gate]
                            expect = 2 if not math.isfinite(s) or not math.isfinite(got) else int(s>got)
                            assert states[r,gi,ci,gate] == expect
    counts = m.select_counts(fc,uc,states)
    denom = np.array([20000,5000,5000,5000])
    assert np.array_equal(uf[...,0]+uf[...,1],uc)
    assert np.array_equal(uf[...,0]+uf[...,2],np.broadcast_to(fc[:,:,None],uc.shape))
    assert np.all(uf.sum(-1) == denom[None,None,None,:,None,None])
    # Six comparisons: each gate versus U and F*. Both/gained/lost/neither.
    pairs = np.zeros(uc.shape+(3,2,4),np.int64)
    for gate in range(3):
        hold = states[...,gate,None,None,None] > 0
        pairs[...,gate,0,0] = np.where(hold,uf[...,0],uc)
        pairs[...,gate,0,1] = np.where(hold,uf[...,2],0)
        pairs[...,gate,0,2] = np.where(hold,uf[...,1],0)
        pairs[...,gate,0,3] = np.where(hold,uf[...,3],denom[None,None,None,:,None,None]-uc)
        of = np.broadcast_to(fc[:,:,None],uc.shape)
        for j in range(4):
            ident = of if j==0 else denom[None,None,None,:,None,None]-of if j==3 else 0
            pairs[...,gate,1,j] = np.where(hold,ident,uf[...,j])
        for ref,pi in [(0,1),(1,0)]:
            assert np.array_equal(pairs[...,gate,ref,1]-pairs[...,gate,ref,2], counts[...,gate+2]-counts[...,pi])
            assert np.array_equal(pairs[...,gate,ref,0]+pairs[...,gate,ref,1],counts[...,gate+2])
    inactive = np.stack([np.broadcast_to(fi[:,:,None],ui.shape),ui] +
        [np.where(states[...,g,None]>0,fi[:,:,None],ui) for g in range(3)],axis=-1)
    np.savez_compressed(out/'results.npz', scores=score, thresholds=thresholds, states=states,
                        U_counts=uc,Fstar_counts=fc,UF_pairs=uf,policy_counts=counts,policy_pairs=pairs,
                        U_inactive=ui,Fstar_inactive=fi,policy_inactive=inactive)
    write(out,'gate_scores.json',details); write(out,'split_audit.json',split_audit)
    write(out,'metadata.json',dict(groups=m.GROUPS,candidates=m.CANDIDATES,gates=m.GATES,policies=m.POLICIES,
        methods=spec.METHODS,alpha_denominators=spec.ALPHA_DENOMINATORS,evaluation=('normal',)+spec.FAULTS,
        dimensions=dict(policy_counts='replicate,group,candidate,evaluation,alpha,method,policy',
            policy_pairs='replicate,group,candidate,evaluation,alpha,method,gate,reference(U/Fstar),cell'),
        pair_cells=('both','gained','lost','neither'),status_labels=('NOT-VETOED','HOLD-INCONSISTENT','HOLD-UNKNOWN'),
        Fstar_bit_files=old_bit_files,scope='post-hoc saved-data shadow policy; no prospective confirmation'))
    assert counters['raw_records_checked']==10909800
    assert counters['old_calibrations']==50 and counters['candidate_batches']==1050
    assert counters['Fstar_binary_decisions']==63000000 and counters['U_binary_decisions_rechecked']==441000000
    for p,sh in snap['protected'].items(): assert h.digest(ROOT/p)==sh,p
    write(out,'validation.json',dict(status='PASS_SAVED_DATA_RECOMPUTATION',counters=counters,
        gate_state_counts={k:int((states==i).sum()) for i,k in enumerate(('not_vetoed','inconsistent','unknown'))},
        thresholds=30,role_split_checks=50,fold_thresholds_independently_checked=True,
        Fstar_bits_independent=True,policy_margins_exact=True,old_outputs_unchanged=True,
        scientific_records_generated=0,manuscript_edited=False,elapsed_seconds=time.perf_counter()-start_time,
        rss_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        audit_type='same-agent independent arithmetic and saved-bit comparison',
        output_hashes={str(p.relative_to(out)):h.digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print('PASS',dict(counters),'seconds',time.perf_counter()-start_time,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=DEFAULT); args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try: run(args.out)
    except BaseException as exc:
        write(args.out,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
