# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""PR-0009 B: fixed nested validation budgets; saved decisions only."""
from pathlib import Path
import hashlib, json, math, time
import numpy as np
from scipy.stats import beta, binom
from analyze_pr0009_mechanisms import Archive
from pr0009_storage import canonical

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_two_axis_v1'
PREV = ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_mechanisms_v1'
OUT = ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_v_budget_v1'
GROUPS = ['B_M', 'B_L', 'K2_M', 'K200_L', 'V05_L', 'V15_L']
SIZES = np.array([500, 1000, 2000, 5000, 10000, 20000])
ALPHAS = [.01, .001]
ETA = .0125
LABELS = ['SUPPORTED', 'EXCEEDED', 'UNRESOLVED']

def h(b): return hashlib.sha256(b).hexdigest()
def load(p): return json.loads(p.read_text())
def dump(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def intervals(k, n):
    k = np.asarray(k)
    lo = np.where(k == 0, 0., beta.ppf(ETA, k, n-k+1))
    hi = np.where(k == n, 1., beta.ppf(1-ETA, k+1, n-k))
    return lo, hi

def classify(k, n, alpha):
    lo, hi = intervals(k, n)
    return np.where(hi <= 1.25*alpha, 0, np.where(lo > 1.25*alpha, 1, 2)), lo, hi

def main():
    started = time.monotonic()
    assert not (OUT/'validation.json').exists(), 'Use a new output version for changed analysis'
    anchors = load(PREV/'closeout_validation.json')
    protected = {}
    for source in [anchors, load(OLD/'closeout_validation.json')]:
        for key in ['outputs', 'sources', 'frozen_sources']:
            protected.update(source.get(key, {}))
    for p in list(ROOT.glob('*.md')) + list((ROOT/'latex').rglob('*.tex')) + list((ROOT/'output/pdf').glob('*.pdf')):
        protected[str(p.relative_to(ROOT))] = h(p.read_bytes())
    for name in ['docs/THROUGHLINE.md', 'audits/fp_corpus.md', 'audits/prereg/README.md']:
        protected[name] = h((ROOT/name).read_bytes())
    for name, digest in protected.items(): assert h((ROOT/name).read_bytes()) == digest, name
    ledger = (ROOT/'audits/ledger_2026-10.md').read_bytes()
    dump('start_snapshot.json', dict(protected=protected, ledger_bytes=len(ledger), ledger_sha256=h(ledger),
        protocol_sha256=h((OUT/'PROTOCOL.md').read_bytes()), previous_anchor_sha256=h((PREV/'closeout_validation.json').read_bytes())))
    arc = Archive()
    counts = np.zeros((50, 6, 6, 2), np.int64)
    segments = np.zeros_like(counts)
    ecounts = np.zeros((50, 6, 2), np.int64)
    provenance = []
    for r in range(50):
        folder = f'shards/{r//5:02d}/rep_{r:02d}'
        frozen = arc.js(folder+'/CALIBRATION_FROZEN.json')
        fh = h(canonical(frozen).encode())
        for name, digest in frozen.items(): assert arc.index[folder+'/'+name][1]['sha256'] == digest
        with np.load(OLD/'recomputed'/f'rep_{r:02d}.npz') as z: old = {k:z[k] for k in z.files}
        for g, group in enumerate(GROUPS):
            pair = []
            for sub, role, nc in [('validation','validation_normal',1), ('decisions','null_eval',8)]:
                chunks = []; raw_hashes = []
                for start in range(0, 20000, 4096):
                    d = arc.npz(f'{folder}/{sub}/{group}/{start:09d}.npz')
                    b = d['packed']; md = json.loads(str(d['metadata'])); task = md['task']
                    n = min(4096,20000-start)
                    assert b.dtype == np.uint32 and b.shape == (n,nc,2)
                    assert md['start'] == start and md['count'] == n and md['calibration_sha256'] == fh
                    assert task == dict(case=group,count=20000,id=f'{r:02d}/{group}/{role}',replicate=r,role=role)
                    assert md['packed_sha256'] == h(b.tobytes())
                    v = b[:,0,:]
                    assert np.all((v & 8) != 0), 'Invalid W0 decisions; do not silently reduce n'
                    chunks.append((v & 7) > 0); raw_hashes.append(md['raw_sha256'])
                alarm = np.concatenate(chunks); assert alarm.shape == (20000,2)
                pair.append(raw_hashes)
                if sub == 'validation':
                    cumulative = np.cumsum(alarm,axis=0,dtype=np.int64)
                    counts[r,g] = cumulative[SIZES-1]
                    # Direct disjoint segment sums independently cross-check prefix indexing.
                    lower = 0
                    for j, upper in enumerate(SIZES):
                        segments[r,g,j] = np.count_nonzero(alarm[lower:upper], axis=0)
                        assert np.array_equal(counts[r,g,j], np.count_nonzero(alarm[:upper],axis=0))
                        lower = upper
                    assert np.array_equal(counts[r,g,-1], old['v_counts'][g,0,:,0,0])
                    assert np.all(old['v_counts'][g,0,:,0,1] == 20000)
                else:
                    ecounts[r,g] = np.count_nonzero(alarm,axis=0)
                    assert np.array_equal(ecounts[r,g],old['counts'][g,0,0,:,0,1:].sum(-1))
                    assert np.all(old['available'][g,0,0,:,0] == 20000)
            assert not set(pair[0]).intersection(pair[1]), 'V/E raw identity overlap'
            provenance.append(dict(replicate=r,group=group,calibration_sha256=fh,V_raw_hashes=pair[0],E_raw_hashes=pair[1]))
        if (r+1)%5 == 0: print(f'{r+1}/50: V and E prefix/count checks matched',flush=True)
    assert np.array_equal(np.cumsum(segments,axis=2),counts)
    states = np.zeros_like(counts); lower = np.zeros_like(counts,dtype=float); upper = lower.copy()
    estates = np.zeros_like(ecounts); elo = np.zeros_like(ecounts,dtype=float); ehi=elo.copy()
    boundaries=[]
    for ai, alpha in enumerate(ALPHAS):
        estates[:,:,ai],elo[:,:,ai],ehi[:,:,ai] = classify(ecounts[:,:,ai],20000,alpha)
        for j,n in enumerate(SIZES):
            states[:,:,j,ai],lower[:,:,j,ai],upper[:,:,j,ai] = classify(counts[:,:,j,ai],n,alpha)
            labels,_,_=classify(np.arange(n+1),n,alpha)
            s=np.flatnonzero(labels==0); x=np.flatnonzero(labels==1)
            ks=int(s[-1]) if len(s) else -1; kx=int(x[0]) if len(x) else n+1
            boundaries.append(dict(alpha=alpha,n=int(n),support_max=ks,exceed_min=kx,
                zero_alarm_upper=float(-np.expm1(np.log(ETA)/n)),
                nominal_support=float(binom.cdf(ks,n,alpha)),
                nominal_exceed=float(binom.sf(kx-1,n,alpha)),
                nominal_unresolved=float(binom.cdf(kx-1,n,alpha)-binom.cdf(ks,n,alpha))))
    original={(v['replicate'],v['group'],v['alpha']):v for v in load(OLD/'validation_evidence.json') if v['variant']=='clean'}
    for r in range(50):
        for g, group in enumerate(GROUPS):
            for ai, alpha in enumerate(ALPHAS):
                v=original[r,group,alpha]
                assert v['result']['label']=='FPR-'+LABELS[states[r,g,-1,ai]]
                assert v['independent_E']['label']=='FPR-'+LABELS[estates[r,g,ai]]
    rows=[]; transitions=[]
    for g,group in enumerate(GROUPS):
        for ai,alpha in enumerate(ALPHAS):
            for j,n in enumerate(SIZES):
                s=states[:,g,j,ai]; es=estates[:,g,ai]
                matrix=np.bincount(3*s+es,minlength=9).reshape(3,3)
                rows.append(dict(group=group,alpha=alpha,n=int(n),labels=np.bincount(s,minlength=3).tolist(),
                    E_labels=np.bincount(es,minlength=3).tolist(),V_E=matrix.tolist(),
                    V_mean_fpr=float(counts[:,g,j,ai].mean()/n),E_mean_fpr=float(ecounts[:,g,ai].mean()/20000),
                    mean_interval_width=float((upper-lower)[:,g,j,ai].mean()),
                    median_upper=float(np.median(upper[:,g,j,ai])),
                    supported_E_mean_fpr=float(ecounts[s==0,g,ai].mean()/20000) if np.any(s==0) else None))
                if j:
                    prev=states[:,g,j-1,ai]
                    transitions.append(dict(group=group,alpha=alpha,n_from=int(SIZES[j-1]),n_to=int(n),
                        matrix=np.bincount(prev*3+s,minlength=9).reshape(3,3).tolist()))
    reversals=[]
    for ai,alpha in enumerate(ALPHAS):
        for g,group in enumerate(GROUPS):
            st=states[:,g,:,ai]; early=np.any(st[:,:-1]==0,axis=1)
            reversals.append(dict(group=group,alpha=alpha,early_support=int(early.sum()),
                early_support_final_unresolved=int(np.sum(early&(st[:,-1]==2))),
                early_support_final_exceeded=int(np.sum(early&(st[:,-1]==1))),
                any_support_then_nonsupport=int(sum(any(st[r,j]==0 and np.any(st[r,j+1:]!=0) for j in range(5)) for r in range(50)))))
    np.savez_compressed(OUT/'counts_and_intervals.npz',sizes=SIZES,counts=counts,segments=segments,E_counts=ecounts,
        states=states,lower=lower,upper=upper,E_states=estates,E_lower=elo,E_upper=ehi)
    dump('resource_curve.json',rows);dump('integer_boundaries.json',boundaries)
    dump('transitions.json',transitions);dump('reversals.json',reversals);dump('provenance.json',provenance)
    for name,digest in protected.items():assert h((ROOT/name).read_bytes())==digest,name
    dump('validation.json',dict(status='PASS_ANALYSIS',repeats=50,V_records=6000000,E_records=6000000,
        W0_record_alpha_decisions_read=24000000,budget_labels=3600,E_labels=600,summary_rows=len(rows),
        missing=0,invalid=0,exclusions=0,archives=arc.archives,checked_members=arc.checked,
        prefix_direct_and_segment_checks=True,original_full_n_counts_and_labels_match=True,
        calibration_hashes_and_role_ids_verified=True,raw_to_detector_recomputation=False,
        new_time_series=0,detector_changes=0,manuscript_edits=0,elapsed_seconds=time.monotonic()-started,
        minimum_zero_alarm_support_n={str(a):math.ceil(math.log(ETA)/math.log1p(-1.25*a)) for a in ALPHAS}))
    print('PASS_ANALYSIS',len(rows),'resource rows;',len(arc.checked),'members checked',flush=True)

if __name__=='__main__': main()
