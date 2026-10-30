# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Four-rule fusion analysis of saved paired scores, never probe simulation.

All thresholds are fitted without alternative or held-out normal records.
CSV/JSON/NPZ intermediates and figures are written only to a new analysis folder.
The Japanese report is edited separately, not generated or overwritten here.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

import analyze_covariance_roc as prior

VERSION = 'fusion-postanalysis-1.0.0'
SEED = 2026101703
BOOTSTRAPS = 1000
METHODS = ('moment', 'CR', 'bonferroni_or', 'joint_or')
PAIRS = ((2, 0), (2, 1), (3, 0), (3, 1), (3, 2))
ROOT = prior.ROOT
DEFAULT_OUT = prior.DEFAULT_OUT / 'fusion_preliminary_v1'
BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.int64)


def rng_for(label):
    words = np.frombuffer(hashlib.sha256(label.encode()).digest(), dtype='<u4')
    return np.random.default_rng(np.random.SeedSequence([SEED, *map(int, words)]))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def protected_hashes(archive):
    paths = set(ROOT.glob('*.md'))
    paths.update((ROOT / 'latex').rglob('*.tex'))
    paths.update((ROOT / 'output').rglob('*.pdf'))
    paths.update(ROOT.glob('*.pdf'))
    paths.update((prior.DEFAULT_OUT).rglob('*'))
    paths = {p for p in paths if p.is_file() and DEFAULT_OUT not in p.parents}
    paths.update([Path(archive), Path(prior.__file__),
                  ROOT / 'numerics/scripts/covariance_consistency_colab_v1.py'])
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def cutoff_order(n, alpha):
    """One-based split-conformal rank; exact rational arithmetic."""
    x = (n + 1) * (1 - Fraction(str(alpha)))
    return (x.numerator + x.denominator - 1) // x.denominator


def cutoff(scores, alpha):
    k = cutoff_order(len(scores), alpha)
    if k > len(scores):
        return math.inf
    return float(np.partition(scores, k - 1)[k - 1])


def reference_ranks(reference, scores):
    ordered = np.sort(reference, axis=0)
    return np.column_stack([np.searchsorted(ordered[:, m], scores[:, m], side='left')
                            for m in range(2)])


def fit_rules(reference, calibration):
    """Only reference/calibration data enter this function.

    A rank L > q is exactly equivalent to raw score > reference_sorted[q].
    Inactive components use +infinity, represented as null in saved JSON.
    """
    nref = len(reference)
    ranks = reference_ranks(reference, calibration)
    rank_cuts = np.full((4, 2), nref, dtype=np.int64)
    for method, col in ((0, 0), (1, 1)):
        q = cutoff(ranks[:, col], .01)
        rank_cuts[method, col] = nref if math.isinf(q) else int(q)
    for col in range(2):
        q = cutoff(ranks[:, col], .005)
        rank_cuts[2, col] = nref if math.isinf(q) else int(q)
    q = cutoff(np.max(ranks, axis=1), .01)
    rank_cuts[3, :] = nref if math.isinf(q) else int(q)
    ordered = np.sort(reference, axis=0)
    raw_cuts = np.full((4, 2), math.inf)
    for m in range(4):
        for col in range(2):
            if rank_cuts[m, col] < nref:
                raw_cuts[m, col] = ordered[rank_cuts[m, col], col]
    via_rank = np.any(ranks[:, None, :] > rank_cuts[None, :, :], axis=2)
    assert np.array_equal(via_rank, decisions(calibration, raw_cuts))
    # Joint OR cannot exceed the uncorrected union of single-1% decisions.
    assert rank_cuts[3, 0] >= rank_cuts[0, 0]
    assert rank_cuts[3, 1] >= rank_cuts[1, 1]
    assert np.all(via_rank.mean(axis=0) <= .01 + 1e-12)
    return raw_cuts, rank_cuts, ranks


def decisions(scores, raw_cuts):
    return ((scores[:, None, 0] > raw_cuts[None, :, 0]) |
            (scores[:, None, 1] > raw_cuts[None, :, 1]))


def patterns(flags):
    codes = (flags * (1 << np.arange(4))).sum(axis=1)
    return np.bincount(codes, minlength=16)


def decomposition(flags, a, b):
    both = int(np.sum(flags[:, a] & flags[:, b]))
    gain = int(np.sum(flags[:, a] & ~flags[:, b]))
    loss = int(np.sum(~flags[:, a] & flags[:, b]))
    neither = int(np.sum(~flags[:, a] & ~flags[:, b]))
    assert both + gain + loss + neither == len(flags)
    assert gain - loss == int(flags[:, a].sum()) - int(flags[:, b].sum())
    return both, gain, loss, neither


def bootstrap_cutoffs(reference, calibration, label, replicates=BOOTSTRAPS):
    rng = rng_for('refit/' + label)
    result = np.empty((replicates, 4, 2))
    for b in range(replicates):
        # Same index for both scores within a record; independent roles.
        ra = reference[rng.integers(0, len(reference), len(reference))]
        ca = calibration[rng.integers(0, len(calibration), len(calibration))]
        result[b] = fit_rules(ra, ca)[0]
    return result


def bootstrap_evaluation(scores, raw_cuts, label):
    """Paired row bootstrap via the sufficient 16-category flag histogram.

    A multinomial draw is exactly the distribution of ordinary empirical row
    resampling for these four fixed binary decisions, conditional on each refit.
    """
    rng = rng_for('evaluate/' + label)
    n, reps = len(scores), len(raw_cuts)
    powers = np.empty((reps, 4))
    gains = np.empty((reps, len(PAIRS)))
    losses = np.empty_like(gains)
    for start in range(0, reps, 24):
        cuts = raw_cuts[start:start + 24]
        flags = ((scores[None, :, None, 0] > cuts[:, None, :, 0]) |
                 (scores[None, :, None, 1] > cuts[:, None, :, 1]))
        codes = (flags * (1 << np.arange(4))).sum(axis=2)
        hist = np.bincount((codes + 16 * np.arange(len(cuts))[:, None]).ravel(),
                           minlength=16 * len(cuts)).reshape(-1, 16)
        counts = rng.multinomial(n, hist / n)
        sl = slice(start, start + len(cuts))
        powers[sl] = counts @ BITS / n
        for j, (a, b) in enumerate(PAIRS):
            gains[sl, j] = counts @ ((BITS[:, a] == 1) & (BITS[:, b] == 0)) / n
            losses[sl, j] = counts @ ((BITS[:, a] == 0) & (BITS[:, b] == 1)) / n
    for j, (a, b) in enumerate(PAIRS):
        np.testing.assert_allclose(gains[:, j] - losses[:, j], powers[:, a] - powers[:, b], atol=1e-15)
    return powers, gains, losses


def summarize(scores, raw_cuts, boot_cuts, common):
    flags = decisions(scores, raw_cuts)
    assert not np.any(flags[:, 3] & ~(flags[:, 0] | flags[:, 1]))
    boot, bg, bl = bootstrap_evaluation(scores, boot_cuts, common['case'])
    intervals = np.quantile(boot, [.025, .975], axis=0)
    power_rows, pair_rows = [], []
    for m, method in enumerate(METHODS):
        hits = int(flags[:, m].sum())
        lo, hi = prior.wilson(hits, len(scores))
        power_rows.append(dict(**common, method=method, n=len(scores), alarms=hits,
            rate=hits / len(scores), conditional_wilson_low=lo, conditional_wilson_high=hi,
            pipeline_bootstrap_low=float(intervals[0, m]), pipeline_bootstrap_high=float(intervals[1, m])))
    for j, (a, b) in enumerate(PAIRS):
        both, gained, lost, neither = decomposition(flags, a, b)
        ci = np.quantile(boot[:, a] - boot[:, b], [.025, .975])
        gc = np.quantile(bg[:, j], [.025, .975]); lc = np.quantile(bl[:, j], [.025, .975])
        pair_rows.append(dict(**common, candidate=METHODS[a], baseline=METHODS[b], n=len(scores),
            both=both, gained=gained, lost=lost, neither=neither,
            gained_rate=gained / len(scores), lost_rate=lost / len(scores),
            net_rate=(gained - lost) / len(scores), net_bootstrap_low=float(ci[0]), net_bootstrap_high=float(ci[1]),
            gained_bootstrap_low=float(gc[0]), gained_bootstrap_high=float(gc[1]),
            lost_bootstrap_low=float(lc[0]), lost_bootstrap_high=float(lc[1])))
    hist = patterns(flags)
    pattern_rows = [dict(**common, pattern=code, moment=int(BITS[code, 0]), CR=int(BITS[code, 1]),
                   bonferroni_or=int(BITS[code, 2]), joint_or=int(BITS[code, 3]), count=int(hist[code]))
                   for code in range(16)]
    union = flags[:, 0] | flags[:, 1]
    accounting = dict(**common, n=len(scores), naive_union_count=int(union.sum()),
        naive_union_rate=float(union.mean()), single_overlap_count=int(np.sum(flags[:, 0] & flags[:, 1])),
        joint_retained_count=int(flags[:, 3].sum()),
        tightening_discard_count=int(np.sum(union & ~flags[:, 3])),
        outside_naive_union_count=int(np.sum(flags[:, 3] & ~union)))
    return power_rows, pair_rows, pattern_rows, accounting, boot


def common_info(case, normal_case, cohort):
    p = case['params']
    return dict(case=case['id'], calibration_null_case=normal_case, cohort=cohort,
        group=case['group'], fault=p.get('fault', 'none'), strength=p.get('strength', 0.),
        T=p['T'], tau_i=p['tau_i'], tau_j=p['tau_j'], dt=p['dt'], kappa=p['kappa'],
        eta_readout=p.get('eta_readout', 0.))


def analyze(archive, out, replicates=BOOTSTRAPS):
    start = time.monotonic()
    out = Path(out)
    protected = protected_hashes(archive)
    protocol = out / 'PROTOCOL.md'
    if not protocol.exists():
        raise ValueError('Place and review PROTOCOL.md in the output directory before analysis')
    arc = prior.Archive(archive)
    cases = [c for c in arc.design if c['experiment'] == 3]
    normals = {c['reference_id']: c for c in cases if c['kind'] == 'null'}
    alternatives = [c for c in cases if c['kind'] == 'alternative']
    shifts = [c for c in cases if c['kind'] == 'null_shift']
    assert (len(normals), len(alternatives), len(shifts)) == (9, 90, 2)
    (out / 'tables').mkdir(exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)
    cache, models, split_arrays, boot_arrays = {}, {}, {}, {}
    normal_rows, normal_pairs, pattern_rows, accounting_rows, cal_rows = [], [], [], [], []
    for i, (rid, c) in enumerate(normals.items()):
        ref = arc.scores(rid); score = arc.scores(c['id'])
        assert ref.shape == (9999, 2) and score.shape == (20000, 2)
        ix = prior.rng_for('holdout-split/' + c['id']).permutation(len(score))
        calix, holdix = ix[:10000], ix[10000:]
        assert not np.intersect1d(calix, holdix).size
        cal, hold = score[calix], score[holdix]
        cuts, ranks, calranks = fit_rules(ref, cal)
        print(f'Normal group {i+1}/9: fitting paired bootstrap calibrations ({replicates})', flush=True)
        boot_cuts = bootstrap_cutoffs(ref, cal, c['id'], replicates)
        model = dict(null_case=c['id'], reference_id=rid, n_reference=len(ref), n_calibration=len(cal),
            n_holdout=len(hold), rank_cutoffs=ranks.tolist(),
            raw_cutoffs=[[None if math.isinf(x) else float(x) for x in row] for row in cuts],
            calibration_indices_sha256=hashlib.sha256(calix.astype('<i8').tobytes()).hexdigest(),
            holdout_indices_sha256=hashlib.sha256(holdix.astype('<i8').tobytes()).hexdigest())
        models[rid] = model
        split_arrays[c['id'] + '_cal'] = calix; split_arrays[c['id'] + '_hold'] = holdix
        boot_arrays[c['id'] + '_cutoffs'] = boot_cuts
        cache[rid] = (c, cuts, boot_cuts)
        p, d, pats, acct, _ = summarize(hold, cuts, boot_cuts, common_info(c, c['id'], 'normal_holdout'))
        normal_rows.extend(p); normal_pairs.extend(d); pattern_rows.extend(pats); accounting_rows.append(acct)
        f = decisions(cal, cuts)
        for m, method in enumerate(METHODS):
            cal_rows.append(dict(null_case=c['id'], method=method, n_reference=len(ref), n_calibration=len(cal),
                calibration_alarms=int(f[:, m].sum()), calibration_rate=float(f[:, m].mean()),
                rank_cut_M=int(ranks[m, 0]), rank_cut_CR=int(ranks[m, 1]),
                raw_cut_M=model['raw_cutoffs'][m][0], raw_cut_CR=model['raw_cutoffs'][m][1],
                nominal_system_alpha=.01, normal_rank_pearson=float(np.corrcoef(calranks.T)[0, 1])))
    power_rows, pairs, shift_rows, shift_pairs = [], [], [], []
    baseline = next(v for v in cache.values() if v[0]['id'] == 'e3_null_T2.048')
    for i, c in enumerate(alternatives + shifts):
        cohort = 'source_shift' if c['kind'] == 'null_shift' else 'alternative'
        nc, cuts, boot_cuts = baseline if cohort == 'source_shift' else cache[c['reference_id']]
        scores = arc.scores(c['id'])
        assert len(scores) == (20000 if cohort == 'source_shift' else 5000)
        p, d, pats, acct, boot = summarize(scores, cuts, boot_cuts, common_info(c, nc['id'], cohort))
        (shift_rows if cohort == 'source_shift' else power_rows).extend(p)
        (shift_pairs if cohort == 'source_shift' else pairs).extend(d)
        pattern_rows.extend(pats); accounting_rows.append(acct)
        boot_arrays[c['id'] + '_power'] = boot
        if (i+1) % 10 == 0 or i+1 == len(alternatives+shifts):
            print(f'Evaluated saved-score conditions {i+1}/{len(alternatives+shifts)}', flush=True)
    outputs = dict(calibration=cal_rows, normal_false_alarm=normal_rows, power=power_rows,
        normal_tradeoffs=normal_pairs, detection_tradeoffs=pairs,
        source_shift=shift_rows, source_shift_tradeoffs=shift_pairs,
        detection_patterns=pattern_rows, union_accounting=accounting_rows)
    for name, rows in outputs.items():
        prior.dump_csv(out / 'tables' / (name + '.csv'), rows)
    prior.dump_json(out / 'models.json', models)
    np.savez_compressed(out / 'split_indices.npz', **split_arrays)
    np.savez_compressed(out / 'pipeline_bootstrap.npz', **boot_arrays)
    assert protected_hashes(archive) == protected, 'Protected input changed during analysis'
    audit = dict(version=VERSION, source_archive=Path(archive).name, archive_sha256=sha(archive),
        source_audit=arc.audit, original_config=arc.config, original_status=arc.status,
        normal_groups=9, alternative_conditions=90, shifted_normal_conditions=2,
        analyzed_reference_records=9*9999, analyzed_normal_records=9*20000,
        analyzed_alternative_records=90*5000, analyzed_shift_records=2*20000,
        split_seed=prior.SEED, bootstrap_seed=SEED, bootstrap_replicates=replicates,
        methods=list(METHODS), protocol_sha256=sha(protocol),
        script_sha256=sha(__file__), loader_sha256=sha(prior.__file__),
        python=platform.python_version(), numpy=np.__version__,
        output_rows={k:len(v) for k,v in outputs.items()},
        protected_hashes_before=protected, protected_hashes_after=protected_hashes(archive),
        source_shift_calibration='Frozen e3_null_T2.048 model; do not recalibrate at shifted rate',
        scope='Post-hoc saved-score analysis. No probe generation, simulation-source execution, or manuscript edits.',
        uncertainty='Pointwise pipeline percentile bootstrap, plus fixed-rule Wilson intervals. Not simultaneous.',
        elapsed_seconds=time.monotonic()-start)
    prior.dump_json(out / 'audit.json', audit)
    print(json.dumps({k:audit[k] for k in ('version','output_rows','elapsed_seconds')}, indent=2), flush=True)


def read_table(out, name):
    with (out / 'tables' / (name + '.csv')).open() as f:
        return list(csv.DictReader(f))


def plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#1768AC', '#D55E00', '#777777', '#16836B']
    labels = ['Moment', 'CR', 'Bonferroni OR', 'Joint-calibrated OR']
    rows = {(r['case'], r['method']):r for r in read_table(out, 'power')}
    trades = {(r['case'],r['candidate'],r['baseline']):r for r in read_table(out,'detection_tradeoffs')}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig, name):
        for ext in ('png','svg'):
            fig.savefig(out/'figures'/f'{name}.{ext}',dpi=175,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11.6,5),layout='constrained')
    for ax,gain in zip(axes,(-.05,.05)):
        for m,method in enumerate(METHODS):
            selected = [rows[(f'e3_gain_{gain:g}_T{T:g}',method)] for T in (.512,2.048,8.192)]
            x=np.arange(3)+(m-1.5)*.055
            y=np.array([float(r['rate'])*100 for r in selected])
            lo=np.array([float(r['pipeline_bootstrap_low'])*100 for r in selected])
            hi=np.array([float(r['pipeline_bootstrap_high'])*100 for r in selected])
            ax.plot(x,y,'o--' if m==2 else 'o-',color=colors[m],label=labels[m],lw=1.8,ms=4)
            ax.vlines(x,lo,hi,color=colors[m],lw=1.2)
        ax.set(xticks=range(3),xticklabels=['0.512','2.048','8.192'],ylim=(0,101),
               xlabel='Evaluation window (s)',ylabel='Detection rate (%)',title=f'Signal gain {gain:+.0%}')
        ax.grid(axis='y',alpha=.18)
    axes[0].legend(loc='upper left',fontsize=9)
    fig.suptitle('Four fixed rules at nominal system FPR 1%\nPointwise 95% pipeline bootstrap; achieved holdout FPRs reported separately',fontsize=12)
    save(fig,'fusion_gain_by_window')
    selected = [('e3_gain_-0.05_T0.512','Gain -5%; 0.512 s'),
                ('e3_gain_0.05_T0.512','Gain +5%; 0.512 s'),
                ('e3_gain_-0.05_T8.192','Gain -5%; 8.192 s'),
                ('e3_gain_0.05_T8.192','Gain +5%; 8.192 s'),
                ('e3_private_0.02_T2.048','Private input 0.02; 2.048 s'),
                ('e3_blind_time_reverse_T2.048','Time reversal; 2.048 s'),
                ('e3_delay_-2_T2.048','Delay -4 ms; 2.048 s'),
                ('e3_delay_2_T2.048','Delay +4 ms; 2.048 s')]
    fig,axes=plt.subplots(1,2,figsize=(13,6.4),sharey=True,layout='constrained')
    for ax,baseline in zip(axes,('moment','CR')):
        vals=[trades[(cid,'joint_or',baseline)] for cid,_ in selected]
        y=np.arange(len(vals))
        gain=np.array([float(r['gained_rate'])*100 for r in vals])
        loss=np.array([float(r['lost_rate'])*100 for r in vals])
        net=gain-loss
        ax.barh(y,gain,color='#16836B',height=.55,label='Gained detections')
        ax.barh(y,-loss,color='#A86A9B',height=.55,label='Lost detections')
        ax.scatter(net,y,color='#202020',marker='D',s=20,zorder=3,label='Net change')
        ax.axvline(0,color='#333333',lw=.8)
        ax.set(yticks=y,yticklabels=[lab for _,lab in selected],xlabel='Fraction of all alternative records (pp)',
               title='Joint OR versus '+('Moment' if baseline=='moment' else 'CR'),xlim=(-105,105))
        ax.set_xticks([-100,-50,0,50,100]); ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis()
    axes[1].legend(loc='lower right',fontsize=9)
    fig.suptitle('Which detections does fusion add or lose?\nPaired records, frozen calibration; uncertainty intervals are in the tables',fontsize=12)
    save(fig,'fusion_rescue_loss')
    prior.dump_json(out/'figure_environment.json',dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__))
    print('Rendered two figures as PNG and SVG.',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=prior.DEFAULT_ZIP)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    stage=parser.add_mutually_exclusive_group()
    stage.add_argument('--analyze-only',action='store_true')
    stage.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    if not args.plots_only: analyze(args.archive,args.out)
    if not args.analyze_only: plots(args.out)
