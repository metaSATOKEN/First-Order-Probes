# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Versioned audit correction: only cross-library t-interval endpoint tolerance."""
import json
import math
import zipfile
import mpmath as mp
from scipy import stats
import analyze_pr0006_budget as base

ORIGINAL_SOURCE = base.__file__
base.OUT = base.ROOT/'numerics/analysis/pr0006_6f23fc7946f8_performance_v2'


def tree_close(a, b, path='aggregate'):
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a)^set(b))
        for k in a: tree_close(a[k], b[k], path+'/'+k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for x, y in zip(a, b): tree_close(x, y, path+'[]')
    elif isinstance(a, float):
        endpoint = path.startswith('aggregate/') and path.rsplit('/', 1)[-1] in ('lower', 'upper')
        assert math.isclose(a, b, abs_tol=1e-10 if endpoint else 2e-12, rel_tol=2e-12), (path, a, b)
        if endpoint:
            base.DIFFS['cross_library_interval_endpoint'] = max(base.DIFFS.get('cross_library_interval_endpoint', 0.), abs(a-b))
    else: assert a == b, (path, a, b)


def quantile_audit():
    mp.mp.dps = 60
    with zipfile.ZipFile(next(base.ROOT.glob('*pr0006*COMPLETE_INDEX*.zip'))) as z:
        agg = json.loads(z.read('analysis/aggregate.json'))
    audit = []
    for family, rows, key in [(1, agg['rates'], 'rate'), (25, agg['primary25'], 'interval'), (216, agg['fpr216'], 'interval')]:
        tail = mp.mpf('.05')/(2*family)
        root = mp.findroot(lambda t: mp.betainc(12, mp.mpf('.5'), 0, 24/(24+t*t), regularized=True)/2-tail,
                           float(stats.t.isf(float(tail), 24)))
        local = float(stats.t.ppf(1-.05/(2*family), 24))
        row = next(r[key] for r in rows if r[key]['mcse'] > 0)
        implied = (row['upper']-row['mean'])/row['mcse']
        assert abs(local-float(root)) < 1e-12 and abs(implied-float(root)) < 1e-9
        audit.append(dict(family=family, df=24, high_precision=str(root), local=local, archive_implied=implied))
    return audit


if __name__ == '__main__':
    base.tree_close = tree_close
    audit = quantile_audit()
    base.save('quantile_audit.json', audit)
    try:
        base.main()
        p = base.OUT/'validation.json'
        v = json.loads(p.read_text())
        v.update(wrapper_source_sha256=base.digest(__file__), base_source_sha256=base.digest(ORIGINAL_SOURCE),
                 interval_endpoint_atol=1e-10, quantile_audit=audit,
                 corrected_audit='v1 overstrict Student-t endpoint comparison; archived decisions unchanged')
        base.save('validation.json', v)
    except Exception as exc:
        base.save('validation_failure.json', dict(status='STOP', error=repr(exc), counts=dict(base.COUNTERS), max_absolute_differences=base.DIFFS))
        raise
