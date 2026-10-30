# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Validate saved ROC outputs against original scores; no simulation runs."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('roc',ROOT/'numerics/scripts/analyze_covariance_roc.py')
roc=importlib.util.module_from_spec(spec)
spec.loader.exec_module(roc)
out=roc.DEFAULT_OUT
tables={p.stem:list(csv.DictReader(p.open())) for p in (out/'tables').glob('*.csv')}
audit=json.loads((out/'audit.json').read_text())
assert hashlib.sha256(Path(roc.__file__).read_bytes()).hexdigest()==audit['analysis_script_sha256']
arc=roc.Archive(roc.DEFAULT_ZIP)
assert hashlib.file_digest(roc.DEFAULT_ZIP.open('rb'),'sha256').hexdigest()==audit['archive_sha256']
normal_ids={r['null_case'] for r in tables['normal_thresholds']}
normal={k:arc.scores(k) for k in normal_ids}
cases={r['case'] for r in tables['matched_fpr_power']}
checked=0
max_auc_error=0.
max_pauc_error=0.
for cid in sorted(cases):
    alt=arc.scores(cid)
    rows=[r for r in tables['matched_fpr_power'] if r['case']==cid]
    null=normal[rows[0]['null_case']]
    for r in rows:
        m=roc.METHODS.index(r['method']); cut=float(r['threshold'])
        assert np.mean(null[:,m]>cut)==float(r['target_fpr'])==float(r['empirical_fpr'])
        assert int(np.sum(alt[:,m]>cut))==int(r['detections'])
        assert np.mean(alt[:,m]>cut)==float(r['tpr'])
        checked+=1
    for m,method in enumerate(roc.METHODS):
        # Independent definition: pooled descending scores and cumulative class counts.
        scores=np.r_[null[:,m],alt[:,m]]
        labels=np.r_[np.zeros(len(null)),np.ones(len(alt))]
        assert len(np.unique(scores))==len(scores), 'Pooled tie handling required'
        ordered=labels[np.argsort(-scores)]
        fpr=np.r_[0,np.cumsum(1-ordered)/len(null)]
        tpr=np.r_[0,np.cumsum(ordered)/len(alt)]
        auc=float(np.trapezoid(tpr,fpr))
        take=fpr<=.05
        pauc=float(np.trapezoid(tpr[take],fpr[take])/.05)
        r=next(r for r in tables['auc'] if r['case']==cid and r['method']==method)
        max_auc_error=max(max_auc_error,abs(auc-float(r['auc'])))
        max_pauc_error=max(max_pauc_error,abs(pauc-float(r['normalized_pauc_0_to_005'])))
assert max_auc_error<1e-12 and max_pauc_error<1e-12
for r in tables['paired_power_difference']:
    rows={x['method']:x for x in tables['matched_fpr_power'] if x['case']==r['case'] and float(x['target_fpr'])==.01}
    assert abs(float(r['difference'])-(float(rows['moment']['tpr'])-float(rows['CR']['tpr'])))<1e-14
report=out/'ROC_AND_LONG_WINDOW_REPORT_JA.md'
for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
    if not target.startswith(('http:','https:')):
        # validation.json is written below after successful checks.
        if target!='validation.json': assert (report.parent/target).exists(),target
suite=unittest.defaultTestLoader.discover(str(ROOT/'numerics/tests'),pattern='test_roc_analysis.py')
log=io.StringIO()
result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
assert result.wasSuccessful(),log.getvalue()
summary=dict(status='PASS',data_scope='Stored scores only; no experiment regenerated',
    matched_fpr_rows_recomputed=checked,roc_auc_pairs_independently_integrated=len(tables['auc']),
    max_auc_difference=max_auc_error,max_normalized_pauc_difference=max_pauc_error,
    paired_differences_checked=len(tables['paired_power_difference']),
    deterministic_tests_passed=result.testsRun,report_links_checked=True,
    source_archive_sha256=audit['archive_sha256'])
(out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
