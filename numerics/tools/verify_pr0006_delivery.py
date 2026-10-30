# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Technical delivery evidence; this script never invokes scientific main/RNG."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sys
import time
import unittest
import uuid
import numpy as np
import scipy
import test_pr0006 as tests
ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    freeze=json.loads((ROOT/'audits/verification/2026-10-17_PR-0006_design_freeze.json').read_text())
    protected={p:sha(ROOT/p)==h for p,h in freeze['protected'].items()}
    ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes()
    ledger_ok=hashlib.sha256(ledger[:freeze['ledger_bytes']]).hexdigest()==freeze['ledger_sha256']
    suite=unittest.defaultTestLoader.loadTestsFromModule(tests);stream=io.StringIO();began=time.monotonic()
    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    log=ROOT/'audits/verification'/('2026-10-17_PR-0006_tests_'+uuid.uuid4().hex[:8]+'.log')
    log.write_text(stream.getvalue())
    outputs=list(ROOT.glob('numerics/scripts/pr0006_*.py'))+list(ROOT.glob('numerics/tools/*pr0006*.py'))
    outputs+=list(ROOT.glob('numerics/colab/pr0006_*'))+[ROOT/freeze['card'],ROOT/'numerics/PR0006_LFDC_B_COLAB_JA.md']
    for p in outputs:
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
    docs=[ROOT/'audits/README.md',ROOT/'audits/prereg/README.md',ROOT/'HANDOFF/LATEST.md',
          ROOT/freeze['card'],ROOT/'numerics/PR0006_LFDC_B_COLAB_JA.md']
    local_links=[];missing=[]
    for p in docs:
        for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',p.read_text()):
            if '://' in target or target.startswith('#'):continue
            target=target.split('#',1)[0]
            if not target:continue
            local_links.append((str(p.relative_to(ROOT)),target))
            if not (p.parent/target).exists():missing.append(local_links[-1])
    passed=result.wasSuccessful() and all(protected.values()) and ledger_ok and not missing
    report=dict(status='PASS_IMPLEMENTATION_LOCAL' if passed else 'FAIL',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.monotonic()-began,
        technical_log=str(log.relative_to(ROOT)),technical_log_sha256=sha(log),
        scientific_records_generated=0,scientific_results_evaluated=False,gpu_executed=False,
        mocked_worker_statistic_fixture_records=448,comment='Fixture records are artificial columns, not PR-0006 physical trials.',
        protected_count=len(protected),protected=protected,old_ledger_prefix_preserved=ledger_ok,
        local_links_checked=len(local_links),missing_links=missing,
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__),
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in outputs},
        limitations=['Same-agent technical audit, not independent scientific confirmation.',
                     'Local test NumPy/SciPy differ from pinned Colab worker; runtime float64 preflight is mandatory.',
                     'Colab installation, actual GPU execution, abrupt VM restore, browser downloads not executed here.',
                     'ZIP gates tested with reduced limits; actual result size and throughput remain to be measured.'])
    out=ROOT/'audits/verification/2026-10-17_PR-0006_implementation.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('protected','artifacts','environment')},ensure_ascii=False,indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
