# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Local small checks and packaging audit only; never calls scientific main."""
import ast
import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import unittest
from urllib.parse import unquote
import zlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
import pr0004_io as disk
import pr0004_spec as spec


def verify(out):
    out=Path(out)
    if out.exists(): raise FileExistsError('Choose a new verification output directory')
    out.mkdir(parents=True)
    disk.write(out/'delivery_validation.json',dict(status='IN_PROGRESS'))
    freeze=disk.read(ROOT/'audits/verification/2026-10-17_PR-0004_design_freeze.json')
    assert disk.digest(ROOT/freeze['card'])==freeze['card_sha256']
    protected=0
    for line in (ROOT/'audits/verification/2026-10-17_setup_baseline.sha256').read_text().splitlines():
        match=re.match(r'([0-9a-f]{64})\s+\*?(.+)',line)
        if match:
            assert disk.digest(ROOT/match[2])==match[1],match[2];protected+=1
    anchors=re.findall(r'\| `([^`]+)` \| `([0-9a-f]{64})` \|',(ROOT/'HANDOFF/2026-10-17_session-01.md').read_text())
    for p,h in anchors: assert disk.digest(ROOT/p)==h,p
    old=disk.read(ROOT/'numerics/colab/probe_cpu_gpu_check_build.json')
    for p,h in (old['sources']|old['artifacts']).items(): assert disk.digest(ROOT/p)==h,p
    old_extra={
      'numerics/scripts/fusion_confirmation_pr0002_v1.py':'0cd62bcd077905cb14a9c22fc9b2391b2ee5b193d2c845886b30a944528a0706',
      'audits/prereg/PR-0002_fusion_independent_confirmation.md':'ac2725d797cb6cdfe85bc0a3f389fa3dbf2bd8a9cd303acafe95a63a381e3e07',
      'numerics/theory/THEORY_NOTE_M_CR_QUADRATIC_FUSION_v1_JA.md':'4264a4ba445a90926e9d78d578d4214bbe0415a2f292c7e7dcad2784f9391e40',
      'numerics/design/LINEAR_FUSION_SOURCE_SHIFT_DESIGN_v1_JA.md':'c8c88434f09b4dd225b216cad380f5d63d96eb41fa9a7c441af88e9ac0091148',
      'probe_cpu_gpu_check_v1_23538c370862_PILOT_COMPLETE_b613933e.zip':'7f652e2356ab79c626d1a19ec012b7729da4a29dbc912aa5603d1b8ebcbd80f9'}
    for p,h in old_extra.items():assert disk.digest(ROOT/p)==h,p
    prefix=(ROOT/'audits/ledger_2026-10.md').read_bytes()[:freeze['previous_ledger_bytes']]
    assert disk.sha(prefix)==freeze['previous_ledger_sha256']
    manifest=disk.read(ROOT/'numerics/colab/pr0004_gpu_main_build.json')
    for p,h in (manifest['sources']|manifest['artifacts']).items(): assert disk.digest(ROOT/p)==h,p
    py=(ROOT/'numerics/colab/pr0004_gpu_main_one_cell.py').read_text()
    nb=disk.read(ROOT/'numerics/colab/pr0004_gpu_main_one_cell.ipynb')
    assert len(nb['cells'])==1 and ''.join(nb['cells'][0]['source'])==py
    assert nb['cells'][0]['outputs']==[] and nb['cells'][0]['execution_count'] is None
    assignments={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(py).body
                 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                 and n.targets[0].id in ('_PAYLOAD','_PAYLOAD_SHA256','_CELL_ID')}
    raw=zlib.decompress(base64.b64decode(assignments['_PAYLOAD']))
    assert disk.sha(raw)==assignments['_PAYLOAD_SHA256']==manifest['payload_sha256']
    sources=json.loads(raw)['sources']
    sys.path.insert(0,str(ROOT/'numerics/tools'))
    import build_pr0004_colab as builder
    for name,path in builder.PATHS.items(): assert sources[name]==(ROOT/path).read_text(),name
    tasks=spec.tasks(); assert sum(t['count'] for t in tasks)==33499400
    assert len(tasks)==3700 and sum((t['count']+4095)//4096 for t in tasks)==10400
    # Recompute the timing proxy independently from the actual pilot medians.
    timing=disk.read(ROOT/'numerics/analysis/pr0003_cpu_gpu_23538c370862_v1/audit_results.json')['timing']
    t={(v['case'],v['batch']):v['median_seconds']['gpu']['end_to_end_seconds'] for v in timing.values()}
    seconds=0.
    for task in tasks:
        c=spec.conditions()[task['case']]; p=c['params']; g=c['group']
        proxy=t['B_S',512] if g=='B_S' else t['B_R10',4096] if g=='B_R10' else t['B_L' if p['T']>3 else 'B_M',4096]
        if p['fault']=='private':proxy*=t['PRIVATE_M',4096]/t['B_M',4096]
        if g in ('K2_L','K200_L') and p['fault']=='none':proxy=t[g,4096]
        seconds+=task['count']/4096*proxy
    assert abs(seconds-freeze['generation_proxy_seconds'])<1e-6
    log=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'numerics/tests'),pattern='test_pr0004.py')
    with redirect_stdout(log):result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    disk.atomic(out/'unit_tests.txt',log.getvalue().encode())
    testinfo=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful())
    disk.write(out/'unit_tests.json',testinfo)
    assert result.wasSuccessful(),log.getvalue()
    docs=['audits/prereg/PR-0004_linear_fusion_source_shift_gpu.md','numerics/PR0004_GPU_MAIN_COLAB_JA.md',
          'audits/verification/2026-10-17_PR-0004_implementation.md','audits/README.md','audits/prereg/README.md',
          'HANDOFF/LATEST.md','audits/ledger_2026-10.md']
    links=0
    for name in docs:
        p=ROOT/name
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            link=unquote(link.strip('<>').split('#')[0])
            if not link or '://' in link or link.startswith('mailto:'):continue
            assert (p.parent/link).exists(),(name,link);links+=1
    import numpy,scipy
    report=dict(status='PASS',card_sha256=freeze['card_sha256'],cell_id=manifest['cell_id'],
      protected_assets=protected,handoff_anchors=len(anchors),pilot_frozen_files=len(old['sources']|old['artifacts']),
      old_extra_hashes=len(old_extra),previous_ledger_prefix_bytes=freeze['previous_ledger_bytes'],
      unit_tests=testinfo,local_python=sys.version,local_numpy=numpy.__version__,local_scipy=scipy.__version__,
      main_simulation_records_generated=0,gpu_executed=False,colab_executed=False,
      generation_proxy_seconds=seconds,expected_records=33499400,expected_tasks=3700,expected_chunks=10400,
      source_and_artifact_hashes=manifest['sources']|manifest['artifacts'],local_links=links,
      verification_sources={p:disk.digest(ROOT/p) for p in ('numerics/tests/test_pr0004.py',
          'numerics/tools/build_pr0004_colab.py','numerics/tools/verify_pr0004_delivery.py')},
      remaining=['GPU execution of PR-0004','Full 50-replicate throughput and storage',
                 'Actual multi-GiB browser download','Real Colab interruption/recovery',
                 'Independent human/third-party review'])
    disk.write(out/'delivery_validation.json',report)
    print(json.dumps(dict(status='PASS',tests=result.testsRun,cell_id=manifest['cell_id'],links=links),ensure_ascii=False))
    return report


if __name__=='__main__':verify(sys.argv[1])
