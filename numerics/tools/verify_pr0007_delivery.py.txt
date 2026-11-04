# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Read-only artifact/plan checks; writes a new verification receipt, never science."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import zlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
import pr0007_spec as s
from probe_check_io import digest,read,write


def verify(destination):
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Preserve earlier verification')
    freeze=read(ROOT/'audits/verification/2026-10-18_PR-0007_design_freeze.json')
    anchors=dict(freeze['inherited_sources']);anchors.update({k:v for k,v in freeze['handoff']['anchors'].items() if k!='audits/fp_corpus.md'})
    anchors[freeze['card']['path']]=s.CARD_SHA
    anchors['audits/prereg/PR-0007_A01_five_replicates_per_zip.md']=s.A01_SHA
    anchors[freeze['handoff']['path']]=freeze['handoff']['sha256']
    for p,h in anchors.items():
        if digest(ROOT/p)!=h:raise AssertionError('Protected anchor changed: '+p)
    for p,n,h in [('audits/ledger_2026-10.md',177353,'8e0c6fccb1b4c7b79e3e3f5796711ec602b9fd5784e2d122421051787fdd36f1'),
                  ('audits/fp_corpus.md',15234,'29502e364442b227c180130578736f2656f6ad6d9e2d820583ce40836293c6f7')]:
        with (ROOT/p).open('rb') as f:actual=hashlib.sha256(f.read(n)).hexdigest()
        if actual!=h:raise AssertionError('Historical prefix changed: '+p)
    build=read(ROOT/'numerics/colab/pr0007_lfdc_calibration_a25_build.json')
    for p,h in {**build['sources'],**build['artifacts']}.items():
        if digest(ROOT/p)!=h:raise AssertionError('Build drift: '+p)
    py=ROOT/'numerics/colab/pr0007_lfdc_calibration_a25_one_cell.py';values={}
    for node in ast.parse(py.read_text()).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('_CELL_ID','_PAYLOAD','_PAYLOAD_SHA256'):
            values[node.targets[0].id]=ast.literal_eval(node.value)
    raw=zlib.decompress(base64.b64decode(values['_PAYLOAD']))
    if hashlib.sha256(raw).hexdigest()!=build['payload_sha256']:raise AssertionError('Payload hash differs')
    if values['_CELL_ID']!=build['cell_id']:raise AssertionError('Cell identity differs')
    nb=read(py.with_suffix('.ipynb'))
    if len(nb['cells'])!=1 or ''.join(nb['cells'][0]['source'])!=py.read_text():raise AssertionError('One-cell notebook differs')
    source=json.loads(raw)['sources']
    for name,body in source.items():
        if name.endswith('.py'):ast.parse(body,filename=name)
    if (len(s.conditions()),len(s.calibrations()),len({c['fit_key'] for c in s.calibrations()}),len(s.contexts()),len(s.contrasts()))!=(24,66,54,264,312):
        raise AssertionError('Frozen plan cardinality differs')
    if sum(t['count'] for t in s.tasks(0))*25!=10909800:raise AssertionError('Record budget differs')
    tests=read(ROOT/'output/runs/pr0007_implementation_tests_v3/results.json')
    if not tests['passed'] or tests['tests']!=40 or tests['scientific_records_generated']!=0:raise AssertionError('Final technical tests not passed')
    docs=['numerics/PR0007_LFDC_CALIBRATION_COLAB_JA.md','audits/verification/2026-10-18_PR-0007_implementation.md']
    links=0
    for name in docs:
        path=ROOT/name
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in link or link.startswith('#'):continue
            target=(path.parent/link.split('#')[0]).resolve()
            if not target.exists():raise AssertionError('Broken artifact link: '+link)
            links+=1
    outputs=list(build['artifacts'])+[
        'numerics/colab/pr0007_lfdc_calibration_a25_build.json','numerics/tools/build_pr0007_colab.py',
        'numerics/tools/test_pr0007.py','numerics/tools/verify_pr0007_delivery.py',
        'numerics/PR0007_LFDC_CALIBRATION_COLAB_JA.md','numerics/PR0007_THIRD_PARTY_NOTICE.txt',
        'audits/verification/2026-10-18_PR-0007_implementation.md',
        'output/runs/pr0007_implementation_tests_v1/results.json','output/runs/pr0007_implementation_tests_v1/test.log',
        'output/runs/pr0007_implementation_tests_v2/results.json','output/runs/pr0007_implementation_tests_v3/results.json',
        'output/runs/pr0007_implementation_tests_v3/test.log']
    report=dict(status='PASS_IMPLEMENTATION_LOCAL_TECHNICAL',date='2026-10-18',date_basis='user-provided Asia/Tokyo date',
        cell_id=build['cell_id'],tests=tests,protected_anchors=anchors,links_checked=links,
        outputs={p:digest(ROOT/p) for p in outputs},source_hashes=build['sources'],
        ledger_prefix=dict(bytes=177353,sha256='8e0c6fccb1b4c7b79e3e3f5796711ec602b9fd5784e2d122421051787fdd36f1'),
        corpus_prefix=dict(bytes=15234,sha256='29502e364442b227c180130578736f2656f6ad6d9e2d820583ce40836293c6f7'),
        current_corpus_sha256=digest(ROOT/'audits/fp_corpus.md'),
        verification_type='Same-agent self-audit; deterministic fixture and mock tests, not independent scientific replication',
        scientific_records_generated=0,production_run=False,manuscript_edited=False,
        pending=['actual Colab/GPU preflight','actual 389MB-class shard packing/transfer','Mac browser memory and permissions',
                 'actual lost-runtime restoration','scientific results and their receipt audit'])
    write(destination,report);print(json.dumps(dict(status=report['status'],tests=tests['tests'],cell_id=build['cell_id'],links=links)))


if __name__=='__main__':verify(sys.argv[1])
