# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Read-only asset audit plus one new verification JSON. No simulation."""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/runs/pr0003_implementation_validation'


@lru_cache(None)
def digest(relative):
    with (ROOT/relative).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    checks = []
    def check(name, ok, **extra):
        checks.append(dict(name=name,passed=bool(ok),**extra))
    baseline = ROOT/'audits/verification/2026-10-17_setup_baseline.sha256'
    protected = [line.split('  ',1) for line in baseline.read_text().splitlines() if line.strip()]
    for expected,path in protected:
        check('protected:'+path,digest(path)==expected)
    handoff = (ROOT/'HANDOFF/2026-10-17_session-01.md').read_text()
    anchors = re.findall(r'\| `([^`]+)` \| `([a-f0-9]{64})` \|',handoff)
    for path,expected in anchors:
        check('handoff:'+path,digest(path)==expected)
    extra_anchors = {
        'loop_kv_nb06_onecell.py':'3ec06ef2c5b7394912f17dceaf8aea5e39e09759ac4f69e815f0e3354f3d1b78',
        'numerics/design/LINEAR_FUSION_SOURCE_SHIFT_DESIGN_v1_JA.md':'c8c88434f09b4dd225b216cad380f5d63d96eb41fa9a7c441af88e9ac0091148',
        'numerics/theory/THEORY_NOTE_M_CR_QUADRATIC_FUSION_v1_JA.md':'4264a4ba445a90926e9d78d578d4214bbe0415a2f292c7e7dcad2784f9391e40',
        'audits/prereg/PR-0002_fusion_independent_confirmation.md':'ac2725d797cb6cdfe85bc0a3f389fa3dbf2bd8a9cd303acafe95a63a381e3e07',
        'numerics/scripts/fusion_confirmation_pr0002_v1.py':'0cd62bcd077905cb14a9c22fc9b2391b2ee5b193d2c845886b30a944528a0706',
        'fusion_confirmation_pr0002_v1_confirm_9bb38b60e748_COMPLETE_20261017T063729_7cb5a0.zip':'1ef8e7c24060b8811594fdf4d83c0c725b8ecd9d70cb6157b99417abdd60b2b3',
    }
    for path,expected in extra_anchors.items():
        check('extra:'+path,digest(path)==expected)
    ledger = (ROOT/'audits/ledger_2026-10.md').read_bytes()
    check('append_only_ledger',hashlib.sha256(ledger[:57571]).hexdigest()==
          'b0a7be1cbabd170c8b97d4055c8d760d4f98bdd27bde0cbdd0e10a34143d68bc')
    manifest = json.loads((ROOT/'numerics/colab/probe_cpu_gpu_check_build.json').read_text())
    for path,expected in {**manifest['sources'],**manifest['artifacts']}.items():
        check('bundle:'+path,digest(path)==expected)
    tests = json.loads((OUT/'unit_tests.json').read_text())
    check('unit_tests_pass',tests['passed'] and tests['tests']==31 and tests['failures']==0 and tests['errors']==0)
    check('gpu_not_executed_locally',tests['gpu_executed'] is False and tests['scientific_performance_run'] is False)
    for path,expected in tests['tested_sources'].items():
        check('tested_source:'+path,digest(path)==expected)
        ast.parse((ROOT/path).read_text(),filename=path)
    documents = ['numerics/CPU_GPU_CHECK_COLAB_JA.md','audits/prereg/PR-0003_cpu_gpu_backend_pilot.md',
                 'audits/verification/2026-10-17_cpu_gpu_pilot_implementation.md',
                 'audits/README.md','audits/prereg/README.md','HANDOFF/LATEST.md','audits/ledger_2026-10.md']
    links = []
    output = OUT/'delivery_validation.json'
    for path in documents:
        for ref in re.findall(r'\[[^\]]*\]\(([^)\n]+)\)',(ROOT/path).read_text()):
            ref = ref.split('#',1)[0].strip('<>')
            if not ref or re.match(r'^\w+://',ref):
                continue
            target = ((ROOT/path).parent/ref).resolve()
            ok = target.exists() or target==output.resolve()
            links.append(dict(document=path,target=ref,exists=ok))
    check('local_links',all(x['exists'] for x in links),count=len(links),missing=[x for x in links if not x['exists']])
    check('asset_counts',len(protected)==124 and len(anchors)==8)
    result = dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',
                  checks=checks,protected_assets=len(protected),handoff_anchors=len(anchors),
                  cell_id=manifest['cell_id'],unit_tests=tests['tests'],local_links=len(links),
                  scientific_run=False,gpu_runtime_verified=False,
                  new_artifacts={p:digest(p) for p in manifest['artifacts']},
                  note='Documentation, small local unit tests and preservation only; Colab GPU execution pending.')
    OUT.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','new_artifacts')},ensure_ascii=False))
    if result['status']!='PASS':
        print(json.dumps([x for x in checks if not x['passed']],ensure_ascii=False,indent=2))
        raise SystemExit(1)


if __name__=='__main__':
    main()
