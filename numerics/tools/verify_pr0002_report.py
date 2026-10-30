#!/usr/bin/env python3
# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Check PR-0002 report transcription and unchanged evidence; no simulations."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'numerics/scripts'))
from fusion_confirmation_pr0002_v1 import sha_file
from verify_pr0002 import protected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Choose a new delivery record, never overwrite'
    folder = ROOT/'numerics/analysis/pr0002_confirm_9bb38b60e748_v1'
    report = folder/'PR0002_CONFIRMATION_REPORT_JA.md'
    text = report.read_text()
    rows = [[c.strip() for c in line.strip('|').split('|')] for line in text.splitlines() if line.startswith('|')]
    results = json.loads((folder/'recomputed_results.json').read_text())
    tables = results['tables']
    checks = {}
    def norm(s): return s.replace('−','-')
    def equal(actual, expected): assert norm(actual)==norm(expected), (actual,expected)
    for p in tables['primary_14']:
        r = next(r for r in rows if r[0]==p['id'])
        equal(r[2],p['A']+'−'+p['D'])
        equal(r[4],f"{p['mean']*100:+.3f}")
        equal(r[5],f"[{p['lower']*100:+.3f}, {p['upper']*100:+.3f}]")
    checks['primary_mean_interval_rows'] = 14
    for c in ('N_S','N_M','N_L','N_R10'):
        r = next(r for r in rows if r[0].startswith(c+'：'))
        for j,m in enumerate(('M','CR','B','J')):
            p=next(p for p in tables['rates_summary'] if (p['condition'],p['method'])==(c,m))
            equal(r[j+1],f"{100*p['mean']:.4f} [{100*p['lower']:.4f}, {100*p['upper']:.4f}]")
    checks['fpr_mean_interval_cells'] = 16
    ids = list(dict.fromkeys(p['condition'] for p in tables['rates_summary'] if p['kind']!='normal'))
    for c in ids:
        r=next(r for r in rows if r[0]==c or r[0].startswith(c+'：'))
        for j,m in enumerate(('M','CR','B','J')):
            p=next(p for p in tables['rates_summary'] if (p['condition'],p['method'])==(c,m))
            equal(r[-4+j],f"{100*p['mean']:.4f}")
    checks['fault_and_structural_rate_cells'] = 48
    names={'G_L_MINUS':'長窓−5%','G_L_PLUS':'長窓+5%','PRIVATE_M':'私的入力'}
    for p in results['diagnostics']['paired_totals']:
        if p['condition'] not in names: continue
        r=next(r for r in rows if len(r)==5 and r[:2]==[names[p['condition']],p['A']+'−'+p['D']])
        equal(r[2],f"{p['gained']:,}"); equal(r[3],f"{p['lost']:,}")
        equal(r[4],f"{100*(p['gained']-p['lost'])/p['n']:+.4f}")
    checks['paired_count_and_difference_rows'] = 15
    handoff=(ROOT/'HANDOFF/2026-10-17_session-01.md').read_text()
    anchors=re.findall(r'\| `([^`]+)` \| `([0-9a-f]{64})` \|',handoff)
    assert len(anchors)==8
    for p,h in anchors: assert sha_file(ROOT/p)==h,p
    checks['handoff_anchors'] = 8
    checks['protected_assets'] = protected()
    build=json.loads((ROOT/'numerics/colab/pr0002_main_build_manifest.json').read_text())
    for p,h in build['artifacts'].items(): assert sha_file(ROOT/'numerics/colab'/p)==h,p
    prior=json.loads((ROOT/'output/runs/pr0002_implementation_validation_v2/validation.json').read_text())
    for p,h in prior['source_hashes'].items(): assert sha_file(ROOT/p)==h,p
    checks['frozen_implementation_hashes'] = len(prior['source_hashes'])
    checks['frozen_main_artifacts'] = len(build['artifacts'])
    ledger=ROOT/'audits/ledger_2026-10.md'
    assert hashlib.sha256(ledger.read_bytes()[:40859]).hexdigest()=='89ff967f8a3baf009aa4b063c9ad758e709a04669765f3590f96844eac085004'
    checks['ledger_prefix_preserved_bytes'] = 40859
    audit=json.loads((folder/'validation.json').read_text())
    assert sha_file(ROOT/audit['input_zip'])==audit['input_sha256']
    checks['input_zip_unchanged'] = True
    docs=[report,ROOT/'audits/README.md',ROOT/'audits/prereg/README.md',ROOT/'HANDOFF/LATEST.md',ledger]
    links=0
    for doc in docs:
        for link in re.findall(r'!?\[[^\]\n]*\]\(([^)\n]+)\)',doc.read_text()):
            if link.startswith(('http://','https://','#')):continue
            path=link.split('#')[0].strip('<>')
            assert (doc.parent/path).exists(),(str(doc),link)
            links+=1
    checks['resolved_local_links'] = links
    for p in (Path(__file__),ROOT/'numerics/scripts/analyze_pr0002_confirmation.py'):
        ast.parse(p.read_text())
    checks['analysis_and_verifier_syntax'] = True
    delivery=dict(status='PASS',scope='Report table transcription, links, evidence immutability; no new simulation or threshold adjustment',
                  checks=checks,source_hashes={str(p.relative_to(ROOT)):sha_file(p) for p in docs[:-1]+[Path(__file__),ROOT/'numerics/scripts/analyze_pr0002_confirmation.py',folder/'validation.json',folder/'recomputed_results.json']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(delivery,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
