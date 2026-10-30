# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Check manually written report tables against recomputed JSON, plus provenance."""
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[2]


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def tables(text):
    groups=[];current=[]
    for line in text.splitlines()+['']:
        if line.startswith('|'):
            current.append([c.strip() for c in line.strip('|').split('|')])
        elif current:
            groups.append(current[2:]);current=[]
    return groups


def run(folder):
    folder=Path(folder);s=json.loads((folder/'recomputed_summary.json').read_text())
    r={(x['case'],x['policy'],x['method']):x for x in s['rates']}
    text=(folder/'PR0004_PERFORMANCE_REPORT_JA.md').read_text();t=tables(text);checked=0
    def number(x):return float(x.replace('−','-').replace(',',''))
    def equal(value,want):
        nonlocal checked
        assert abs(number(value)-100*want)<.000051,(value,want)
        checked+=1
    def bounds(value,row):
        v=re.findall(r'[+−-]?\d+\.\d+',value)
        assert len(v)==2,(value,v)
        equal(v[0],row['lower']);equal(v[1],row['upper'])
    cases=['B_L__GM05','B_L__GP05','B_M__P02']
    assert len(t)==10
    for row,case in zip(t[2],cases):
        for value,m in zip(row[1:],s['methods'][:5]):equal(value,r[case,'B',m]['mean'])
    for row,p in zip(t[3],s['primary12']):
        assert row[1]==p['right'];equal(row[2],p['mean']);bounds(row[3],p)
        assert ('損失' if p['classification']=='loss' else '改善') in row[4]
    for i,row in enumerate(t[4]):
        case=cases[i//4];method=s['methods'][i%4]
        p=next(x for x in s['paired'] if x['case']==case and x['policy']=='B' and x['right']==method)
        assert row[1]==method
        for value,k in zip(row[2:],['both','gained','lost','neither']):equal(value,p['cells'][k]['mean'])
    for row,policy in zip(t[5],['B','F','C','R']):
        a=[x for x in s['fpr140'] if x['policy']==policy]
        assert int(row[1])==len(a) and int(row[2].strip('*'))==sum(x['excess_flag'] for x in a)
        a_min,a_max=row[3].split('–');equal(a_min,min(x['mean'] for x in a));equal(a_max,max(x['mean'] for x in a))
    groups=['K2_M','K2_L','K200_M','K200_L','V05_M','V05_L','V15_M','V15_L']
    for row,g in zip(t[6],groups):
        for value,p in zip(row[1:],['F','C','R']):equal(value,r[g,p,'LF']['mean'])
    for row,case in zip(t[7],['K2_M__GP05','K200_L__P02','V05_L__GP05']):
        for value,p in zip(row[1:4],['F','C','R']):equal(value,r[case,p,'LF']['mean'])
        equal(row[4],r[case,'R','J']['mean'])
        d=next(x for x in s['policy_differences'] if x['case']==case and x['method']=='LF' and x['right']=='C')
        equal(row[5].split(' ')[0],d['mean']);bounds(row[5].split(' ',1)[1],d)
    aux=['B_L__GM02','B_L__GP02','B_M__P005','B_M__DM2','B_M__DP2','B_R10__GP05']
    for row,case in zip(t[8],aux):
        for value,m in zip(row[1:],s['methods'][:5]):equal(value,r[case,'B',m]['mean'])
    for row,case in zip(t[9],cases):
        equal(row[1],r[case,'B','LF']['mean']);equal(row[2],r[case,'B','LF-D']['mean'])
        d=next(x for x in s['paired'] if x['case']==case and x['policy']=='B' and x['right']=='LF-D')
        equal(row[3].split(' ')[0],d['mean']);bounds(row[3].split(' ',1)[1],d)
    appendix=tables((folder/'ALL_RESULTS.md').read_text())
    assert [len(x) for x in appendix]==[114,12,140,570,384,12,28]
    m=json.loads((ROOT/'numerics/colab/pr0004_gpu_main_build.json').read_text())
    for p,h in (m['sources']|m['artifacts']).items():assert digest(ROOT/p)==h,p
    validation=json.loads((folder/'validation.json').read_text())
    assert validation['status']=='PASS'
    assert digest(ROOT/'numerics/scripts/analyze_pr0004_performance.py')==validation['analysis_source_sha256']
    ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes()
    assert hashlib.sha256(ledger[:89026]).hexdigest()=='5faca803d89a0ead50594a6fb72936e75fd204b935afc96dfcfcb3b231014a84'
    corpus=(ROOT/'audits/fp_corpus.md').read_bytes()
    assert hashlib.sha256(corpus[:6018]).hexdigest()=='52eeb8729becbc64c23b363a9befac6ed9b1f14b6430dee49039ea85d40ab88a'
    anchors=re.findall(r'\| `([^`]+)` \| `([0-9a-f]{64})` \|',(ROOT/'HANDOFF/2026-10-17_session-01.md').read_text())
    for p,h in anchors:assert digest(ROOT/p)==h,p
    link_count=0
    for p in [folder/'PR0004_PERFORMANCE_REPORT_JA.md',ROOT/'audits/README.md',ROOT/'audits/prereg/README.md',ROOT/'HANDOFF/LATEST.md',ROOT/'audits/ledger_2026-10.md',ROOT/'audits/fp_corpus.md']:
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            link=unquote(link.strip('<>').split('#')[0])
            if not link or link.startswith(('http:','https:','mailto:')):continue
            assert (p.parent/link).exists(),(p,link);link_count+=1
    record=dict(status='PASS',main_report_numeric_values_checked=checked,appendix_table_rows=[len(x) for x in appendix],
                frozen_hashes=15,handoff_anchors=len(anchors),local_links=link_count,
                old_ledger_and_corpus_prefixes_unchanged=True,
                figures='Two PNGs visually inspected by the same assistant; labels readable, no clipped data.',
                files={p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='delivery_validation.json'},
                sources={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'numerics/scripts/analyze_pr0004_performance.py',ROOT/'numerics/tools/render_pr0004_performance.py',Path(__file__)]})
    (folder/'delivery_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['files','sources']},ensure_ascii=False,indent=2))


if __name__=='__main__':run(sys.argv[1])
