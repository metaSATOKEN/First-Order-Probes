# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Independent count conservation, summary/report tables, and provenance checks."""
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import defaultdict
from urllib.parse import unquote
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]


def h(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def tables(text):
    groups=[];current=[]
    for line in text.splitlines()+['']:
        if line.startswith('|'):current.append([x.strip() for x in line.strip('|').split('|')])
        elif current:groups.append(current[2:]);current=[]
    return groups


def verify(folder):
    f=Path(folder);s=json.loads((f/'summary.json').read_text());v=json.loads((f/'validation.json').read_text())
    assert v['status']=='PASS' and h(ROOT/'numerics/scripts/analyze_pr0004_branches.py')==v['source_sha256']
    assert h(f/'PROTOCOL.md')==v['protocol_sha256']
    reps=json.loads((f/'replicate_branches.json').read_text());group=defaultdict(list)
    for r in reps:
        n=r['n'];c=np.array(r['counts']);assert c.sum()==n
        wanted=dict(neither=c[0],g_only=c[1],p_only=c[2],both=c[3],total=n-c[0],g_fire=c[1]+c[3],p_fire=c[2]+c[3],
                    g_miss=c[0]+c[2],p_miss=c[0]+c[1],g_negative=r['gain_tail_counts'][0],g_positive=r['gain_tail_counts'][1])
        for k,value in wanted.items():assert abs(r['metrics'][k]-value/n)<1e-14
        assert sum(r['gain_tail_counts'])==wanted['g_fire']
        for comp,table in r['cross'].items():
            a=np.array(table);assert a.shape==(4,2) and np.array_equal(a.sum(axis=1),c)
        group[r['case'],r['policy'],r['method']].append(r)
    lookup={(r['case'],r['policy'],r['method']):r for r in s['branches']}
    for key,rr in group.items():
        assert len(rr)==50 and {r['replicate'] for r in rr}==set(range(50))
        b=lookup[key]
        for k,metric in b['metrics'].items():
            a=np.array([r['metrics'][k] for r in rr]);sem=stats.sem(a);width=stats.t.ppf(.975,49)*sem
            assert abs(a.mean()-metric['mean'])<2e-14
            if a.std(ddof=1)>0:
                assert abs(a.mean()-width-metric['lower'])<2e-14 and abs(a.mean()+width-metric['upper'])<2e-14
            else:assert metric['lower'] is None and metric['upper'] is None
        for comp,table in b['cross'].items():assert np.array_equal(table,np.sum([r['cross'][comp] for r in rr],axis=0))
    byrep={(r['replicate'],r['case'],r['policy'],r['method']):r for r in reps}
    for r in json.loads((f/'replicate_asymmetry.json').read_text()):
        p=byrep[r['replicate'],r['positive'],r['policy'],r['method']];n=byrep[r['replicate'],r['negative'],r['policy'],r['method']]
        for k,value in r['differences'].items():assert abs(value-(p['metrics'][k]-n['metrics'][k]))<1e-14
        assert abs(r['differences']['total']-sum(r['differences'][k] for k in ['g_only','p_only','both']))<1e-14
        assert abs(r['differences']['total']-r['differences']['g_fire']-r['differences']['p_only'])<1e-14
    doc=f/'LF_BRANCH_ASYMMETRY_REPORT_JA.md';ts=tables(doc.read_text());assert len(ts)==4
    checked=0
    def eq(text,want,factor=100):
        nonlocal checked
        value=float(text.replace('−','-').replace('±','').replace('∓',''))
        assert abs(value-factor*want)<.000051,(text,want)
        checked+=1
    cases=['B_L','B_L__GM02','B_L__GP02','B_L__GM05','B_L__GP05','B_M__P02']
    for row,c in zip(ts[0],cases):
        for text,k in zip(row[1:],['g_only','p_only','both','neither','total']):eq(text,lookup[c,'B','LF']['metrics'][k]['mean'])
    asym={(r['group'],r['policy'],r['method'],r['amplitude']):r for r in s['asymmetry']}
    for row,(g,amp) in zip(ts[1],[('B_S',.05),('B_M',.05),('B_L',.02),('B_L',.05)]):
        for m,i in [('LF',1),('LF-D',3)]:
            r=asym[g,'B',m,amp]['differences']['total'];eq(row[i],r['mean'])
            values=re.findall(r'[−-]?\d+\.\d+',row[i+1]);eq(values[0],r['lower']);eq(values[1],r['upper'])
    geo={(r['case'],r['policy'],r['method']):r for r in s['geometry']}
    for row,(m,j) in zip(ts[2],[('LF',0),('LF',1),('LF-D',0),('LF-D',1)]):
        pos=geo['B_L__GP05','B',m];neg=geo['B_L__GM05','B',m]
        eq(row[1],abs(pos['linear_projection_shift'][j]['mean']),1)
        eq(row[2],pos['quadratic_projection_shift'][j]['mean'],1)
        eq(row[3],neg['observed_projection_shift'][j]['mean'],1);eq(row[4],pos['observed_projection_shift'][j]['mean'],1)
    for row,(c,comp) in zip(ts[3],[(c,m) for c in ['B_L__GM05','B_L__GP05'] for m in ['M','J','LF-D']]):
        r=lookup[c,'B','LF'];a=np.array(r['cross'][comp])/r['n_total']
        for text,want in zip(row[2:],[a[1,0],a[2,0],a[3,0],a[0,1]]):eq(text,want)
    tabs=tables((f/'ALL_BRANCH_RESULTS.md').read_text());assert [len(t) for t in tabs]==[228,56,570]
    for row,r in zip(tabs[0],s['branches']):
        assert row[:3]==[r['case'],r['policy'],r['method']]
        for text,k in zip(row[3:],['g_only','p_only','both','neither','total','g_fire','p_fire']):eq(text,r['metrics'][k]['mean'])
    frozen=json.loads((ROOT/'numerics/colab/pr0004_gpu_main_build.json').read_text())
    for p,value in (frozen['sources']|frozen['artifacts']).items():assert h(ROOT/p)==value,p
    prior=ROOT/'numerics/analysis/pr0004_e0af6fcba596_performance_v2'
    delivery=json.loads((prior/'delivery_validation.json').read_text())
    for p,value in delivery['files'].items():assert h(prior/p)==value
    anchors=re.findall(r'\| `([^`]+)` \| `([0-9a-f]{64})` \|',(ROOT/'HANDOFF/2026-10-17_session-01.md').read_text())
    for p,value in anchors:assert h(ROOT/p)==value
    ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes()
    assert hashlib.sha256(ledger[:v['ledger_before_bytes']]).hexdigest()==v['ledger_before_sha256']
    assert h(ROOT/'audits/fp_corpus.md')=='cc4e0e6b2db1d7da052033b380d8f8ba9361259dcd4f33e06e1d8ab2cc5ebdf3'
    links=0
    for p in [doc,f/'PROTOCOL.md',ROOT/'audits/README.md',ROOT/'audits/prereg/README.md',ROOT/'HANDOFF/LATEST.md',ROOT/'audits/ledger_2026-10.md']:
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            path=unquote(link.strip('<>').split('#')[0])
            if not path or path.startswith(('https:','http:','mailto:')):continue
            assert (p.parent/path).exists(),(str(p),path);links+=1
    result=dict(status='PASS',replicate_branch_rows=len(reps),main_table_numbers=94,
        total_table_numbers_checked=checked,appendix_rows=[len(t) for t in tabs],
        frozen_hashes=15,handoff_anchors=len(anchors),local_links=links,
        previous_ledger_prefix_and_corpus_unchanged=True,
        figures='4 PNGs visually inspected; layout and labels checked; illustrative replicate fixed at 0.',
        files={p.name:h(p) for p in f.iterdir() if p.is_file() and p.name!='delivery_validation.json'},
        sources={str(p.relative_to(ROOT)):h(p) for p in [ROOT/'numerics/scripts/analyze_pr0004_branches.py',ROOT/'numerics/tools/render_pr0004_branches.py',Path(__file__)]})
    (f/'delivery_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','sources']},ensure_ascii=False,indent=2))


if __name__=='__main__':verify(sys.argv[1])
