# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Separate delivery audit: saved counts, Markdown values, figures and preservation."""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
import pr0006_spec as spec

OUT=ROOT/'numerics/analysis/pr0006_6f23fc7946f8_performance_v2'


def digest(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def tables(text):
    lines=text.splitlines();out=[];i=0
    while i<len(lines)-1:
        if lines[i].startswith('|') and re.fullmatch(r'\|(?:\s*:?-+:?\s*\|)+',lines[i+1]):
            header=[v.strip() for v in lines[i].strip('|').split('|')];rows=[];i+=2
            while i<len(lines) and lines[i].startswith('|'):
                row=[v.strip() for v in lines[i].strip('|').split('|')]
                assert len(row)==len(header),(len(row),len(header),lines[i])
                rows.append(row);i+=1
            out.append((header,rows))
        else:i+=1
    return out


def percent(v):return f'{100*v:.4f}'
def interval(v):
    n=lambda x:'未定義' if x is None else percent(x)
    return f"{n(v['mean'])} [{n(v['lower'])}, {n(v['upper'])}]"


def main():
    checks={}
    data=json.loads((OUT/'recomputed_results.json').read_text())
    valid=json.loads((OUT/'validation.json').read_text())
    assert valid['status']=='PASS_SAVED_DATA_RECOMPUTATION'
    R={(r['case'],r['policy'],r['alpha'],r['method']):r for r in data['rates']}
    P={(r['case'],r['policy'],r['alpha'],r['left'],r['right']):r for r in data['paired']}
    with np.load(OUT/'recomputed_counts.npz',allow_pickle=False) as f:
        arrays={k:f[k] for k in f.files}
    counts=arrays['counts'];available=arrays['available'];pairs=arrays['pairs'];pairn=arrays['pair_n']
    assert counts.shape==(25,76,2,6,8)
    assert np.array_equal(counts.sum(axis=-1),available)
    assert np.array_equal(pairs.sum(axis=-1),pairn)
    assert not counts[...,3].any() and not counts[...,7].any(),'Incompatible simultaneous positive and negative tails'
    for k,(case,policy) in enumerate(spec.contexts()):
        kind=spec.conditions()[case]['kind'];n=spec.COUNTS['null' if kind=='normal' else kind]
        assert np.all(available[:,k]==n) and np.all(pairn[:,k]==n)
        for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
            for mi,m in enumerate(spec.METHODS):
                r=R[case,policy,1/den,m]
                assert r['available']==r['planned']==25*n
                assert r['pattern_counts']==counts[:,k,ai,mi].sum(axis=0).tolist()
                assert r['rate']['mean']==np.mean((n-counts[:,k,ai,mi,0])/n)
            for pi,(left,right) in enumerate(spec.PAIRS):
                assert np.array_equal(pairs[:,k,ai,pi,1]-pairs[:,k,ai,pi,2],counts[:,k,ai,right,0]-counts[:,k,ai,left,0])
    checks['exact_count_conservation_all_conditions']=True
    checks['unavailable_records']=0
    checks['method_alpha_decisions']=int(available.sum())
    assert checks['method_alpha_decisions']==195000000
    # Exhaustive generated tables, checked by content rather than only row count.
    rows=tables((OUT/'PRIMARY_RESULTS.md').read_text())[0][1]
    assert len(rows)==25
    for row,r in zip(rows,data['primary25']):
        assert row[:2]==[r['case'],r['policy']] and row[4]==interval(r['interval'])
        assert row[5:]==[r['classification'],r['magnitude']]
    blocks=tables((OUT/'ALL_RESULTS.md').read_text())
    assert [len(b[1]) for b in blocks]==[len(data['rates']),216,len(data['contrasts'])]
    for row,r in zip(blocks[0][1],data['rates']):
        assert row[:3]==[r['case'],r['policy'],r['kind']]
        assert row[4]==r['method'] and row[5]==interval(r['rate']) and row[6]==interval(r['miss'])
    for row,r in zip(blocks[1][1],data['fpr216']):
        assert row[4]==interval(r['interval']) and row[5]==str(r['excess_flag'])
    for row,r in zip(blocks[2][1],data['contrasts']):assert row[-1]==interval(r['interval'])
    rows=tables((OUT/'ALL_PAIRED_RESULTS.md').read_text())[0][1]
    assert len(rows)==2280
    for row,r in zip(rows,data['paired']):
        assert row[4:8]==[interval(r['cells'][k]) for k in ('both','gained','lost','neither')]
        assert row[8]==interval(r['difference'])
    rows=tables((OUT/'ALL_BRANCH_RESULTS.md').read_text())[0][1]
    assert len(rows)==912
    for row,r in zip(rows,data['rates']):
        assert row[4:7]==[interval(x) for x in r['branches']]
        assert row[7]=='/'.join(map(str,r['pattern_counts']))
    checks['all_table_rows']={'primary':25,'rates':912,'fpr':216,'contrasts':len(data['contrasts']),'pairs':2280,'branches':912}
    text=(OUT/'PR0006_BUDGET_REPORT_JA.md').read_text();blocks=tables(text)
    assert [len(b[1]) for b in blocks]==[6,6,6,6,3]
    gaincases=[('B_L__GM05','B'),('B_L__GP05','B'),('V05_L__GM05','C'),('V05_L__GP05','C'),('B_L__GM02','B'),('B_L__GP02','B')]
    for row,(c,p) in zip(blocks[1][1],gaincases):assert row[1:]==[percent(R[c,p,.01,m]['rate']['mean']) for m in spec.METHODS]
    paircases=[('K200_L__P02','C','WP'),('K200_L__P02','C','WG'),('K200_L__P02','C','LF0'),('B_L__GM05','B','WP'),('B_L__GP05','B','WG'),('V05_L__GM05','C','LF0')]
    for row,(c,p,m) in zip(blocks[2][1],paircases):
        r=P[c,p,.01,m,'W0'];assert row[2:4]==[percent(r['cells'][k]['mean']) for k in ('gained','lost')]
        assert abs(float(row[4].replace('−','-'))-100*r['difference']['mean'])<.00005
        assert row[5]==percent(r['cells']['neither']['mean'])
    for row,m in zip(blocks[3][1],spec.METHODS):
        fault=R['K2_M__P02','C',.001,m];null=R['K2_M','C',.001,m]
        assert row[1:4]==[percent(fault['rate']['mean']),percent(fault['miss']['mean']),percent(null['rate']['mean'])]
        inactive=null['lf0_structurally_inactive_replicates'] if m=='LF0' else 0
        assert row[4]==f'{inactive}/25'
    flags={p:sum(r['excess_flag'] for r in data['fpr216'] if r['policy']==p) for p in ('B','F','C')}
    assert flags=={'B':0,'F':48,'C':0}
    for row,p in zip(blocks[4][1],('B','F','C')):
        for j,alpha in enumerate((.01,.001),1):
            v=[r['interval']['mean'] for r in data['fpr216'] if r['policy']==p and r['alpha']==alpha]
            assert row[j]==percent(min(v))+'–'+percent(max(v))
        total=sum(r['policy']==p for r in data['fpr216']);assert row[3]==f'{flags[p]}/{total}'
    checks['main_report_tables']=True
    # Prespecified signs rather than a new total-win criterion.
    pri={(r['case'],r['policy'],r['left']):r for r in data['primary25']}
    h1=[('K200_L__P02','C','WP',1),('K200_L__P02','C','WG',-1)]+[(c,'B',m,sign) for c in ('B_L__GM05','B_L__GP05') for m,sign in [('WP',-1),('WG',1)]]
    h2=[(g+'__'+tag,p,m,sign) for g,p in [('B_L','B'),('V05_L','C')] for tag in ('GM05','GP05') for m,sign in [('W+',1 if tag=='GP05' else -1),('W−',-1 if tag=='GP05' else 1)]]
    for hypothesis in (h1,h2):
        for c,p,m,sign in hypothesis:
            r=pri[c,p,m]['interval'];assert r['lower']>0 if sign>0 else r['upper']<0
    checks['H1_parts_supported']=6;checks['H2_parts_supported']=8
    assert data['primary25'][-1]['classification']=='improved'
    checks['H3_direction_supported']=True;checks['H4_fpr_flags']=flags
    # Every local link in the delivered report/tables resolves.
    links=0
    for path in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' not in target and not target.startswith('#'):
                if target=='delivery_validation.json':continue
                assert (path.parent/target.split('#')[0]).exists(),(path,target)
                links+=1
    checks['links_resolved']=links
    figures=[]
    for p in sorted((OUT/'figures').glob('*.png')):
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:assert min(im.size)>700;size=im.size
        figures.append(dict(name=p.name,size=size,sha256=digest(p)))
    assert len(figures)==7
    checks['figure_files']=figures
    snapshot=json.loads((OUT/'start_snapshot.json').read_text())
    for p,h in snapshot['baseline'].items():assert digest(ROOT/p)==h,p
    freeze=json.loads((ROOT/'audits/verification/2026-10-17_PR-0006_design_freeze.json').read_text())
    for p,h in freeze['protected'].items():
        if p=='audits/fp_corpus.md':
            # Approved append-only QA-008 closes this work unit after the scientific audit.
            corpus=(ROOT/p).read_bytes()
            assert hashlib.sha256(corpus[:10473]).hexdigest()==h
            assert b'QA-008' in corpus[10473:]
        else:assert digest(ROOT/p)==h,p
    for r in json.loads((OUT/'archive_inventory.json').read_text()):assert digest(ROOT/r['path'])==r['sha256']
    ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes()
    assert hashlib.sha256(ledger[:snapshot['ledger_bytes']]).hexdigest()==snapshot['ledger_sha256']
    assert digest(ROOT/'numerics/scripts/analyze_pr0006_budget.py')==valid['base_source_sha256']
    assert digest(ROOT/'numerics/scripts/analyze_pr0006_budget_v2.py')==valid['wrapper_source_sha256']
    checks['protected_files_unchanged']=len(freeze['protected'])-1
    checks['protected_corpus_append_only']={'path':'audits/fp_corpus.md','prefix_bytes':10473}
    checks['original_archives_unchanged']=6
    checks['ledger_prefix_preserved']=snapshot['ledger_bytes']
    outputs={str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_validation.json'}
    sources={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'numerics/scripts/render_pr0006_budget.py']}
    result=dict(status='PASS_DELIVERY_SELF_AUDIT',checks=checks,output_sha256=outputs,source_sha256=sources,
                scope='same agent, separate count/table/preservation checks; not independent human review')
    (OUT/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(result['status'],checks['all_table_rows'])


if __name__=='__main__':main()
