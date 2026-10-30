# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Separate reconstruction from integer score histograms and report checks."""
from pathlib import Path
from urllib.parse import unquote
import hashlib
import json
import re
import sys
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]


def h(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify(out):
    out=Path(out);v=json.loads((out/'validation.json').read_text())
    assert v['status']=='PASS' and h(ROOT/'numerics/scripts/analyze_pr0004_lowfpr.py')==v['source_sha256']
    assert h(out/'PROTOCOL.md')==v['protocol_sha256']
    md=json.loads((out/'metadata.json').read_text());s=json.loads((out/'summary.json').read_text())
    with np.load(out/'replicate_metrics.npz',allow_pickle=False) as f:
        a={k:f[k] for k in f.files}
    keys=[tuple(k) for k in md['count_keys']];ki={k:i for i,k in enumerate(keys)}
    nk={tuple(k):i for i,k in enumerate(md['null_keys'])};fk={tuple(k):i for i,k in enumerate(md['fault_keys'])}
    points=0;integrals=0;orders=np.array(md['orders']);grid=np.array(md['grid']);anchor=np.array(md['anchor_indices'])
    for rep in range(50):
        with np.load(out/f'counts/rep_{rep:02d}.npz',allow_pickle=False) as f:hist=f['counts'].astype(np.int64)
        assert hist.shape==(114,5,10001) and np.all(hist>=0)
        # Forward cumulative complement, unlike the analysis reverse-tail implementation.
        n=hist.sum(axis=2)
        tails=n[:,:,None]-np.concatenate((np.zeros((114,5,1),np.int64),hist.cumsum(axis=2)),axis=2)
        for (group,p),ni in nk.items():
            ci=ki[group,p];assert np.all(n[ci]==20000)
            k=a['thresholds'][rep,ni].astype(int)
            got=np.take_along_axis(tails[ci],k,axis=1)
            assert np.array_equal(got/20000,a['actual_fpr'][rep,ni])
            assert np.all(got<=np.arange(401))
            assert np.all(np.take_along_axis(tails[ci],k-1,axis=1)>np.arange(401))
            assert np.array_equal(np.take_along_axis(tails[ci],k[:,anchor]-1,axis=1)/20000,a['next_fpr'][rep,ni])
            assert np.array_equal(np.take_along_axis(tails[ci],orders,axis=1)/20000,a['nominal_fpr'][rep,ni])
        for (case,p),fi in fk.items():
            ci=ki[case,p];ni=nk[md['conditions'][case]['group'],p];nc=ki[md['conditions'][case]['group'],p]
            k=a['thresholds'][rep,ni].astype(int);assert np.all(n[ci]==5000)
            got=np.take_along_axis(tails[ci],k,axis=1)/5000
            assert np.array_equal(got,a['power'][rep,fi])
            assert np.array_equal(np.take_along_axis(tails[ci],orders,axis=1)/5000,a['nominal_tpr'][rep,fi])
            points+=got.size
            # Exact integration from distinct normal rejection counts, no grid averaging.
            for j in range(5):
                u,first=np.unique(tails[nc,j,1:],return_index=True)
                use=u<400;xx=np.r_[u[use],400]/20000
                yy=tails[ci,j,first[use]+1]/5000
                area=np.sum(np.diff(xx)*yy)/.02
                assert abs(area-a['npauc'][rep,fi,j])<3e-15
                integrals+=1
            tab=a['overlap'][rep,fi]
            assert np.all(tab.sum(-1)==5000)
            for ai,g in enumerate(anchor):
                for j in range(4):
                    both,gain,loss,neither=map(int,tab[ai,j])
                    assert abs((both+gain)/5000-got[4,g])<1e-14
                    assert abs((both+loss)/5000-got[j,g])<1e-14
        if (rep+1)%10==0:print(f'Independent histogram check {rep+1}/50',flush=True)
    intervals=0
    def eq(metric,values):
        nonlocal intervals
        x=np.asarray(values);mu=float(x.mean());se=stats.sem(x)
        assert abs(metric['mean']-mu)<2e-14
        if se>0:
            w=stats.t.ppf(.975,49)*se
            assert abs(metric['lower']-(mu-w))<2e-14 and abs(metric['upper']-(mu+w))<2e-14
        else:assert metric['lower'] is None and metric['upper'] is None
        intervals+=1
    for r in s['rates']:
        fi=fk[r['case'],r['policy']];ni=nk[md['conditions'][r['case']]['group'],r['policy']]
        j=md['methods'].index(r['method']);ai=md['alphas'].index(r['beta']);g=anchor[ai]
        for key,values in [('tpr',a['power'][:,fi,j,g]),('actual_fpr',a['actual_fpr'][:,ni,j,g]),
                           ('nominal_tpr',a['nominal_tpr'][:,fi,j,ai]),('nominal_fpr',a['nominal_fpr'][:,ni,j,ai])]:eq(r[key],values)
    for r in s['areas']:eq(r['npauc'],a['npauc'][:,fk[r['case'],r['policy']],md['methods'].index(r['method'])])
    for r in s['paired']:
        fi=fk[r['case'],r['policy']];j=md['methods'].index(r['other']);ai=md['alphas'].index(r['beta']);g=anchor[ai]
        eq(r['difference'],a['power'][:,fi,4,g]-a['power'][:,fi,j,g])
        eq(r['npauc_difference'],a['npauc'][:,fi,4]-a['npauc'][:,fi,j])
        for i,k in enumerate(md['overlap_cells']):eq(r[k],a['overlap'][:,fi,ai,j,i]/5000)
    for r in s['normal']:
        ni=nk[r['group'],r['policy']];j=md['methods'].index(r['method']);ai=md['alphas'].index(r['beta']);g=anchor[ai]
        eq(r['actual_fpr'],a['actual_fpr'][:,ni,j,g]);eq(r['next_fpr'],a['next_fpr'][:,ni,j,ai]);eq(r['nominal_fpr'],a['nominal_fpr'][:,ni,j,ai])
        x=a['actual_fpr'][:,ni,j,g]
        assert r['exact_replicates']==int(np.sum(x==r['beta'])) and r['min_fpr']==x.min() and r['max_fpr']==x.max()
    # Text table validation: map source values to every rendered numeric cell.
    def tables(text):
        groups=[];current=[]
        for line in text.splitlines()+['']:
            if line.startswith('|'):current.append([x.strip() for x in line.strip('|').split('|')])
            elif current:groups.append(current[2:]);current=[]
        return groups
    tails=json.loads((out/'tail_diagnostics.json').read_text())
    assert tails['status']=='PASS' and h(ROOT/'numerics/tools/diagnose_pr0004_lowfpr_tails.py')==tails['source_sha256']
    rr=json.loads((out/'tail_replicates.json').read_text())
    from collections import defaultdict
    tailgroups=defaultdict(list)
    for r in rr:
        ni=nk[r['group'],r['policy']];j=md['methods'].index(r['method']);ai=md['alphas'].index(r['beta'])
        assert r['selected_threshold']==a['thresholds'][r['replicate'],ni,j,anchor[ai]]
        assert r['always_negative']==(r['selected_threshold']>r['maximum_possible_score'])
        assert r['nominal_always_negative']==(orders[j,ai]>r['maximum_possible_score'])
        tailgroups[r['group'],r['policy'],r['method'],r['beta']].append(r)
    for r in tails['summary']:
        group=tailgroups[r['group'],r['policy'],r['method'],r['beta']];assert len(group)==50
        assert r['always_negative_replicates']==sum(x['always_negative'] for x in group)
        assert r['nominal_always_negative_replicates']==sum(x['nominal_always_negative'] for x in group)
        for k in ['rank_cap_fraction','calibrated_cap_fraction']:eq(r[k],[x[k] for x in group])
    tabs=tables((out/'ALL_LOW_FPR_RESULTS.md').read_text());assert [len(t) for t in tabs]==[1720,1376,430,560,560]
    nums=0
    def number(text,want):
        nonlocal nums
        assert abs(float(text.replace('−','-'))-want)<.000051,(text,want);nums+=1
    def bound(text,r):
        if r['lower'] is None:assert text=='未定義'
        else:
            q=re.findall(r'[+-]?\d+\.\d+',text.replace('−','-'));assert len(q)==2
            number(q[0],100*r['lower']);number(q[1],100*r['upper'])
    for row,r in zip(tabs[0],s['rates']):
        assert row[:3]==[r['case'],r['policy'],r['method']]
        for col,key in [(4,'actual_fpr'),(5,'tpr'),(7,'nominal_fpr'),(8,'nominal_tpr')]:number(row[col],100*r[key]['mean'])
        number(row[3],100*r['beta']);bound(row[6],r['tpr'])
    for row,r in zip(tabs[1],s['paired']):
        assert row[:3]==[r['case'],r['policy'],r['other']]
        number(row[3],100*r['beta']);number(row[4],100*r['difference']['mean']);bound(row[5],r['difference'])
        for col,key in enumerate(['LF_only','other_only','both','neither'],6):number(row[col],100*r[key]['mean'])
    for row,r in zip(tabs[2],s['areas']):
        assert row[:3]==[r['case'],r['policy'],r['method']];number(row[3],100*r['npauc']['mean']);bound(row[4],r['npauc'])
    for row,r in zip(tabs[3],s['normal']):
        assert row[:3]==[r['group'],r['policy'],r['method']]
        number(row[3],100*r['beta']);number(row[4],100*r['actual_fpr']['mean'])
        q=re.findall(r'\d+\.\d+',row[5]);number(q[0],100*r['min_fpr']);number(q[1],100*r['max_fpr'])
        number(row[6],100*r['next_fpr']['mean']);number(row[7],r['exact_replicates'])
        number(row[8],100*r['nominal_fpr']['mean']);bound(row[9],r['nominal_fpr'])
    for row,r in zip(tabs[4],tails['summary']):
        assert row[:3]==[r['group'],r['policy'],r['method']]
        number(row[3],100*r['beta']);number(row[4],100*r['rank_cap_fraction']['mean']);number(row[5],100*r['calibrated_cap_fraction']['mean'])
        number(row[6],r['always_negative_replicates']);number(row[7],r['nominal_always_negative_replicates'])
    appendix_nums=nums
    main=tables((out/'LOW_FPR_ROC_REPORT_JA.md').read_text())
    assert [len(t) for t in main]==[6,6,3,5,8,3]
    rates={(r['case'],r['policy'],r['method'],r['beta']):r for r in s['rates']}
    paired={(r['case'],r['policy'],r['other'],r['beta']):r for r in s['paired']}
    normals={(r['group'],r['policy'],r['method'],r['beta']):r for r in s['normal']}
    ar={(r['case'],r['policy'],r['method']):r for r in s['areas']}
    ta={(r['group'],r['policy'],r['method'],r['beta']):r for r in tails['summary']}
    cases=['B_L__GM05','B_L__GP05','B_M__P02']
    casepoints=[(case,b) for case in cases for b in [.001,.01]]
    for row,(case,b) in zip(main[0],casepoints):
        number(row[1],100*b)
        for j,method in enumerate(md['methods'],2):number(row[j],100*rates[case,'B',method,b]['tpr']['mean'])
    for row,(case,b) in zip(main[1],casepoints):
        r=paired[case,'B','J',b];number(row[1],100*b);number(row[2],100*r['difference']['mean']);bound(row[3],r['difference'])
        for col,k in enumerate(['LF_only','other_only','neither'],4):number(row[col],100*r[k]['mean'])
    for row,case in zip(main[2],cases):
        for j,method in enumerate(md['methods'],1):number(row[j],100*ar[case,'B',method]['npauc']['mean'])
    for row,method in zip(main[3],md['methods']):
        assert row[0]==method
        for ai,b in enumerate([.001,.01]):
            r=normals['B_L','B',method,b];col=1+3*ai
            number(row[col],100*r['actual_fpr']['mean']);number(row[col+1],100*r['next_fpr']['mean']);number(row[col+2],r['exact_replicates'])
    groups=[g for g,p in md['null_keys'] if p=='F']
    for row,g in zip(main[4],groups):
        assert row[0]==g
        for col,(b,p) in enumerate([(b,p) for b in [.001,.01] for p in ['F','C','R']],1):number(row[col],100*normals[g,p,'LF',b]['nominal_fpr']['mean'])
    for row,p in zip(main[5],['F','C','R']):
        assert row[0]==p
        number(row[1],100*normals['K2_M',p,'LF',.001]['actual_fpr']['mean'])
        number(row[2],100*rates['K2_M__GP05',p,'LF',.001]['tpr']['mean'])
        number(row[3],100*rates['K2_M__P02',p,'LF',.001]['tpr']['mean'])
        number(row[4],ta['K2_M',p,'LF',.001]['always_negative_replicates'])
        number(row[5],ta['K2_M',p,'LF',.001]['nominal_always_negative_replicates'])
    for section,files in v['preserved'].items():
        for p,sha in files.items():assert h(ROOT/p)==sha,(section,p)
    ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes()
    assert hashlib.sha256(ledger[:v['ledger_before_bytes']]).hexdigest()==v['ledger_before_sha256']
    assert h(ROOT/'audits/fp_corpus.md')==v['corpus_sha256']
    links=0
    docs=list(out.glob('*.md'))+[ROOT/'audits/README.md',ROOT/'HANDOFF/LATEST.md',ROOT/'audits/prereg/README.md',ROOT/'audits/ledger_2026-10.md']
    for p in docs:
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            path=unquote(link.strip('<>').split('#')[0])
            if not path or path.startswith(('http:','https:','mailto:')):continue
            assert (p.parent/path).exists(),(p,path);links+=1
    sources=[ROOT/'numerics/scripts/analyze_pr0004_lowfpr.py',ROOT/'numerics/tools/render_pr0004_lowfpr.py',ROOT/'numerics/tools/diagnose_pr0004_lowfpr_tails.py',Path(__file__)]
    result=dict(status='PASS',same_agent=True,independent_scientific_replication=False,
        histogram_roc_points=points,distinct_jump_integrals=integrals,summary_intervals=intervals,
        appendix_numeric_cells=appendix_nums,main_numeric_cells=nums-appendix_nums,appendix_rows=[len(t) for t in tabs],local_links=links,
        frozen_and_old_outputs_preserved=True,ledger_prefix_preserved=True,
        files={str(p.relative_to(out)):h(p) for p in out.rglob('*') if p.is_file() and p.name!='delivery_validation.json'},
        sources={str(p.relative_to(ROOT)):h(p) for p in sources})
    (out/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','sources']},indent=2))


if __name__=='__main__':verify(sys.argv[1])
