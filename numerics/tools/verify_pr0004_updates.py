# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Separate, read-only reconstruction checks for C/R update decomposition."""
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote
import hashlib
import json
import re
import sys
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def tables(text):
    tabs=[];current=[]
    for line in text.splitlines()+['']:
        if line.startswith('|'):current.append([x.strip() for x in line.strip('|').split('|')])
        elif current:
            if len(current)>1 and all(re.fullmatch(r':?-{3,}:?',cell) for cell in current[1]):tabs.append(current[2:])
            current=[]
    return tabs


def verify(out):
    out=Path(out);read=lambda n:json.loads((out/n).read_text())
    # Positive and negative parser controls: an absolute-value equation is not a table.
    assert tables('|x| > t\n')==[]
    assert tables('| A | B |\n|---|---|\n| 1 | 2 |\n')==[[['1','2']]]
    val=read('validation.json');s=read('summary.json');g=read('geometry.json')
    assert val['status']=='PASS' and sha(out/'PROTOCOL.md')==val['protocol_sha256']
    assert sha(ROOT/'numerics/scripts/analyze_pr0004_updates.py')==val['source_sha256']
    rr=read('replicate_rates.json');tt=read('replicate_transitions.json');mm=read('replicate_margins.json')
    ri={(r['replicate'],r['case'],r['method'],r['alpha']):r for r in rr}
    ti={(r['replicate'],r['case'],r['alpha']):r for r in tt}
    mi={(r['replicate'],r['case'],r['direction']):r for r in mm}
    assert len(ri)==len(rr)==16000 and len(ti)==len(tt)==3200 and len(mi)==len(mm)==3200
    cases=list(dict.fromkeys(r['case'] for r in rr));checks=0
    for rep in range(50):
        with np.load(out/f'decisions/rep_{rep:02d}.npz',allow_pickle=False) as z:
            assert set(z.files)=={c+suffix for c in cases for suffix in ['_bits','_branches']}
            for case in cases:
                bits=z[case+'_bits'];br=z[case+'_branches'];n=20000 if '__' not in case else 5000
                assert bits.shape==br.shape==(n,2,2) and bits.dtype==br.dtype==np.uint8
                assert np.all(bits<32) and np.all(br<4) and np.array_equal((bits&16)>0,br>0)
                for ai,a in enumerate([.001,.01]):
                    for j,m in enumerate(['M','CR','B','J','LF']):
                        # Reconstruct joint-state histogram rather than four Boolean sums.
                        code=((bits[:,ai,0]>>j)&1)+2*((bits[:,ai,1]>>j)&1)
                        counts=np.bincount(code,minlength=4)[[3,2,1,0]]
                        r=ri[rep,case,m,a];assert r['n']==n and np.array_equal(counts,r['counts']);checks+=1
                    tr=ti[rep,case,a];hist=np.histogram2d(br[:,ai,0],br[:,ai,1],bins=[np.arange(5)-.5,np.arange(5)-.5])[0]
                    assert tr['n']==n and np.array_equal(hist,tr['table']) and hist.sum()==n
                for direction,index in [('R_only',1),('C_only',2)]:
                    r=mi[rep,case,direction];assert r['n']==n
                    assert r['flip_count']==ri[rep,case,'LF',.01]['counts'][index]==sum(r['face_counts'])
                    assert abs(np.sum(r['term_sums'])-r['margin_sum'])<1e-8
        if (rep+1)%10==0:print(f'Decision / transition reconstruction {rep+1}/50',flush=True)
    intervals=0
    def eq(o,x):
        nonlocal intervals
        x=np.asarray(x,float);m=float(x.mean());sd=float(x.std(ddof=1));w=stats.t.ppf(.975,len(x)-1)*stats.sem(x)
        assert abs(o['mean']-m)<1e-13 and abs(o['sd']-sd)<1e-13 and o['replicates']==50
        if sd:assert abs(o['lower']-(m-w))<1e-13 and abs(o['upper']-(m+w))<1e-13
        else:assert o['lower'] is None and o['upper'] is None
        intervals+=1
    for r in s['rates']:
        vals=np.array([ri[k,r['case'],r['method'],r['alpha']]['counts'] for k in range(50)],float)
        vals/=ri[0,r['case'],r['method'],r['alpha']]['n']
        for j,name in enumerate(['both','R_only','C_only','neither']):eq(r['metrics'][name],vals[:,j])
        for name,x in [('C',vals[:,0]+vals[:,2]),('R',vals[:,0]+vals[:,1]),('difference',vals[:,1]-vals[:,2])]:eq(r['metrics'][name],x)
    for r in s['transitions']:
        a=np.array([ti[k,r['case'],r['alpha']]['table'] for k in range(50)])
        assert np.array_equal(a.sum(0),r['table_counts']) and a.sum()==r['n_total']
        for name,i,j in [('R_g_only',0,1),('R_p_only',0,2),('R_both',0,3),('C_g_only',1,0),('C_p_only',2,0),('C_both',3,0)]:
            eq(r['metrics'][name],a[:,i,j]/ti[0,r['case'],r['alpha']]['n'])
    for r in s['margins']:
        q=[mi[k,r['case'],r['direction']] for k in range(50)]
        sums=np.array([x['term_sums'] for x in q]);count=sum(x['flip_count'] for x in q)
        assert count==r['flip_count'] and np.array_equal(np.sum([x['face_counts'] for x in q],0),r['face_counts'])
        assert np.allclose(sums.sum(0),r['term_sums'],rtol=0,atol=1e-9)
        if count:assert np.allclose(sums.sum((0,1))/count,r['conditional_mean'],atol=1e-13,rtol=0)
        else:assert r['conditional_mean'] is None
        assert abs(sum(x['margin_sum'] for x in q)-r['margin_sum'])<1e-8
        for j in range(4):eq(r['per_record'][j],sums.sum(1)[:,j]/q[0]['n'])
    assert len(g)==800
    for r in g:
        for pol in ['C','R']:
            a=r[pol];assert np.allclose(np.linalg.norm(a['u'],axis=1),1,atol=1e-14,rtol=0)
            if pol=='R':assert a['b']==[0.,0.]
            for j in range(2):
                if a['h'][j] is None:assert a['t'][j] is None and r['cuts_'+pol][4][j]==9999
                else:assert abs(a['t'][j]/a['L'][j]-a['h'][j])<1e-13
        if r['alpha']==.01:
            assert np.allclose(np.array(r['R']['h'])-r['C']['h'],np.array(r['delta_h_table'])+r['delta_h_cut'],atol=2e-13,rtol=0)
        else:assert r['delta_h_table'] is None and r['delta_h_cut'] is None
    # Independently re-express the original two branch inequalities on retained examples.
    gi={(r['replicate'],r['group'],r['alpha']):r for r in g};example_records=0
    with np.load(out/'examples_rep00.npz',allow_pickle=False) as z:
        for key in z.files:
            if not key.endswith('_z'):continue
            case=key[:-2];x=z[key];b=z[case+'_branches']
            for ai,a in enumerate([.001,.01]):
                for pi,pol in enumerate(['C','R']):
                    o=gi[0,case.split('__')[0],a][pol];u=np.array(o['u']);offset=np.array(o['b']);h=np.array([np.inf if v is None else v for v in o['h']])
                    v=np.einsum('ij,kj->ik',x,u)+offset
                    code=(np.abs(v[:,0])>h[0]).astype(np.uint8)+2*(v[:,1]>h[1]).astype(np.uint8)
                    assert np.array_equal(code,b[:,ai,pi]);example_records+=len(x)
    nums=0
    def num(text,want,tol=.000051):
        nonlocal nums
        assert abs(float(text.replace('−','-'))-want)<tol,(text,want);nums+=1
    def bounds(text,r):
        if r['lower'] is None:assert text=='未定義'
        else:
            x=re.findall(r'[+-]?\d+\.\d+',text.replace('−','-'));assert len(x)==2
            num(x[0],100*r['lower']);num(x[1],100*r['upper'])
    tabs=tables((out/'ALL_UPDATE_RESULTS.md').read_text());assert list(map(len,tabs))==[320,16,64,64]
    for row,r in zip(tabs[0],s['rates']):
        assert row[:2]==[r['case'],r['method']];num(row[2],100*r['alpha'])
        for j,k in enumerate(['C','R','difference'],3):num(row[j],100*r['metrics'][k]['mean'])
        bounds(row[6],r['metrics']['difference'])
        for j,k in enumerate(['R_only','C_only','both','neither'],7):num(row[j],100*r['metrics'][k]['mean'])
    tr={(r['case'],r['alpha']):r for r in s['transitions']}
    for row in tabs[1]:
        r=tr[row[0],.01];a=np.array(r['table_counts'])/r['n_total'];code=a.sum(1 if row[1]=='C' else 0)
        for j,x in enumerate([*code[1:],1-code[0]],2):num(row[j],100*x)
    for row,r in zip(tabs[2],s['transitions']):
        assert row[0]==r['case'];num(row[1],100*r['alpha'])
        for j,k in enumerate(['R_g_only','R_p_only','R_both','C_g_only','C_p_only','C_both'],2):num(row[j],100*r['metrics'][k]['mean'])
    for row,r in zip(tabs[3],s['margins']):
        assert row[:2]==[r['case'],r['direction']]
        for j,x in enumerate([r['flip_count'],*r['face_counts']],2):num(row[j],x)
        if r['conditional_mean'] is None:assert row[6:]==['未定義']*5
        else:
            for j,x in enumerate([*r['conditional_mean'],sum(r['conditional_mean'])],6):num(row[j],x,.00000051)
    appendix_nums=nums;main=tables((out/'UPDATE_DECOMPOSITION_REPORT_JA.md').read_text())
    assert list(map(len,main))==[4,6,3,4,5]
    rates={(r['case'],r['method'],r['alpha']):r['metrics'] for r in s['rates']}
    for row in main[0]:
        r=rates[row[0],'LF',.01]
        for j,k in enumerate(['C','R','difference'],1):num(row[j],100*r[k]['mean'])
        bounds(row[4],r['difference'])
        for j,k in enumerate(['R_only','C_only','neither'],5):num(row[j],100*r[k]['mean'])
    for row in main[1]:
        r=tr[row[0],.01];a=np.array(r['table_counts'])/r['n_total'];code=a.sum(1 if row[1]=='C' else 0)
        for j,x in enumerate([code[1]+code[3],code[2]+code[3],code[3],1-code[0]],2):num(row[j],100*x)
    for row in main[2]:
        a=[r for r in g if r['group']==row[0] and r['alpha']==.01]
        x=[np.mean([r['angle_difference'][j] for r in a]) for j in [0,1]]
        x+=[np.mean([r[p]['h'][0] for r in a]) for p in ['C','R']]
        x+=[np.mean([r[p]['h'][1]-r[p]['b'][1] for r in a]) for p in ['C','R']]
        for j,v in enumerate(x,1):num(row[j],v,.00000051)
    for row in main[3]:
        for j,m in enumerate(['M','CR','B','J','LF'],1):num(row[j],100*rates[row[0],m,.01]['difference']['mean'])
    for row in main[4]:
        r=rates['K2_M__P02',row[0],.001]
        for j,k in enumerate(['C','R','difference'],1):num(row[j],100*r[k]['mean'])
    for section,files in val['preserved'].items():
        for p,h in files.items():assert sha(ROOT/p)==h,(section,p)
    ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes()
    assert hashlib.sha256(ledger[:val['ledger_before_bytes']]).hexdigest()==val['ledger_before_sha256']
    correction=read('INSPECTION_CORRECTION.json');corpus=(ROOT/'audits/fp_corpus.md').read_bytes()
    assert correction['corpus_before_sha256']==val['corpus_sha256']
    assert hashlib.sha256(corpus[:correction['corpus_before_bytes']]).hexdigest()==val['corpus_sha256']
    links=0
    docs=list(out.glob('*.md'))+[ROOT/'audits/README.md',ROOT/'HANDOFF/LATEST.md',ROOT/'audits/prereg/README.md',ROOT/'audits/ledger_2026-10.md']
    for p in docs:
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            q=unquote(link.strip('<>').split('#')[0])
            if not q or q.startswith(('http:','https:','mailto:')):continue
            # This manifest is about to be written; all other links must exist now.
            if (p.parent/q).resolve()==(out/'delivery_validation.json').resolve():continue
            assert (p.parent/q).exists(),(p,q);links+=1
    sources=[ROOT/'numerics/scripts/analyze_pr0004_updates.py',ROOT/'numerics/tools/render_pr0004_updates.py',Path(__file__)]
    report=dict(status='PASS',same_agent=True,independent_scientific_replication=False,
        decision_tables=checks,branch_transition_tables=len(tt),margin_records=len(mm),summary_intervals=intervals,
        geometry_rows=len(g),example_boundary_records=example_records,appendix_rows=list(map(len,tabs)),
        appendix_numeric_cells=appendix_nums,main_numeric_cells=nums-appendix_nums,local_links=links,
        preserved=True,ledger_prefix_preserved=True,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='delivery_validation.json'},
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources})
    (out/'delivery_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['files','sources']},indent=2))


if __name__=='__main__':verify(sys.argv[1])
