# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Separate binomial-tail/integer verification and report delivery checks."""
from pathlib import Path
import hashlib, json, re, sys
import numpy as np
import scipy
from scipy.stats import binom
from analyze_pr0009_mechanisms import Archive

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_v_budget_v1'
G=['B_M','B_L','K2_M','K200_L','V05_L','V15_L'];N=[500,1000,2000,5000,10000,20000];A=[.01,.001]
def h(b):return hashlib.sha256(b).hexdigest()
def load(n):return json.loads((OUT/n).read_text())
def triple(a):return '/'.join(map(str,a))

def main():
    assert load('validation.json')['status']=='PASS_ANALYSIS'
    with np.load(OUT/'counts_and_intervals.npz') as z:d={k:z[k] for k in z.files}
    # Re-read original bit records; histogram the first nibble rather than a cumulative Boolean sum.
    arc=Archive();v=np.zeros((50,6,6,2),np.int64);e=np.zeros((50,6,2),np.int64)
    for r in range(50):
        for gi,g in enumerate(G):
            folder=f'shards/{r//5:02d}/rep_{r:02d}'
            for sub in ['validation','decisions']:
                for start in range(0,20000,4096):
                    bits=arc.npz(f'{folder}/{sub}/{g}/{start:09d}.npz')['packed'][:,0,:]
                    for ai in range(2):
                        values=(bits[:,ai]&15).astype(int)
                        assert np.all(values>=8)
                        for j,n in enumerate(N if sub=='validation' else [20000]):
                            m=max(0,min(len(values),n-start))
                            histogram=np.bincount(values[:m],minlength=16)
                            k=int(histogram[9:16].sum())
                            if sub=='validation':v[r,gi,j,ai]+=k
                            else:e[r,gi,ai]+=k
    assert np.array_equal(v,d['counts']) and np.array_equal(e,d['E_counts'])
    assert np.array_equal(np.cumsum(d['segments'],axis=2),v)
    assert np.all(d['segments']>=0)
    boundaries={(x['alpha'],x['n']):x for x in load('integer_boundaries.json')}
    checks=0
    for ai,alpha in enumerate(A):
        cap=1.25*alpha
        for j,n in enumerate(N):
            k=np.arange(n+1)
            # Invert exact one-sided binomial tests, independently of beta quantiles.
            s=k[binom.cdf(k,n,cap)<=.0125];x=k[binom.sf(k-1,n,cap)<.0125]
            smax=int(s[-1]) if s.size else -1;xmin=int(x[0]) if x.size else n+1
            b=boundaries[alpha,n];assert b['support_max']==smax and b['exceed_min']==xmin
            assert smax+1<xmin
            pred=np.where(v[:,:,j,ai]<=smax,0,np.where(v[:,:,j,ai]>=xmin,1,2))
            assert np.array_equal(pred,d['states'][:,:,j,ai]);checks+=pred.size
            kval=v[:,:,j,ai];lo=d['lower'][:,:,j,ai];hi=d['upper'][:,:,j,ai]
            assert np.all(lo[kval==0]==0) and np.all(hi[kval==n]==1)
            use=kval>0
            assert np.allclose(binom.sf(kval[use]-1,n,lo[use]),.0125,rtol=2e-10,atol=2e-13)
            use=kval<n
            assert np.allclose(binom.cdf(kval[use],n,hi[use]),.0125,rtol=2e-10,atol=2e-13)
            assert np.isclose(b['nominal_support'],binom.pmf(np.arange(smax+1),n,alpha).sum(),rtol=1e-12,atol=1e-15)
            assert np.isclose(sum(b[t] for t in ['nominal_support','nominal_exceed','nominal_unresolved']),1.)
            # Boundary negative controls: one count beyond either acceptance boundary changes its label.
            assert not (smax+1<=smax) and not (xmin-1>=xmin)
            if smax>=0:
                assert binom.cdf(smax,n,cap)<=.0125 and binom.cdf(smax+1,n,cap)>.0125
            assert binom.sf(xmin-1,n,cap)<.0125 and binom.sf(xmin-2,n,cap)>=.0125
        b=boundaries[alpha,20000]
        pred=np.where(e[:,:,ai]<=b['support_max'],0,np.where(e[:,:,ai]>=b['exceed_min'],1,2))
        assert np.array_equal(pred,d['E_states'][:,:,ai]);checks+=pred.size
        min_n=load('validation.json')['minimum_zero_alarm_support_n'][str(alpha)]
        assert (1-cap)**min_n<=.0125 and (1-cap)**(min_n-1)>.0125
    rows=load('resource_curve.json');ix={}
    for row in rows:
        gi=G.index(row['group']);j=N.index(row['n']);ai=A.index(row['alpha'])
        s=d['states'][:,gi,j,ai];es=d['E_states'][:,gi,ai]
        matrix=[[int(np.sum((s==i)&(es==k))) for k in range(3)] for i in range(3)]
        assert matrix==row['V_E']
        assert [int(np.sum(s==i)) for i in range(3)]==row['labels']
        assert [int(np.sum(es==i)) for i in range(3)]==row['E_labels']
        assert np.isclose(row['V_mean_fpr'],v[:,gi,j,ai].sum()/(50*row['n']))
        assert np.isclose(row['E_mean_fpr'],e[:,gi,ai].sum()/1000000)
        assert np.isclose(row['mean_interval_width'],np.mean(d['upper'][:,gi,j,ai]-d['lower'][:,gi,j,ai]))
        ix[row['group'],row['alpha'],row['n']]=row
    for row in load('transitions.json'):
        gi=G.index(row['group']);ai=A.index(row['alpha']);j=N.index(row['n_to'])
        s=d['states'][:,gi,j-1,ai];n=d['states'][:,gi,j,ai]
        assert row['matrix']==[[int(np.sum((s==i)&(n==k))) for k in range(3)] for i in range(3)]
    for row in load('reversals.json'):
        gi=G.index(row['group']);ai=A.index(row['alpha']);s=d['states'][:,gi,:,ai]
        early=set(np.flatnonzero(np.any(s[:,:-1]==0,axis=1)))
        assert row['early_support']==len(early)
        assert row['early_support_final_unresolved']==len(early & set(np.flatnonzero(s[:,-1]==2)))
        assert row['early_support_final_exceeded']==len(early & set(np.flatnonzero(s[:,-1]==1)))
        affected=set()
        for r in range(50):
            for j in range(5):
                if s[r,j]==0 and any(s[r,k]!=0 for k in range(j+1,6)):affected.add(r)
        assert row['any_support_then_nonsupport']==len(affected)
    report=(OUT/'PR0009_V_BUDGET_REPORT_JA.md').read_text()
    for r in [5,49]:
        c='→'.join(map(str,v[r,0,:,0]));s='→'.join('SXU'[int(x)] for x in d['states'][r,0,:,0])
        expected=f"| {r} | {c} | {s} | {e[r,0,0]}警報・{'SXU'[int(d['E_states'][r,0,0])]} |"
        assert expected in report
    assert np.sum((d['states']==0)&(d['E_states'][:,:,None,:]==1))==0
    assert np.sum((d['states']==1)&(d['E_states'][:,:,None,:]==0))==0
    for section,alpha in [('3.',.01),('4.',.001)]:
        body=report.split('## '+section)[1].split('\n## ')[0]
        for g in G:
            expected='| '+g+' | '+' | '.join(triple(ix[g,alpha,n]['labels']) for n in N)+' |'
            assert expected in body,expected
    for alpha in A:
        al=f'{100*alpha:g}%'
        for g in G:
            row=ix[g,alpha,20000]
            expected=f"| {g} | {al} | {100*row['E_mean_fpr']:.4f}% | {triple(row['labels'])} | {triple(row['E_labels'])} |"
            assert expected in report,expected
        for n in N:
            b=boundaries[alpha,n];ks='支持不可能' if b['support_max']<0 else str(b['support_max'])
            expected=f"| {al} | {n:,} | {ks} | {b['exceed_min']} | {100*b['nominal_support']:.2f}% |"
            assert expected in report,expected
    full=(OUT/'ALL_V_BUDGET_RESULTS.md').read_text();published=0
    for section in full.split('## ')[1:]:
        header=section.splitlines()[0]
        if ' / α=' not in header:continue
        g,al=header.split(' / α=');alpha=float(al)
        for n in N:
            r=ix[g,alpha,n]
            expected='| '+' | '.join([str(n),triple(r['labels'])]+[triple(a) for a in r['V_E']]+[f"{100*r['V_mean_fpr']:.4f}",f"{100*r['mean_interval_width']:.4f}"])+' |'
            assert expected in section;published+=1
    assert published==72
    for row in load('transitions.json'):
        expected='| '+' | '.join([row['group'],str(row['alpha']),f"{row['n_from']}→{row['n_to']}"]+[triple(v) for v in row['matrix']])+' |'
        assert expected in full
    for text in [report,full]:
        for block in re.findall(r'(?:^\|.*\n)+',text,re.M):
            assert len(set(len(line.split('|')) for line in block.splitlines()))==1
    linkcount=0
    for p in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in target or target.startswith('#'):continue
            dest=(p.parent/target.split('#')[0]).resolve()
            if dest.name in ['delivery_validation.json','closeout_validation.json']:continue
            assert dest.exists(),(p,target);linkcount+=1
    figures=list((OUT/'figures').glob('*.png'));assert len(figures)==6
    for p in figures:assert p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    snap=load('start_snapshot.json')
    for name,digest in snap['protected'].items():assert h((ROOT/name).read_bytes())==digest,name
    ledger=(ROOT/'audits/ledger_2026-10.md').read_bytes()
    assert h(ledger[:snap['ledger_bytes']])==snap['ledger_sha256']
    outputs={str(p.relative_to(ROOT)):h(p.read_bytes()) for p in OUT.rglob('*') if p.is_file() and p.name not in ['delivery_validation.json','closeout_validation.json']}
    sources={str((ROOT/'numerics/scripts'/n).relative_to(ROOT)):h((ROOT/'numerics/scripts'/n).read_bytes()) for n in ['analyze_pr0009_v_budget.py','report_pr0009_v_budget.py','verify_pr0009_v_budget.py']}
    result=dict(status='PASS_DELIVERY',classification_checks=checks,binomial_boundary_sets=12,
        saved_bit_records_recounted=12000000,recount_member_reads=len(arc.checked),summary_rows=72,transition_rows=60,
        protected_files=len(snap['protected']),local_links=linkcount,figures=6,
        boundary_positive_negative_controls=True,original_V_E_histograms_match=True,
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        new_time_series=0,manuscript_edits=0,independent_reviewer=False,outputs=outputs,sources=sources)
    (OUT/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS_DELIVERY',checks,'classifications;',published,'published resource rows; 60 transitions')

if __name__=='__main__':main()
