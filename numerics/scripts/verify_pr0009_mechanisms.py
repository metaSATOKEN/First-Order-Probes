# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Separate arithmetic/delivery check; does not call the analysis functions."""
from pathlib import Path
import hashlib,json,re
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_mechanisms_v1'
OLD=ROOT/'numerics/analysis/pr0009_9adfa7a8c6e0_two_axis_v1'
G=['B_M','B_L','K2_M','K200_L','V05_L','V15_L'];CL=['clean','GM160','GM800','GP160','GP800','P160','P800']
EV=['normal','GM05','GP05','P02'];ME=['W0','WP','WG','W+','W−','LF0'];GA=['G-old','G-M','G-MC']
def load(n):return json.loads((OUT/n).read_text())
def h(b):return hashlib.sha256(b).hexdigest()
def close(a,b):assert np.allclose(a,b,rtol=2e-12,atol=2e-14),(a,b)

def main():
    old=[];new=[]
    for r in range(50):
        with np.load(OLD/'recomputed'/f'rep_{r:02d}.npz') as z:old.append({k:z[k] for k in z.files})
        with np.load(OUT/'recomputed'/f'rep_{r:02d}.npz') as z:new.append({k:z[k] for k in z.files})
    a={k:np.stack([v[k] for v in new]) for k in new[0]}
    o={k:np.stack([v[k] for v in old]) for k in old[0]}
    assert np.array_equal(a['states'],o['gate_states'])
    err=np.max(np.abs(a['T']-a['T'][:,:,0,None,:]-a['mean_effect']-a['sd_effect']))
    assert err<1e-10
    close(a['variance_parts'].sum(-2),a['sd']**2)
    assert np.all(a['joint']>=0)
    policy=load('policy_decomposition.json');ix={}
    maxround=0
    for v in policy:
        r=np.arange(50);g=G.index(v['group']);c=v['candidate'];e=EV.index(v['evaluation'])
        ai=[.01,.001].index(v['alpha']);mi=ME.index(v['method']);gi=GA.index(v['gate']);n=20000 if e==0 else 5000
        F=o['counts'][:,g,e,0,ai,mi,1:].sum(1);U=o['counts'][:,g,e,c+1,ai,mi,1:].sum(1)
        C=o['counts'][:,g,e,1,ai,mi,1:].sum(1);hold=o['gate_states'][:,g,c,gi]==1
        # Integer selection and integer subtraction BEFORE conversion to floating rates.
        sel=np.where(hold,F,U);delta=(sel-U)/n
        close(delta,v['delta_per_rep']);close(delta.mean(),v['delta']['mean'])
        for key,count in [('F',F),('U',U),('C_clean',C),('G',sel)]:close(count.mean()/n,v[key])
        close((sel-C).mean()/n,v['G_minus_C']);close((U-C).mean()/n,v['U_minus_C'])
        jt=a['joint'][:,g,c,e,ai,mi];assert np.all(jt.sum((1,2))==n)
        assert np.array_equal(jt.sum(1),o['counts'][:,g,e,c+1,ai,mi])
        assert np.array_equal(jt.sum(2),o['counts'][:,g,e,0,ai,mi])
        gain=jt[hold,1:,0].sum();loss=jt[hold,0,1:].sum()
        assert gain-loss==np.sum(sel-U)
        close(gain/(50*n),v['gained']);close(loss/(50*n),v['lost'])
        gp=np.zeros(8,np.int64);lp=np.zeros(8,np.int64)
        gp[1:]=jt[hold,1:,0].sum(0);lp[1:]=jt[hold,0,1:].sum(0)
        assert gp.tolist()==v['gain_patterns'] and lp.tolist()==v['loss_patterns']
        p=hold.mean();conditional=((F-U)[hold]/n).mean() if hold.any() else 0
        er=abs(delta.mean()-p*conditional);maxround=max(maxround,er);assert er<2e-14
        # Negative control: one artificial lost alarm must violate exact integer accounting.
        assert gain-(loss+1)!=np.sum(sel-U)
        ix[v['group'],c,v['evaluation'],v['alpha'],v['method'],v['gate']]=v
    assert len(policy)==6048 and len(ix)==6048
    events=load('gate_events.json');slips=[v for v in events if v['mixed'] and v['states'][2]==0]
    assert len(slips)==285 and sum(v['fixed_sd_crosses'] for v in slips)==11
    assert sum(v['candidate'] in (1,2) for v in slips)==128
    assert sum(v['candidate'] in (3,4) for v in slips)==145
    assert sum(v['candidate'] in (5,6) for v in slips)==12
    assert [sum(v['group']==g for v in slips) for g in G]==[45,4,18,164,54,0]
    mixed=a['states'][:,:,1:];m=mixed[...,1].astype(bool);mc=mixed[...,2].astype(bool)
    assert [int(x.sum()) for x in [~m&~mc,~m&mc,m&~mc,m&mc]]==[239,386,46,1129]
    clean=a['states'][:,:,0];assert clean[...,2].sum()==2 and ((clean[...,1]==1)&(clean[...,2]==0)).sum()==3
    assert not np.any((clean[...,1]==0)&(clean[...,2]==1))
    cross=ix['V05_L',2,'GP05',.01,'W0','G-MC']
    assert cross['hold_n']==50 and cross['loss_patterns']==[0,48378,0,0,0,0,0,0] and cross['gain_patterns']==[0,0,0,0,2274,0,0,0]
    close(cross['delta']['mean'],-.184416);close(cross['G_minus_C'],-.190432)
    assert np.all(a['cuts'][:,4,0,0,0,0]>a['cuts'][:,4,3,0,0,0])
    assert np.all(a['cuts'][:,4,0,0,0,2]<a['cuts'][:,4,3,0,0,2])
    # Verify every published compact row against its JSON source.
    full=(OUT/'ALL_POLICY_DECOMPOSITIONS.md').read_text();rows=0
    for section in full.split('## ')[1:]:
        header=section.splitlines()[0];gate,method,al=header.split(' / ');alpha=float(al.split('=')[1])
        for line in section.splitlines():
            if not any(line.startswith('| '+g+' |') for g in G):continue
            cells=[s.strip() for s in line.strip('|').split('|')];g,c=cells[:2];ci=CL.index(c)
            v=[ix[g,ci,e,alpha,method,gate] for e in EV]
            expected=[g,c,str(v[0]['hold_n'])]+[f'{100*x["delta"]["mean"]:.3f}' for x in v]+[f'{100*x["gained"]:.3f}/{100*x["lost"]:.3f}' for x in v]
            assert cells==expected;rows+=1
    assert rows==1512
    report=(OUT/'PR0009_MECHANISM_REPORT_JA.md').read_text()
    for ev in EV:
        v=ix['V05_L',2,ev,.01,'W0','G-MC']
        expected=f'{100*v["U"]:.4f}% | {100*v["G"]:.4f}% | {100*v["C_clean"]:.4f}% | {100*v["delta"]["mean"]:+.4f} pp'
        # Main table leaves negative signs and positive plus signs explicit.
        assert expected in report.replace('−','-'),expected
        assert expected.replace(' pp','9 pp') not in report.replace('−','-')
    # Table rendering: no unescaped equation pipes inside table cells.
    for block in re.findall(r'(?:^\|.*\n)+',report,re.M):
        counts=[len(s.split('|')) for s in block.splitlines()];assert len(set(counts))==1,block
    linkcount=0
    for file in [OUT/'PR0009_MECHANISM_REPORT_JA.md',OUT/'PROTOCOL.md',OUT/'ALL_POLICY_DECOMPOSITIONS.md']:
        for target in re.findall(r'\]\(([^)]+)\)',file.read_text()):
            if '://' in target or target.startswith('#'):continue
            dest=(file.parent/target.split('#')[0]).resolve()
            if dest==OUT/'delivery_validation.json':continue
            assert dest.exists(),(file,target);linkcount+=1
    snap=load('start_snapshot.json')
    unchanged=0;append_only={}
    for p,digest in snap['protected'].items():
        b=(ROOT/p).read_bytes()
        if p=='audits/fp_corpus.md':
            assert h(b[:18833])==digest,p
            append_only[p]=dict(original_bytes=18833,prefix_sha256=digest)
        else:assert h(b)==digest,p;unchanged+=1
    frozen=json.loads((ROOT/'numerics/colab/pr0009_two_axis_a50_build.json').read_text())['sources']
    for p,digest in frozen.items():assert h((ROOT/p).read_bytes())==digest,p
    b=(ROOT/'audits/ledger_2026-10.md').read_bytes();assert h(b[:snap['ledger_bytes']])==snap['ledger_sha256']
    outputs={str(p.relative_to(ROOT)):h(p.read_bytes()) for p in OUT.rglob('*') if p.is_file() and p.name not in ['delivery_validation.json','closeout_validation.json']}
    result=dict(status='PASS_DELIVERY',method='integer margins and selected integer counts, separate from floating-rate analysis',
                policy_rows=6048,published_table_rows=rows,gate_events=len(events),figures=7,local_links=linkcount,
                max_product_identity_error=maxround,max_T_decomposition_error=float(err),
                integer_negative_control='one altered lost alarm rejected in every policy cell',
                protected_files_unchanged=unchanged,append_only_prefix_checks=append_only,
                frozen_sources_verified=frozen,ledger_prefix_preserved=True,
                outputs=outputs,sources={str(p.relative_to(ROOT)):h(p.read_bytes()) for p in [Path(__file__).resolve(),ROOT/'numerics/scripts/analyze_pr0009_mechanisms.py',ROOT/'numerics/scripts/report_pr0009_mechanisms.py']},
                new_time_series=0,manuscript_edits=0,independent_reviewer=False)
    (OUT/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS_DELIVERY',rows,'published rows;',len(outputs),'outputs;',linkcount,'links')

if __name__=='__main__':main()
