# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Verify report numbers/links, saved output identity and integer accounting."""
from pathlib import Path
import json,hashlib,re
import numpy as np
from scipy.stats import t
from analyze_pr0010_mixture import OUT,G,EV,AL,METHODS,load,sha

def main():
    assert load(OUT/'validation.json')['status']=='PASS_SAVED_DATA_RECONSTRUCTION'
    assert load(OUT/'second_path_validation.json')['status']=='PASS_SECOND_PATH'
    assert load(OUT/'summary_validation.json')['status']=='PASS_SUMMARY_IDENTITIES'
    recipe=load(OUT/'recipes.json');ids={x['id']:i for i,x in enumerate(recipe['unique'])}
    rows=load(OUT/'results.json');parent=load(OUT/'parent_contrasts.json')
    assert len(rows)==5472 and len(parent)==2304
    vals=[];states=[];pairs=[];CP=[];JP=[]
    for r in range(50):
        path=OUT/'recomputed'/f'rep_{r:02d}.npz';rr=load(path.with_suffix('.json'));assert sha(path)==rr['output_sha256']
        with np.load(path) as z:
            jt=z['joint_F'];n=np.array([20000,5000,5000,5000])[None,None,:,None,None]
            F=jt[...,1:,:].sum((-1,-2));U=jt[...,1:].sum((-1,-2));st=z['states'];states.append(st)
            cs=np.stack([F,U]+[np.where((st[...,i]>0)[...,None,None,None],F,U) for i in range(3)],-1)
            vals.append(cs/n[...,None]);pairs.append(jt);CP.append(z['joint_C']);JP.append(z['joint_parent'])
            for key in ['joint_F','joint_C','joint_parent','joint_methods']:assert (z[key]>=0).all()
    vals=np.stack(vals);states=np.stack(states);pairs=np.stack(pairs);CP=np.stack(CP);JP=np.stack(JP)
    def check(x,s):
        width=t.ppf(.975,49)*x.std(ddof=1)/np.sqrt(50)
        for k,v in [('mean',x.mean()),('sd',x.std(ddof=1)),('lower',x.mean()-width),('upper',x.mean()+width)]:
            assert np.isclose(s[k],v,rtol=2e-12,atol=2e-14),(k,s[k],v)
    for row in rows:
        g=G.index(row['group']);c=ids[row['candidate']];e=EV.index(row['evaluation']);a=AL.index(row['alpha']);m=METHODS.index(row['method']);n=20000 if e==0 else 5000
        rates=vals[:,g,c,e,a,m]
        for pi,pol in enumerate(['F','U','G-old','G-M','G-MC']):check(rates[:,pi],row['rates'][pol])
        cj=CP[:,g,c,e,a,m];clean=cj[:,1:,:].sum((1,2))/n
        check(clean,row['C0']);check(rates[:,1]-clean,row['U_minus_C0'])
        check(cj[:,0,1:].sum(1)/n,row['U_only_C0']);check(cj[:,1:,0].sum(1)/n,row['C0_only_U'])
        j=pairs[:,g,c,e,a,m]
        for gi,effect in enumerate(row['gate_effects']):
            h=(states[:,g,c,gi]>0);check(h*j[:,1:,0].sum(1)/n,effect['gained_over_U']);check(h*j[:,0,1:].sum(1)/n,effect['lost_from_U'])
            check(rates[:,gi+2]-rates[:,1],effect['delta_U']);check(rates[:,gi+2]-rates[:,0],effect['delta_F']);check(rates[:,gi+2]-clean,effect['delta_C0'])
    edges={e['id']:i for i,e in enumerate(recipe['edges'])}
    for row in parent:
        g=G.index(row['group']);a=AL.index(row['alpha']);m=METHODS.index(row['method']);e=EV.index(row['evaluation']);n=20000 if e==0 else 5000
        j=JP[:,g,edges[row['child_alias']],e,a,m];p=ids[row['parent']];c=ids[row['child']]
        check(j[:,1:,0].sum(1)/n,row['gained']);check(j[:,0,1:].sum(1)/n,row['lost'])
        check(vals[:,g,p,e,a,m,1]-vals[:,g,c,e,a,m,1],row['delta'])
        for gi,eff in enumerate(row['gate_policy_contrasts']):check(vals[:,g,p,e,a,m,gi+2]-vals[:,g,c,e,a,m,gi+2],eff['difference'])
    gates=load(OUT/'gate_summary.json')
    for r in gates:
        actual=(states[:,G.index(r['group']),ids[r['candidate']]]==1).sum(0)
        assert np.array_equal(actual,r['holds'])
    table=(OUT/'ALL_MIXTURE_RESULTS.md').read_text();lines=[s for s in table.splitlines() if s.startswith('| ') and not s.startswith('| 候補')]
    assert len(lines)==342+6840
    index={(r['group'],r['candidate'],r['alpha'],r['method'],r['evaluation']):r for r in rows}
    for g in G:
        for c in ids:
            for a in AL:
                for m in METHODS:
                    for pol in ['F','U','G-old','G-M','G-MC']:
                        line='| '+c+' | '+pol+' | '+' | '.join(f'{100*index[g,c,a,m,e]["rates"][pol]["mean"]:.4f}' for e in EV)+' |'
                        assert line in table,line
    # Check every displayed rate table row in the narrative against canonical summaries.
    body=(OUT/'PR0010_MIXTURE_REPORT_JA.md').read_text();report_rows=0
    for g in G:
        for c in ['GMGP11_n800','GMGPP112_n800']:
            for pol in ['C0','U','G-MC']:
                v=[index[g,c,.01,'W0',e]['C0']['mean'] if pol=='C0' else index[g,c,.01,'W0',e]['rates'][pol]['mean'] for e in EV]
                line='| '+g+' | '+c+' | '+pol+' | '+' | '.join(f'{100*x:.4f}' for x in v)+' |'
                assert line in body,line;report_rows+=1
    # All twelve matched-addition rows, not just a favorable illustrative row.
    px={(r['group'],r['child_alias'],r['alpha'],r['method'],r['evaluation']):r for r in parent}
    for g in G:
        for n in [160,800]:
            rr=[px[g,f'GMGPP112_n{n}_keep01',.01,'W0',e] for e in EV];p=rr[3]
            line=f'| {g} | {n} | {p["gate_hold_difference"][2]:+d}/50 | '+' | '.join(f'{100*r["delta"]["mean"]:+.4f}' for r in rr)+f' | {100*p["gained"]["mean"]:.4f} | {100*p["lost"]["mean"]:.4f} |'
            assert line in body,line;report_rows+=1
    # Headline witnesses are frozen-source/condition specific, not selected maxima.
    assert [(states[:,g,ids['GMGP11_n800'],1]==1).sum() for g in range(6)]==[1,0,1,1,0,0]
    assert [(states[:,g,ids['GMGP11_n800'],2]==1).sum() for g in range(6)]==[50,50,50,10,50,50]
    for group,case,ev,pol,expected in [('K200_L','GMGPP112_n800_keep01','P02','U',62.9604),('K200_L','GMGPP112_n800','P02','U',19.8960),
      ('K2_M','GMGPP112_n160_keep01','normal','G-MC',1.2919),('K2_M','GMGPP112_n160','normal','G-MC',3.6213)]:
        actual=100*index[group,case,.01,'W0',ev]['rates'][pol]['mean'];assert abs(actual-expected)<5e-5
    links=[]
    for filename in ['PR0010_MIXTURE_REPORT_JA.md','ALL_MIXTURE_RESULTS.md']:
        text=(OUT/filename).read_text()
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:','#')):continue
            p=(OUT/target).resolve()
            assert p.exists() or p==OUT/'delivery_validation.json',(filename,target)
            links.append([filename,target])
        for block in re.findall(r'(?:^\|.*\|\n){2,}',text,re.M):assert len({len(re.split(r'(?<!\\)\|',x)) for x in block.splitlines()})==1
    zips=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((OUT/'bundles').glob('*.zip'))]
    assert len(zips)==10 and all(x['bytes']<128_000_000 for x in zips)
    snap=load(OUT/'start_snapshot.json')
    for p,h in snap['protected'].items():assert sha(OUT.parents[2]/p)==h,p
    result=dict(status='PASS_DELIVERY_NUMBERS_AND_LINKS',rate_rows_checked=len(rows),parent_rows=len(parent),exhaustive_table_rows=7182,
      narrative_rate_rows=report_rows,links=links,figures=load(OUT/'figure_manifest.json'),
      figures_visually_inspected=False,visual_inspection_record='See closeout validation after display inspection',
      source_files_unchanged=len(snap['protected']),zips=zips,independent_reviewer=False,
      technical_tests=10,initial_fixture_error=True,scientific_missing=0,scientific_excluded=0)
    (OUT/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS_DELIVERY',len(rows),len(links),report_rows,flush=True)

if __name__=='__main__':main()
