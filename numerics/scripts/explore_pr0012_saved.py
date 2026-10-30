# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Four descriptive views of fixed PR-0012. No simulation or decision refit."""
from pathlib import Path
import hashlib,json
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/pr0012_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'numerics/analysis/pr0012_64bf0f2c7697_confirmation_v1'
OUT=ROOT/'numerics/analysis/pr0012_64bf0f2c7697_exploration_v2'
G=['B_L','K2_M','K200_L','V05_L'];C=['clean','GM800','GP800','BAL800','BAL400','TRI800'];E=['normal','GM05','GP05','P02'];N=[20000,5000,5000,5000]
CUT=np.array([[2.454714455705117,3.0310874379402915],[2.333890445586029,2.333890445586029],[2.454714455705117,3.0310874379402915],[2.454714455705117,3.0310874379402915]])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p):return json.loads(p.read_text())
def dump(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(h,rows):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def pct(v):return 'NA' if v is None else f'{100*v:.4f}'
def mean(x):return float(np.mean(x)) if len(x) else None

def main():
    assert not (OUT/'results.json').exists(),'Do not overwrite completed exploration'
    close=js(SRC/'closeout_validation.json');anchors={**close['protected_sources'],**close['outputs'],**close['current_navigation']}
    for p,h in anchors.items():assert sha(ROOT/p)==h,p
    for key,path in [('ledger','audits/ledger_2026-10.md'),('corpus','audits/fp_corpus.md')]:
        assert sha(ROOT/path)==close[key]['sha256']
    dump('start_snapshot.json',dict(anchors=anchors,previous_closeout_sha=sha(SRC/'closeout_validation.json'),ledger=close['ledger'],corpus=close['corpus']))
    a=dict(np.load(SRC/'outer_arrays.npz'));details=[js(SRC/f'details/rep_{i:02d}.json') for i in range(50)]
    decomposition=[];transfer=[];branches=[];passrisk=[]
    errors=dict(mean_closure=0.,variance_closure=0.)
    for g,gn in enumerate(G):
        for c,cn in enumerate(C):
            ad=np.array([[r['scores'] for r in d if r['group']==gn and r['candidate']==cn and r['role']=='audit'][0] for d in details])
            ud=np.array([[r['scores'] for r in d if r['group']==gn and r['candidate']==cn and r['role']=='update'][0] for d in details])
            ah=ad>CUT[g];uh=ud>CUT[g]
            assert np.array_equal(ah,a['gates'][:,g,c].astype(bool))
            for b,bn in enumerate(('G-M','G-MC')):
                counts=np.zeros((2,2),int)
                for x,y in zip(ah[:,b],uh[:,b]):counts[int(x),int(y)]+=1
                assert counts.sum()==50
                transfer.append(dict(group=gn,candidate=cn,gate=bn,matrix=counts.tolist(),
                   A_holds=int(ah[:,b].sum()),U_shadow_exceeds=int(uh[:,b].sum()),disagree=int((ah[:,b]!=uh[:,b]).sum()),
                   A_pass_U_exceed=int((~ah[:,b]&uh[:,b]).sum()),A_hold_U_pass=int((ah[:,b]&~uh[:,b]).sum())))
            for ai,alpha in enumerate((.01,.001)):
                sel=a['selection'][:,g,c,ai,2]
                assert np.all(np.isin(sel,[0,c+1,7,-1]))
                for e,en in enumerate(E):
                    h=a['policy'][:,g,c,e,ai];delta=(h[:,2,0]-h[:,0,0])/N[e];mu=delta.mean()
                    pieces=[];within=between=0.
                    for name,key in [('F',0),('U',c+1),('C',7),('DEFER',-1)]:
                        mask=sel==key;n=int(mask.sum());weight=n/50;cond=mean(delta[mask]);contrib=float(np.sum(delta[mask])/50)
                        if n:
                            within+=float(np.sum((delta[mask]-cond)**2)/50)
                            between+=weight*(cond-mu)**2
                        pieces.append(dict(path=name,n=n,conditional_delta=cond,contribution=contrib,
                             P0=mean(h[mask,0,0]/N[e]),P2=mean(h[mask,2,0]/N[e])))
                    total=float(delta.var(ddof=0))
                    errors['mean_closure']=max(errors['mean_closure'],abs(sum(x['contribution'] for x in pieces)-mu))
                    errors['variance_closure']=max(errors['variance_closure'],abs(total-within-between))
                    decomposition.append(dict(group=gn,candidate=cn,alpha=alpha,evaluation=en,mean=float(mu),
                      variance=total,within=within,between=between,between_fraction=between/total if total else None,pieces=pieces,
                      per_repeat=delta.tolist(),selection=sel.tolist()))
                    mask=~ah[:,1];h1=a['wcounts'][:,g,e,c+1,ai];hc=a['wcounts'][:,g,e,1,ai]
                    v=h1[:,1:8].sum(1)/N[e];base=hc[:,1:8].sum(1)/N[e]
                    passrisk.append(dict(group=gn,candidate=cn,alpha=alpha,evaluation=en,passed=int(mask.sum()),
                      U=mean(v[mask]),clean_U=mean(base[mask]),delta=mean((v-base)[mask]),
                      lower_than_clean=int((v[mask]<base[mask]).sum()),higher_than_clean=int((v[mask]>base[mask]).sum()),
                      empirical_above_cap=int((v[mask]>1.25*alpha).sum()) if e==0 else None))
        for e,en in enumerate(E):
            for ai,alpha in enumerate((.01,.001)):
                for context,ci in [('F',0),('clean_U',1),('C',7)]:
                    hist=a['wcounts'][:,g,e,ci,ai].sum(0);assert hist[8]==0
                    alarm=int(hist[1:8].sum());go=int(sum(hist[k] for k in (1,2,3)));po=int(hist[4]);both=int(sum(hist[k] for k in (5,6,7)))
                    assert go+po+both==alarm
                    branches.append(dict(group=gn,evaluation=en,alpha=alpha,context=context,
                      alarm=alarm/(50*N[e]),G_only=go/(50*N[e]),P_only=po/(50*N[e]),both=both/(50*N[e]),
                      P_only_share=po/alarm if alarm else None,G_only_share=go/alarm if alarm else None,
                      denominator=50*N[e],counts=dict(alarm=alarm,G_only=go,P_only=po,both=both)))
    assert errors['mean_closure']<1e-14 and errors['variance_closure']<1e-14
    # Separate loop-based population moment calculation validates the decomposition.
    for r in decomposition:
        d=r['per_repeat'];s=r['selection'];mu=sum(d)/50
        total=sum((v-mu)**2 for v in d)/50;within=between=0.
        for k in set(s):
            values=[d[i] for i in range(50) if s[i]==k];m=sum(values)/len(values)
            within+=sum((v-m)**2 for v in values)/50;between+=len(values)/50*(m-mu)**2
        assert abs(total-r['variance'])<1e-14 and abs(within-r['within'])<1e-14 and abs(between-r['between'])<1e-14
    result=dict(decomposition=decomposition,transfer=transfer,branches=branches,passrisk=passrisk)
    dump('results.json',result)
    dump('validation.json',dict(status='PASS_EXPLORATORY_IDENTITIES',errors=errors,
      rows={k:len(v) for k,v in result.items()},all_inputs_hashed=True,independent_reviewer=False,
      primary_classification_unchanged=js(SRC/'second_path_validation.json')['primary'],
      generator_calls=0,refits=0,policy_changes=0))
    (OUT/'ALL_EXPLORATORY_RESULTS.md').write_text('# PR-0012 探索的全条件\n\n率は%、差はpp。条件は同じ50反復の事後層別。小標本層も件数を保持し0件はNA。\n\n## 経路分解\n\n'+
      table(['group','candidate','α','E','path','n','conditional Δ','contribution','between variance %'],
       [[r['group'],r['candidate'],r['alpha'],r['evaluation'],s['path'],s['n'],pct(s['conditional_delta']),pct(s['contribution']),pct(r['between_fraction'])] for r in decomposition for s in r['pieces']])+
      '\n## AとUの固定境界照合\n\n行A、列U、0通過／1超過。Uは影点検で実政策には不使用。\n\n'+
      table(['group','candidate','gate','matrix','A holds','U exceeds','A pass U exceeds','A hold U passes'],
        [[r['group'],r['candidate'],r['gate'],r['matrix'],r['A_holds'],r['U_shadow_exceeds'],r['A_pass_U_exceed'],r['A_hold_U_pass']] for r in transfer])+
      '\n## 枝構成\n\nPのみ比率は全警報に占める比率。\n\n'+table(['group','α','context','E','OR','G only','P only','both','P only share'],
        [[r['group'],r['alpha'],r['context'],r['evaluation']]+[pct(r[k]) for k in ('alarm','G_only','P_only','both','P_only_share')] for r in branches])+
      '\n## A通過層の混入Uとclean U\n\nabove capは評価正常の経験率が1.25αを超える反復数で、CP認証ではない。\n\n'+
      table(['group','candidate','α','E','n pass','U','clean U','Δ','lower','higher','above cap'],
        [[r['group'],r['candidate'],r['alpha'],r['evaluation'],r['passed'],pct(r['U']),pct(r['clean_U']),pct(r['delta']),r['lower_than_clean'],r['higher_than_clean'],r['empirical_above_cap']] for r in passrisk]))
    figures(result)
    print('PASS', {k:len(v) for k,v in result.items()})

def pick(rows,**kw):return next(r for r in rows if all(r[k]==v for k,v in kw.items()))
def figures(r):
    f=OUT/'figures';f.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):fig.tight_layout();fig.savefig(f/name,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ai,alpha in enumerate((.01,.001)):
        ax=axes[ai];rr=[pick(r['decomposition'],group='K2_M',candidate='TRI800',alpha=alpha,evaluation=e) for e in E]
        for i,path in enumerate(('C','DEFER')):
            v=[next(x['contribution'] for x in z['pieces'] if x['path']==path)*100 for z in rr]
            ax.bar(np.arange(4)+(i-.5)*.35,v,.35,label=path,color=['#238d9f','#e4b65b'][i])
        ax.set_xticks(range(4),E);ax.set_ylabel('Contribution to P2 - P0 (pp)');ax.set_title(f'K2_M / TRI800, alpha={alpha}');ax.legend();ax.axhline(0,color='k',lw=.7)
    fig.suptitle('The average loss has two distinct sources');save(fig,'01_path_contributions.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ax,en in zip(axes,['GM05','P02']):
        q=pick(r['decomposition'],group='K2_M',candidate='TRI800',alpha=.01,evaluation=en)
        for name,key,col in [('C recovery',7,'#238d9f'),('DEFER',-1,'#e4b65b')]:
            ids=[i for i,s in enumerate(q['selection']) if s==key];ax.scatter(ids,[q['per_repeat'][i]*100 for i in ids],label=name,color=col,s=25)
        ax.set_xlabel('Outer repeat');ax.set_ylabel('P2 - P0 detection (pp)');ax.set_title(en+f": between-path variance {100*q['between_fraction']:.2f}%");ax.legend()
    fig.suptitle('R05 uncertainty is a policy-path mixture, not noisy per-record detection');save(fig,'02_repeat_mixture.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ax,key,title in [(axes[0],'A_pass_U_exceed','A passes, U shadow exceeds'),(axes[1],'A_hold_U_pass','A holds, U shadow passes')]:
        z=np.array([[pick(r['transfer'],group=g,candidate=c,gate='G-MC')[key] for c in C] for g in G]);ax.imshow(z,cmap='Blues',vmin=0,vmax=max(1,int(z.max())))
        for g in range(4):
            for c in range(6):ax.text(c,g,str(z[g,c]),ha='center',va='center',color='white' if z[g,c]>z.max()*.55 else 'black')
        ax.set_xticks(range(6),C,rotation=30,ha='right');ax.set_yticks(range(4),G);ax.set_title(title+' / 50')
    fig.suptitle('Same mixture composition does not imply identical sample decisions');save(fig,'03_A_U_disagreement.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ax,en in zip(axes,['GM05','GP05']):
        rr=[pick(r['branches'],group=g,context='F',alpha=.01,evaluation=en) for g in G];bottom=np.zeros(4)
        for k,col in [('G_only','#238d9f'),('P_only','#c45143'),('both','#707e89')]:
            v=np.array([x[k]/x['alarm']*100 if x['alarm'] else 0 for x in rr]);ax.bar(range(4),v,bottom=bottom,label=k,color=col);bottom+=v
        ax.set_xticks(range(4),G);ax.set_ylim(0,110);ax.set_ylabel('Share of all detected faults (%)');ax.set_title(en+', old F, alpha=0.01');ax.legend(fontsize=8)
    fig.suptitle('A branch label is not a fault diagnosis');save(fig,'04_branch_cross_sensitivity.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ax,ai in zip(axes,range(2)):
        alpha=(.01,.001)[ai];cs=C[1:];x=np.arange(len(cs))
        for i,e in enumerate(E):
            rr=[pick(r['passrisk'],group='K200_L',candidate=c,alpha=alpha,evaluation=e) for c in cs]
            v=[q['delta']*100 if q['delta'] is not None else np.nan for q in rr]
            ax.plot(x,v,'o-',label=e)
        ns=[pick(r['passrisk'],group='K200_L',candidate=c,alpha=alpha,evaluation='normal')['passed'] for c in cs]
        ax.set_xticks(x,[c+f'\nn={n}' for c,n in zip(cs,ns)],rotation=15);ax.axhline(0,color='k',lw=.7);ax.set_title(f'K200_L, alpha={alpha}');ax.set_ylabel('Accepted U - clean U (pp)');ax.legend(fontsize=8)
    fig.suptitle('Among audit passes, fewer alarms can mean lost sensitivity');save(fig,'05_accepted_U_cost.png')

if __name__=='__main__':main()
