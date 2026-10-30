# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Generate PR-0009 descriptive tables and figures from reproduced integer counts."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pr0009-matplotlib')
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.stats import beta,t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pr0009_spec as spec
from analyze_pr0009_return import OUT,ROOT,dump,equal

G=spec.groups();C=spec.CANDIDATES;METHODS=list(spec.METHODS)
CL=['clean','GM160','GM800','GP160','GP800','P160','P800']
POL=list(spec.POLICIES)+['C-clean']
LAB=['FPR-SUPPORTED','FPR-EXCEEDED','FPR-UNRESOLVED','UNAVAILABLE']
COL=['#298c70','#ce544b','#c59a33','#88929d']
def load(n):return json.loads((OUT/n).read_text())
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def fmt(x,d=3):return f'{100*x:.{d}f}'
def figsave(name):
    plt.savefig(OUT/'figures'/name,dpi=160,bbox_inches='tight');plt.close()
def interval(x,family=1):
    x=np.asarray(x);v=dict(n=len(x),mean=float(x.mean()),sd=float(x.std(ddof=1)),median=float(np.median(x)),lower=None,upper=None)
    if len(x)==50 and np.isfinite(x).all() and v['sd']>0:
        h=t.isf(.05/(2*family),49)*v['sd']/np.sqrt(50);v.update(lower=float(v['mean']-h),upper=float(v['mean']+h))
    return v

def main():
    assert load('validation.json')['status']=='PASS_SAVED_DATA_REPRODUCTION'
    (OUT/'figures').mkdir(exist_ok=True)
    data=[]
    for r in range(50):
        with np.load(OUT/'recomputed'/f'rep_{r:02d}.npz') as f:data.append({k:f[k] for k in f.files})
    a={k:np.stack([r[k] for r in data]) for k in data[0]}
    assert np.all(a['available']==np.array([20000,5000,5000,5000])[None,None,:,None,None,None])
    assert np.all(a['gate_states']<2) and np.all(a['same_states']<2)
    rates=a['counts'][...,1:].sum(-1)/a['available']
    labels=np.full((50,6,5,2),3,int);evlabels=np.full((50,6,2),3,int)
    evidence=load('validation_evidence.json')
    for v in evidence:
        r=v['replicate'];g=G.index(v['group']);vi=['clean','stale','mix_GM05','mix_GP05','mix_P02'].index(v['variant']);ai=[.01,.001].index(v['alpha'])
        k=v['result']['alarms'];n=v['result']['available']
        if v['result']['label']!='UNAVAILABLE':
            # Independent inversion using beta.isf for the upper endpoint.
            lo=0. if not k else beta.ppf(.0125,k,n-k+1)
            hi=1. if k==n else beta.isf(.0125,k+1,n-k)
            label=0 if hi<=1.25*v['alpha'] else 1 if lo>1.25*v['alpha'] else 2
            assert LAB[label]==v['result']['label']
        labels[r,g,vi,ai]=LAB.index(v['result']['label']);evlabels[r,g,ai]=LAB.index(v['independent_E']['label'])
    def indices(g,c,p):
        if p==0:return np.zeros(50,int)
        if p==1:return np.full(50,c+1)
        if p==6:return np.ones(50,int)
        states=a['gate_states'][:,g,c,p-2] if p<5 else a['same_states'][:,g,c]
        return np.where(states>0,0,c+1)
    def rate(g,c,p,e=0,ai=0,mi=0):return rates[np.arange(50),g,e,indices(g,c,p),ai,mi]
    def ci_text(iv):return '未解決（SD=0）' if iv['lower'] is None else f"[{fmt(iv['lower'])}, {fmt(iv['upper'])}]"
    primary=[]
    for p in spec.primary():
        g=G.index(p['group']);c=p['candidate'];e=('normal',)+spec.FAULTS;e=e.index(p['evaluation'])
        x=rate(g,c,4,e)-rate(g,c,1,e);iv=interval(x,18)
        orig=next(j for j in load('primary18.json') if j['id']==p['id'])
        for key in iv:equal(iv[key],orig['interval'][key],p['id']+'/'+key)
        primary.append(dict(**p,**iv,U=float(rate(g,c,1,e).mean()),GMC=float(rate(g,c,4,e).mean()),
            classification='positive' if iv['lower'] is not None and iv['lower']>0 else 'negative' if iv['upper'] is not None and iv['upper']<0 else 'unresolved'))
    dump(OUT/'primary_independent.json',primary)
    # Full compact tables, with every outer replicate retained in recomputed/*.npz.
    records=[];full=[];paired=[];conditional=[];twoaxis=[];same=[]
    for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
        for mi,method in enumerate(METHODS):
            full+=['',f'## {method} — α={100/den:g}%', '']
            rows=[]
            for g,group in enumerate(G):
                for c,cl in enumerate(CL):
                    for p,policy in enumerate(POL):
                        rr=[rate(g,c,p,e,ai,mi) for e in range(4)]
                        idx=indices(g,c,p);inact=int((a['inactive'][np.arange(50),g,idx,ai]==1).sum()) if mi==5 else None
                        item=dict(group=group,candidate=c,policy=policy,method=method,alpha=1/den,
                            normal=interval(rr[0]),GM05=interval(rr[1]),GP05=interval(rr[2]),P02=interval(rr[3]),
                            positive_minus_negative=interval(rr[2]-rr[1]),
                            zero_alarm_replicates=[int((x==0).sum()) for x in rr],lf0_inactive_replicates=inact,
                            planned_per_replicate=[20000,5000,5000,5000],unavailable=0)
                        records.append(item)
                        rows.append([group,cl,policy]+[fmt(x.mean()) for x in rr]+[fmt(1-x.mean()) for x in rr[1:]]+[inact if inact is not None else '—'])
                    for e,name in enumerate(('normal',)+spec.FAULTS):
                        n=20000 if e==0 else 5000
                        for gi,gate in enumerate(list(spec.GATES)+['G-MC-same-U']):
                            for ri,ref in enumerate(('U','F','C-clean')):
                                cells=a['paired'][:,g,c,e,gi,ri,ai,mi]
                                assert np.all(cells.sum(-1)==n)
                                paired.append(dict(group=group,candidate=c,evaluation=name,method=method,alpha=1/den,gate=gate,reference=ref,
                                    both=float(cells[:,0].mean()/n),gained=float(cells[:,1].mean()/n),lost=float(cells[:,2].mean()/n),neither=float(cells[:,3].mean()/n),
                                    delta=interval((cells[:,1]-cells[:,2])/n)))
                    # Required conditional summaries: all / hold / nonhold / each V label.
                    states=a['gate_states'][:,g,c,2]
                    masks={'all':np.ones(50,bool),'hold':states>0,'nonhold':states==0}
                    masks.update({l:labels[:,g,0,ai]==j for j,l in enumerate(LAB)})
                    for state in ('hold','nonhold'):
                        masks.update({state+' & '+l:masks[state]&(labels[:,g,0,ai]==j) for j,l in enumerate(LAB)})
                    for name,mask in masks.items():
                        vals=[rate(g,c,4,e,ai,mi)[mask] for e in range(4)]
                        conditional.append(dict(group=group,candidate=c,method=method,alpha=1/den,subset=name,outer_n=int(mask.sum()),
                            denominator=[int(mask.sum())*n for n in (20000,5000,5000,5000)],
                            means=[float(x.mean()) if len(x) else None for x in vals],selected_descriptive=True))
                    same.append(dict(group=group,candidate=c,method=method,alpha=1/den,
                        same_U_minus_separate_A=[interval(rate(g,c,5,e,ai,mi)-rate(g,c,4,e,ai,mi)) for e in range(4)]))
            full.append(table(['群','候補','政策','FPR %','GM検出 %','GP検出 %','P検出 %','GM見逃し %','GP見逃し %','P見逃し %','LF0非発火 /50'],rows))
    dump(OUT/'all_policy_summary.json',records);dump(OUT/'paired_summary.json',paired)
    dump(OUT/'conditional_policy_summary.json',conditional);dump(OUT/'same_batch_comparison.json',same)
    (OUT/'ALL_POLICY_RESULTS.md').write_text('# 全政策・方式・水準の結果\n\n全50反復の平均率。F/Uは候補間共有を含む。C-cleanは実験参照。全件成立、欠測0。非発火は構造的上限で、実測0警報とは別。[全数値・SD・区間](all_policy_summary.json)、[対応得失](paired_summary.json)、[条件付き記述](conditional_policy_summary.json)、各反復の整数値は`recomputed/`。表の通常区間は多重性未調整、主18だけ別途調整。単位は%（差はpp）。\n'+'\n'.join(full)+'\n')
    gsummary=[]
    for g,group in enumerate(G):
        for c,cl in enumerate(CL):
            for j,gate in enumerate(spec.GATES):
                states=a['gate_states'][:,g,c,j]
                gsummary.append(dict(group=group,candidate=c,gate=gate,hold=int((states==1).sum()),unknown=int((states==2).sum()),pass_count=int((states==0).sum())))
            for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
                cell=np.zeros((3,4),int)
                np.add.at(cell,(a['gate_states'][:,g,c,2],labels[:,g,0,ai]),1)
                twoaxis.append(dict(group=group,candidate=c,alpha=1/den,gate='G-MC',counts=cell.tolist(),n=50,
                                    warning='V shared across 7 candidates; not 7 independent V batches'))
    dump(OUT/'gate_summary.json',gsummary);dump(OUT/'two_axis_table.json',twoaxis)
    vs=[]
    for g,group in enumerate(G):
        for ai,den in enumerate(spec.ALPHA_DENOMINATORS):
            for vi,name in enumerate(('clean','stale','mix_GM05','mix_GP05','mix_P02')):
                transition=np.zeros((4,4),int);np.add.at(transition,(labels[:,g,0,ai],labels[:,g,vi,ai]),1)
                vs.append(dict(group=group,alpha=1/den,variant=name,labels=np.bincount(labels[:,g,vi,ai],minlength=4).tolist(),
                    clean_to_variant=transition.tolist(),supported_but_E_exceeded=int(((labels[:,g,vi,ai]==0)&(evlabels[:,g,ai]==1)).sum()),
                    changed=int((labels[:,g,vi,ai]!=labels[:,g,0,ai]).sum())))
    dump(OUT/'validation_transitions.json',vs)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(13,6),sharey=True)
    det=primary[::2];fpr=primary[1::2];ys=np.arange(9)
    for ax,rr,title in zip(axs,(det,fpr),('Detection change (pp)','False-alarm change (pp)')):
        x=np.array([r['mean'] for r in rr])*100;lo=np.array([r['lower'] for r in rr])*100;hi=np.array([r['upper'] for r in rr])*100
        ax.errorbar(x,ys,xerr=[x-lo,hi-x],fmt='o',color='#286ca1',capsize=3);ax.axvline(0,color='grey',lw=.8);ax.set_title(title);ax.grid(axis='x',alpha=.2)
    axs[0].set_yticks(ys,[f"{p['group']} {p['mix_fault']}:{p['mix_count']} -> {p['evaluation']}" for p in det]);axs[0].invert_yaxis()
    fig.suptitle('G-MC minus U | W0, alpha=1% | 18-contrast family 95% intervals');fig.tight_layout();figsave('01_primary.png')
    fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for j,ax in enumerate(axs):
        val=(a['gate_states'][:,:,:,j]==1).mean(0)*100;im=ax.imshow(val,vmin=0,vmax=100,cmap='Blues')
        for y in range(6):
            for x in range(7):ax.text(x,y,f'{val[y,x]:.0f}',ha='center',va='center',color='white' if val[y,x]>60 else 'black')
        ax.set_xticks(range(7),CL,rotation=45,ha='right');ax.set_yticks(range(6),G);ax.set_title(spec.GATES[j]+' hold % (50/cell)')
    fig.tight_layout();figsave('02_gate_holds.png')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ai,ax in enumerate(axs):
        for p,lab,offset,color in [(0,'F',-.23,'#ce544b'),(1,'U clean',0,'#298c70'),(4,'G-MC clean',.23,'#286ca1')]:
            means=[rate(g,0,p,0,ai).mean()*100 for g in range(6)];ax.bar(np.arange(6)+offset,means,.23,label=lab,color=color)
        ax.axhline([1,.1][ai],c='black',ls=':',label='nominal alpha');ax.axhline([1.25,.125][ai],c='grey',ls='--',label='1.25 alpha')
        ax.set_xticks(range(6),G,rotation=35);ax.set_ylabel('False alarms (%)');ax.set_title(f'W0 alpha={[1,.1][ai]}%');ax.legend(fontsize=8)
    fig.tight_layout();figsave('03_source_fpr.png')
    fig,axs=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ai,ax in enumerate(axs):
        counts=np.array([np.bincount(labels[:,g,0,ai],minlength=4) for g in range(6)]);bottom=np.zeros(6)
        for j,lab in enumerate(LAB):ax.bar(range(6),counts[:,j],bottom=bottom,label=lab.replace('FPR-',''),color=COL[j]);bottom+=counts[:,j]
        ax.set_xticks(range(6),G,rotation=35);ax.set_title(f'Clean V -> old F W0, alpha={[1,.1][ai]}%');ax.set_ylabel('Independent episodes (50/group)')
    axs[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2);fig.tight_layout();figsave('04_v_labels.png')
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    for g,ax in enumerate(axs.flat):
        x=np.array([[(rate(g,c,4,e)-rate(g,c,1,e)).mean()*100 for e in (1,2,3)] for c in (2,4,6)])
        ax.imshow(x,cmap='RdBu',vmin=-90,vmax=90)
        for y in range(3):
            for j in range(3):ax.text(j,y,f'{x[y,j]:+.1f}',ha='center',va='center',color='white' if abs(x[y,j])>50 else 'black')
        ax.set_xticks(range(3),['GM05 eval','GP05 eval','P02 eval']);ax.set_yticks(range(3),['GM800 mix','GP800 mix','P800 mix']);ax.set_title(G[g])
    fig.suptitle('Cross-fault detection change: G-MC minus U (pp), W0 alpha=1%');fig.tight_layout();figsave('05_cross_fault.png')
    fig,axs=plt.subplots(1,2,figsize=(13,6),sharey=True)
    for ax,ri,title in zip(axs,(0,2),('Against mixed U','Against C-clean reference')):
        gain=[];loss=[]
        for p in det:
            g=G.index(p['group']);e=spec.FAULTS.index(p['evaluation'])+1;c=p['candidate'];cells=a['paired'][:,g,c,e,2,ri,0,0]
            gain.append(cells[:,1].mean()/50);loss.append(cells[:,2].mean()/50)
        ax.barh(ys,gain,color='#298c70',label='gained');ax.barh(ys,-np.array(loss),color='#ce544b',label='lost');ax.axvline(0,color='grey');ax.set_title(title);ax.set_xlabel('Paired records (%)');ax.legend()
    axs[0].set_yticks(ys,[f"{p['group']} {p['mix_fault']}:{p['mix_count']} -> {p['evaluation']}" for p in det]);axs[0].invert_yaxis();fig.tight_layout();figsave('06_gains_losses.png')
    fig,axs=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ai,ax in enumerate(axs):
        matrix=np.array([[(labels[:,g,v,ai]==0).sum() for v in range(5)] for g in range(6)])
        ax.imshow(matrix,vmin=0,vmax=50,cmap='YlGn');ax.set_xticks(range(5),['clean','stale','mix GM','mix GP','mix P'],rotation=25);ax.set_yticks(range(6),G)
        for g in range(6):
            for v in range(5):ax.text(v,g,'n/a' if v==1 and g<2 else str(matrix[g,v]),ha='center',va='center',color='white' if matrix[g,v]>30 else 'black')
        ax.set_title(f'V support episodes /50, alpha={[1,.1][ai]}%')
    fig.tight_layout();figsave('07_v_provenance.png')
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    for j,(g,c,e,title) in enumerate([(2,6,3,'K2_M P800 -> P02'),(4,2,2,'V05_L GM800 -> GP05')]):
        for ai in range(2):
            ax=axs[j,ai]
            for p,lab,off,color in [(1,'U',-.25,'#b7bcc1'),(4,'G-MC',0,'#286ca1'),(6,'C-clean',.25,'#298c70')]:
                ax.bar(np.arange(6)+off,[rate(g,c,p,e,ai,mi).mean()*100 for mi in range(6)],.25,label=lab,color=color)
            ax.set_xticks(range(6),[x.replace('−','-') for x in METHODS]);ax.set_title(f'{title} | alpha={[1,.1][ai]}%');ax.set_ylabel('Detection (%)');ax.legend(fontsize=8)
    fig.tight_layout();figsave('08_methods.png')

    holds=(a['gate_states']==1).sum(0)
    mixed=[int((a['gate_states'][:,:,1:,j]==1).sum()) for j in range(3)]
    clean=[int((a['gate_states'][:,:,0,j]==1).sum()) for j in range(3)]
    added=int(((a['gate_states'][:,:,1:,1]==0)&(a['gate_states'][:,:,1:,2]==1)).sum())
    released=int(((a['gate_states'][:,:,1:,1]==1)&(a['gate_states'][:,:,1:,2]==0)).sum())
    runtime=load('runtime_evidence.json');receipt=load('receipt.json');validation=load('validation.json')
    fresh=next(v for k,v in runtime['sessions'].items() if k.endswith('/launch.json') and v['mode']=='fresh')
    env=next(v for k,v in runtime['sessions'].items() if k.endswith('/environment.json'))
    summary=dict(mixed_hold=mixed,clean_hold=clean,MC_adds_over_M=added,MC_releases_M=released,
        unknown=int((a['gate_states']==2).sum()),same_batch_gate_disagreements=int((a['gate_states'][:,:,:,2]!=a['same_states']).sum()),
        primary=primary,validation=vs,run='9adfa7a8c6e0',fresh_wall_minutes=fresh['wall_seconds']/60,
        max_zip_bytes=max(x['bytes'] for x in receipt['archives']))
    dump(OUT/'report_summary.json',summary)
    ptable=table(['ID','群・混入→評価','U %','G-MC %','差 pp','族95%区間 pp'],[[p['id'],f"{p['group']} {p['mix_fault']} {p['mix_count']} → {p['evaluation']}",fmt(p['U']),fmt(p['GMC']),fmt(p['mean']),ci_text(p)] for p in primary])
    gate_table=table(['ゲート','混入保留 /1,800','すり抜け /1,800','clean誤保留 /300','UNKNOWN'],[[spec.GATES[j],mixed[j],1800-mixed[j],clean[j],0] for j in range(3)])
    clean_table=table(['群','G-MC clean保留 /50','6群同時片側95%上限'],[[r['group'],r['held_inconsistent'],fmt(r['upper'])+'%'] for r in load('clean_gate_summary.json') if r['gate']=='G-MC'])
    vtable=table(['群','α','V支持 /50','V超過 /50','V未解決 /50','E旧F平均FPR','V支持→E超過'],[[v['group'],f"{v['alpha']*100:g}%",*v['labels'][:3],fmt(rate(G.index(v['group']),0,0,0,[.01,.001].index(v['alpha'])).mean())+'%',v['supported_but_E_exceeded']] for v in vs if v['variant']=='clean'])
    stress=table(['α','V副版','対象エピソード','cleanから変更','支持なのにE超過'],[[f'{100/den:g}%',variant,200 if variant=='stale' else 300,
        sum(v['changed'] for v in vs if v['alpha']==1/den and v['variant']==variant and (variant!='stale' or not v['group'].startswith('B_'))),
        sum(v['supported_but_E_exceeded'] for v in vs if v['alpha']==1/den and v['variant']==variant)] for den in (100,1000) for variant in ('stale','mix_GM05','mix_GP05','mix_P02')])
    axes_rows=[]
    for g,group in enumerate(G):
        hold=a['gate_states'][:,g,6,2]>0
        axes_rows.append([group,int(hold.sum())]+[int((hold&(labels[:,g,0,0]==j)).sum()) for j in range(3)]+[fmt(rate(g,6,4,0).mean())+'%',fmt(rate(g,6,4,3).mean())+'%'])
    axis_table=table(['群（P800候補）','MC保留 /50','うちFPR支持','超過','未解決','政策FPR','P02検出'],axes_rows)
    same_table=table(['群','A/Uゲート不一致 /350','cleanでの不一致 /50'],[[group,int((a['gate_states'][:,g,:,2]!=a['same_states'][:,g]).sum()),int((a['gate_states'][:,g,0,2]!=a['same_states'][:,g,0]).sum())] for g,group in enumerate(G)])
    keyrows=[]
    for g,c,e in [(2,6,3),(4,2,2)]:
        for p in (0,1,4,6):keyrows.append([G[g],CL[c],POL[p],fmt(rate(g,c,p,0).mean()),fmt(rate(g,c,p,e).mean())])
    keytable=table(['群','候補','政策','FPR %','指定故障検出 %'],keyrows)
    inactive_table=table(['群','α','F LF0非発火 /50','clean U非発火 /50','W0 clean U P検出 %','LF0 clean U P検出 %','LF0 clean U FPR %'],[[group,f'{100/den:g}%',int((a['inactive'][:,g,0,ai]==1).sum()),int((a['inactive'][:,g,1,ai]==1).sum()),fmt(rate(g,0,1,3,ai,0).mean()),fmt(rate(g,0,1,3,ai,5).mean()),fmt(rate(g,0,1,0,ai,5).mean())] for ai,den in enumerate((100,1000)) for g,group in enumerate(G)])
    report=f'''# PR-0009：混入保留と旧校正のFPR根拠を分けた独立確認

返却分析 v1／2026-10-18。run `9adfa7a8c6e0`。科学条件の正本は[PR-0009](../../../..//audits/prereg/PR-0009_two_axis_independent_confirmation.md)。元ZIPは[Pr0009](../../../../Pr0009/)に保持。新規生成・融合則変更・論文改訂なし。

## 1. 結論

**事前指定した8セルの検出回復と、1セルの交差検出損失が、新しい50反復でも確認された。だが、混入を保留して旧閾値へ戻すことと、その旧閾値が現在の源で適切な誤警報率を持つことは別だった。** 二軸表示がこの違いを実測で分離した。

- G-MCは混入{mixed[2]}/1,800候補を保留し、{1800-mixed[2]}候補を通した。clean誤保留は{clean[2]}/300、UNKNOWNは0。候補間共有があるので1,800件を独立反復とみなさない。
- K2_M・P02 800混入では、G-MC−Uの検出差は{fmt(primary[14]['mean'])} pp、FPR差は{fmt(primary[15]['mean'])} pp。両方の主調整区間が正。回復した検出を、同じ実測FPRでの優越とは解釈しない。
- V05_L・負ゲイン800混入では、正ゲイン検出を{fmt(-primary[16]['mean'])} pp失った。混入への防御が他故障の感度を一律に改善するわけではない。
- 独立したclean Vは旧FのFPRについて支持・超過・未解決を区別した。古い源の正常データをVへ流用すると、現在条件での超過を支持と誤表示し得る。正常という属性だけでなく、**現在の条件に対応する来歴**が必要である。

これは既知条件を選んだ後に固定した独立確認であり、過去25反復との合算でも、未知源一般への保証でもない。

## 2. 受領と再現性

12 ZIP＝10シャード＋2完了索引を照合した。2索引の科学データは同一で、後の索引には完了後のexisting再起動ログが加わる。50反復00–49は欠落・重複なし。最大ZIPは{summary['max_zip_bytes']:,} bytesで、全て十進1 GB未満。

全{receipt['indexed_member_occurrences']:,}索引メンバー出現のCRC／SHA-256／長さ、固定ソース26件、カード・仕様、全31,039,200保存記録を照合。100参照モデル・100旧校正・2,100更新校正・A/U各2,100候補を再構成し、評価E **1,008,000,000判定**、V **82,800,000判定**を別成分式から再計算して一致。全carryの整数集計と保存出力表も一致した。数値未成立・欠測・科学的除外は0。

実行GPUは{env['gpu']}（A100ではない）。fresh RUNの壁時計は{fresh['wall_seconds']/60:.2f}分、保存active時間は{runtime['active']['seconds']/60:.2f}分。実機CPU/GPU点検72件は合格し、科学標本には含めていない。ローカルでは保存統計だけを処理し、原時系列を再生成していない。

[受領照合](receipt.json)／[全件再計算](validation.json)／[実行ログ抜粋](runtime_evidence.json)／[解析範囲](PROTOCOL.md)。同一エージェントの自己監査で、第三者の独立レビューではない。実ファイルの到着は確認できたが、実VM消失からの復旧やブラウザ資源解放の実機保証とは別である。

## 3. 比較の読み方

50独立外側反復。固定LF方向と6方式（W0/WP/WG/W+/W−/LF0）、α=1%・0.1%。候補監査A、更新cal U、旧Fを調べるV、最終評価Eは独立な役割として取得。各役割内の候補は対応比較のためプールを共有する。歴史的開発Dの閾値を再調整していない。

G-old/G-M/G-MCは保留時に旧F、非保留時に候補Uを使う。Vラベルは警報を変更しない。C-cleanは同じ源の清浄Uを与えた実験参照で、混入を自動除去した運用方式や性能上限ではない。

主18対比はW0・1%のG-MC−U。区間は反復差のdf49 t区間にBonferroniを適用した族95%の近似区間。表の率は%、差はpp。副比較と条件付き表は記述的であり、主比較の追加成功へ昇格させない。

## 4. 主比較：利益と損失は両方再現

{ptable}

![主18比較](figures/01_primary.png)

H1の8つの回復セルは全て区間下端>0、交差損失セルP17は上端<0。H2のP15/P16もともに正で、検出回復とFPR増加の共存を支持する。FPR差は回復と常に同方向ではなく、P10とP18は低下、P12はゼロをまたいで未解決である。検出だけの一括「成功」にはまとめない。

全50の値・SD・中央値は[主結果](primary18.json)、差を政策率から別計算した値は[独立算術照合](primary_independent.json)。

## 5. 候補ゲート：すり抜け、誤保留、非単調性

{gate_table}

![ゲート保留率](figures/02_gate_holds.png)

G-MCはG-Mに比べ、混入候補を{added}件追加保留した一方、G-Mが止めた{released}件を通した。MCの閾値も別に校正されており、単純な上位包含ではない。

{clean_table}

cleanの観測保留率は低いが、50反復では全群の真の誤保留率≤5%を保証できない。0/50でも6群同時上限は約9.13%。群をまとめた2/300等を使って、この群別の保証不足を隠さない。

全42候補×3ゲートは[ゲート表](gate_summary.json)、生状態は[50反復状態](gate_results.json)。

## 6. 保留後の旧F：二軸を分ける意味

{vtable}

![旧Fの根拠ラベル](figures/04_v_labels.png)

VのSUPPORTEDは**旧F・W0のFPRが1.25α以下という限定的な根拠**である。装置全体の正常性・故障感度・更新先Uを認証しない。EXCEEDEDとUNRESOLVEDも分ける。特にα=0.1%では正常V=20,000件の支持分解能が低く、未解決を不適合と読み替えない。

代表としてP800候補を示す。Vは群・反復ごとに一つで、7候補へ共有される。同じVを7回の独立根拠と数えない。

{axis_table}

![源変化とFPR](figures/03_source_fpr.png)

保留は「候補Uを採用しない」という判断で、Fが今も適切という肯定的根拠ではない。K2_Mでは混入を止めても旧FのFPR超過が残る。一方、低いFPRやSUPPORTEDだけで故障感度を保証することもできない。全候補の[二軸クロス表](two_axis_table.json)、保留／非保留／V根拠別の[条件付き政策表](conditional_policy_summary.json)を分母付きで保持した。後者は選択後の記述値である。

## 7. 検出の回復、交差損失、警報配分

{keytable}

![同一記録の拾得と喪失](figures/06_gains_losses.png)

左は混入Uに対する拾得／喪失、右はC-clean参照に対するもの。同じ正の純差でも両方向の入替があり得る。正常記録の拾得は追加誤警報、喪失は抑制誤警報として別計上した。

![混入×評価の交差9通り](figures/05_cross_fault.png)

800件混入の3×3表は、同じ故障を回復させる政策が別の故障を見落とす場合を示す。全160/800・全方式・両αについて[対応得失](paired_summary.json)を保存。評価で最良配分を選択せず、W0を主表示のまま維持した。

![6方式の代表条件](figures/08_methods.png)

配分とLF0の違いは残るが、この図は同じ名目αでの政策比較であり、完全に同じ実測FPRでのROC優越を示さない。全条件のFPR・検出・見逃し・正負差は[全政策表](ALL_POLICY_RESULTS.md)と[数値表](all_policy_summary.json)を参照。

## 8. Vの来歴破綻：正常でも古ければ別の根拠

{stress}

![V来歴と支持数](figures/07_v_provenance.png)

STALE-Vは源変化4群の200エピソードだけに適用。MIX-Vは現在Vの固定5%を別故障プールへ置換した意図的な誤来歴で、正常FPRの正確二項区間とは呼ばない。clean→副版の全ラベル遷移と、支持されたVに対し独立EがEXCEEDEDとなる衝突を[遷移表](validation_transitions.json)へ保存した。Eも有限標本であり、Eを真のFPRそのものとは扱わない。

主1%では、STALE-Vによる支持と独立Eの超過が79/200エピソードで同居した。対して今回のMIX-Vは支持数を増やさず、主に支持を失わせる方向だった。この二つを「どちらも危険」という一語で潰さず、古い正常条件への誤移送と、混入による評価対象そのものの変化を区別する。

H3はラベルが来歴誤指定で変わるという記述的確認。支持数が増えたことを改善と呼ばず、減ったことも単独で正常性検査の成功としない。これは取得根拠の必要性を示すストレス試験であり、その根拠の現場取得法は未検証である。

## 9. 同じUで監査する対照と、LF0の非発火

{same_table}

Aを分けた主政策とU自身を監査した副対照では計{summary['same_batch_gate_disagreements']}候補で判断が異なった。独立化を一律の性能改善とは解釈しない。全方式・α・候補でsame-U−separate-AのFPR・3故障検出差を[副比較](same_batch_comparison.json)へ残した。

W0・1%の記述例では、same-UはB_L・GM160の負ゲイン検出を分離版より1.5004 pp下げ、一方K2_M・GM160では1.2860 pp上げた。同じK2_M条件のFPRも0.2022 pp上がる。これらは全セルを見た後の例示であり、新たな主検定や方式選択には使わない。

{inactive_table}

LF0の順位上限による構造的非発火と、有限Eでたまたま0警報だった場合を区別した。FPRが低いだけでは有用な計器にならず、低αでは検出感度と併記する必要がある。W0のV根拠はLF0へ転用しない。

K2_M・clean U・0.1%ではLF0が46/50反復で構造的に非発火となり、私的入力検出は7.326%。W0は93.886%だった。ただし実測FPRも異なるため、これを同一実測FPRでの性能差とは呼ばない。

## 10. 今回確かめられたことと、残る判断

今回の確認は「不審な更新を止める」ことを否定しない。その効用と、停止後に使う校正の根拠を分けて測る必要を示した。既知6群において、混入保留・検出回復・交差損失・旧FのFPR超過を同じ固定実験で確認できた。

次に進めるなら、推奨は新しい融合式の追加より、**旧Fが超過／未解決のとき、現行条件の正常校正資源をどう取得し、得られるまで何を出力するか**を定めること。これは次の設計候補で、今回の結果から運用政策へ自動昇格しない。今回のVは外部正常来歴の理想化であり、未知源・非ガウス・A/U間の状態変化・連続監視・実機は未検証。

原ZIP・固定コード・過去の結果・統合論文・TeX/PDFは保持した。生成コードは[受領／再計算](../../../scripts/analyze_pr0009_return.py)と[報告生成](../../../scripts/report_pr0009_two_axis.py)。
'''
    # Script links from analysis/run directory go two levels up to numerics, not three.
    report=report.replace('../../../scripts/','../../scripts/').replace('../../../..//audits/','../../../audits/').replace('../../../../Pr0009/','../../../Pr0009/')
    (OUT/'PR0009_TWO_AXIS_REPORT_JA.md').write_text(report)
    print('REPORT',OUT/'PR0009_TWO_AXIS_REPORT_JA.md')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('primary','validation')},ensure_ascii=False))
if __name__=='__main__':main()
