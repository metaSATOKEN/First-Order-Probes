# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""PR-0010 exhaustive descriptive summaries and fixed scientific figures."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pr0010-matplotlib')
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_pr0010_mixture import OUT,ROOT,G,EV,AL,METHODS,PATTERNS,recipes,load,dump

def stats(x):
    x=np.asarray(x,float);mean=x.mean();sd=x.std(ddof=1);w=t.ppf(.975,49)*sd/np.sqrt(50)
    return dict(mean=mean,sd=sd,lower=mean-w,upper=mean+w,n=50)
def exchange(j):
    # Left minus right: both, left-only, right-only, neither.
    n=j.sum((-2,-1));return np.stack([j[...,1:,1:].sum((-2,-1)),j[...,1:,0].sum(-1),j[...,0,1:].sum(-1),j[...,0,0]],-1)/n[...,None]
def pct(x):return f'{100*x:.4f}'
def heat(ax,mat,rows,cols,title,lim=None):
    mat=np.asarray(mat);v=max(abs(mat.min()),abs(mat.max()),.01) if lim is None else lim
    im=ax.imshow(mat,cmap='BrBG',vmin=-v,vmax=v,aspect='auto')
    ax.set_xticks(range(len(cols)),cols);ax.set_yticks(range(len(rows)),rows);ax.set_title(title)
    for (i,j),x in np.ndenumerate(mat):ax.text(j,i,f'{x:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(x)>.55*v else '#17212c')
    return im

def main():
    validation=load(OUT/'validation.json');assert validation['status']=='PASS_SAVED_DATA_RECONSTRUCTION'
    cs,lays,mp,edges=recipes();keys=['mean','sd','T','bf','varparts','Tparts','states','scores','cut','rank_cut','top','ties','joint_F','joint_C','joint_methods','joint_parent']
    aa={k:[] for k in keys}
    for r in range(50):
        with np.load(OUT/'recomputed'/f'rep_{r:02d}.npz') as z:
            for k in keys:aa[k].append(z[k])
    a={k:np.stack(v) for k,v in aa.items()};del aa
    # shape R,G,C,E,A,M; saved full per-replicate branch tables remain canonical.
    u=a['joint_F'][...,1:].sum((-2,-1))/np.array([20000,5000,5000,5000])[None,None,None,:,None,None]
    f=a['joint_F'][...,1:,:].sum((-2,-1))/np.array([20000,5000,5000,5000])[None,None,None,:,None,None]
    c0=a['joint_C'][...,1:,:].sum((-2,-1))/np.array([20000,5000,5000,5000])[None,None,None,:,None,None]
    policies=np.stack([f,u]+[np.where((a['states'][...,gi]>0)[...,None,None,None],f,u) for gi in range(3)],-1)
    assert np.all(a['states']<2);assert np.isfinite(policies).all()
    fx=exchange(a['joint_F']);cx=exchange(a['joint_C']);px=exchange(a['joint_parent']);mx=exchange(a['joint_methods'])
    assert np.allclose(f-u,fx[...,1]-fx[...,2],atol=2e-14,rtol=0)
    assert np.allclose(c0-u,cx[...,1]-cx[...,2],atol=2e-14,rtol=0)
    rows=[];gates=[];parentrows=[];mechanisms=[];asym=[];methodrows=[]
    table=['# 全条件表：PR-0010混入量・異種混入','',
      '全て50外側反復の平均。割合は%。列Nは正常FPR、GM/GP/Pは各故障の検出率。主37＋成分除去の実体20、別名4はrecipes.jsonで参照する。全50値・枝対応はrecomputed/のNPZ。0警報と欠測を混同しない。','']
    for gi,g in enumerate(G):
        table += [f'## {g}：候補監査','', '| 候補 | n | G-old HOLD | G-M HOLD | G-MC HOLD | LF0非発火 1% / 0.1% |','|---|---:|---:|---:|---:|---:|']
        for ci,cc in enumerate(cs):
            hold=(a['states'][:,gi,ci]==1).sum(0);off=(a['rank_cut'][:,gi,ci]==9999).sum(0)
            gates.append(dict(group=g,candidate=cc['id'],counts=cc['counts'],holds=hold,pass_count=50-hold,unknown=[0,0,0],inactive=off))
            table.append(f'| {cc["id"]} | {cc["n"]} | {hold[0]} | {hold[1]} | {hold[2]} | {off[0]} / {off[1]} |')
            for ri,role in enumerate(['A','U']):
                b=a['bf'][:,gi,ci,ri,:,0];den=abs(b[:,0])+abs(b[:,1]);eta=np.divide(abs(b[:,0]+b[:,1]),den,out=np.full(50,np.nan),where=den>0);eta=1-eta
                mechanisms.append(dict(group=g,candidate=cc['id'],role=role,mean=a['mean'][:,gi,ci,ri].mean(0),sd=a['sd'][:,gi,ci,ri].mean(0),
                  bf=a['bf'][:,gi,ci,ri].mean(0),eta_mean=np.nanmean(eta) if np.isfinite(eta).any() else None,
                  eta_denominator_mean=den.mean(),eta_per_rep=eta,between_fraction=(a['varparts'][:,gi,ci,ri,1]/a['sd'][:,gi,ci,ri,:2]**2).mean(0),
                  T_mean=a['T'][:,gi,ci,ri].mean(0),Tparts=a['Tparts'][:,gi,ci,ri].mean(0)))
            for ai,al in enumerate(AL):
                for mi,met in enumerate(METHODS):
                    for ei,ev in enumerate(EV):
                        rates=policies[:,gi,ci,ei,ai,mi]
                        gr=[]
                        for gatei in range(3):
                            h=(a['states'][:,gi,ci,gatei]>0);diff=f[:,gi,ci,ei,ai,mi]-u[:,gi,ci,ei,ai,mi]
                            delta=rates[:,gatei+2]-rates[:,1]
                            assert abs(delta.mean()-(h.mean()*diff[h].mean() if h.any() else 0))<2e-14
                            gr.append(dict(hold=int(h.sum()),delta_U=stats(delta),delta_F=stats(rates[:,gatei+2]-rates[:,0]),
                              delta_C0=stats(rates[:,gatei+2]-c0[:,gi,ci,ei,ai,mi]),
                              gained_over_U=stats(h*fx[:,gi,ci,ei,ai,mi,1]),lost_from_U=stats(h*fx[:,gi,ci,ei,ai,mi,2]),
                              hold_conditional_F_minus_U=diff[h].mean() if h.any() else None))
                        rows.append(dict(group=g,candidate=cc['id'],alpha=al,method=met,evaluation=ev,
                          planned_each=20000 if ei==0 else 5000,available_replicates=50,
                          rates={p:stats(rates[:,pi]) for pi,p in enumerate(['F','U','G-old','G-M','G-MC'])},
                          C0=stats(c0[:,gi,ci,ei,ai,mi]),U_minus_C0=stats(u[:,gi,ci,ei,ai,mi]-c0[:,gi,ci,ei,ai,mi]),
                          U_only_C0=stats(cx[:,gi,ci,ei,ai,mi,2]),C0_only_U=stats(cx[:,gi,ci,ei,ai,mi,1]),
                          gate_effects=gr))
                        if mi==0:
                            ex=mx[:,gi,ci,ei,ai]
                            methodrows.append(dict(group=g,candidate=cc['id'],alpha=al,evaluation=ev,
                              W0_minus_LF0=stats(ex[:,1]-ex[:,2]),W0_only=stats(ex[:,1]),LF0_only=stats(ex[:,2])))
                    ph=a['joint_F'][:,gi,ci,2,ai,mi].sum(-2);mh=a['joint_F'][:,gi,ci,1,ai,mi].sum(-2)
                    gb=np.array([(j&3)>0 for j in range(8)])
                    dg=(ph[:,gb].sum(1)-mh[:,gb].sum(1))/5000;dp=(ph[:,4]-mh[:,4])/5000
                    total=u[:,gi,ci,2,ai,mi]-u[:,gi,ci,1,ai,mi]
                    assert np.allclose(dg+dp,total,atol=2e-14,rtol=0)
                    asym.append(dict(group=g,candidate=cc['id'],alpha=al,method=met,GP_minus_GM=stats(total),G_any=stats(dg),P_only=stats(dp)))
        for ai,al in enumerate(AL):
            for mi,met in enumerate(METHODS):
                table += ['',f'### {met} / α={al:g}','', '| 候補 | 政策 | N | GM | GP | P |','|---|---|---:|---:|---:|---:|']
                for ci,cc in enumerate(cs):
                    for pi,pol in enumerate(['F','U','G-old','G-M','G-MC']):
                        v=policies[:,gi,ci,:,ai,mi,pi].mean(0)
                        table.append('| '+cc['id']+' | '+pol+' | '+' | '.join(map(pct,v))+' |')
        for ei,e in enumerate(edges):
            pi,ci=e['parent'],e['child']
            for ai,al in enumerate(AL):
                for mi,met in enumerate(METHODS):
                    for ev,outcome in enumerate(EV):
                        delta=u[:,gi,pi,ev,ai,mi]-u[:,gi,ci,ev,ai,mi]
                        assert np.allclose(delta,px[:,gi,ei,ev,ai,mi,1]-px[:,gi,ei,ev,ai,mi,2],atol=2e-14,rtol=0)
                        parentrows.append(dict(group=g,parent=cs[pi]['id'],child_alias=e['id'],child=cs[ci]['id'],alpha=al,method=met,evaluation=outcome,
                          delta=stats(delta),gained=stats(px[:,gi,ei,ev,ai,mi,1]),lost=stats(px[:,gi,ei,ev,ai,mi,2]),
                          gate_hold_difference=((a['states'][:,gi,pi]==1).astype(int)-(a['states'][:,gi,ci]==1)).sum(0),
                          W0_cut_difference=(a['cut'][:,gi,pi,ai,0]-a['cut'][:,gi,ci,ai,0]).mean(0)))
                        effects=[]
                        for gatei in range(3):
                            hp=a['states'][:,gi,pi,gatei]>0;hc=a['states'][:,gi,ci,gatei]>0
                            base=px[:,gi,ei,ev,ai,mi].copy()
                            base[hp&~hc]=fx[:,gi,ci,ev,ai,mi][hp&~hc]
                            base[~hp&hc]=fx[:,gi,pi,ev,ai,mi][:,[0,2,1,3]][~hp&hc]
                            ff=f[:,gi,pi,ev,ai,mi]
                            base[hp&hc]=np.column_stack([ff,np.zeros(50),np.zeros(50),1-ff])[hp&hc]
                            diff=policies[:,gi,pi,ev,ai,mi,gatei+2]-policies[:,gi,ci,ev,ai,mi,gatei+2]
                            assert np.allclose(base[:,1]-base[:,2],diff,rtol=0,atol=2e-14)
                            effects.append(dict(gate=['G-old','G-M','G-MC'][gatei],difference=stats(diff),gained=stats(base[:,1]),lost=stats(base[:,2])))
                        parentrows[-1]['gate_policy_contrasts']=effects
    dump(OUT/'results.json',rows);dump(OUT/'gate_summary.json',gates);dump(OUT/'mechanisms.json',mechanisms)
    dump(OUT/'parent_contrasts.json',parentrows);dump(OUT/'gain_asymmetry.json',asym)
    dump(OUT/'method_comparisons.json',methodrows)
    (OUT/'ALL_MIXTURE_RESULTS.md').write_text('\n'.join(table)+'\n')
    # Compact arrays retain all 50 paired rates, including policy harms and misses.
    np.savez_compressed(OUT/'summary_arrays.npz',rates=policies,C0=c0,parent_exchange=px,method_exchange=mx)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);figures=[]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors=list(plt.get_cmap('tab10').colors[:9]);kinds=list(PATTERNS);dose=[0,80,160,400,800];x=100*np.array(dose)/15999
    def save(fig,name):
        fig.savefig(figdir/name,dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig);figures.append(name)
    for gatei,gate in enumerate(['G-old','G-M','G-MC']):
        fig,axs=plt.subplots(2,3,figsize=(13,7.8),sharex=True,sharey=True)
        for gi,ax in enumerate(axs.flat):
            for ki,kind in enumerate(kinds):
                ids=[0]+[mp[f'{kind}_n{n}'] for n in dose[1:]]
                y=(a['states'][:,gi,ids,gatei]==1).mean(0)*100
                ax.plot(x,y,'o-',lw=1.25,ms=3,color=colors[ki],label=kind)
            ax.set_title(G[gi]);ax.set_ylim(-3,103);ax.grid(alpha=.15)
            ax.set_xlabel('Contamination (%)');ax.set_ylabel('HOLD batches (%)')
        fig.legend(*axs[0,0].get_legend_handles_labels(),ncol=5,loc='lower center',fontsize=9,frameon=False)
        fig.suptitle(f'{gate}: dose and composition, independent A batches\n50 outer replicates; clean is shared once; fixed gate thresholds',fontsize=13)
        fig.tight_layout(rect=[0,.10,1,.92]);save(fig,f'0{gatei+1}_dose_{gate.replace("-","")}.png')
    fig,axs=plt.subplots(2,3,figsize=(12.6,7.8),sharex=True,sharey=True)
    for gi,ax in enumerate(axs.flat):
        for kind,col in [('GMGP31','#367daf'),('GMGP11','#24856d'),('GMGP13','#bc6d34')]:
            ids=[mp[f'{kind}_n{n}'] for n in dose[1:]];b=np.take(a['bf'][:,gi],ids,axis=1)[:,:,0,:,0]
            assert b.shape==(50,4,3)
            den=abs(b[:,:,0])+abs(b[:,:,1])
            eta=1-abs(b[:,:,0]+b[:,:,1])/den
            ax.plot(x[1:],eta.mean(0),'o-',label=kind,color=col)
        ax.set_title(G[gi]);ax.set_ylim(0,1);ax.grid(alpha=.15);ax.set_xlabel('Contamination (%)');ax.set_ylabel('Mean cancellation ratio eta_M')
    fig.legend(*axs[0,0].get_legend_handles_labels(),ncol=3,loc='lower center',frameon=False)
    fig.suptitle('Cancellation of signed replacement contributions in A\nRatio includes the removed normal records; not a new gate or normality proof',fontsize=13)
    fig.tight_layout(rect=[0,.07,1,.92]);save(fig,'04_cancellation.png')
    fig,axs=plt.subplots(2,3,figsize=(13,8.0))
    show=['GM','GMGP31','GMGP11','GMGP13','GP','GMGPP112']
    for gi,ax in enumerate(axs.flat):
        ids=[mp[f'{k}_n800'] for k in show];parts=np.take(a['Tparts'][:,gi],ids,axis=1)[:,:,0,:,0].mean(0)
        assert parts.shape==(6,2)
        xx=np.arange(len(show));ax.bar(xx-.17,parts[:,0],.34,label='Mean contribution',color='#327ba4');ax.bar(xx+.17,parts[:,1],.34,label='SD contribution',color='#b96842')
        ax.plot(xx,parts.sum(1),'ko',ms=3,label='Total delta T_M');ax.axhline(0,color='gray',lw=.7)
        ax.set_xticks(xx,show,rotation=45,ha='right');ax.set_title(G[gi]);ax.set_ylabel('Change in signed T_M')
    fig.legend(*axs[0,0].get_legend_handles_labels(),ncol=3,loc='lower center',frameon=False)
    fig.suptitle('Mean shift and self-normalization are different mechanisms\nA, n=800; exact symmetric decomposition relative to clean A',fontsize=13)
    fig.tight_layout(rect=[0,.06,1,.92]);save(fig,'05_mean_sd.png')
    # Preselected component additions at both anchor doses, all sources and outcomes.
    def edgeid(kind,n,keep):return next(i for i,e in enumerate(edges) if e['id']==f'{kind}_n{n}_keep{keep}')
    additions=[('GMP11','0','P to GM'),('GPP11','1','P to GP'),('GMGPP112','01','P to balanced G')]
    fig,axs=plt.subplots(1,2,figsize=(12.2,10.8));labels=[g+' / '+label for g in G for _,_,label in additions]
    matrices=[]
    for n in [160,800]:
        mat=[]
        for gi,g in enumerate(G):
            for kind,keep,_ in additions:
                ei=edgeid(kind,n,keep);mat.append(100*(px[:,gi,ei,:,0,0,1]-px[:,gi,ei,:,0,0,2]).mean(0))
        matrices.append(np.array(mat))
    lim=max(abs(np.concatenate(matrices)).max(),1)
    for ax,n,mat in zip(axs,[160,800],matrices):heat(ax,mat,labels,['FPR','GM detect','GP detect','P detect'],f'Parent total n={n}',lim)
    fig.suptitle('Add P while preserving the gain record IDs and positions\nW0 U, alpha=1%; parent minus component-removed control (pp); positive FPR is a cost',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.93]);save(fig,'06_private_addition.png')
    fig,axs=plt.subplots(1,2,figsize=(12,8.8));focus=['GMGP11','GMP11','GPP11','GMGPP112'];labels=[g+' / '+k for g in G for k in focus]
    m1=[];m2=[]
    for gi,g in enumerate(G):
        for k in focus:
            ci=mp[k+'_n800'];m1.append(100*(u[:,gi,ci,:,0,0]-c0[:,gi,ci,:,0,0]).mean(0));m2.append(100*(policies[:,gi,ci,:,0,0,4]-u[:,gi,ci,:,0,0]).mean(0))
    heat(axs[0],m1,labels,['FPR','GM','GP','P'],'Mixed U minus clean C0')
    heat(axs[1],m2,labels,['FPR','GM','GP','P'],'G-MC policy minus mixed U')
    fig.suptitle('Calibration harm and fallback effects must be read separately\nW0, alpha=1%, n=800; percentage points; normal column is not a benefit scale',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.93]);save(fig,'07_policy_tradeoffs.png')
    fig,axs=plt.subplots(1,2,figsize=(13,7.8));idlist=[0]+[mp[k+'_n800'] for k in kinds];labels=['clean']+kinds
    off=(a['rank_cut'][:,:,idlist,1]==9999).sum(0);im=axs[0].imshow(off,aspect='auto',vmin=0,vmax=50,cmap='Reds')
    for (i,j),v in np.ndenumerate(off):axs[0].text(j,i,str(v),ha='center',va='center',fontsize=9,color='white' if v>25 else 'black')
    axs[0].set_xticks(range(10),labels,rotation=65,ha='right');axs[0].set_yticks(range(6),G);axs[0].set_title('LF0 structurally inactive / 50')
    delta=100*(u[:,:,idlist,3,1,0]-u[:,:,idlist,3,1,1]).mean(0)
    heat(axs[1],delta,G,labels,'W0 minus LF0, private detection (pp)');axs[1].tick_params(axis='x',rotation=65)
    fig.suptitle('Low alpha = 0.1%, mixed U; n=800 except clean\nFinite-rank nonfire remains an outcome; realized FPRs are not matched',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.92]);save(fig,'08_low_alpha.png')
    dump(OUT/'figure_manifest.json',figures)
    dump(OUT/'summary_validation.json',dict(status='PASS_SUMMARY_IDENTITIES',rows=len(rows),gate_rows=len(gates),parent_rows=len(parentrows),asymmetry_rows=len(asym),figures=len(figures),unknown=0,unavailable=0,excluded=0))
    ix={(r['group'],r['candidate'],r['alpha'],r['method'],r['evaluation']):r for r in rows}
    gx={(r['group'],r['candidate']):r for r in gates};bx={(r['group'],r['candidate'],r['role']):r for r in mechanisms}
    tx={(r['group'],r['child_alias'],r['alpha'],r['method'],r['evaluation']):r for r in parentrows}
    doc=['# PR-0010：混入量・異種混入の保存データ分析','',
      '2026-10-18／保存PR-0009 run `9adfa7a8c6e0`、50外側反復。D-Aを採択した[研究カードPR-0010](../../../audits/prereg/PR-0010_mixture_dose_saved_data.md)に基づく事後的ストレス分析。新規波形生成・再学習・配分最適化・論文改訂はない。','',
      '## 結果の要旨','',
      '**相殺は確認できた。ただし「Mが相殺する」「混入を保留できる」「保留後の計器が妥当」は三つの別問題だった。**','',
      '- 正負半々・総混入800では、G-Mの保留は全6群で0〜1/50まで落ちた。一方G-MCは5群で50/50を保留し、CR側に残る応答が働いた。高速源K200_Lだけは10/50で、40/50がすり抜けた。相殺比ηMの平均は0.8297〜0.9735。','',
      '- 同じGM200・GP200へP400を足すと、K200_LのG-MC保留は2/50→50/50になった。しかし混入Uをそのまま使ったP検出は62.9604%→19.8960%。P追加は監査を助ける一方、採用された校正の検出を損なう。W0・1%では、このP追加によるP検出の純損失が全6群で43.0644〜77.5860 ppあった。','',
      '- 保留後の旧Fも自動的には適切でない。V05_LでGM40・GP40へP80を足すとG-MCは0/50→50/50保留となり、政策全体の負／正ゲイン検出はそれぞれ14.5348／14.5976 pp低下した。K2_Mの同じ比較では検出が増える一方、政策FPRが1.2919%→3.6213%へ上昇した。','',
      '- 三種混合800のLF0はα=0.1%で全6群・全50反復が構造的非発火。W0は発火を残すが、同条件のP検出は約2〜3%まで落ちており、順位上端の問題だけを直せば混入校正の害が解消するわけではない。','',
      '## 1. 問いと比較の単位','',
      '校正バッチに負5%ゲイン・正5%ゲイン・私的入力P02の記録が混在すると、Aの混入監査と、別Uから作った閾値の性能がどう変わるかを調べた。一つの時系列に複合故障を作用させた実験ではない。P02の2%は私的信号の分散比で、バッチの混入率ではない。','',
      'AとUは独立原記録・独立保存順序。参照Rと旧校正Oは固定、Aで3ゲート、UでW0/LF0・2αの校正を確定・保存した後、共通Eで評価した。Eは正常20,000・各故障5,000件／群・反復。混入ラベルは候補の構成・事後の説明だけに使い、ゲートや閾値選択へ渡していない。Vは新Uや選択政策を認証しない。','',
      'バッチ総数15,999、混入80/160/400/800件（約0.5/1/2.5/5%）。単独3・正負3:1/1:1/1:3・GM/PとGP/P各1:1・GM/GP/P=1:1:2の9組成、cleanを加えて37主条件。成分除去24設計IDのうち4は同一配置の別名で、実体57配置とした。別名や用量を独立反復へ水増ししない。','',
      '基準源BはOU相関率κ=20 s⁻¹・源分散比1。M/L窓は2.048/8.192秒、K2_Mはκ=2、K200_Lはκ=200、V05_L/V15_Lは源分散比0.5/1.5で他の基準設定を保つ。τi=0.04秒・τj=0.01秒、標本間隔0.002秒。読出し雑音の絶対量は基準に固定しており、源変化後もSNRを揃えた比較ではない。','',
      '本文の率は50反復平均（%）、差はpercentage points（pp）。すべての源群・組成・方式・水準の値は[全条件表](ALL_MIXTURE_RESULTS.md)、対応95% t区間とSDは[結果JSON](results.json)にある。区間は点ごとの記述用で、同時保証・確認仮説の有意判定ではない。','',
      '## 2. 正負の相殺とゲート','',
      '均衡混在では期待値の一次ゲイン項が消える一方、二次項が残る。実現値の相殺比ηMは、追加故障から取り除いた正常値を引いた各成分の寄与で計算した。ηM=1はその二成分の平均寄与の完全相殺を表すが、正常分布への復帰やゲート通過を意味しない。','',
      '| 群 | GM800 MC保留 | GP800 MC保留 | 均衡800 M保留 | 均衡800 MC保留 | 均衡ηM | 三種800 MC保留 |','|---|---:|---:|---:|---:|---:|---:|']
    for g in G:
        aa0=gx[g,'GM_n800'];bb0=gx[g,'GP_n800'];bb=gx[g,'GMGP11_n800'];tt=gx[g,'GMGPP112_n800'];eta=bx[g,'GMGP11_n800','A']['eta_mean']
        doc.append(f'| {g} | {aa0["holds"][2]}/50 | {bb0["holds"][2]}/50 | {bb["holds"][1]}/50 | {bb["holds"][2]}/50 | {eta:.4f} | {tt["holds"][2]}/50 |')
    doc += ['', 'この表の単独800対均衡800は総混入数を揃えた組成交替。均衡800対三種800ではゲイン記録数が減るため、私的入力の純粋な追加比較ではない。次節では同じゲイン記録・位置を保持した対照を使う。','',
      '### 混入量の全曲線','']
    for fn,caption in zip(figures[:3],['旧Fの警報割合を使うG-old','符号付きM平均を使うG-M','MとCRを使うG-MC']):doc += [f'![{caption}](figures/{fn})','']
    doc += ['![相殺比](figures/04_cancellation.png)','',
      '![平均とSDの対称分解](figures/05_mean_sd.png)','',
      '平均移動とSD膨張は別の機序として残した。符号付きTの変化は両寄与の厳密和であるが、計器部品を交換した因果効果ではない。分散の群内／群間項、A/U別の成分寄与とηMの分母は[機序JSON](mechanisms.json)と各反復NPZに保存した。','',
      '## 3. 同じゲイン混在へ私的入力を加える','',
      '三種親n=160はGM40/GP40/P80、n=800はGM200/GP200/P400。それぞれからPだけを元正常へ戻した対照と比べる。残るゲインのID・位置は完全に同一。以下はW0・α=1%、親−成分除去対照。MC保留差の分母は50バッチ、性能差は同じE上の対応差。','',
      '| 群 | 親n | MC保留差 | ΔFPR pp | ΔGM検出 pp | ΔGP検出 pp | ΔP検出 pp | P拾得 % | P喪失 % |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in G:
        for n in [160,800]:
            cc=f'GMGPP112_n{n}_keep01';rr=[tx[g,cc,.01,'W0',ev] for ev in EV];p=rr[3]
            doc.append(f'| {g} | {n} | {p["gate_hold_difference"][2]:+d}/50 | '+' | '.join(f'{100*r["delta"]["mean"]:+.4f}' for r in rr)+f' | {pct(p["gained"]["mean"])} | {pct(p["lost"]["mean"])} |')
    doc += ['', 'Pの追加でP検出が低下しても計算矛盾ではない。ここで変わるのは評価故障でなく校正Uであり、Pの異常値を校正へ取り込むとP枝の閾値を押し上げうる。保留が増えれば、その後の政策は旧Fの適合性にも依存する。','',
      '![同じゲイン記録へのP追加](figures/06_private_addition.png)','',
      '同様にGMだけ、GPだけへPを加える対照も全群で実施した。全24親子ID・2α・2方式・4評価の対応拾得／喪失、3ゲート政策の親子差、W0の各枝閾値差は[成分除去対照JSON](parent_contrasts.json)にある。二つの方式・政策が異なる記録を拾う場合、純差だけで隠さない。','',
      '## 4. 混入Uの害と、旧Fへ戻す効果','',
      '次表は均衡ゲイン800と三種混合800、W0・1%の固定断面。C0は清浄な現在源Uからの参照校正で、運用中に自動で得られる正常認定ではない。NはFPR、他は検出率（%）。','',
      '| 群 | 組成 | 校正／政策 | N | GM | GP | P |','|---|---|---|---:|---:|---:|---:|']
    for g in G:
        for cc in ['GMGP11_n800','GMGPP112_n800']:
            rr=[ix[g,cc,.01,'W0',ev] for ev in EV]
            for pol in ['C0','U','G-MC']:
                vals=[r['C0']['mean'] if pol=='C0' else r['rates'][pol]['mean'] for r in rr]
                doc.append('| '+g+' | '+cc+' | '+pol+' | '+' | '.join(map(pct,vals))+' |')
    doc += ['', '![校正と政策の得失](figures/07_policy_tradeoffs.png)','',
      '全条件で「ゲート政策−U＝保留頻度×保留時のF−U」を照合した。源変化後にFが適合していなければ、混入を正しく止めてもFPR増大や別故障の喪失が残る。保留率だけから政策の妥当性を認定しない。αと1.25αは参照線・比較基準であり、この事後評価から新たな認証ラベルを作らない。','',
      '正負の評価記録は別々なので、GP−GMは反復内の率差として扱う。全条件でG枝成分＋P-only成分への整数分解を照合した。[正負差JSON](gain_asymmetry.json)。','',
      '## 5. 低αとLF0の非発火','',
      'LF0では9,999件の有限参照順位の上端に、15,999件の校正中16件以上が集まると、α=0.1%のcutが最大順位になり、strict不等号の下で全枝が発火できない。これは技術的不成立や欠測ではない。','',
      '| 群 | 組成 | LF0非発火 /50 | W0 FPR % | LF0 FPR % | W0 P検出 % | LF0 P検出 % |','|---|---|---:|---:|---:|---:|---:|']
    for g in G:
        for cc in ['clean','GMGP11_n800','GMGPP112_n800']:
            vals=[ix[g,cc,.001,met,ev]['rates']['U']['mean'] for ev in ['normal','P02'] for met in METHODS]
            doc.append(f'| {g} | {cc} | {gx[g,cc]["inactive"][1]} | '+' | '.join(map(pct,vals))+' |')
    doc += ['', '![低alpha](figures/08_low_alpha.png)','',
      'W0とLF0は同じ公称αだが、実測FPRは揃っていない。検出率だけで等FPRの優越を主張しない。全条件の方式間拾得・喪失は[方式比較JSON](method_comparisons.json)、全枝の対応は反復NPZに残した。','',
      '## 6. 検証・件数・再現','',
      f'主解析は50反復、34,200のA/Uバッチ、実体57配置。元A/U 11,039,400記録とE 10,500,000記録を再利用した。主解析の累積反復処理時間は{validation["elapsed_rep_seconds"]/60:.2f}分。新規科学標本生成0、欠測0、UNKNOWN0、除外0。候補の延べ評価は2,394,000,000記録×方式水準、旧F再照合42,000,000。これは独立標本数ではない。','',
      '既存clean＋単独160/800の7候補は全50反復・全6群でゲート・校正・E枝bitと一致した。LF0順位判定と等価連続半平面も全新候補で一致した。原12ZIPのSHA／サイズ、読込メンバーのSHA／長さ／CRC、原記録ハッシュを照合した。[主解析検証](validation.json)。','',
      '検証は同一エージェントの自己点検であり、第三者の独立追試ではない。別経路では記録ごとの配置、スカラー射影、ソート閾値、add.at整数表を使い、34,200バッチ・34,200校正水準とE1,050万記録を再照合した。799,200枝対応表が厳密一致。実測の二座標恒等式最大差は6.9614e-15だった。[別経路検証](second_path_validation.json)と[納品点検](delivery_validation.json)はPASS。','',
      '技術fixtureは実行前の6件＋役割混線・不正メタデータ・SDゼロ・整数改変の負対照4件。初回のテスト側添字不備を保存・修正して全件通過した。主解析の`checks.q_identity_max`は更新されない初期値カウンタだったため最大誤差として解釈しない（恒等式assertion自体は実行済み）。実測最大誤差は別経路検証の`q_identity_measured_max`を用いる。いずれも統計的FP/FNへ混ぜない。','',
      '最初の図生成は、相殺図のadvanced indexingで用量軸と反復軸を取り違えて停止した。途中版ソースと3図を[失敗記録](attempts/report01/FAILURE.md)へ保持し、候補軸を明示した選択と形状assertへ修正した。元の成分値・JSON集計・科学判定は不変。全8図は修正後の反復平均から作り直した。','',
      '再現コマンド（新規実行先は旧成果を上書きしない版を用意する）：','',
      '```sh','python3 numerics/scripts/test_pr0010_mixture.py','python3 numerics/scripts/test_pr0010_negative_controls.py','python3 numerics/scripts/resume_pr0010_mixture.py','python3 numerics/scripts/verify_pr0010_mixture.py','python3 numerics/scripts/report_pr0010_mixture.py','python3 numerics/scripts/verify_pr0010_delivery.py','```','',
      '既存の完了receiptがある主解析は入力・コードハッシュを確認して再利用する。再開入口`resume_pr0010_mixture.py`は50反復完了済みなら検証だけで終了し、経過時間等の完了証拠を再上書きしない。改訂解析・再検証結果の保存は別出力版を用意する。各5反復の非累積ZIPは`bundles/`、固定A/U状態は`frozen_candidates/`、全反復値は`recomputed/`に保存。元科学ZIPを再同梱していない。','',
      '## 7. 解釈の境界','',
      '代数の閉合H1と性能の問いQ1–Q3を分ける。この解析は既知結果を踏まえた混合ストレス評価であり、独立確認への昇格、最適混合率や普遍的なゲート選択はしない。均衡平均だけから分布全体の識別不能は結論できず、正負比率を少し動かす探索や新しい複合故障の生成も行っていない。','',
      '扱ったのは固定3故障強度・6群のバッチ内混在。A/Uの混入率が違う場合、連続ドリフト、実機の故障頻度、旧F不適合時の停止・復旧政策は別の問いである。検出損失・非発火・FPR悪化を残したまま、次の問いを人間が選ぶ。今回の結果は論文へまだ反映していない。','']
    (OUT/'PR0010_MIXTURE_REPORT_JA.md').write_text('\n'.join(doc)+'\n')
    print('REPORT_DATA_READY',len(rows),len(parentrows),len(figures),flush=True)

if __name__=='__main__':main()
