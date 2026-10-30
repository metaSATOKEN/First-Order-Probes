# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Render fixed PR-0005 comparisons; never chooses thresholds or allocations."""
from pathlib import Path
import json,sys,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'numerics/scripts'))
from analyze_pr0005_three_config import METHODS,PRIMARY,BUDGETS,put,file_hash
COLORS=['#667785','#087e8b','#d37a25']


def render(out):
    start=time.monotonic();out=Path(out);s=json.loads((out/'summary.json').read_text());meta=json.loads((out/'metadata.json').read_text())
    rate={(r['case'],r['policy'],r['alpha'],r['budget'],r['method']):r for r in s['rates']}
    pair={(r['case'],r['policy'],r['alpha'],r['budget'],r['new'],r['reference']):r for r in s['comparisons']}
    with np.load(out/'replicate_statistics.npz',allow_pickle=False) as f:cuts=f['cuts'];av=f['available']
    def r(case,policy,method,alpha=.01,budget='W0'):return rate[case,policy,alpha,budget,method]
    def value(case,policy,method,alpha=.01,budget='W0'):return 100*r(case,policy,method,alpha,budget)['rate']['mean']
    def save(fig,name):
        fig.savefig(out/name,dpi=160,bbox_inches='tight');plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    # All normal contexts; log color scale, numerical annotations are percentages.
    fig,axes=plt.subplots(1,2,figsize=(12,10),layout='constrained')
    for ax,alpha in zip(axes,[.01,.001]):
        mat=np.array([[value(c,p,m,alpha) for m in METHODS] for c,p in meta['null_keys']])
        im=ax.imshow(np.maximum(mat,1e-4),norm=LogNorm(vmin=.001,vmax=100),cmap='magma',aspect='auto')
        ax.set_xticks(range(3),METHODS);ax.set_yticks(range(20),[c+' / '+p for c,p in meta['null_keys']])
        ax.set_title(f'Normal FPR (%) | nominal {100*alpha:g}%')
        for i in range(20):
            for j in range(3):ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',color='white' if mat[i,j]<3 else 'black',fontsize=8)
    fig.colorbar(im,ax=axes,label='Actual FPR (%)',shrink=.7);save(fig,'fig01_false_alarm.png')
    # Signed gains at independent short/medium/long windows, plus paired asymmetry.
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for mi,m in enumerate(METHODS):
        pp=[value(g+'__GP05','B',m) for g in ['B_S','B_M','B_L']]
        mm=[value(g+'__GM05','B',m) for g in ['B_S','B_M','B_L']]
        for ax,ys in zip(axes,[mm,pp,np.array(pp)-mm]):ax.plot(range(3),ys,'o-',color=COLORS[mi],label=m)
    for ax,title in zip(axes,['Negative gain: -5%','Positive gain: +5%','Positive minus negative']):
        ax.set_xticks(range(3),['0.512','2.048','8.192']);ax.set_xlabel('Window T (s)');ax.set_title(title);ax.grid(alpha=.2)
    axes[0].set_ylabel('Detection (%)');axes[1].set_ylabel('Detection (%)');axes[2].set_ylabel('Difference (pp)')
    axes[0].legend();fig.suptitle('W0; nominal 1%; independent records at each window');save(fig,'fig02_signed_gain.png')
    # Six prespecified contexts: actual FPR against detection, not a matched-FPR ROC.
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,(case,p) in zip(axes.flat,PRIMARY):
        group=meta['conditions'][case]['group']
        for mi,m in enumerate(METHODS):
            rr=r(case,p,m);ff=r(group,p,m);x=100*ff['rate']['mean'];y=100*rr['rate']['mean']
            ax.scatter(x,y,color=COLORS[mi],s=55,label=m)
            ax.errorbar(x,y,xerr=100*1.96*ff['rate']['mcse'],yerr=100*1.96*rr['rate']['mcse'],color=COLORS[mi],capsize=3)
        ax.axvline(1,color='black',ls=':',lw=1);ax.set_title(case+' / '+p);ax.set_xlabel('Actual FPR (%)');ax.set_ylabel('Detection (%)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9);fig.suptitle('Same nominal 1% budget, not identical realized FPR; bars: 1.96 MCSE')
    save(fig,'fig03_fpr_detection.png')
    # Actual branch usage and paired gains/losses for all main contrasts.
    fig,(ax,bx)=plt.subplots(1,2,figsize=(16,8),layout='constrained',gridspec_kw={'width_ratios':[1,1.5]})
    labels=[];br=[]
    for c,p in [('B_L','B'),('K200_L','C'),('V05_L','C')]:
        for m in METHODS:
            rr=r(c,p,m);h=np.array(rr['pattern_counts'])/rr['planned']*100
            br.append([h[1],h[2],h[4],h[3]+h[5]+h[6]+h[7]]);labels.append(c+' '+m)
    br=np.array(br);left=np.zeros(9)
    for j,(lab,col) in enumerate(zip(['positive only','negative only','private only','overlap'],['#b74a51','#4a70a8','#48956a','#9272a6'])):
        ax.barh(range(9),br[:,j],left=left,label=lab,color=col);left+=br[:,j]
    ax.set_yticks(range(9),labels);ax.invert_yaxis();ax.set_xlabel('Normal alarms (%)');ax.set_title('Realized branch allocation (W0, 1%)',pad=42)
    ax.legend(fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2)
    labs=[];gain=[];loss=[]
    for case,p in PRIMARY:
        for new,ref in [('LF-DC','LF0'),('PH-DC','LF0'),('PH-DC','LF-DC')]:
            rr=pair[case,p,.01,'W0',new,ref];labs.append(case+' '+new+' - '+ref)
            gain.append(rr['gained']['mean']*100);loss.append(rr['lost']['mean']*100)
    yy=np.arange(18);bx.barh(yy,gain,color='#238b65',label='new only');bx.barh(yy,-np.array(loss),color='#c96c57',label='reference only')
    bx.set_yticks(yy,labs,fontsize=8);bx.invert_yaxis();bx.axvline(0,color='black',lw=.8);bx.set_xlabel('Paired record fraction (%)');bx.set_title('Gains and losses, not just net difference');bx.legend()
    save(fig,'fig04_branches_and_switches.png')
    # Prespecified low-FPR diagnostic plus numerical validity.
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,case,title in [(axes[0,0],'K2_M__P02','K2_M private input: detection'),(axes[0,1],'K2_M','K2_M normal: false alarms')]:
        for mi,m in enumerate(METHODS):ax.bar(mi,value(case,'C',m,.001),color=COLORS[mi])
        ax.set_xticks(range(3),METHODS);ax.set_ylabel('%');ax.set_title(title)
    axes[0,1].axhline(.1,color='black',ls=':')
    ni=[tuple(k) for k in meta['null_keys']].index(('K2_M','C'))
    dead=[int(np.sum(cuts[:,ni,1,0,0,0]>=9999)),int(np.sum(np.isinf(cuts[:,ni,1,0,1]).all(1))),int(np.sum(np.isinf(cuts[:,ni,1,0,2]).all(1)))]
    axes[1,0].bar(range(3),dead,color=COLORS);axes[1,0].set_xticks(range(3),METHODS);axes[1,0].set_ylim(0,50);axes[1,0].set_ylabel('Replicates out of 50');axes[1,0].set_title('Structurally inactive at 0.1% (not observed-zero FP)')
    diag=json.loads((out/'diagnostics.json').read_text());base=['B_S','B_M','B_L','B_R10']
    axes[1,1].boxplot([[d['physical']['condition'] for d in diag if d['group']==b and d['physical']['condition'] is not None] for b in base],tick_labels=base)
    axes[1,1].set_ylabel('Column-normalized condition number');axes[1,1].set_title('Physical D gate: threshold 1e6; no discarded cases')
    fig.suptitle('Supplement: W0, nominal 0.1%, threshold update C');save(fig,'fig05_low_fpr_and_validity.png')
    # Every frozen allocation, never conditionwise maxima.
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for ax,(case,p) in zip(axes.flat,PRIMARY):
        mat=np.array([[value(case,p,m,.01,b) for m in METHODS] for b in BUDGETS])
        im=ax.imshow(mat,cmap='viridis',vmin=0,vmax=100,aspect='auto');ax.set_xticks(range(3),METHODS);ax.set_yticks(range(5),BUDGETS)
        ax.set_title(case+' / '+p)
        for i in range(5):
            for j in range(3):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if mat[i,j]<55 else 'black',fontsize=9)
    fig.colorbar(im,ax=axes,label='Detection (%)',shrink=.75);fig.suptitle('All five fixed allocations, nominal 1% (no winner selection)')
    save(fig,'fig06_budget_sensitivity.png')
    # Exhaustive human-readable result tables, numerical uncertainty stays in JSON/NPZ.
    lines=['# PR-0005 全条件表','', '保存済み50反復の事後解析。値は外側反復平均。全率は%、差はpp。方式が不成立なら条件付き率と成立数を併記する。',
           '', '区間・SD・MCSE・分位点・枝8状態は[summary.json](summary.json)、各反復とWilson区間は[replicate_statistics.npz](replicate_statistics.npz)。', '']
    for alpha in [.01,.001]:
        for budget in BUDGETS:
            lines+= [f'## α={100*alpha:g}%／{budget}', '', '| 条件 | 方針 | 種別 | LF0 | LF-DC | PH-DC | 成立反復(LF0/DC/PH) |','|---|---|---|---:|---:|---:|---|']
            for case,p in meta['keys']:
                rows=[r(case,p,m,alpha,budget) for m in METHODS]
                vals=['NA' if z['rate']['mean'] is None else f"{100*z['rate']['mean']:.4f}" for z in rows]
                lines.append('| '+' | '.join([case,p,rows[0]['kind'],*vals,'/'.join(str(z['valid_replicates']) for z in rows)])+' |')
            lines+=['','| 条件 | 方針 | 新−比較 | 両方 | 拾得 | 損失 | 両方非発火 | 純差 | 共通成立反復 |','|---|---|---|---:|---:|---:|---:|---:|---:|']
            for case,p in meta['keys']:
                for new,ref in [('LF-DC','LF0'),('PH-DC','LF0'),('PH-DC','LF-DC')]:
                    z=pair[case,p,alpha,budget,new,ref];nn=z['n_joint'];vals=[f'{100*x/nn:.4f}' if nn else 'NA' for x in z['counts']]
                    diff='NA' if z['difference']['mean'] is None else f"{100*z['difference']['mean']:.4f}"
                    lines.append('| '+' | '.join([case,p,new+'−'+ref,*vals,diff,str(z['valid_replicates'])])+' |')
            lines.append('')
    (out/'ALL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    put(out/'render_validation.json',dict(status='GENERATED_PENDING_VISUAL_INSPECTION',elapsed_seconds=time.monotonic()-start,
        source_sha256=file_hash(Path(__file__)),figures=[p.name for p in sorted(out.glob('fig*.png'))],table_rate_rows=820,table_pair_rows=2460))


if __name__=='__main__':render(sys.argv[1])
