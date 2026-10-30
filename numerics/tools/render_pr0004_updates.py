# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Render fixed C/R decision and geometric decomposition; no detector fitting."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def iv(r):return '未定義' if r['lower'] is None else f"[{100*r['lower']:.4f}, {100*r['upper']:.4f}]"
def tab(head,rows):return ['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows]


def render(folder):
    f=Path(folder);s=json.loads((f/'summary.json').read_text());g=json.loads((f/'geometry.json').read_text())
    assert json.loads((f/'validation.json').read_text())['status']=='PASS'
    rates={(r['case'],r['method'],r['alpha']):r['metrics'] for r in s['rates']}
    tr={(r['case'],r['alpha']):r for r in s['transitions']};mar={(r['case'],r['direction']):r for r in s['margins']}
    groups=list(dict.fromkeys(r['group'] for r in g))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':165})
    vals=np.array([[[100*rates[group+'__'+tag,'LF',alpha]['difference']['mean'] for tag in ['GM05','GP05','P02']] for group in groups] for alpha in [.01,.001]])
    fig,axes=plt.subplots(1,2,figsize=(11,5.7),sharey=True,layout='constrained')
    for ax,alpha,x in zip(axes,[.01,.001],vals):
        lim=max(2,float(abs(x).max()));im=ax.imshow(x,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
        for i in range(8):
            for j in range(3):ax.text(j,i,f'{x[i,j]:+.2f}',ha='center',va='center',color='white' if abs(x[i,j])>.6*lim else 'black')
        ax.set(title=f'LF: nominal {100*alpha:g}%',xticks=range(3),xticklabels=['Gain -5%','Gain +5%','Private 0.02'],yticks=range(8),yticklabels=groups)
        fig.colorbar(im,ax=ax,label='R minus C detection (pp)',shrink=.8)
    fig.suptitle('Relearning gains AND losses (independent normal calibration)\nSeparate colour scales; no evaluation-based FPR matching')
    fig.savefig(f/'update_detection.png');plt.close(fig)

    cases=['K200_L__P02','K2_M__P02','V05_L__GM05','V05_L__GP05','K2_M__GM05','K2_M__GP05']
    fig,ax=plt.subplots(figsize=(11,5.2),layout='constrained');up=np.zeros(6);down=np.zeros(6)
    for i,(label,color) in enumerate([('G only','#2878b5'),('P only','#dc8b32'),('Both branches','#4f9e8b')]):
        pos=np.array([100*tr[c,.01]['metrics'][['R_g_only','R_p_only','R_both'][i]]['mean'] for c in cases])
        neg=np.array([100*tr[c,.01]['metrics'][['C_g_only','C_p_only','C_both'][i]]['mean'] for c in cases])
        ax.bar(np.arange(6),pos,bottom=up,color=color,label=label);ax.bar(np.arange(6),-neg,bottom=-down,color=color,hatch='//',alpha=.75)
        up+=pos;down+=neg
    met=[rates[c,'LF',.01]['difference'] for c in cases];y=np.array([100*r['mean'] for r in met])
    ax.errorbar(np.arange(6),y,yerr=[100*(r['upper']-r['mean']) for r in met],fmt='ko',capsize=3,label='Net R-C / pointwise 95% CI')
    ax.axhline(0,color='black',lw=.7);ax.set(xticks=range(6),xticklabels=[c.replace('__','\n') for c in cases],ylabel='Share of all fault records (%)')
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='outside lower center',fontsize=9)
    ax.set_title('Nominal 1%: newly detected by R (above) and lost from C (below)')
    fig.savefig(f/'branch_gains_losses.png');plt.close(fig)

    with np.load(f/'examples_rep00.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    lookup={(r['replicate'],r['group'],r['alpha']):r for r in g}
    plot_meta={};fig,axes=plt.subplots(1,2,figsize=(12,5.7),layout='constrained')
    for ax,case,ai in zip(axes,['K200_L__P02','K2_M__P02'],[1,0]):
        group=case.split('__')[0];normal=data[group+'_z'];fault=data[case+'_z'];both=np.concatenate((normal,fault))
        lo=np.quantile(both,.001,axis=0)-.5;hi=np.quantile(both,.999,axis=0)+.5
        xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],350),np.linspace(lo[1],hi[1],350));zz=np.column_stack((xx.ravel(),yy.ravel()))
        meta=lookup[0,group,[.001,.01][ai]];disabled=[]
        ax.hist2d(normal[:,0],normal[:,1],bins=60,range=[[lo[0],hi[0]],[lo[1],hi[1]]],cmap='Greys',cmin=1,alpha=.4)
        bits=data[case+'_bits'][:,ai];c=(bits[:,0]&16)!=0;r=(bits[:,1]&16)!=0
        cat=c.astype(int)+2*r.astype(int);shown=np.arange(min(1200,len(fault)))
        for k,color in [(0,'#b9bfc7'),(3,'#548c68'),(1,'#2878b5'),(2,'#cc3e4b')]:
            ix=shown[cat[shown]==k];ax.scatter(fault[ix,0],fault[ix,1],c=color,s=7,alpha=.55)
        for pol,color,ls in [('C','#222222','--'),('R','#8d40a5','-')]:
            obj=meta[pol];u=np.array(obj['u']);b=np.array(obj['b']);h=np.array([np.inf if x is None else x for x in obj['h']])
            v=zz@u.T+b;alarm=((abs(v[:,0])>h[0])|(v[:,1]>h[1])).reshape(xx.shape)
            if alarm.any() and (~alarm).any():ax.contour(xx,yy,alarm.astype(float),levels=[.5],colors=[color],linestyles=ls,linewidths=1.6)
            elif not alarm.any():disabled.append(pol+' has no alarm in displayed region')
        ax.set(title=f'{case}, nominal {100*[.001,.01][ai]:g}%',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='Signed moment coordinate / target normal SD',ylabel='CR coordinate / target normal SD')
        if disabled:ax.text(.02,.98,'\n'.join(disabled),transform=ax.transAxes,va='top',fontsize=8)
        plot_meta[case]=dict(limits=[lo.tolist(),hi.tolist()],displayed_fault_records=len(shown),outside_full_fault=int(np.any((fault<lo)|(fault>hi),axis=1).sum()))
    legend=[Line2D([0],[0],marker='o',ls='',color=c,label=l) for c,l in [('#b9bfc7','Neither'),('#2878b5','C only (lost)'),('#cc3e4b','R only (gained)'),('#548c68','Both')]]
    legend+=[Line2D([0],[0],color='#222222',ls='--',label='C boundary'),Line2D([0],[0],color='#8d40a5',label='R boundary')]
    fig.legend(handles=legend,loc='outside lower center',ncol=3);fig.suptitle('Illustrative replicate 0: actual fixed C/R decision regions\nGrey normal density; first 1,200 fault records, not outcome-selected')
    fig.savefig(f/'decision_regions.png');plt.close(fig)

    terms=s['terms'];colors=['#2878b5','#8c6bb1','#d79532','#43a08c'];picked=cases[:4]
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),sharey=True,layout='constrained')
    for ax,direction in zip(axes,['R_only','C_only']):
        arr=np.array([mar[c,direction]['conditional_mean'] for c in picked]);pos=np.zeros(4);neg=np.zeros(4)
        for j,(term,color) in enumerate(zip(terms,colors)):
            v=arr[:,j];ax.bar(np.arange(4),v,bottom=np.where(v>=0,pos,neg),color=color,label=term.replace('_',' '));pos+=np.maximum(v,0);neg+=np.minimum(v,0)
        ax.plot(np.arange(4),arr.sum(1),'ko',label='Total margin change')
        ax.axhline(0,color='black',lw=.7);ax.set(title=direction+' records',xticks=range(4),xticklabels=[c.replace('__','\n') for c in picked],ylabel='Mean normalized boundary-margin change')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Nominal 1%: algebraic margin decomposition on discordant records\nPositive = towards the winning policy; not causal contributions to detection rate')
    fig.savefig(f/'margin_decomposition.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for ax,alpha in zip(axes,[.01,.001]):
        for pol,marker in [('C','s'),('R','o')]:
            v=[rates[group,'LF',alpha][pol] for group in groups];y=np.array([100*r['mean'] for r in v]);w=[0 if r['lower'] is None else 100*(r['upper']-r['mean']) for r in v]
            ax.errorbar(np.arange(8)+(0.08 if pol=='R' else -.08),y,yerr=w,fmt=marker,capsize=3,label=pol)
        ax.axhline(100*alpha,color='black',ls='--',lw=1);ax.set(title=f'LF nominal {100*alpha:g}%',xticks=range(8),xticklabels=groups,ylabel='Held-out FPR (%)');ax.tick_params(axis='x',rotation=35);ax.set_ylim(bottom=0);ax.legend()
    fig.suptitle('Normal evaluation is held out: no ROC-based threshold adjustment')
    fig.savefig(f/'calibration_fpr.png');plt.close(fig)

    allocation=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.9),sharey=True,layout='constrained')
    for ax,pol in zip(axes,['C','R']):
        x=[]
        for group in groups:
            row=tr[group,.01];a=np.array(row['table_counts'])/row['n_total']
            codes=a.sum(axis=1 if pol=='C' else 0)
            x.append(100*codes[1:]);allocation.append([group,pol]+(100*np.array([codes[1],codes[2],codes[3],1-codes[0]])).tolist())
        x=np.array(x);bottom=np.zeros(8)
        for j,(label,color) in enumerate([('G only','#2878b5'),('P only','#dc8b32'),('Both','#4f9e8b')]):
            ax.bar(np.arange(8),x[:,j],bottom=bottom,color=color,label=label);bottom+=x[:,j]
        ax.axhline(1,color='black',ls='--',lw=1)
        ax.set(title=pol,xticks=range(8),xticklabels=groups,ylabel='Held-out normal alarm rate (%)',ylim=(0,1.17));ax.tick_params(axis='x',rotation=35)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.suptitle('Nominal 1%: similar total FPR, different branch allocation\nObserved rates, not imposed branch budgets')
    fig.savefig(f/'normal_branch_allocation.png');plt.close(fig)

    lines=['# C/R更新差：全条件表','',
        '率は%、差はpp。公称水準は正常校正だけで決定。正常行の追加／損失は誤警報の増加／除去であり、故障の検出と意味が逆になる。点別95%外側t区間、多重比較調整なし。','',
        '## 全5方式の対応比較','']
    lines+=tab(['条件','方式','公称 %','C %','R %','R−C pp','点別95%区間','Rのみ %','Cのみ %','両方 %','非発火 %'],
        [[r['case'],r['method'],f"{100*r['alpha']:.1f}"]+[f"{100*r['metrics'][k]['mean']:.4f}" for k in ['C','R','difference']]+[iv(r['metrics']['difference'])]+[f"{100*r['metrics'][k]['mean']:.4f}" for k in ['R_only','C_only','both','neither']] for r in s['rates']])
    lines+=['','## LF正常誤警報の枝配分（公称1%）','',
        'Gのみ・Pのみ・両枝は排他的。合計は正常OR発火率。枝ごとの固定予算ではなく評価正常で観測された率。','']
    lines+=tab(['条件','更新','Gのみ %','Pのみ %','両枝 %','全体 %'],[[*r[:2],*[f'{v:.4f}' for v in r[2:]]] for r in allocation])
    lines+=['','## LFの追加・損失の枝分解','']
    lines+=tab(['条件','公称 %','追加Gのみ %','追加Pのみ %','追加両枝 %','損失Gのみ %','損失Pのみ %','損失両枝 %'],
        [[r['case'],f"{100*r['alpha']:.1f}"]+[f"{100*r['metrics'][k]['mean']:.4f}" for k in ['R_g_only','R_p_only','R_both','C_g_only','C_p_only','C_both']] for r in s['transitions']])
    lines+=['','## 公称1%の境界余裕分解','',
        '不一致記録に条件づけた件数重み付き平均。方向・中心・参照表・校正順位の4項は選択面の余裕変化へ加算する。検出率の寄与率ではない。Cのみの場合はC−Rへ向きを反転。G+/G−/Pは勝者側最大余裕面の件数で、同時発火を重複計数しない。','']
    lines+=tab(['条件','不一致側','記録数','G+件数','G−件数','P件数','方向','中心','参照表','校正順位','合計'],
        [[r['case'],r['direction'],r['flip_count']]+r['face_counts']+
         ([f'{x:.6f}' for x in r['conditional_mean']]+[f"{sum(r['conditional_mean']):.6f}"] if r['conditional_mean'] is not None else ['未定義']*5) for r in s['margins']])
    (f/'ALL_UPDATE_RESULTS.md').write_text('\n'.join(lines)+'\n')
    (f/'figure_metadata.json').write_text(json.dumps(dict(figures=['update_detection.png','branch_gains_losses.png','decision_regions.png','margin_decomposition.png','calibration_fpr.png','normal_branch_allocation.png'],examples=plot_meta,
        tables=[len(s[k]) for k in ['rates','transitions','margins']],conditional_margin_units='Euclidean distance in target-reference standardized q; not pp'),indent=2)+'\n')
    print('Rendered six figures and all-condition tables.')


if __name__=='__main__':render(sys.argv[1])
