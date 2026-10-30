# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Figures and complete tables for the saved-data low-FPR analysis."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t

COLORS=['#2878b5','#e07b29','#809239','#8373af','#c52b45']
STYLES=['-','-',':','--','-']


def interval(r, scale=100):
    return '未定義' if r['lower'] is None else f"[{scale*r['lower']:.4f}, {scale*r['upper']:.4f}]"


def table(header, rows):
    return ['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows]


def render(out):
    out=Path(out);s=json.loads((out/'summary.json').read_text());m=json.loads((out/'metadata.json').read_text())
    assert json.loads((out/'validation.json').read_text())['status']=='PASS'
    with np.load(out/'replicate_metrics.npz',allow_pickle=False) as f:
        a={k:f[k] for k in f.files}
    fk={tuple(k):i for i,k in enumerate(m['fault_keys'])};nk={tuple(k):i for i,k in enumerate(m['null_keys'])}
    grid=np.array(m['grid'])*100;methods=m['methods']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':165})
    def curve(ax,case,policy,title):
        fi=fk[case,policy];ni=nk[m['conditions'][case]['group'],policy]
        for j,method in enumerate(methods):
            x=a['power'][:,fi,j]*100;y=x.mean(0)
            ax.step(grid,y,where='post',label=method,color=COLORS[j],ls=STYLES[j],lw=2 if j==4 else 1.5)
            if j==4:
                w=t.ppf(.975,49)*x.std(0,ddof=1)/np.sqrt(50)
                ax.fill_between(grid,np.maximum(0,y-w),np.minimum(100,y+w),color=COLORS[j],alpha=.13,step='post')
            ax.scatter(100*a['nominal_fpr'][:,ni,j,2].mean(),100*a['nominal_tpr'][:,fi,j,2].mean(),
                color=COLORS[j],marker='x',s=38,zorder=5)
        ax.set(title=title,xlim=(0,2),ylim=(0,100),xlabel='Empirical FPR budget (%)',ylabel='Detection (%)')
        ax.grid(alpha=.16)
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.7),layout='constrained')
    for ax,case,label in zip(axes,['B_L__GM05','B_L__GP05','B_M__P02'],['Gain -5%, 8.192 s','Gain +5%, 8.192 s','Private input 0.02, 2.048 s']):
        curve(ax,case,'B',label)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5)
    fig.suptitle('Low-FPR attainable ROC: 50-replicate means\nLF shading: pointwise 95% CI; x: independent-calibration nominal 1% point')
    fig.savefig(out/'primary_roc.png');plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,8.3),layout='constrained')
    for ax,case,label in zip(axes.ravel(),['B_L__GM02','B_L__GP02','B_M__P005','B_M__DP2'],
                            ['Gain -2%, 8.192 s','Gain +2%, 8.192 s','Private input 0.005, 2.048 s','Delay +2 samples, 2.048 s']):
        curve(ax,case,'B',label)
    fig.legend(handles,labels,loc='outside lower center',ncol=5)
    fig.suptitle('Secondary conditions: same frozen methods and threshold families\nAll other conditions, including negative delay and control, are tabulated')
    fig.savefig(out/'secondary_roc.png');plt.close(fig)

    groups=[g for g,p in m['null_keys'] if p=='F'];policies=['F','C','R']
    areas={(r['case'],r['policy'],r['method']):r['npauc']['mean'] for r in s['areas']}
    vals=np.array([[[100*(areas[g+'__'+tag,p,'LF']-areas[g+'__'+tag,p,'J']) for tag in ['GM05','GP05','P02']] for g in groups] for p in policies])
    limit=max(5,float(abs(vals).max()))
    fig,axes=plt.subplots(1,3,figsize=(12.4,5.7),sharey=True,layout='constrained')
    for ax,p,x in zip(axes,policies,vals):
        im=ax.imshow(x,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        for i in range(len(groups)):
            for j in range(3):ax.text(j,i,f'{x[i,j]:+.2f}',ha='center',va='center',fontsize=9,color='white' if abs(x[i,j])>.6*limit else 'black')
        ax.set(title=p,xticks=range(3),xticklabels=['Gain -5%','Gain +5%','Private 0.02'],yticks=range(len(groups)),yticklabels=groups)
    fig.colorbar(im,ax=axes,label='LF minus J: normalized partial area (pp)',shrink=.8)
    fig.suptitle('Source-shift discrimination at empirical FPR budgets 0-2%\nDescriptive threshold sweep; not a guarantee of transferred false-alarm control')
    fig.savefig(out/'source_shift_npauc.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,ai,label in zip(axes,[0,2],['Nominal 0.1%','Nominal 1%']):
        for pi,p in enumerate(policies):
            inds=[nk[g,p] for g in groups];v=a['nominal_fpr'][:,inds,4,ai]*100
            y=v.mean(0);w=t.ppf(.975,49)*v.std(0,ddof=1)/np.sqrt(50)
            ax.errorbar(np.arange(8)+(pi-1)*.16,y,yerr=w,fmt=['o','s','^'][pi],capsize=2,label=p,ms=4)
        ax.axhline(100*m['alphas'][ai],color='black',ls='--',lw=1)
        ax.set(title='LF: '+label,xticks=range(8),xticklabels=groups,ylabel='Held-out false-alarm rate (%)')
        ax.tick_params(axis='x',rotation=35);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.16);ax.legend(title='Policy')
    fig.suptitle('Independent calibration operating points (no evaluation-based threshold selection)\nPointwise 95% intervals across the 50 outer replicates')
    fig.savefig(out/'source_shift_calibration.png');plt.close(fig)

    tails=json.loads((out/'tail_diagnostics.json').read_text())['summary']
    tr={(r['group'],r['policy'],r['method'],r['beta']):r for r in tails}
    fig,axes=plt.subplots(1,3,figsize=(12.4,5.7),sharey=True,layout='constrained')
    for ax,p in zip(axes,policies):
        x=np.array([[tr[g,p,method,.001]['always_negative_replicates'] for method in methods] for g in groups])
        im=ax.imshow(x,vmin=0,vmax=50,cmap='YlOrRd',aspect='auto')
        for i in range(8):
            for j in range(5):ax.text(j,i,str(x[i,j]),ha='center',va='center',color='white' if x[i,j]>30 else 'black',fontsize=9)
        ax.set(title=p,xticks=range(5),xticklabels=methods,yticks=range(8),yticklabels=groups)
    fig.colorbar(im,ax=axes,label='Always-negative replicates / 50',shrink=.8)
    fig.suptitle('At 0.1% empirical FPR budget: when the frozen rank path cannot alarm\nThreshold exceeds the largest possible calibrated score; no randomized tie breaking')
    fig.savefig(out/'rank_resolution.png');plt.close(fig)

    lines=['# 全条件の低FPR ROC・独立校正運用点','',
        '事後的記述解析。率と差の単位は%／pp。点別95%外側t区間で、多重比較調整なし。ROC予算は評価正常から最も緩い許容閾値を選ぶ記述値で、運用FPR保証ではない。独立校正運用点は正常校正だけで選ぶ別指標。方針Bは基準環境、方式BはBonferroni OR。時間反転REVは構造対照で故障に含めない。','',
        '## ROC代表点と独立校正運用点','']
    lines+=table(['条件','方針','方式','FPR予算/公称 %','到達FPR %','ROC検出 %','ROC点別95%区間','独立校正FPR %','独立校正検出 %'],
        [[r['case'],r['policy'],r['method'],f"{100*r['beta']:.1f}",f"{100*r['actual_fpr']['mean']:.4f}",
          f"{100*r['tpr']['mean']:.4f}",interval(r['tpr']),f"{100*r['nominal_fpr']['mean']:.4f}",f"{100*r['nominal_tpr']['mean']:.4f}"] for r in s['rates']])
    lines+=['','## LF対既存4方式：対応する検出・見逃し','',
        '同じFPR予算の下、各方式の到達可能な保守点を比較。両方式の実FPRが完全に等しいとは限らない。','']
    lines+=table(['条件','方針','比較先','FPR予算 %','差 pp','点別95%区間 pp','LFのみ %','比較先のみ %','両方 %','両方見逃し %'],
        [[r['case'],r['policy'],r['other'],f"{100*r['beta']:.1f}",f"{100*r['difference']['mean']:+.4f}",interval(r['difference'])]+
         [f"{100*r[k]['mean']:.4f}" for k in ['LF_only','other_only','both','neither']] for r in s['paired']])
    lines+=['','## 正規化部分面積（0〜2%）','',
        'n-pAUC×100。段階関数の面積を0.02で割った平均検出率相当の値。AUC全域や偶然水準補正AUCではない。','']
    lines+=table(['条件','方針','方式','n-pAUC×100','点別95%区間'],
        [[r['case'],r['policy'],r['method'],f"{100*r['npauc']['mean']:.4f}",interval(r['npauc'])] for r in s['areas']])
    lines+=['','## 正常群の到達性と校正','',
        '次のFPRは、予算を超える一段緩い閾値のFPRの平均。厳密到達は50反復中の件数。最小・最大は保守点の反復間範囲。','']
    lines+=table(['正常群','方針','方式','予算/公称 %','到達FPR %','到達範囲 %','次のFPR %','厳密到達 /50','独立校正FPR %','独立校正95%区間'],
        [[r['group'],r['policy'],r['method'],f"{100*r['beta']:.1f}",f"{100*r['actual_fpr']['mean']:.4f}",
          f"[{100*r['min_fpr']:.4f}, {100*r['max_fpr']:.4f}]",f"{100*r['next_fpr']['mean']:.4f}",r['exact_replicates'],
          f"{100*r['nominal_fpr']['mean']:.4f}",interval(r['nominal_fpr'])] for r in s['normal']])
    lines+=['','## 順位上限と常時非発火への退化','',
        '参照上限率は正常評価で参照順位が上限9999に達した率。Bでは元の二成分のいずれかの上限率であり、校正済みBスコアの最大値のビンとは同一でない。校正スコア上限率は理論上の最大Uに入った率。常時非発火件数は閾値が最大Uを超えた反復数で、単に正常標本の誤警報が0だったという意味ではない。','']
    lines+=table(['正常群','方針','方式','予算/公称 %','参照上限率 %','校正スコア上限率 %','ROC常時非発火 /50','独立校正常時非発火 /50'],
        [[r['group'],r['policy'],r['method'],f"{100*r['beta']:.1f}",f"{100*r['rank_cap_fraction']['mean']:.4f}",
          f"{100*r['calibrated_cap_fraction']['mean']:.4f}",r['always_negative_replicates'],r['nominal_always_negative_replicates']] for r in tails])
    (out/'ALL_LOW_FPR_RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'figure_metadata.json').write_text(json.dumps(dict(figures=['primary_roc.png','secondary_roc.png','source_shift_npauc.png','source_shift_calibration.png','rank_resolution.png'],
        all_table_rows=[len(s[k]) for k in ['rates','paired','areas','normal']]+[len(tails)],
        averaging='per-replicate attainable-floor curve, not pooled threshold ROC',
        interpolation='none; post steps on exact 1/20000 FPR grid',bands='pointwise, not simultaneous'),indent=2)+'\n')
    print('Rendered 5 figures and complete tables.')


if __name__=='__main__':render(sys.argv[1])
