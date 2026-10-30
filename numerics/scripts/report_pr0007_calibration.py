# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Render the complete, fixed PR-0007 analysis after saved-data verification."""
from pathlib import Path
import json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pr0007_spec as spec

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'numerics/analysis/pr0007_60acab7af20d_performance_v1'
FIG = OUT/'figures'
METHODS = list(spec.METHODS)
COLORS = ['#2166ac', '#1b9e77', '#d95f02', '#984ea3', '#e7298a', '#444444']
LABELS = {'GM05':'Gain -5%', 'GP05':'Gain +5%', 'P02':'Private h=0.02'}


def read(name): return json.loads((OUT/name).read_text())
def save(name, value): (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def md(name, lines): (OUT/name).write_text('\n'.join(lines)+'\n')
def fmt(x, digits=4): return '未定義' if x is None else f'{100*x:.{digits}f}'
def interval(d): return f"[{fmt(d['lower'])}, {fmt(d['upper'])}]"
def table(headers, rows): return ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]
def figsave(fig, name):
    fig.savefig(FIG/name, dpi=155, bbox_inches='tight', facecolor='white'); plt.close(fig)


def main():
    assert read('validation.json')['status'] == 'PASS_SAVED_DATA_RECOMPUTATION'
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    rates = read('recomputed_rates.json'); primary = read('recomputed_primary8.json')
    cross = read('recomputed_cross1296.json'); fpr = read('recomputed_fpr792.json')
    policy = read('recomputed_calibration_contrasts.json'); within = read('recomputed_within_methods.json')
    cal = read('calibration_audit.json')
    lookup = {(r['context']['id'],r['context']['case'],r['method'],r['alpha']):r for r in rates}
    def rate(calid, case, method='W0', alpha=.01): return lookup[calid,case,method,alpha]
    primary_rows = []
    for p in primary:
        g,case,n = p['group'],p['case'],p['mix_count']
        clean = rate(spec.clean_id(g,15999),case); mix=rate(p['cal_id'],case)
        nc=rate(spec.clean_id(g,15999),g); nm=rate(p['cal_id'],g)
        contrast=next(r for r in policy if r['left']['id']==p['cal_id'] and r['contrast']['case']==case and r['method']=='W0' and r['alpha']==.01)
        primary_rows.append(dict(id=p['id'],group=g,fault=p['fault'],mix=n,clean=clean['rate']['mean'],mixed=mix['rate']['mean'],
            delta=p['interval']['mean'],interval=p['interval'],gained=contrast['cells']['gained']['mean'],lost=contrast['cells']['lost']['mean'],
            clean_fpr=nc['rate']['mean'],mixed_fpr=nm['rate']['mean'],classification=p['classification']))
    save('report_primary.json', primary_rows)
    fig,ax=plt.subplots(figsize=(10,5.5))
    y=np.arange(8); v=np.array([100*r['delta'] for r in primary_rows]); lo=np.array([100*r['interval']['lower'] for r in primary_rows]); hi=np.array([100*r['interval']['upper'] for r in primary_rows])
    ax.errorbar(v,y,xerr=[v-lo,hi-v],fmt='o',color='#b2182b',capsize=4)
    ax.set_yticks(y,[f"{r['id']}  {r['group']} / {LABELS[r['fault']]} / {r['mix']}" for r in primary_rows]); ax.invert_yaxis()
    ax.axvline(0,color='black',lw=.8); ax.set_xlabel('Detection change: mixed - clean (percentage points)')
    ax.set_title('Eight fixed comparisons: W0, alpha=1%, m=15,999\nBonferroni family-8 approximate outer-t intervals, 25 replicates')
    ax.grid(axis='x',alpha=.2); fig.tight_layout(); figsave(fig,'01_primary8.png')

    # All 12 cross-grid figures use one global symmetric color scale.
    limit=float(np.ceil(max(abs(r['difference']['mean'])*100 for r in cross)/10)*10)
    crosslookup={(r['group'],r['mix_count'],r['mix_fault'],r['evaluation_fault'],r['method'],r['alpha']):r for r in cross}
    cross_figures=[]
    for method in METHODS:
        for alpha in (.01,.001):
            fig,axes=plt.subplots(6,2,figsize=(10,17),layout='constrained')
            for gi,g in enumerate(spec.groups()):
                for ni,n in enumerate(spec.MIX_COUNTS):
                    ax=axes[gi,ni]
                    mat=np.array([[100*crosslookup[g,n,a,b,method,alpha]['difference']['mean'] for b in spec.FAULTS] for a in spec.FAULTS])
                    im=ax.imshow(mat,cmap='RdBu',vmin=-limit,vmax=limit)
                    ax.set_xticks(range(3),['Gain -5%','Gain +5%','Private']);ax.set_yticks(range(3),['Gain -5%','Gain +5%','Private'])
                    ax.set_title(f'{g} | mixed {n}/15,999')
                    if ni==0: ax.set_ylabel('Fault in calibration')
                    if gi==5: ax.set_xlabel('Fault in evaluation')
                    for i in range(3):
                        for j in range(3):ax.text(j,i,f'{mat[i,j]:+.1f}',ha='center',va='center',color='white' if abs(mat[i,j])>limit*.57 else 'black',fontsize=11)
            fig.suptitle(f'{method}: detection change (mixed - clean), alpha={100*alpha:g}%\nExploratory paired means; common scale across all methods and levels',fontsize=14)
            fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6,label='Percentage points')
            name=f'cross_{METHODS.index(method)}_a{round(1/alpha)}.png';figsave(fig,name);cross_figures.append(dict(method=method,alpha=alpha,file='figures/'+name))
    save('figure_manifest.json',dict(cross_scale_pp=[-limit,limit],cross=cross_figures))

    # F and recalibrated C-clean: all shifted groups, both levels, all m.
    fig,axes=plt.subplots(2,4,figsize=(14,6),layout='constrained')
    for j,g in enumerate(spec.groups()[2:]):
        for i,alpha in enumerate((.01,.001)):
            ax=axes[i,j]
            for pol,color in [('F','#d95f02'),('C-clean','#2166ac')]:
                rr=[rate(f'{g}__F_{m}' if pol=='F' else spec.clean_id(g,m),g,alpha=alpha) for m in spec.SIZES]
                ax.errorbar(range(3),[r['rate']['mean']/alpha for r in rr],yerr=[r['rate']['sd']/alpha for r in rr],fmt='o-',label=pol,color=color,capsize=3)
            ax.axhline(1,color='black',ls=':',lw=1);ax.set_xticks(range(3),['7,999','15,999','31,999']);ax.set_title(f'{g} | alpha={100*alpha:g}%')
            ax.set_yscale('log');ax.grid(axis='y',alpha=.2)
            if j==0:ax.set_ylabel('Mean FPR / nominal alpha\n(error bar: outer SD)')
    axes[0,0].legend();figsave(fig,'02_fpr_shift_W0.png')
    fig,axes=plt.subplots(2,6,figsize=(16,6),layout='constrained')
    stability=[]
    for gi,g in enumerate(spec.groups()):
        for ai,alpha in enumerate((.01,.001)):
            ax=axes[ai,gi]
            for mi,method in enumerate(METHODS):
                rr=[rate(spec.clean_id(g,m),g,method,alpha) for m in spec.SIZES]
                for m,r in zip(spec.SIZES,rr):stability.append(dict(group=g,alpha=alpha,method=method,m=m,mean_fpr=r['rate']['mean'],sd_fpr=r['rate']['sd'],inactive=sum(r['lf0_structurally_inactive'] or [])))
                ax.plot(range(3),[r['rate']['sd']/alpha for r in rr],'o-',label=method,color=COLORS[mi],ms=3)
            ax.set_title(f'{g} | {100*alpha:g}%');ax.set_xticks(range(3),['8k','16k','32k']);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
            if gi==0:ax.set_ylabel('Outer FPR SD / nominal alpha')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=6)
    figsave(fig,'03_fpr_stability.png');save('report_stability.json',stability)

    threshold_rows=[]
    for g in spec.groups():
        for m in spec.SIZES:
            cc=[r for r in cal if r['calibration']['fit_key']==spec.clean_id(g,m)]
            assert len(cc)==25
            cuts=np.array([r['cut'] for r in cc])
            for ai,alpha in enumerate((.01,.001)):
                for mi,method in enumerate(METHODS):
                    for branch in range(3 if mi<5 else 1):
                        x=cuts[:,ai,mi,branch]
                        threshold_rows.append(dict(group=g,m=m,alpha=alpha,method=method,branch=branch,
                            mean=float(x.mean()),sd=float(x.std(ddof=1)),values=x.tolist()))
    save('report_threshold_stability.json',threshold_rows)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,g in zip(axes.ravel(),spec.groups()):
        for ai,alpha in enumerate((.01,.001)):
            for b,label in enumerate(('s_g','-s_g','s_p')):
                rr=[next(r for r in threshold_rows if r['group']==g and r['m']==m and r['alpha']==alpha and r['method']=='W0' and r['branch']==b) for m in spec.SIZES]
                ax.plot(range(3),[r['sd'] for r in rr],marker='o',ls='-' if ai==0 else '--',color=COLORS[b],label=f'{label}, {100*alpha:g}%')
        ax.set_title(g);ax.set_xticks(range(3),['7,999','15,999','31,999']);ax.set_ylabel('Outer SD of score cutoff');ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncols=3)
    fig.suptitle('W0 clean-calibration cutoff variability; no evaluation-based tuning');figsave(fig,'04_threshold_stability.png')

    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    saturation=[]
    for ax,g in zip(axes.ravel(),spec.groups()):
        for ai,alpha in enumerate((.01,.001)):
            rr=[rate(spec.clean_id(g,m),g,'LF0',alpha) for m in spec.SIZES]
            nums=[sum(r['lf0_structurally_inactive']) for r in rr]
            ax.plot(range(3),nums,'o-',label=f'alpha={100*alpha:g}%',color=COLORS[ai])
            for m,r,n in zip(spec.SIZES,rr,nums): saturation.append(dict(group=g,alpha=alpha,m=m,inactive=n,
                zero_null_alarm_replicates=r['zero_alarm_full_replicates'],fpr=r['rate']['mean'],
                private_detection=rate(spec.clean_id(g,m),g+'__P02','LF0',alpha)['rate']['mean'],
                w0_private_detection=rate(spec.clean_id(g,m),g+'__P02','W0',alpha)['rate']['mean']))
        ax.set_title(g);ax.set_xticks(range(3),['7,999','15,999','31,999']);ax.set_ylim(-.5,25.5);ax.set_yticks([0,5,10,15,20,25]);ax.grid(alpha=.2)
        ax.set_ylabel('Structurally inactive / 25')
    axes[0,0].legend();fig.suptitle('LF0 finite-reference saturation: clean calibration (reference fixed at 9,999)')
    figsave(fig,'05_lf0_saturation.png');save('report_saturation.json',saturation)

    fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
    scenarios=[('B_L','GM05'),('B_L','GP05'),('K200_L','P02'),('K2_M','P02')]
    trade=[]
    for ni,n in enumerate(spec.MIX_COUNTS):
        for gi,(g,f) in enumerate(scenarios):
            ax=axes[ni,gi]
            for mi,method in enumerate(METHODS):
                clean=rate(spec.clean_id(g,15999),g+'__'+f,method);mixed=rate(spec.mix_id(g,f,n),g+'__'+f,method)
                ax.plot([0,1],[100*clean['rate']['mean'],100*mixed['rate']['mean']],'o-',color=COLORS[mi],label=method,ms=4)
                nm=rate(spec.mix_id(g,f,n),g,method)
                trade.append(dict(group=g,fault=f,mix=n,method=method,clean_detection=clean['rate']['mean'],mixed_detection=mixed['rate']['mean'],mixed_fpr=nm['rate']['mean']))
            ax.set_xticks([0,1],['Clean',f'Mixed {n}']);ax.set_ylim(0,100);ax.set_title(g+' / '+LABELS[f]);ax.grid(axis='y',alpha=.2)
            if gi==0:ax.set_ylabel('Detection (%)')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncols=6)
    fig.suptitle('Fixed budgets and LF0 under matching-fault contamination; alpha=1%')
    figsave(fig,'06_budget_tradeoffs.png');save('report_budget_tradeoffs.json',trade)

    # Full tables, not only examples that look favorable.
    md('ALL_RATES.md',['# 全率・見逃し・枝の表','', '率は%。区間は点95%の外側t区間で同時保証なし。全25値・Wilson区間・予定分母範囲は[JSON](recomputed_rates.json)。同じ評価記録を複数文脈で再使用する。','']+
       table(['cal ID','評価case','方式','α %','検出/FP %','見逃し %','SD pp','点95%','成立/予定','非発火/25'],
       [[r['context']['id'],r['context']['case'],r['method'],fmt(r['alpha'],1),fmt(r['rate']['mean']),fmt(r['miss']['mean']),fmt(r['rate']['sd']),interval(r['rate']),f"{r['available']}/{r['planned']}",sum(r['lf0_structurally_inactive']) if r['lf0_structurally_inactive'] else '—'] for r in rates]))
    md('ALL_CROSS_RESULTS.md',['# 混入3種×評価3種：全1,296セル','', '探索的対応差。率差はpp、点95%区間に同時保証なし。正常FPRは同一行の3評価故障で共有され、独立な3正常群ではない。','']+
       table(['群','混入件数','混入故障','評価故障','方式','α %','検出差 pp','点95%','clean FPR %','mix FPR %'],
       [[r['group'],r['mix_count'],r['mix_fault'],r['evaluation_fault'],r['method'],fmt(r['alpha'],1),fmt(r['difference']['mean']),interval(r['difference']),
         fmt(rate(spec.clean_id(r['group'],15999),r['group'],r['method'],r['alpha'])['rate']['mean']),fmt(rate(spec.mix_id(r['group'],r['mix_fault'],r['mix_count']),r['group'],r['method'],r['alpha'])['rate']['mean'])] for r in cross]))
    md('ALL_FPR.md',['# FPR全792系列','', '家族792補正、主8とは別家族。超過フラグなしは上限制御や正常認定の証明ではない。','']+
       table(['cal ID','方式','α %','FPR %','SD pp','家族792区間 %','超過'],[[r['context']['id'],r['method'],fmt(r['alpha'],1),fmt(r['interval']['mean']),fmt(r['interval']['sd']),interval(r['interval']),r['excess_flag']] for r in fpr]))
    md('ALL_PAIRED_TRADEOFFS.md',['# 校正政策・方式間の対応得失','', '率は%。差は左−右（pp）。両側発火・左のみ取得・右のみ喪失・両側非発火は同じ記録の対応四分表。点95%区間は探索的。','', '## 校正政策','']+
       table(['評価case','左cal','右cal','方式','α %','差 pp','取得 %','喪失 %','両発火 %','両非発火 %'],
       [[r['contrast']['case'],r['left']['id'],r['right']['id'],r['method'],fmt(r['alpha'],1),fmt(r['difference']['mean'])]+[fmt(r['cells'][k]['mean']) for k in ('gained','lost','both','neither')] for r in policy])+
       ['', '## 同一校正内の方式比較','']+table(['cal ID','評価case','左方式','右方式','α %','差 pp','取得 %','喪失 %','両発火 %','両非発火 %'],
       [[r['context']['id'],r['context']['case'],r['left'],r['right'],fmt(r['alpha'],1),fmt(r['difference']['mean'])]+[fmt(r['cells'][k]['mean']) for k in ('gained','lost','both','neither')] for r in within]))
    md('ALL_CALIBRATION_STABILITY.md',['# 正常校正の件数・変動・LF0上限','', '閾値はスコア座標、FPRは%。SDは25外側反復の変動であり、評価記録だけの二項誤差ではない。','']+
       table(['群','方式','α %','m','FPR %','SD pp','LF0構造非発火/25'],[[r['group'],r['method'],fmt(r['alpha'],1),r['m'],fmt(r['mean_fpr']),fmt(r['sd_fpr']),r['inactive'] if r['method']=='LF0' else '—'] for r in stability])+
       ['','## 閾値の外側変動','']+table(['群','方式','α %','m','枝','閾値平均','閾値SD'],[[r['group'],r['method'],fmt(r['alpha'],1),r['m'],r['branch'],f"{r['mean']:.8g}",f"{r['sd']:.8g}"] for r in threshold_rows]))
    md('FIGURES.md',['# 図一覧','', '全交差図は共通対称色域。固定した全方式・両水準を掲載する。','']+
       [f"- [{r['method']}・α={100*r['alpha']:g}%]({r['file']})" for r in cross_figures]+
       ['', '- [主8比較](figures/01_primary8.png)', '- [W0源変化FPR](figures/02_fpr_shift_W0.png)', '- [全方式FPR変動](figures/03_fpr_stability.png)', '- [W0閾値変動](figures/04_threshold_stability.png)', '- [LF0非発火](figures/05_lf0_saturation.png)', '- [予算別得失](figures/06_budget_tradeoffs.png)'])
    save('report_digest.json',dict(primary_classes=dict(Counter(r['classification'] for r in primary)),
        fpr_flags={p:dict(series=sum(r['context']['policy']==p for r in fpr),flags=sum(r['context']['policy']==p and r['excess_flag'] for r in fpr)) for p in ('B','C-clean','F','C-mix')},
        cross_scale_pp=limit,figure_count=len(list(FIG.glob('*.png'))),
        tables=dict(rates=len(rates),cross=len(cross),fpr=len(fpr),policy=len(policy),within=len(within))))
    print('Figures and complete tables written:',len(list(FIG.glob('*.png'))),flush=True)


if __name__=='__main__': main()
