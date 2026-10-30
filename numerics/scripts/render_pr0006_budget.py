# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Figures and exhaustive Markdown tables from audited saved-data results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pr0006_spec as spec

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'numerics/analysis/pr0006_6f23fc7946f8_performance_v2'
DATA = json.loads((OUT/'recomputed_results.json').read_text())
assert json.loads((OUT/'validation.json').read_text())['status'] == 'PASS_SAVED_DATA_RECOMPUTATION'
METHODS = spec.METHODS
COLORS = ['#303c52', '#17945b', '#d18000', '#c33b43', '#2b70bd', '#8756a6']
RATE = {(r['case'], r['policy'], r['alpha'], r['method']):r for r in DATA['rates']}
PAIR = {(r['case'], r['policy'], r['alpha'], r['left'], r['right']):r for r in DATA['paired']}
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.titleweight':'bold', 'savefig.dpi':180, 'figure.facecolor':'white'})
(OUT/'figures').mkdir(exist_ok=True)


def rate(c, p, a, m): return RATE[c, p, a, m]['rate']['mean']*100
def fmt(x, digits=4): return '未定義' if x is None else f'{100*x:.{digits}f}'
def ci(r): return f"{fmt(r['mean'])} [{fmt(r['lower'])}, {fmt(r['upper'])}]"
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])+'\n'
def write(name, s): (OUT/name).write_text(s, encoding='utf-8')
def savefig(fig, name):
    fig.savefig(OUT/'figures'/name, bbox_inches='tight'); plt.close(fig)


def exhaustive_tables():
    primary = table(['条件','方針','α%','差','平均 [同時95%区間] pp','符号判定','1 pp目安'],
        [[r['case'],r['policy'],fmt(r['alpha'],1),r['left']+'−'+r['right'],ci(r['interval']),r['classification'],r['magnitude']] for r in DATA['primary25']])
    write('PRIMARY_RESULTS.md', '# 主要25比較\n\n家族25、df24、両側Bonferroni95%。1 ppは記述上の目安。\n\n'+primary)
    s = '# 全条件の率・見逃し・誤警報\n\n率と区間の単位は%。区間は25反復の点95% t区間。SD=0の区間は未定義。正常行の「見逃し」は正常非警報率であって故障見逃しではない。種別で区別する。全予定分母／成立数を保持。\n\n'
    s += table(['条件','方針','種別','α%','方式','警報率 [95%区間]','非警報率 [95%区間]','SD%','MCSE%','q10/50/90%','成立/予定','全予定分母の率下限%','FPR範囲%','0警報反復','LF0常時非発火'],
        [[r['case'],r['policy'],r['kind'],fmt(r['alpha'],1),r['method'],ci(r['rate']),ci(r['miss']),fmt(r['rate']['sd']),fmt(r['rate']['mcse']),
          '/'.join(fmt(r['rate'][q]) for q in ('q10','q50','q90')),f"{r['available']}/{r['planned']}",fmt(r['detection_lower']),
          '/'.join(fmt(x) for x in r['fpr_range']),r['zero_alarm_full_replicates'],r['lf0_structurally_inactive_replicates']] for r in DATA['rates']])
    s += '\n## 正常FPR216系列の同時区間\n\n家族216。下限>公称αを超過フラグとする。フラグなしは条件付き保証ではない。\n\n'
    s += table(['条件','方針','α%','方式','FPR [同時95%区間] %','超過','不成立'],
        [[r['case'],r['policy'],fmt(r['alpha'],1),r['method'],ci(r['interval']),r['excess_flag'],r['unavailable']] for r in DATA['fpr216']])
    s += '\n## 正負差・C−F差\n\n点95%記述区間。異なる符号故障は同一波形ではなく、同じ外側反復番号での対比。C/Fは同じ評価記録。\n\n'
    s += table(['対比','条件','対照条件','方針','α%','方式','差 [95%区間] pp'],
        [[r['kind'],r['case'],r['other'],r['policy'],fmt(r['alpha'],1),r['method'],ci(r['interval'])] for r in DATA['contrasts']])
    write('ALL_RESULTS.md', s)
    s = '# 全15方式対の同一記録における拾得・損失\n\n左方式−右方式。全四分表の分母は同時成立記録で、今回は全予定記録と一致。単位%、差だけpp。25反復の点95%区間。正常のgained/lostは追加誤警報／除去誤警報を意味し、性能向上と呼ばない。\n\n'
    s += table(['条件','方針','α%','左/右','両方警報','左のみ(拾得)','右のみ(損失)','両方非警報','正味差 pp','成立/予定'],
        [[r['case'],r['policy'],fmt(r['alpha'],1),r['left']+'/'+r['right'],*[ci(r['cells'][k]) for k in ('both','gained','lost','neither')],
          ci(r['difference']),f"{r['joint_available']}/{r['planned']}"] for r in DATA['paired']])
    write('ALL_PAIRED_RESULTS.md', s)
    s = '# 全枝・パターン\n\n枝率は重複を含む。G+とG−は同時発火せず、Pとは重なりうる。枝名は原因診断ではない。パターン0–7はG+ bit0、G− bit1、P bit2。件数の総和=成立数。\n\n'
    s += table(['条件','方針','α%','方式','G+ [95%] %','G− [95%] %','P [95%] %','パターン0/1/2/3/4/5/6/7件'],
        [[r['case'],r['policy'],fmt(r['alpha'],1),r['method'],*[ci(x) for x in r['branches']],'/'.join(map(str,r['pattern_counts']))] for r in DATA['rates']])
    write('ALL_BRANCH_RESULTS.md', s)


def primary_plot():
    fig, axes = plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax, k in zip(axes.flat, range(0,24,4)):
        rows = DATA['primary25'][k:k+4]
        for j, r in enumerate(rows):
            q = r['interval']; x = 100*q['mean']
            ax.plot(x,j,'o',color=COLORS[j+1])
            if q['lower'] is not None: ax.hlines(j,100*q['lower'],100*q['upper'],color=COLORS[j+1],lw=2)
        ax.axvspan(-1,1,color='#d8dee5',alpha=.45); ax.axvline(0,color='black',lw=.7)
        ax.set_yticks(range(4), METHODS[1:5]); ax.invert_yaxis()
        ax.set_title(rows[0]['case']+' / '+rows[0]['policy'])
        ax.set_xlabel('Detection difference vs W0 (pp)'); ax.grid(axis='x',alpha=.2)
    fig.suptitle('Fixed-budget trade-offs | nominal 1% | simultaneous 95% intervals (family 25)')
    savefig(fig,'01_primary_differences.png')


def gain_plot():
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(group,pol,alpha) in zip(axes.flat,[('B_L','B',.01),('V05_L','C',.01),('B_L','B',.001),('V05_L','C',.001)]):
        for j,m in enumerate(METHODS):
            y=[rate(group+'__'+tag,pol,alpha,m) for tag in ('GM05','GP05')]
            x=np.arange(2)+(j-2.5)*.12
            ax.bar(x,y,.11,color=COLORS[j],label=m)
            for xx,tag in zip(x,('GM05','GP05')):
                q=RATE[group+'__'+tag,pol,alpha,m]['rate']
                if q['lower'] is not None: ax.vlines(xx,100*q['lower'],100*q['upper'],color='black',lw=.7)
        ax.set_xticks([0,1],['Gain -5%','Gain +5%']);ax.set_ylim(0,100)
        ax.set_ylabel('Detection (%)');ax.set_title(f'{group} / {pol} | nominal {100*alpha:g}%');ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=6)
    fig.suptitle('Signed gain sensitivity | long window 8.192 s | pointwise 95% intervals')
    savefig(fig,'02_signed_gain.png')


def private_plot():
    groups=[s+'_'+w for s in ('B','K2','K200','V05','V15') for w in ('M','L')]
    fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained',sharex=True)
    for ax,alpha in zip(axes,[.01,.001]):
        for j,m in enumerate(METHODS):
            y=[rate(g+'__P02','B' if g.startswith('B_') else 'C',alpha,m) for g in groups]
            ax.plot(range(10),y,marker='o',ms=4,color=COLORS[j],label=m)
        ax.set_ylim(-2,102);ax.set_ylabel('Detection (%)');ax.set_title(f'Private input h=0.02 | nominal {100*alpha:g}%')
        ax.grid(alpha=.2)
    axes[-1].set_xticks(range(10),[g.replace('_','\n') for g in groups])
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=6)
    fig.suptitle('Private-input sensitivity across sources | baseline B, shifted calibration C')
    savefig(fig,'03_private_source_shift.png')


def paired_plot():
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for ax,(case,pol) in zip(axes.flat,[('B_L__GM05','B'),('B_L__GP05','B'),('B_M__P02','B'),('K200_L__P02','C'),('V05_L__GM05','C'),('V05_L__GP05','C')]):
        for j,m in enumerate(METHODS[1:]):
            r=PAIR[case,pol,.01,m,'W0'];g=r['cells']['gained']['mean']*100;l=r['cells']['lost']['mean']*100
            ax.barh(j,g,color=COLORS[j+1]);ax.barh(j,-l,color=COLORS[j+1],alpha=.35)
            ax.plot(g-l,j,'kd',ms=4)
        ax.set_yticks(range(5),METHODS[1:]);ax.invert_yaxis();ax.axvline(0,color='k',lw=.7)
        ax.set_title(case+' / '+pol);ax.set_xlabel('Lost < 0 | gained > 0 (% of all fault records)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Same-record gains and losses versus W0 | nominal 1% | diamonds: net difference')
    savefig(fig,'04_paired_gains_losses.png')


def fpr_plot():
    contexts=[(c,p) for c,p in spec.contexts() if spec.conditions()[c]['kind']=='normal']
    flag={(r['case'],r['policy'],r['alpha'],r['method']):r['excess_flag'] for r in DATA['fpr216']}
    fig,axes=plt.subplots(1,2,figsize=(13,10),layout='constrained')
    for ax,alpha in zip(axes,[.01,.001]):
        values=np.array([[rate(c,p,alpha,m) for m in METHODS] for c,p in contexts])
        image=ax.imshow(np.maximum(values/(alpha*100),.01),cmap='RdYlBu_r',norm=LogNorm(.01,10),aspect='auto')
        for i,(c,p) in enumerate(contexts):
            for j,m in enumerate(METHODS):
                mark='*' if flag[c,p,alpha,m] else ''
                ax.text(j,i,f'{values[i,j]:.3f}{mark}',ha='center',va='center',fontsize=8,
                        color='white' if values[i,j]/(alpha*100)>5 or values[i,j]/(alpha*100)<.03 else 'black')
        ax.set_xticks(range(6),METHODS);ax.set_yticks(range(len(contexts)),[c+' / '+p for c,p in contexts])
        ax.set_title(f'Nominal {alpha*100:g}% | annotations: actual FPR (%)')
        for y in [1.5,5.5,9.5,13.5]:ax.axhline(y,color='gray',lw=.7)
    fig.colorbar(image,ax=axes,label='Actual / nominal FPR (log scale; colors clipped to 0.01–10)',shrink=.65)
    fig.suptitle('Source transfer: frozen thresholds F versus recalibration C\n* lower simultaneous 95% bound exceeds nominal (family 216)')
    savefig(fig,'05_false_alarm_source_shift.png')


def low_plot():
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,c,label in zip(axes,['K2_M__P02','K2_M'],['Detection (%)','False alarm (%)']):
        for j,m in enumerate(METHODS):
            row=RATE[c,'C',.001,m];q=row['rate'];y=q['mean']*100
            ax.bar(j,y,color=COLORS[j]);ax.text(j,y+.03 if c=='K2_M' else y+1,f'{y:.3f}',ha='center',fontsize=9)
            if q['lower'] is not None:ax.vlines(j,100*q['lower'],100*q['upper'],color='black',lw=1)
        ax.set_xticks(range(6),METHODS);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
        ax.set_title(c+' / C')
    axes[0].set_ylim(0,106);axes[1].axhline(.1,ls='--',color='black',lw=.8);axes[1].set_ylim(-.015,.19)
    inactive=RATE['K2_M','C',.001,'LF0']['lf0_structurally_inactive_replicates']
    fig.suptitle(f'Low-level direct calibration | nominal 0.1% | LF0 structurally inactive: {inactive}/25\nPointwise 95% intervals; not a matched-realized-FPR comparison')
    savefig(fig,'06_low_level_rank_ceiling.png')


def branch_plot():
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained',sharey=True)
    for ax,(c,p) in zip(axes,[('B_L','B'),('K2_M','C'),('K200_L','C')]):
        bottom=np.zeros(6)
        # Disjoint assignment: G+ including P overlap; G- including P overlap; P only.
        for pats,label,color in [([1,5],'G+ (incl. overlap)','#c33b43'),([2,6],'G- (incl. overlap)','#2b70bd'),([4],'P only','#17945b')]:
            y=[100*sum(RATE[c,p,.01,m]['pattern_counts'][i] for i in pats)/RATE[c,p,.01,m]['available'] for m in METHODS]
            ax.bar(range(6),y,bottom=bottom,color=color,label=label);bottom+=y
        ax.axhline(1,color='black',ls='--',lw=.8);ax.set_xticks(range(6),METHODS);ax.set_title(c+' / '+p);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Disjoint false-alarm contribution (%)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.suptitle('Where the normal false alarms go | nominal 1%\nDisjoint display partition, not causal attribution')
    savefig(fig,'07_normal_alarm_components.png')


if __name__=='__main__':
    exhaustive_tables()
    for f in (primary_plot,gain_plot,private_plot,paired_plot,fpr_plot,low_plot,branch_plot):f()
    print('Created four exhaustive tables and seven figures.')
