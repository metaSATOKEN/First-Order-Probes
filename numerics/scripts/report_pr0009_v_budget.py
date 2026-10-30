# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Figures and complete descriptive tables for the fixed-budget V audit."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from analyze_pr0009_v_budget import OUT, GROUPS, SIZES, ALPHAS, LABELS

def main():
    load=lambda n:json.loads((OUT/n).read_text())
    assert load('validation.json')['status']=='PASS_ANALYSIS'
    rows=load('resource_curve.json'); bd=load('integer_boundaries.json'); rev=load('reversals.json')
    ix={(r['group'],r['alpha'],r['n']):r for r in rows}
    d=np.load(OUT/'counts_and_intervals.npz'); states=d['states']
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#247a67','#b34344','#d6dbe2']; names=['Supported','Exceeded','Unresolved']
    handles=[Patch(facecolor=c,label=n) for c,n in zip(colors,names)]
    files=[]
    def save(fig,name):
        fig.savefig(figdir/name,dpi=160,bbox_inches='tight',facecolor='white');plt.close(fig);files.append(name)
    for ai,alpha in enumerate(ALPHAS):
        fig,axs=plt.subplots(2,3,figsize=(13,7),sharey=True)
        for ax,g in zip(axs.flat,GROUPS):
            vals=np.array([ix[g,alpha,int(n)]['labels'] for n in SIZES]);bottom=np.zeros(6)
            for k,c in enumerate(colors):
                ax.bar(range(6),vals[:,k],bottom=bottom,color=c,width=.72)
                for j,v in enumerate(vals[:,k]):
                    if v>=4:ax.text(j,bottom[j]+v/2,str(v),ha='center',va='center',color='white' if k<2 else '#323942',fontsize=9)
                bottom+=vals[:,k]
            ax.set_xticks(range(6),['0.5k','1k','2k','5k','10k','20k']);ax.set_ylim(0,52)
            ax.set_title(g);ax.set_xlabel('Current clean V records');ax.set_ylabel('Outer replicates (of 50)')
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        fig.suptitle(f'Fixed old F / W0: validation evidence at alpha = {100*alpha:g}%\nFPR cap = {125*alpha:g}%; nested prefixes, not a sequential stopping rule',fontsize=14)
        fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False)
        fig.tight_layout(rect=[0,.055,1,.91]);save(fig,f'0{ai+1}_resource_{ai}.png')
    for ai,alpha in enumerate(ALPHAS):
        fig,axs=plt.subplots(2,3,figsize=(12,7.3))
        for ax,g in zip(axs.flat,GROUPS):
            v=ix[g,alpha,20000];m=np.array(v['V_E'])
            ax.imshow(m,cmap='Blues',vmin=0,vmax=50)
            for (i,j),val in np.ndenumerate(m):ax.text(j,i,str(val),ha='center',va='center',color='white' if val>=25 else '#172331',fontsize=13)
            ax.set_xticks(range(3),['S','X','U']);ax.set_yticks(range(3),['S','X','U'])
            ax.set_xlabel('Independent E label');ax.set_ylabel('V label');ax.set_title(g)
        fig.suptitle(f'V versus independent E: 20,000 records each, alpha = {100*alpha:g}%\nCounts of 50 replicates; S = supported, X = exceeded, U = unresolved',fontsize=14)
        fig.tight_layout(rect=[0,0,1,.91]);save(fig,f'0{ai+3}_V_E_{ai}.png')
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.8),sharey=True)
    for ax,alpha in zip(axs,ALPHAS):
        v=[r for r in bd if r['alpha']==alpha]
        for key,c,name in [('nominal_support',colors[0],'Supported'),('nominal_unresolved','#778494','Unresolved'),('nominal_exceed',colors[1],'Exceeded')]:
            ax.plot(SIZES,100*np.array([r[key] for r in v]),'o-',color=c,label=name)
        ax.set_xscale('log');ax.set_xticks(SIZES,['500','1k','2k','5k','10k','20k'])
        ax.set_title(f'True p = alpha = {100*alpha:g}%');ax.set_xlabel('Fixed V size');ax.set_ylim(-3,103);ax.grid(alpha=.2)
        ax.annotate(f"Support at 20k: {v[-1]['nominal_support']*100:.2f}%",xy=(20000,v[-1]['nominal_support']*100),xytext=(-12,12),textcoords='offset points',ha='right',color=colors[0])
    axs[0].set_ylabel('Exact binomial probability (%)');axs[0].legend(loc='center left',frameon=False)
    fig.suptitle('Sample budget benchmark at p = alpha (analytic, not a new simulation)\nUnchanged CP tails eta = 0.0125; FPR cap = 1.25 alpha',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.88]);save(fig,'05_nominal_benchmark.png')
    fig,axs=plt.subplots(1,3,figsize=(12,9),sharey=True)
    cmap=ListedColormap(colors); norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    for ax,(g,ai) in zip(axs,[('B_M',0),('B_M',1),('K200_L',1)]):
        ax.imshow(states[:,GROUPS.index(g),:,ai],aspect='auto',interpolation='nearest',cmap=cmap,norm=norm)
        ax.set_xticks(range(6),['500','1k','2k','5k','10k','20k']);ax.set_yticks([0,9,19,29,39,49],[1,10,20,30,40,50])
        ax.set_title(f'{g}, alpha = {100*ALPHAS[ai]:g}%');ax.set_xlabel('Fixed V size')
        ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.grid(which='minor',axis='x',color='white',lw=.5)
    axs[0].set_ylabel('Outer replicate (original order)')
    fig.suptitle('Same-replicate paths: a supported label need not remain supported\nExamples are descriptive; no stopping decision is applied',fontsize=13)
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=[0,.055,1,.93]);save(fig,'06_label_paths.png')
    text=['# 全条件表：PR-0009正常確認Vの資源曲線','',
          'S=支持、X=超過、U=未解決。各セル50外側反復。Eは常に独立20,000件。件数間は同じVの入れ子であり、反復数を増やさない。区間幅はFPRのpercentage points（pp）。対応表の各三つ組はEのS/X/Uの順。','']
    for alpha in ALPHAS:
        for g in GROUPS:
            v=ix[g,alpha,20000]
            text += [f'## {g} / α={alpha}', '', f"固定E：S/X/U={v['E_labels']}、平均FPR={100*v['E_mean_fpr']:.4f}%。",'',
                     '| V件数 | V S/X/U | V支持→E S/X/U | V超過→E S/X/U | V未解決→E S/X/U | V平均FPR（%） | 平均区間幅（pp） |',
                     '|---|---|---|---|---|---|---|']
            for n in SIZES:
                r=ix[g,alpha,int(n)];triple=lambda a:'/'.join(map(str,a))
                text.append('| '+ ' | '.join([str(n),triple(r['labels'])]+[triple(a) for a in r['V_E']]+[f"{100*r['V_mean_fpr']:.4f}",f"{100*r['mean_interval_width']:.4f}"])+' |')
            text.append('')
    text+=['## 隣接件数間の全ラベル遷移','', '各三つ組は次の件数でのS/X/U。前後とも同じ反復で対応付ける。','',
           '| 群 | α | 件数の増加 | 旧S→S/X/U | 旧X→S/X/U | 旧U→S/X/U |','|---|---|---|---|---|---|']
    for r in load('transitions.json'):
        text.append('| '+' | '.join([r['group'],str(r['alpha']),f"{r['n_from']}→{r['n_to']}"]+['/'.join(map(str,v)) for v in r['matrix']])+' |')
    (OUT/'ALL_V_BUDGET_RESULTS.md').write_text('\n'.join(text)+'\n')
    facts=dict(figures=files,summary_rows=len(rows),transition_rows=len(load('transitions.json')),
               contradictory_pairs={str(a):sum(r['V_E'][0][1]+r['V_E'][1][0] for r in rows if r['alpha']==a) for a in ALPHAS},
               reversals=rev)
    (OUT/'report_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n')
    print('Figures:',len(files),'complete resource rows:',len(rows),'transition rows:',len(load('transitions.json')))

if __name__=='__main__': main()
