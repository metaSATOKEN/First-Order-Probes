# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Figures and exhaustive descriptive tables for PR-0009 reuse analysis A."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pr0009-mechanisms-mpl')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_pr0009_mechanisms import OUT,OLD,GROUPS,CL,EVAL,METHODS,dump,stat,spec

def load(n):return json.loads((OUT/n).read_text())
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def save(n):plt.savefig(OUT/'figures'/n,dpi=170,bbox_inches='tight');plt.close()
def main():
    assert load('validation.json')['status']=='PASS_MECHANISM_RECONSTRUCTION'
    (OUT/'figures').mkdir(exist_ok=True)
    ev=load('gate_events.json');gs=load('gate_summary.json');pol=load('policy_decomposition.json')
    ix={(v['group'],v['candidate'],v['evaluation'],v['alpha'],v['method'],v['gate']):v for v in pol}
    def row(g,c,e,a=.01,m='W0',gate='G-MC'):return ix[g,c,e,a,m,gate]
    def mat(key):return np.array([[next(v[key] for v in gs if v['group']==g and v['candidate']==c) for c in range(1,7)] for g in GROUPS])
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for ax,key,title in zip(axes,['MC_slips','MC_only','M_only'],['MC passes contamination','MC holds, M passes','M holds, MC passes']):
        a=mat(key);ax.imshow(a,vmin=0,vmax=50,cmap='Blues');ax.set_title(title+' (of 50)')
        ax.set_xticks(range(6),CL[1:],rotation=45,ha='right');ax.set_yticks(range(6),GROUPS)
        for (i,j),v in np.ndenumerate(a):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>27 else 'black')
    save('01_gate_map.png')
    colors={'both-pass':'#a0a5ad','both-hold':'#278b7a','MC-only':'#d77a13','M-only':'#b64b77'}
    fig,axes=plt.subplots(1,2,figsize=(11,5.6),layout='constrained')
    for wi,ax in enumerate(axes):
        cut=spec.GATE_CUTS[wi];items=[v for v in ev if v['mixed'] and v['group'].endswith('_M' if wi==0 else '_L')]
        for label,color in colors.items():
            vs=[v for v in items if v['M_MC_class']==label]
            ax.scatter([abs(v['T'][0]) for v in vs],[v['T'][1] for v in vs],s=11,alpha=.55,c=color,label=f'{label} ({len(vs)})')
        ax.axvline(cut[1],color='#b64b77',ls='--',lw=1,label='M threshold')
        ax.axvline(cut[2],color='black',lw=1);ax.axhline(cut[2],color='black',lw=1,label='MC boundaries')
        ax.set_xscale('symlog',linthresh=3);ax.set_yscale('symlog',linthresh=3)
        ax.set_xlabel('|T_M| (symlog)');ax.set_ylabel('T_C (symlog)');ax.set_title('M window' if wi==0 else 'L window')
        ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2)
    save('02_gate_geometry.png')
    # All cells, not only successful/failed examples. Effects are signed-T changes.
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for j in range(2):
        lim=max(abs(v[k][j]) for v in gs if v['candidate']>0 for k in ('mean_effect','sd_effect'))
        for col,key,title in [(0,'mean_effect','Mean contribution'),(1,'sd_effect','SD contribution')]:
            ax=axes[j,col];a=np.array([[next(v[key][j] for v in gs if v['group']==g and v['candidate']==c) for c in range(1,7)] for g in GROUPS])
            im=ax.imshow(a,cmap='coolwarm',vmin=-lim,vmax=lim)
            ax.set_title(title+f' to signed T_{"M" if j==0 else "C"} change')
            ax.set_xticks(range(6),CL[1:],rotation=35,ha='right');ax.set_yticks(range(6),GROUPS)
            for (i,k),v in np.ndenumerate(a):ax.text(k,i,f'{v:.2f}',ha='center',va='center',color='white' if abs(v)>.58*lim else 'black',fontsize=9)
        fig.colorbar(im,ax=axes[j,:],shrink=.75,label='Signed-T contribution')
    save('03_mean_sd_decomposition.png')
    panels=[('B_L',1,'GM05'),('K200_L',5,'P02'),('K2_M',6,'P02'),('V05_L',2,'GP05')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for ax,kind in zip(axes,['fault','normal']):
        x=np.arange(len(panels));width=.24
        for k,(key,label,color) in enumerate([('F','Old F','#8697a2'),('U','Candidate U','#df9f40'),('G','G-MC','#278b7a')]):
            ys=[100*row(g,c,e if kind=='fault' else 'normal')[key] for g,c,e in panels]
            ax.bar(x+(k-1)*width,ys,width,label=label,color=color)
        ys=[100*row(g,c,e if kind=='fault' else 'normal')['C_clean'] for g,c,e in panels]
        ax.scatter(x,ys,marker='D',c='black',label='C-clean reference',zorder=5,s=30)
        ax.set_xticks(x,[f'{g}\n{CL[c]} -> {e}' for g,c,e in panels],fontsize=8)
        ax.set_ylabel('Detection (%)' if kind=='fault' else 'False alarms (%)');ax.set_title('W0, nominal alpha = 1%')
        if kind=='normal':ax.axhline(1,color='black',ls=':',lw=1)
        ax.legend(fontsize=8)
    save('04_fallback_tradeoffs.png')
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    for ax,n,candidates in zip(axes,[160,800],[[1,3,5],[2,4,6]]):
        vals=np.array([[100*row(g,c,e)['delta']['mean'] for c in candidates for e in EVAL[1:]] for g in GROUPS])
        ax.imshow(vals,cmap='coolwarm',vmin=-90,vmax=90);ax.set_title(f'G-MC - U detection (pp), mix {n}')
        ax.set_xticks(range(9),[f'{CL[c]}\n{e}' for c in candidates for e in EVAL[1:]],fontsize=7)
        ax.set_yticks(range(6),GROUPS)
        for (i,j),v in np.ndenumerate(vals):ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>50 else 'black')
    save('05_all_cross_losses.png')
    a=[]
    for r in range(50):
        with np.load(OUT/'recomputed'/f'rep_{r:02d}.npz') as z:a.append({k:z[k] for k in z.files})
    cuts=np.stack([v['cuts'][GROUPS.index('V05_L'),:,0,0] for v in a]);jm=np.stack([v['joint'][4,2,2,0,0] for v in a])
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for b,ax in enumerate(axes):
        ax.errorbar(range(8),cuts[:,:,b].mean(0),yerr=cuts[:,:,b].std(0,ddof=1),fmt='o',capsize=3)
        ax.set_xticks(range(8),['F']+CL,rotation=55,ha='right');ax.set_title(['Positive-gain tail','Negative-gain tail','Private-input tail'][b]);ax.set_ylabel('LF score threshold')
    fig.suptitle('V05_L, W0, alpha = 1%: threshold mean +/- outer SD')
    save('06_thresholds.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    patterns=['none','G+','G-','G+G-','P','G+P','G-P','G+G-P']
    joint=jm.sum(0)/(50*5000)*100
    axes[0].imshow(joint,cmap='Blues',vmin=0,vmax=joint.max());axes[0].set_xticks(range(8),patterns,rotation=45,ha='right');axes[0].set_yticks(range(8),patterns)
    axes[0].set_xlabel('U (GM800) pattern');axes[0].set_ylabel('F pattern');axes[0].set_title('Same-record joint pattern (%)')
    for (i,j),v in np.ndenumerate(joint):
        if v:axes[0].text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8,color='white' if v>joint.max()*.6 else 'black')
    rec=row('V05_L',2,'GP05');x=np.arange(1,8)
    axes[1].bar(x-.18,np.array(rec['gain_patterns'])[1:]/250000*100,.36,label='Gained by fallback',color='#278b7a')
    axes[1].bar(x+.18,-np.array(rec['loss_patterns'])[1:]/250000*100,.36,label='Lost by fallback',color='#b64b77')
    axes[1].set_xticks(x,patterns[1:],rotation=45,ha='right');axes[1].set_ylabel('Fraction of all GP05 records (%)');axes[1].axhline(0,color='black',lw=.8);axes[1].legend(fontsize=8)
    fig.suptitle('V05_L, GM800 -> GP05, W0, alpha = 1%')
    save('07_cross_loss_branches.png')
    # Whole table, grouped by condition: normal and all three faults on the same row.
    full=['# 全条件の政策差と拾得・喪失','',
          '事後的記述分析。全50外側反復の平均。差はpp、拾得／喪失は全評価記録に対する%。正常での拾得は追加誤警報、喪失は誤警報抑制。Gは各行のゲート政策、比較相手は同じ候補U。全件成立、欠測・除外0。個別反復・SD・条件付き値は[JSON](policy_decomposition.json)。','']
    for ai,alpha in enumerate([.01,.001]):
        for method in METHODS:
            for gate in spec.GATES:
                full += [f'## {gate} / {method} / α={alpha:g}','']
                rows=[]
                for g in GROUPS:
                    for c in range(7):
                        records=[row(g,c,e,alpha,method,gate) for e in EVAL]
                        rows.append([g,CL[c],records[0]['hold_n']]+[f'{100*v["delta"]["mean"]:.3f}' for v in records]+
                                    [f'{100*v["gained"]:.3f}/{100*v["lost"]:.3f}' for v in records])
                full += [table(['群','候補','保留/50','ΔFPR','ΔGM','ΔGP','ΔP','正常 拾/喪','GM 拾/喪','GP 拾/喪','P 拾/喪'],rows),'']
    (OUT/'ALL_POLICY_DECOMPOSITIONS.md').write_text('\n'.join(full)+'\n')
    # Paired gate difference split into MC-only and M-only selections.
    gd=[]
    for g,group in enumerate(GROUPS):
        for ci in range(7):
            masks=[np.array([v['states'][g,ci,2]==1 and v['states'][g,ci,1]==0 for v in a]),
                   np.array([v['states'][g,ci,2]==0 and v['states'][g,ci,1]==1 for v in a])]
            for e,evn in enumerate(EVAL):
                jt=np.stack([v['joint'][g,ci,e,0,0] for v in a]);n=20000 if e==0 else 5000
                d=(jt[:,1:,0].sum(1)-jt[:,0,1:].sum(1))/n
                add=float((d*masks[0]).mean());rel=float((-d*masks[1]).mean())
                actual=row(group,ci,evn)['G']-row(group,ci,evn,gate='G-M')['G']
                assert abs(add+rel-actual)<1e-15
                gd.append(dict(group=group,candidate=ci,evaluation=evn,MC_only_n=int(masks[0].sum()),M_only_n=int(masks[1].sum()),
                               MC_only_contribution=add,M_only_release_contribution=rel,MC_minus_M=actual))
    dump('gate_difference_effects.json',gd)
    mixed=[v for v in ev if v['mixed']];slip=[v for v in mixed if not v['states'][2]]
    details=dict(slips=sum(1 for v in slip),slip_fixed_sd_crosses=sum(v['fixed_sd_crosses'] for v in slip),
        slips_by_type={f:sum(CL[v['candidate']].startswith(f) for v in slip) for f in ['GM','GP','P']},
        clean_false_holds=[v for v in ev if not v['mixed'] and v['states'][2]],
        cross_loss=row('V05_L',2,'GP05'),cross_fpr=row('V05_L',2,'normal'),
        cross_thresholds_mean=cuts.mean(0).tolist(),cross_joint_percent=joint.tolist(),
        gate_geometry_all_verified=True,figures=7,full_table_rows=1512)
    dump('report_facts.json',details)
    print(json.dumps({k:v for k,v in details.items() if k not in ['cross_loss','cross_fpr','cross_joint_percent','clean_false_holds']},ensure_ascii=False,indent=2))
    print('CROSS',details['cross_loss'])

if __name__=='__main__':main()
