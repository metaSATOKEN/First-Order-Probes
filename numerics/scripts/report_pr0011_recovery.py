# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Descriptive summaries and figures for frozen PR-0011 saved-data replay."""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy.stats import t, binom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'numerics/analysis/pr0011_9adfa7a8c6e0_recovery_v2'
S=json.loads((OUT/'schema.json').read_text())
G=S['groups']; C=S['candidates']; TC=float(t.ppf(.975,49))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stat(x):
    x=np.asarray(x,float);m=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(50))
    return dict(mean=m,ci95=[m-TC*se,m+TC*se],sd=float(x.std(ddof=1)),n=50)
def ratio(a,b):
    den=int(b.sum());active=int(np.count_nonzero(b))
    if den==0:return dict(value=None,ci95=None,active_replicates=0,issued_records=0)
    theta=float(a.sum()/den);u=a-theta*b
    se=float(u.std(ddof=1)/(np.sqrt(50)*b.mean()))
    ci=[theta-TC*se,theta+TC*se] if active>=2 and np.isfinite(se) and np.any(u!=u[0]) else None
    return dict(value=theta,ci95=ci,active_replicates=active,issued_records=den)
def pct(x,d=3):return 'NA' if x is None else f'{100*x:.{d}f}'
def figsave(fig,name):
    fig.savefig(OUT/'figures'/name,dpi=165,bbox_inches='tight');plt.close(fig)
def heat(ax,x,title,cols):
    im=ax.imshow(x,vmin=0,vmax=100,cmap='viridis',aspect='auto')
    ax.set(title=title,xticks=range(len(cols)),xticklabels=cols,yticks=range(6),yticklabels=G)
    ax.tick_params(axis='x',rotation=50)
    for i in range(6):
        for j in range(len(cols)):ax.text(j,i,f'{x[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if x[i,j]<45 else 'black')
    return im
def main():
    for p in ['validation.json','second_path_validation.json']:assert json.loads((OUT/p).read_text())['status'].startswith('PASS')
    (OUT/'figures').mkdir(exist_ok=True)
    names=['counts','paired','actions','states','v_counts','v_labels','old_labels','e_labels','e_bounds','base','shadows']
    rows=[]
    for r in range(50):
        with np.load(OUT/'recomputed'/f'rep_{r:02d}.npz') as z:rows.append({k:z[k] for k in names})
    d={k:np.stack([z[k] for z in rows]) for k in names};del rows
    rates=[];pairs=[];lookup={};plook={}
    for g in range(6):
        for k,c in enumerate(C):
            for gate in range(3):
                for a,alpha in enumerate(S['alphas']):
                    for e,ev in enumerate(S['evaluations']):
                        n=20000 if e==0 else 5000
                        common=dict(group=G[g],candidate=c['id'],gate=S['gates'][gate],alpha=alpha,evaluation=ev,per_rep_N=n)
                        for pol in range(3):
                            x=d['counts'][:,g,k,gate,a,pol,e];z=d['actions'][:,g,k,gate,a,pol]
                            row=dict(**common,policy=S['policies'][pol],totals=x.sum(0).tolist(),
                                alarm=stat(x[:,0]/n),no_alarm=stat(x[:,1]/n),defer=stat(x[:,2]/n),invalid=stat(x[:,3]/n),
                                coverage=stat((x[:,0]+x[:,1])/n),issued_rate=ratio(x[:,0],x[:,0]+x[:,1]),
                                action_replicates=np.bincount(z,minlength=5).tolist(),
                                suppressed_shadow_alarms_U_F_C0=d['shadows'][:,g,k,gate,a,pol,e].sum(0).tolist())
                            rates.append(row);lookup[g,k,gate,a,pol,e]=row
                        for pi,(lp,rp) in enumerate(S['pairs']):
                            j=d['paired'][:,g,k,gate,a,pi,e]
                            gain=j[:,0,1:].sum(1);loss=j[:,1:,0].sum(1)
                            row=dict(**common,left=S['policies'][lp],right=S['policies'][rp],table=j.sum(0).tolist(),
                                gain_alarm=stat(gain/n),loss_alarm=stat(loss/n),alarm_difference=stat((gain-loss)/n))
                            pairs.append(row);plook[g,k,gate,a,pi,e]=row
    dump(OUT/'rates_all.json',rates);dump(OUT/'paired_all.json',pairs)
    # Every condition, all gates, both alphas, all evaluation faults.
    lines=['# PR-0011 全条件表','', '全件率の分母は各条件の50反復×N。%表記。正常行の警報は誤警報、故障行の非警報は提供見逃し。未判定はどちらでもない。提供時率の区間・対応差・反復値はJSON／NPZを参照。','',
        '|群|候補|ゲート|α|政策|評価|警報%|提供非警報%|未判定%|不成立%|提供率%|提供時警報%|提供反復|',
        '|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rates:
        lines.append('|'+ '|'.join([r['group'],r['candidate'],r['gate'],str(r['alpha']),r['policy'],r['evaluation'],
            *[pct(r[x]['mean']) for x in ['alarm','no_alarm','defer','invalid','coverage']],pct(r['issued_rate']['value']),str(r['issued_rate']['active_replicates'])])+'|')
    (OUT/'ALL_CONDITIONS.md').write_text('\n'.join(lines)+'\n')
    labcount=np.zeros((6,2,2,4),int);cross=np.zeros((6,2,2,4,4),int)
    for g in range(6):
        for c in range(2):
            for a in range(2):
                labcount[g,c,a]=np.bincount(d['v_labels'][:,g,c,a],minlength=4)
                cross[g,c,a]=np.bincount((4*d['v_labels'][:,g,c,a]+d['e_labels'][:,g,c,a]).astype(int),minlength=16).reshape(4,4)
    oldchange=np.zeros((6,2,4,4),int)
    for g in range(6):
        for a in range(2):oldchange[g,a]=np.bincount((4*d['old_labels'][:,g,a]+d['v_labels'][:,g,0,a]).astype(int),minlength=16).reshape(4,4)
    dump(OUT/'validation_evidence_summary.json',dict(V_labels=labcount.tolist(),V_by_E_labels=cross.tolist(),old_by_new_F_labels=oldchange.tolist(),
        label_order=S['labels'],group_order=G,cal_order=['F','C0'],alpha_order=S['alphas']))
    colors=['#207a78','#c45143','#d6ad4f','#555555'];polcolors=['#707e89','#c45143','#238d9f']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,7),sharey=True)
    for a in range(2):
        for c in range(2):
            ax=axs[a,c];bottom=np.zeros(6)
            for l in range(3):
                v=labcount[:,c,a,l];ax.bar(G,v,bottom=bottom,color=colors[l],label=S['labels'][l]);bottom+=v
            ax.set(title=f'{["F (old)","C0 (clean current)"][c]} | alpha={S["alphas"][a]:g}',ylim=(0,54),ylabel='Replicates / 50')
            for g in range(6):ax.text(g,51,str(labcount[g,c,a,0]),ha='center',fontsize=9)
    axs[0,0].legend(fontsize=8);fig.suptitle('V: support for FPR <= 1.25 alpha (8-tail family, eta=0.00625)')
    fig.tight_layout();figsave(fig,'01_validation_support.png')
    ix=[0,*range(28,37)];cols=['clean','GM','GP','P','GMGP 3:1','GMGP 1:1','GMGP 1:3','GM/P 1:1','GP/P 1:1','GM/GP/P 1:1:2']
    fig,axs=plt.subplots(2,2,figsize=(16,9))
    for a in range(2):
        for p in [1,2]:
            xx=np.array([[100*lookup[g,k,2,a,p,0]['coverage']['mean'] for k in ix] for g in range(6)])
            im=heat(axs[a,p-1],xx,f'P{p} coverage | alpha={S["alphas"][a]:g}',cols)
    fig.suptitle('G-MC: issued decisions (% of all planned records), n=800 contamination')
    fig.tight_layout();figsave(fig,'02_coverage.png')
    for a in range(2):
        fig,axs=plt.subplots(2,2,figsize=(13,8));xx=np.arange(6)
        for e,ax in enumerate(axs.flat):
            for p in range(3):
                vv=np.array([100*lookup[g,36,2,a,p,e]['alarm']['mean'] for g in range(6)])
                ax.bar(xx+(p-1)*.24,vv,width=.24,label=f'P{p}',color=polcolors[p])
            ax.set(xticks=xx,xticklabels=G,ylabel='% of planned records',title=['Normal: false alarms','Negative gain GM05','Positive gain GP05','Private input P02'][e])
            if e==0:ax.axhline(125*S['alphas'][a],ls='--',color='black',lw=1,label='1.25 alpha (reference)')
            else:ax.set_ylim(0,105)
        axs[0,0].legend(fontsize=8)
        fig.suptitle(f'G-MC | GM:GP:P=200:200:400 | alpha={S["alphas"][a]:g}\nDeferral reduces both full-record false alarms and detections; it is not a negative decision')
        fig.tight_layout();figsave(fig,f'0{3+a}_full_record_performance.png')
    fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for e,ax in enumerate(axs,1):
        ga=np.array([100*plook[g,36,2,0,2,e]['gain_alarm']['mean'] for g in range(6)])
        lo=np.array([100*plook[g,36,2,0,2,e]['loss_alarm']['mean'] for g in range(6)])
        ax.barh(np.arange(6)-.15,ga,height=.3,color='#238d9f',label='New alarm')
        ax.barh(np.arange(6)+.15,-lo,height=.3,color='#c45143',label='Lost alarm')
        ax.set(yticks=range(6),yticklabels=G,xlabel='Percentage points of all records',title=S['evaluations'][e]);ax.axvline(0,color='black',lw=.6)
        for g in range(6):ax.text(.35,g-.15,f'+{ga[g]:.3f}',va='center',fontsize=8)
    axs[0].legend(fontsize=8,loc='lower left');fig.suptitle('P2 vs P0: matched gains and losses | G-MC, alpha=0.01, triple n=800')
    fig.tight_layout();figsave(fig,'05_paired_gain_loss.png')
    fig,axs=plt.subplots(2,3,figsize=(13,7),sharey=True)
    for gi,g in enumerate([2,5]):
        for e in range(1,4):
            ax=axs[gi,e-1];bot=np.zeros(3)
            for st,color in enumerate(['#238d9f','#7a8a96','#e4b65b']):
                vv=np.array([100*d['counts'][:,g,36,2,0,p,e,st].mean()/5000 for p in range(3)])
                ax.bar(['P0','P1','P2'],vv,bottom=bot,color=color,label=['ALARM','NO-ALARM (miss)','DEFER'][st])
                for p in range(3):
                    if vv[p]>7:ax.text(p,bot[p]+vv[p]/2,f'{vv[p]:.1f}',ha='center',va='center',fontsize=9)
                bot+=vv
            ax.set(title=G[g]+' | '+S['evaluations'][e],ylim=(0,100),ylabel='% of planned fault records')
    axs[0,0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,1.38),ncol=3)
    fig.suptitle('Lost decisions are DEFER, not ordinary misses | G-MC, alpha=0.01, triple n=800',y=1.02)
    fig.tight_layout();figsave(fig,'06_miss_vs_defer.png')
    fig,axs=plt.subplots(1,2,figsize=(16,5))
    for ni,inds in enumerate([list(range(1,10)),list(range(28,37))]):
        x=np.take(d['states'][:,:,:,2],inds,axis=2)
        assert x.shape==(50,6,9)
        heat(axs[ni],(x==0).mean(0)*100,f'Candidate contamination n={[80,800][ni]}',cols[1:])
    fig.suptitle('Unverified U path: gate passes, so all three policies use U (%) | G-MC')
    fig.tight_layout();figsave(fig,'07_unverified_update_path.png')
    fig,axs=plt.subplots(2,2,figsize=(12,10))
    points=[]
    for a,alpha in enumerate(S['alphas']):
        grid=np.arange(20001);support=int(grid[binom.cdf(grid,20000,1.25*alpha)<=.00625].max())
        exceed=int(grid[binom.sf(grid-1,20000,1.25*alpha)<.00625].min())
        for c,b in enumerate([1,2]):
            ax=axs[a,c]
            for g,group in enumerate(G):
                vx=d['v_counts'][:,g,c,a];ey=d['base'][:,g,0,0,a,b,1:].sum(-1)
                ax.scatter(vx,ey,s=15,alpha=.65,label=group)
                points.append(dict(group=group,cal=['F','C0'][c],alpha=alpha,V=vx.tolist(),E=ey.tolist()))
            for z,ls in [(support,':'),(exceed,'--')]:ax.axhline(z,c='gray',ls=ls,lw=.8);ax.axvline(z,c='gray',ls=ls,lw=.8)
            ax.set(xlabel='V normal alarms / 20,000',ylabel='Independent E normal alarms / 20,000',title=f'{["F","C0"][c]} | alpha={alpha:g}\nSupport <= {support}; exceed >= {exceed}')
    axs[0,0].legend(fontsize=8);fig.suptitle('V and E are separate finite samples; E labels are not ground-truth FPR')
    fig.tight_layout();figsave(fig,'08_validation_vs_evaluation.png');dump(OUT/'figure08_points.json',points)
    # Compact machine-readable headline values; prose is authored separately.
    focus={}
    for g,group in enumerate(G):
        for a,alpha in enumerate(S['alphas']):
            focus[f'{group}/{alpha}']=dict(
                policies=[[lookup[g,36,2,a,p,e] for e in range(4)] for p in range(3)],
                paired=[[plook[g,36,2,a,pi,e] for e in range(4)] for pi in range(3)])
    dump(OUT/'focus_triple800.json',focus)
    dump(OUT/'summary_manifest.json',dict(rate_rows=len(rates),paired_rows=len(pairs),figures=8,independent_replicates=50,
        statuses=S['statuses'],actions=S['actions'],full_rate_ci='pointwise paired outer-50 t, df=49',issued_ci='outer-replicate ratio delta method',
        inputs={str((OUT/'recomputed'/f'rep_{r:02d}.npz').relative_to(ROOT)):hashlib.sha256((OUT/'recomputed'/f'rep_{r:02d}.npz').read_bytes()).hexdigest() for r in range(50)},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('SUMMARY PASS',len(rates),len(pairs),'8 figures')
if __name__=='__main__':main()
