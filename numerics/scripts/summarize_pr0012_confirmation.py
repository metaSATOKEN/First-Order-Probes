# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Report fixed PR-0012 results; no simulation, threshold fit, or E selection."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import binom, t
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'numerics/analysis/pr0012_64bf0f2c7697_confirmation_v1'
G = ['B_L','K2_M','K200_L','V05_L']
C = ['clean','GM800','GP800','BAL800','BAL400','TRI800']
E = ['normal','GM05','GP05','P02']
N = [20000,5000,5000,5000]
CON = ['F']+C+['C']
LABEL = ['SUPPORTED','UNRESOLVED','EXCEEDED']
COL = ['#238d9f','#707e89','#c45143','#e4b65b']

def read(name): return json.loads((OUT/name).read_text())
def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def pct(x): return 'NA' if x is None else f'{100*x:.4f}'
def table(head,rows):
    return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def stat(x):
    x=np.asarray(x,float);se=x.std(ddof=1)/np.sqrt(len(x));h=t.ppf(.975,len(x)-1)*se
    return dict(mean=float(x.mean()),lower=float(x.mean()-h) if se>0 else None,
                upper=float(x.mean()+h) if se>0 else None,n=len(x))
def ratio(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if not b.sum():return dict(mean=None,lower=None,upper=None,active=0)
    r=a.sum()/b.sum();u=a-r*b;se=u.std(ddof=1)/(np.sqrt(50)*b.mean())
    h=t.ppf(.975,49)*se;ok=(b>0).sum()>=2 and se>0
    return dict(mean=float(r),lower=float(r-h) if ok else None,upper=float(r+h) if ok else None,active=int((b>0).sum()))
def cp_tail(k,n,eta):
    lo=0 if k==0 else brentq(lambda p:binom.sf(k-1,n,p)-eta,0,1,xtol=1e-14)
    hi=1 if k==n else brentq(lambda p:binom.cdf(k,n,p)-eta,0,1,xtol=1e-14)
    return lo,hi
def classify(k,alpha):
    cap=alpha*1.25
    return 0 if binom.cdf(k,20000,cap)<=.00625 else 2 if binom.sf(k-1,20000,cap)<.00625 else 1
def verify_primary(a,primary):
    z=a['gates'];args=[(z[:,0,3,0],z[:,0,1,0]),(z[:,0,3,0],z[:,0,2,0]),
                      (z[:,0,3,1],z[:,0,3,0]),(z[:,2,5,1],z[:,2,4,1])]
    vals=[]
    for left,right in args:
        n10=int(((left==1)&(right==0)).sum());n01=int(((left==0)&(right==1)).sum())
        l,u=cp_tail(n10,50,.05/52);ll,uu=cp_tail(n01,50,.05/52)
        vals.append(((n10-n01)/50,l-uu,u-ll))
    def ci(x):
        mu=x.mean();sd=x.std(ddof=1)
        h=t.isf(.05/26,49)*sd/np.sqrt(50) if sd else np.sqrt(2*np.log(520)/50)
        return mu,max(-1,mu-h),min(1,mu+h)
    for e,n in enumerate(N):
        h=a['wcounts'][:,2,e,:,0]
        vals.append(ci(((h[:,6,1:8].sum(1)-h[:,5,1:8].sum(1))/n)))
    k=int((a['selection'][:,1,5,0,2]==7).sum());lo,hi=cp_tail(k,50,.05/26)
    vals.append((k/50,lo,hi))
    for e,n in enumerate(N):
        h=a['policy'][:,1,5,e,0];vals.append(ci((h[:,2,0]-h[:,0,0])/n))
    classifications=[]
    for p,v in zip(primary,vals):
        np.testing.assert_allclose([p['interval'][k] for k in ('mean','lower','upper')],v,rtol=1e-10,atol=2e-12)
        _,lo,hi=v
        if p['id']=='R01':lab='SUPPORTED' if lo>.5 else 'OPPOSITE' if hi<.5 else 'UNRESOLVED'
        elif p['expected']==1:lab='SUPPORTED' if lo>0 else 'OPPOSITE' if hi<0 else 'UNRESOLVED'
        else:lab='SUPPORTED' if hi<0 else 'OPPOSITE' if lo>0 else 'UNRESOLVED'
        assert lab==p['classification'];classifications.append(lab)
    return {k:classifications.count(k) for k in ('SUPPORTED','UNRESOLVED','OPPOSITE','TECHNICAL_UNESTABLISHED')}

def main():
    assert read('validation.json')['status']=='PASS_SAVED_DATA_REPRODUCTION'
    a=dict(np.load(OUT/'outer_arrays.npz'));primary=read('primary13.json')
    verification=dict(primary=verify_primary(a,primary))
    # Independently check all finite-state pair margins and alarm gains/losses.
    for e,n in enumerate(N):
        for g in range(4):
            for c in range(6):
                for ai in range(2):
                    for j,(left,right) in enumerate(((1,0),(2,0),(2,1))):
                        h=a['pairs'][:,g,c,e,ai,j];p=a['policy'][:,g,c,e,ai]
                        assert np.array_equal(h.sum(2),p[:,left])
                        assert np.array_equal(h.sum(1),p[:,right])
                        assert np.all(h.sum((1,2))==n)
                        assert np.array_equal(h[:,0,1:].sum(1)-h[:,1:,0].sum(1),p[:,left,0]-p[:,right,0])
    verification['policy_pair_tables_checked']=50*4*6*4*2*3
    verification['invalid']={k:int(a[k][...,-1].sum()) for k in ('wcounts','lcounts','policy','lf_pairs')}
    assert not any(verification['invalid'].values())
    assert not np.any(a['selection']==-2)
    # All information used for descriptive summaries remains indexed by outer replicate.
    policy=[];paired=[];cal=[];lf=[];support=[];selection=[];mechanism=[];strata=[]
    for g,gn in enumerate(G):
        for c,cn in enumerate(C):
            for ai,alpha in enumerate((.01,.001)):
                for pi in range(3):
                    sel=a['selection'][:,g,c,ai,pi]
                    selection.append(dict(group=gn,candidate=cn,alpha=alpha,policy=f'P{pi}',
                       counts={name:int((sel==k).sum()) for name,k in [('F',0),('U',c+1),('C',7),('DEFER',-1),('INVALID',-2)]}))
                for e,en in enumerate(E):
                    for pi in range(3):
                        h=a['policy'][:,g,c,e,ai,pi].astype(float);n=N[e]
                        policy.append(dict(group=gn,candidate=cn,alpha=alpha,evaluation=en,policy=f'P{pi}',
                          alarm=stat(h[:,0]/n),coverage=stat(h[:,:2].sum(1)/n),quiet=stat(h[:,1]/n),defer=stat(h[:,2]/n),
                          invalid=stat(h[:,3]/n),provided_alarm=ratio(h[:,0],h[:,:2].sum(1))))
                    for j,name in enumerate(('P1-P0','P2-P0','P2-P1')):
                        h=a['pairs'][:,g,c,e,ai,j];n=N[e]
                        gain=h[:,0,1:].sum(1)/n;loss=h[:,1:,0].sum(1)/n
                        paired.append(dict(group=gn,candidate=cn,alpha=alpha,evaluation=en,pair=name,
                          gain=stat(gain),loss=stat(loss),net=stat(gain-loss),matrix=h.sum(0).tolist()))
                    # Post-selection strata: descriptive, not causal identification.
                    sel=a['selection'][:,g,c,ai,2]
                    for name,mask in [('C',sel==7),('DEFER',sel==-1),('F',sel==0),('U',sel==c+1)]:
                        p=a['policy'][:,g,c,e,ai]
                        den=mask*N[e]
                        strata.append(dict(group=gn,candidate=cn,alpha=alpha,evaluation=en,stratum=name,n_repeats=int(mask.sum()),
                          P0=ratio(p[:,0,0]*mask,den),P2=ratio(p[:,2,0]*mask,den),
                          delta=ratio((p[:,2,0]-p[:,0,0])*mask,den)))
        for e,en in enumerate(E):
            for ai,alpha in enumerate((.01,.001)):
                for key,method,contexts in [('wcounts','W0',CON),('lcounts','LF0',['F','clean','C'])]:
                    for c,cn in enumerate(contexts):
                        h=a[key][:,g,e,c,ai];n=N[e]
                        branches=[]
                        for bit in (1,2,4):branches.append(stat(h[:,[k for k in range(8) if k&bit]].sum(1)/n))
                        cal.append(dict(group=gn,evaluation=en,alpha=alpha,method=method,context=cn,
                           alarm=stat(h[:,1:8].sum(1)/n),branches=branches,P_only=stat(h[:,4]/n),invalid=int(h[:,8].sum())))
                for c,cn in enumerate(('F','clean','C')):
                    h=a['lf_pairs'][:,g,e,c,ai];n=N[e]
                    lf.append(dict(group=gn,evaluation=en,alpha=alpha,context=cn,
                       W0_only=stat(h[:,1]/n),LF0_only=stat(h[:,2]/n),net=stat((h[:,1]-h[:,2])/n),
                       inactive=int((a['inactive'][:,g,c,ai]==1).sum())))
        for ci,context in enumerate(('F','C')):
            for ai,alpha in enumerate((.01,.001)):
                v=a['v_counts'][:,g,ci,ai,0]
                en=a['wcounts'][:,g,0,0 if ci==0 else 7,ai,1:8].sum(1)
                vl=np.array([classify(k,alpha) for k in v]);el=np.array([classify(k,alpha) for k in en])
                cross=np.zeros((3,3),int)
                for i,j in zip(vl,el):cross[i,j]+=1
                assert cross.sum()==50
                support.append(dict(group=gn,context=context,alpha=alpha,labels=LABEL,
                  V=np.bincount(vl,minlength=3).tolist(),E=np.bincount(el,minlength=3).tolist(),cross=cross.tolist(),
                  E_alarm=stat(en/20000)))
    for g,gn in enumerate(G):
        for role in ('audit','update'):
            for cn in C:
                rows=[[r for r in read(f'details/rep_{rep:02d}.json') if r['group']==gn and r['role']==role and r['candidate']==cn][0] for rep in range(50)]
                contrib=np.array([r['component_contributions'] for r in rows]);net=contrib.sum(1);mag=np.abs(contrib).sum(1)
                cancel=np.divide(np.abs(net),mag,out=np.zeros_like(net),where=mag>0)
                mechanism.append(dict(group=gn,role=role,candidate=cn,mean=np.mean([r['mean'] for r in rows],0).tolist(),
                  sd=np.mean([r['sd'] for r in rows],0).tolist(),signed_T=np.mean([r['signed_T'] for r in rows],0).tolist(),
                  components=contrib.mean(0).tolist(),absolute_net_to_sum_abs=cancel.mean(0).tolist()))
    for name,rows in [('policy_summary',policy),('paired_summary',paired),('calibration_summary',cal),('lf_summary',lf),
                      ('support_summary',support),('selection_summary',selection),('mechanism_summary',mechanism),('strata_summary',strata)]:dump(name+'.json',rows)
    for row in read('V_evidence.json'):
        assert LABEL[classify(row['alarms'],row['alpha'])]==row['label']
    verification['V_classifications_checked']=800
    verification['status']='PASS_SECOND_PATH_COUNTS_AND_PRIMARY_INFERENCE';dump('second_path_validation.json',verification)
    # Machine-generated complete tables; percentages and pp stated explicitly.
    (OUT/'ALL_POLICIES.md').write_text('# PR-0012 全政策・対応得失\n\n全50反復、率は%。主α=1%、0.1%は補助。quietは正常なら正しい非警報、故障なら提供下見逃し。\n\n'+
      table(['group','candidate','α','policy','E','alarm/N','coverage','alarm/provided','quiet/N','DEFER/N','INVALID/N'],
        [[r['group'],r['candidate'],r['alpha'],r['policy'],r['evaluation']]+[pct(r[k]['mean']) for k in ('alarm','coverage','provided_alarm','quiet','defer','invalid')] for r in policy])+
      '\n## 対応得失（pp）\n\nnormalの拾得は誤警報増。全4×4計数はpaired_summary.json。\n\n'+
      table(['group','candidate','α','pair','E','gain','loss','net'],[[r['group'],r['candidate'],r['alpha'],r['pair'],r['evaluation']]+[pct(r[k]['mean']) for k in ('gain','loss','net')] for r in paired]))
    (OUT/'ALL_CALIBRATIONS.md').write_text('# PR-0012 校正・読出し全条件\n\n全50反復の率（%）。枝は重複あり。LF0/W0は同じ公称αで実測FPR一致の比較ではない。\n\n'+
      table(['group','E','α','method','context','OR','G+','G−','P','P only','invalid count'],
       [[r['group'],r['evaluation'],r['alpha'],r['method'],r['context'],pct(r['alarm']['mean'])]+[pct(x['mean']) for x in r['branches']]+[pct(r['P_only']['mean']),r['invalid']] for r in cal])+
      '\n## W0−LF0 対応得失（pp）／LF0非発火反復\n\n'+table(['group','E','α','context','W0 only','LF0 only','net','inactive /50'],
       [[r['group'],r['evaluation'],r['alpha'],r['context']]+[pct(r[k]['mean']) for k in ('W0_only','LF0_only','net')]+[r['inactive']] for r in lf]))
    (OUT/'ALL_GATES_AND_SUPPORT.md').write_text('# PR-0012 ゲート・選択・V/E対応\n\n保留数/50（cleanでは誤保留）。\n\n'+table(['group','candidate','G-M','G-MC'],
       [[G[g],C[c],int((a['gates'][:,g,c,0]==1).sum()),int((a['gates'][:,g,c,1]==1).sum())] for g in range(4) for c in range(6)])+
      '\n## 政策選択数 /50\n\n'+table(['group','candidate','α','policy','F','U','C','DEFER','INVALID'],
       [[r['group'],r['candidate'],r['alpha'],r['policy']]+[r['counts'][k] for k in ('F','U','C','DEFER','INVALID')] for r in selection])+
      '\n## Vと独立E（支持／未解決／超過）\n\n同じCP規準で記述。Eを選択に用いない。crossの行はV、列はE。\n\n'+
      table(['group','context','α','V S/U/X','E S/U/X','cross','E FPR %'],[[r['group'],r['context'],r['alpha'],r['V'],r['E'],r['cross'],pct(r['E_alarm']['mean'])] for r in support]))
    figures(a,primary,cal,support,paired,mechanism)
    print(json.dumps(verification,indent=2))

def pick(rows,**kw):return next(r for r in rows if all(r[k]==v for k,v in kw.items()))
def figures(a,p,cal,support,paired,mech):
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):fig.savefig(figdir/name,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,7),gridspec_kw={'width_ratios':[1,1.25]})
    for ax,ids,title in [(axes[0],[0,1,2,3,8],'Gate / recovery probability'),(axes[1],[4,5,6,7,9,10,11,12],'Alarm-rate contrasts')]:
        for j,i in enumerate(ids):
            r=p[i];iv=r['interval'];offset=.5 if i==8 else 0
            x=100*(iv['mean']-offset);lo=100*(iv['lower']-offset);hi=100*(iv['upper']-offset)
            ax.errorbar(x,j,xerr=[[x-lo],[hi-x]],fmt='o',color=COL[0] if r['classification']=='SUPPORTED' else COL[3],capsize=3)
        ax.set_yticks(range(len(ids)),[p[i]['id']+(' (C rate - 50%)' if i==8 else ' '+p[i].get('evaluation','')) for i in ids]);ax.invert_yaxis()
        ax.axvline(0,color='k',lw=.8);ax.set_title(title);ax.set_xlabel('Difference (percentage points)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Fixed 13 contrasts: 12 supported, R05 unresolved\nFamily-adjusted intervals; R01 centered on the majority threshold');fig.tight_layout();save(fig,'01_primary13.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for b,ax in enumerate(axes):
        z=(a['gates'][...,b]==1).sum(0);ax.imshow(z,vmin=0,vmax=50,cmap='Blues')
        for g in range(4):
            for c in range(6):ax.text(c,g,str(z[g,c]),ha='center',va='center',color='white' if z[g,c]>28 else 'black')
        ax.set_xticks(range(6),C,rotation=35,ha='right');ax.set_yticks(range(4),G);ax.set_title(['G-M','G-MC'][b]+' holds / 50')
    fig.suptitle('Candidate audit only; clean holds are false holds');fig.tight_layout();save(fig,'02_gate_holds.png')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    vals=np.array([pick(mech,group='B_L',role='audit',candidate=c)['components'] for c in ('GM800','GP800','BAL800')])
    for i in range(3):axes[0].bar(np.arange(3)+(i-1)*.23,vals[:,i,0]*1e4,width=.23,label=['GM','GP','P'][i],color=COL[i])
    axes[0].axhline(0,color='k',lw=.7);axes[0].set_xticks(range(3),['GM800','GP800','BAL800']);axes[0].set_ylabel('Contribution to mean qM (x 1e-4)');axes[0].set_title('B_L: signed cancellation');axes[0].legend()
    for ax,es,title in [(axes[1],[0,1,2],'K200_L: FPR / gain detection'),(axes[2],[3],'K200_L: private-input detection')]:
        for c,cn in enumerate(('BAL400','TRI800')):
            vs=[pick(cal,group='K200_L',evaluation=E[e],alpha=.01,method='W0',context=cn)['alarm']['mean']*100 for e in es]
            ax.bar(np.arange(len(es))+(c-.5)*.32,vs,.32,label=cn,color=COL[c])
        ax.set_xticks(range(len(es)),[E[e] for e in es]);ax.set_title(title);ax.set_ylabel('Alarm rate (%)');ax.legend()
    fig.suptitle('Adding private-input records helps audit but contaminates update calibration');fig.tight_layout();save(fig,'03_cancellation_and_private_addition.png')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ai,alpha in enumerate((.01,.001)):
        for ci,context in enumerate(('F','C')):
            ax=axes[ai,ci];rows=[pick(support,group=g,context=context,alpha=alpha) for g in G];bottom=np.zeros(4)
            for j,lab in enumerate(LABEL):
                v=np.array([r['V'][j] for r in rows]);ax.bar(range(4),v,bottom=bottom,color=[COL[0],COL[3],COL[2]][j],label=lab);bottom+=v
            ax.set_xticks(range(4),G);ax.set_ylim(0,55);ax.set_title(f'{context}: alpha={alpha:g}');ax.set_ylabel('Outer repeats / 50')
    axes[0,0].legend(loc='upper center',fontsize=8);fig.suptitle('Independent normal V: evidence for the FPR cap, not sensitivity');fig.tight_layout();save(fig,'04_V_support.png')
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for ai,alpha in enumerate((.01,.001)):
        for pi in range(3):
            ax=axes[ai,pi];sel=a['selection'][:,:,5,ai,pi];bottom=np.zeros(4)
            for label,k,color in [('U',6,COL[3]),('F',0,COL[1]),('C',7,COL[0]),('DEFER',-1,COL[2])]:
                v=(sel==k).sum(0)*2;ax.bar(range(4),v,bottom=bottom,label=label,color=color);bottom+=v
            ax.set_xticks(range(4),G,rotation=20);ax.set_ylim(0,105);ax.set_title(f'P{pi}, alpha={alpha:g}');ax.set_ylabel('Selection fraction (%)')
    axes[0,0].legend(fontsize=8,ncol=2);fig.suptitle('TRI800: stop / recovery / unvalidated-update pathways');fig.tight_layout();save(fig,'05_policy_paths.png')
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for ai,alpha in enumerate((.01,.001)):
        for pi in range(3):
            ax=axes[ai,pi];z=a['policy'][:,1,5,:,ai,pi].sum(0);bot=np.zeros(4)
            for j,(name,color) in enumerate(zip(['alarm','provided quiet / miss','DEFER','INVALID'],[COL[0],COL[1],COL[3],COL[2]])):
                v=z[:,j]/(50*np.array(N))*100;ax.bar(range(4),v,bottom=bot,label=name,color=color);bot+=v
            ax.set_xticks(range(4),E);ax.set_ylim(0,105);ax.set_title(f'P{pi}, alpha={alpha:g}');ax.set_ylabel('All planned records (%)')
    axes[0,0].legend(fontsize=7);fig.suptitle('K2_M / TRI800: quiet, missed, deferred are not interchangeable');fig.tight_layout();save(fig,'06_recovery_tradeoffs.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ai,alpha in enumerate((.01,.001)):
        ax=axes[ai];rows=[pick(paired,group='K2_M',candidate='TRI800',alpha=alpha,evaluation=e,pair='P2-P0') for e in E]
        ax.bar(np.arange(4)-.18,[r['gain']['mean']*100 for r in rows],.36,label='New alarms',color=COL[0]);ax.bar(np.arange(4)+.18,[-r['loss']['mean']*100 for r in rows],.36,label='Lost alarms',color=COL[2])
        ax.axhline(0,color='k',lw=.7);ax.set_xticks(range(4),E);ax.set_ylabel('Percentage points, all planned E');ax.set_title(f'P2 - P0, alpha={alpha:g}');ax.legend()
    fig.suptitle('Paired gains and losses: normal alarms are false alarms');fig.tight_layout();save(fig,'07_paired_gain_loss.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,g in zip(axes,['K2_M','K200_L']):
        for j,method in enumerate(('W0','LF0')):
            rows=[pick(cal,group=g,evaluation=e,alpha=.001,method=method,context='C') for e in E]
            ax.bar(np.arange(4)+(j-.5)*.35,[r['alarm']['mean']*100 for r in rows],.35,label=method,color=COL[j])
        ax.set_xticks(range(4),E);ax.set_ylabel('Alarm rate (%)');ax.set_title(g+', independent C, alpha=0.001');ax.legend()
    fig.suptitle('Low-alpha readout: direct thresholds help some regimes, not all');fig.tight_layout();save(fig,'08_LF0_W0.png')

if __name__=='__main__':main()
