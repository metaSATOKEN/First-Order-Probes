# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Tables and static scientific figures for the frozen PR-0008 saved-data analysis."""
from pathlib import Path
import json
import os
import math
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pr0008-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pr0008_shadow_methods as m
from analyze_pr0008_shadow import DEFAULT, write

OUT = DEFAULT
METHODS = ('W0','WP','WG','W+','W−','LF0')
EVALS = ('normal','GM05','GP05','P02')
DEN = np.array([20000,5000,5000,5000])
COLORS = ['#737373','#c0392b','#d89416','#2577b5','#18856b']


def summarize(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), minimum=float(x.min()), maximum=float(x.max()),
                fold_means=[float(x[i:i+5].mean()) for i in range(0,25,5)])


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def plot_save(name, fig):
    fig.savefig(OUT/'figures'/name,dpi=170,bbox_inches='tight'); plt.close(fig)


def main():
    if (OUT/'PR0008_SHADOW_GATE_REPORT_JA.md').exists(): raise FileExistsError('Preserve delivered reports')
    with np.load(OUT/'results.npz',allow_pickle=False) as f: d={k:f[k] for k in f.files}
    s=d['states']; count=d['policy_counts']; pair=d['policy_pairs']; rates=count/DEN[None,None,None,:,None,None,None]
    assert s.shape==(25,6,7,3) and count.shape==(25,6,7,4,2,6,5)
    validation=json.loads((OUT/'validation.json').read_text()); assert validation['status'].startswith('PASS')
    gates=[]; rate_rows=[]; paired_rows=[]
    gr=[]
    for gi,g in enumerate(m.GROUPS):
        for ci,(fault,n) in enumerate(m.CANDIDATES):
            for gate,gn in enumerate(m.GATES):
                st=s[:,gi,ci,gate]
                row=dict(group=g,candidate=ci,fault=fault,mix_count=n,gate=gn,
                    not_vetoed=int((st==0).sum()),inconsistent=int((st==1).sum()),unknown=int((st==2).sum()),
                    states=st.tolist(),fold_inconsistent=[int((st[i:i+5]==1).sum()) for i in range(0,25,5)])
                gates.append(row);gr.append([g,fault,n,gn,row['not_vetoed'],row['inconsistent'],row['unknown']])
            for ei,e in enumerate(EVALS):
                for ai,alpha in enumerate((.01,.001)):
                    for mi,method in enumerate(METHODS):
                        base=dict(group=g,candidate=ci,mix_fault=fault,mix_count=n,evaluation=e,alpha=alpha,method=method)
                        for pi,p in enumerate(m.POLICIES):
                            a=rates[:,gi,ci,ei,ai,mi,pi]
                            row=dict(**base,policy=p,rate=summarize(a),miss=summarize(1-a),
                                per_replicate_alarms=count[:,gi,ci,ei,ai,mi,pi].tolist(),planned_per_replicate=int(DEN[ei]),
                                zero_alarm_replicates=int((a==0).sum()),
                                lf0_structurally_inactive=int(d['policy_inactive'][:,gi,ci,ai,pi].sum()) if mi==5 else None,
                                difference_from_Cclean_pp=float(100*(a-rates[:,gi,0,ei,ai,mi,1]).mean()))
                            if pi>=2:
                                for tag,status in [('accepted',0),('held',None)]:
                                    mask=s[:,gi,ci,pi-2]==0 if status==0 else s[:,gi,ci,pi-2]>0
                                    row[tag]=dict(replicates=int(mask.sum()),records=int(mask.sum()*DEN[ei]),
                                        rate=float(a[mask].mean()) if mask.any() else None)
                            rate_rows.append(row)
                        for gate,gn in enumerate(m.GATES):
                            for ri,ref in enumerate(('U','F*')):
                                cells=pair[:,gi,ci,ei,ai,mi,gate,ri]/DEN[ei]
                                delta=(cells[:,1]-cells[:,2])*100
                                paired_rows.append(dict(**base,policy=gn,reference=ref,
                                    delta_pp=summarize(delta),cells_percent={k:summarize(cells[:,j]*100)
                                    for j,k in enumerate(('both','gained','lost','neither'))}))
    assert len(gates)==126 and len(rate_rows)==10080 and len(paired_rows)==12096
    write(OUT,'gate_results.json',gates);write(OUT,'policy_rates.json',rate_rows);write(OUT,'paired_tradeoffs.json',paired_rows)
    (OUT/'ALL_GATE_RESULTS.md').write_text('# 全ゲート結果\n\n各行25候補。cleanの不整合保留は誤保留、mixedの非保留はすり抜け。\n\n'+table(
        ['group','mixed fault','n','gate','非保留','不整合保留','不明'],gr))
    rr=[]
    for r in rate_rows:
        f=lambda x:'—' if x is None else f'{x*100:.4f}'
        ac=r.get('accepted',{}); he=r.get('held',{})
        rr.append([r['group'],r['mix_fault'],r['mix_count'],r['evaluation'],r['method'],r['alpha'],r['policy'],
            f(r['rate']['mean']),f(r['miss']['mean']),f(r['rate']['sd']),r['zero_alarm_replicates'],
            r['lf0_structurally_inactive'],ac.get('replicates','—'),f(ac.get('rate')),he.get('replicates','—'),f(he.get('rate'))])
    (OUT/'ALL_POLICY_RATES.md').write_text('# 全10,080政策率\n\n%単位。SDは反復間の記述量で信頼区間ではない。正常行の「miss」は非警報率。非保留／保留の条件付き率と分母を併記。構造非発火はLF0のみ。\n\n'+table(
        ['group','mix','n','eval','method','alpha','policy','rate %','miss/nonalarm %','SD pp','警報0/25','LF0非発火/25','非保留n','非保留率%','保留n','保留率%'],rr))
    pr=[]
    for r in paired_rows:
        pr.append([r['group'],r['mix_fault'],r['mix_count'],r['evaluation'],r['method'],r['alpha'],r['policy'],r['reference'],
            f"{r['delta_pp']['mean']:.4f}"]+[f"{r['cells_percent'][k]['mean']:.4f}" for k in ('both','gained','lost','neither')])
    (OUT/'ALL_PAIRED_TRADEOFFS.md').write_text('# 全12,096対応得失\n\nleft=ゲート政策、right=参照政策。正常行の取得は追加誤警報、喪失は抑制誤警報。%／pp単位。\n\n'+table(
        ['group','mix','n','eval','method','alpha','left','right','差pp','両方%','取得%','喪失%','両方なし%'],pr))
    (OUT/'figures').mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),constrained_layout=True)
    labels=['clean','GM160','GM800','GP160','GP800','P160','P800']
    for gate,ax in enumerate(axes):
        z=(s[...,gate]==1).sum(0)
        im=ax.imshow(z,vmin=0,vmax=25,cmap='YlOrRd',aspect='auto')
        ax.set_xticks(range(7),labels,rotation=50,ha='right');ax.set_yticks(range(6),m.GROUPS);ax.set_title(m.GATES[gate])
        for g in range(6):
            for c in range(7):ax.text(c,g,str(z[g,c]),ha='center',va='center',color='white' if z[g,c]>17 else 'black')
    fig.colorbar(im,ax=axes,label='HOLD-INCONSISTENT count / 25')
    fig.suptitle('Batch vetoes: clean column = false holds; mixed columns = contamination holds')
    plot_save('01_gate_holds.png',fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.4),constrained_layout=True)
    for ai,ax in enumerate(axes):
        for pi,p in enumerate(m.POLICIES):
            vals=rates[:,:,0,0,ai,0,pi]
            ax.errorbar(np.arange(6)+(pi-2)*.1,vals.mean(0)*100,yerr=vals.std(0,ddof=1)*100,
                        label=p,fmt='o',ms=4,color=COLORS[pi],capsize=2)
        ax.axhline([1,.1][ai],color='black',ls='--',lw=1);ax.set_xticks(range(6),m.GROUPS,rotation=25)
        ax.set_title(f'Clean update candidates; alpha = {[1,0.1][ai]}%');ax.set_ylabel('Normal FPR (%)')
    axes[0].legend(ncols=3,fontsize=8);fig.suptitle('Policy-level false alarms (W0); error bars = descriptive outer SD, not CI')
    plot_save('02_clean_update_fpr.png',fig)
    picks=[(1,1,1),(1,2,1),(1,3,2),(1,4,2),(3,5,3),(3,6,3),(2,5,3),(2,6,3)]
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for ax,(g,c,e) in zip(axes.flat,picks):
        a=rates[:,g,c,e,0,0].mean(0)*100
        ax.bar(range(5),a,color=COLORS);ax.axhline(rates[:,g,0,e,0,0,1].mean()*100,color='black',ls='--',label='C-clean reference')
        ax.set_xticks(range(5),m.POLICIES,rotation=35,ha='right');ax.set_ylim(0,105)
        ax.set_title(f'{m.GROUPS[g]} / {EVALS[e]} / mix {m.CANDIDATES[c][1]}');ax.set_ylabel('Detection (%)')
    axes.flat[0].legend(fontsize=7);fig.suptitle('Same-fault detection: legacy primary-eight locations (W0, alpha=1%); descriptive')
    plot_save('03_reference_eight_detection.png',fig)
    for gate,gn in enumerate(m.GATES):
        fig,axes=plt.subplots(3,4,figsize=(14,10),constrained_layout=True)
        for g in range(6):
            for ni,n in enumerate((160,800)):
                ax=axes.flat[g*2+ni]
                z=np.array([[100*(rates[:,g,1+fi2*2+ni,e,0,0,gate+2]-rates[:,g,1+fi2*2+ni,e,0,0,1]).mean()
                             for e in range(1,4)] for fi2 in range(3)])
                im=ax.imshow(z,cmap='RdBu',vmin=-100,vmax=100)
                ax.set_xticks(range(3),EVALS[1:]);ax.set_yticks(range(3),EVALS[1:]);ax.set_title(f'{m.GROUPS[g]} / mix {n}')
                for i in range(3):
                    for j in range(3):ax.text(j,i,f'{z[i,j]:+.1f}',ha='center',va='center',color='white' if abs(z[i,j])>60 else 'black')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Detection change (pp)',shrink=.65)
        fig.suptitle(f'{gn} minus unconditional update U (W0, alpha=1%); rows=mixed fault, columns=evaluated fault')
        plot_save(f'0{4+gate}_cross_gate{gate}.png',fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    for ax,z,title in [(axes[0],((s[...,2]>0)&(s[...,1]==0)).sum(0),'Additional holds by G-MC'),
                       (axes[1],((s[...,2]==0)&(s[...,1]>0)).sum(0),'Released holds by G-MC')]:
        im=ax.imshow(z,vmin=0,vmax=25,cmap='Blues',aspect='auto');ax.set_title(title)
        ax.set_xticks(range(7),labels,rotation=40,ha='right');ax.set_yticks(range(6),m.GROUPS)
        for g in range(6):
            for c in range(7):ax.text(c,g,str(z[g,c]),ha='center',va='center',color='white' if z[g,c]>15 else 'black')
    fig.colorbar(im,ax=axes,label='Count / 25');fig.suptitle('G-MC versus G-M: neither gate is assumed to contain the other')
    plot_save('07_gate_increment.png',fig)
    summary=dict(clean_holds={gn:(s[:,:,0,i]==1).sum(0).tolist() for i,gn in enumerate(m.GATES)},
        mixed_holds={gn:int((s[:,:,1:,i]==1).sum()) for i,gn in enumerate(m.GATES)},
        mixed_slips={gn:int((s[:,:,1:,i]==0).sum()) for i,gn in enumerate(m.GATES)},
        additional_mc_clean=int(((s[:,:,0,2]>0)&(s[:,:,0,1]==0)).sum()),
        additional_mc_mixed=int(((s[:,:,1:,2]>0)&(s[:,:,1:,1]==0)).sum()),
        released_mc_clean=int(((s[:,:,0,2]==0)&(s[:,:,0,1]>0)).sum()),
        released_mc_mixed=int(((s[:,:,1:,2]==0)&(s[:,:,1:,1]>0)).sum()),
        unknown=int((s==2).sum()),rates_series=len(rate_rows),paired_series=len(paired_rows))
    summary['H1_descriptive']=bool((s[:,:,1:,1]==1).any() and ((rates[:,:,1:,1:,0,0,3]-rates[:,:,1:,1:,0,0,1]).mean(0)>0).any())
    summary['H2_descriptive']=bool((s[...,1]!=s[...,2]).any())
    summary['H3_descriptive']=bool(((rates[:,2:,:,0,0,0,2:]-rates[:,2:,:,0,0,0,1,None]).mean(0)>0).any())
    write(OUT,'report_summary.json',summary)
    # Scientific interpretation is written after inspecting these fixed tables.
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Primary-eight locations:')
    for g,c,e in picks: print(m.GROUPS[g],m.CANDIDATES[c],EVALS[e],(100*rates[:,g,c,e,0,0].mean(0)).round(4).tolist())
    print('Clean FPR W0 1%:')
    print((100*rates[:,:,0,0,0,0].mean(0)).round(4))


if __name__=='__main__': main()
