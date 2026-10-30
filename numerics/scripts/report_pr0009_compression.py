# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Static figures and exhaustive tables; no detector choices from evaluation."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_pr0009_compression import OUT,G,EV,CAL,AL

def main():
    load=lambda n:json.loads((OUT/n).read_text())
    assert load('validation.json')['status']=='PASS_ANALYSIS'
    a={}
    for k in ['mean','sd','cut','cal_top','inactive','rank_cut']:
        a[k]=np.stack([np.load(OUT/'recomputed'/f'rep_{r:02d}.npz')[k] for r in range(50)])
    rows=load('comparison.json');ix={(r['group'],r['calibration'],r['alpha'],r['evaluation']):r for r in rows}
    asym=load('gain_asymmetry.json');axix={(r['group'],r['calibration'],r['alpha'],r['method']):r for r in asym}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    out=OUT/'figures';out.mkdir(exist_ok=True);files=[]
    def save(fig,n):fig.savefig(out/n,dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig);files.append(n)
    colors=['#173f70','#347fc4','#bf582f','#a57eb5','#218a69','#bc9034']
    ex=np.load(OUT/'example_rep00.npz')
    fig,axs=plt.subplots(2,2,figsize=(11.5,8.4))
    for row,(base,groups) in enumerate([('B_M',['B_M','K2_M']),('B_L',['B_L','K200_L','V05_L','V15_L'])]):
        for group in groups:
            d=ex[group+'__normal'];q=(d[:,:2]-ex[base+'_mean'])/ex[base+'_scale'];s=d[:,2:]
            for ax,z in zip(axs[row],[q,s]):ax.scatter(z[:,0],z[:,1],s=8,alpha=.25,color=colors[G.index(group)],label=group,rasterized=True)
        for ax in axs[row]:ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5);ax.grid(alpha=.12)
        axs[row,0].set_xlabel('qM, centered / reference SD');axs[row,0].set_ylabel('qC, centered / reference SD')
        axs[row,1].set_xlabel('Signed gain projection sG');axs[row,1].set_ylabel('Signed private projection sP')
        axs[row,0].set_title(base+' window: two physical statistics');axs[row,1].set_title(base+' window: same data after invertible LF map')
        axs[row,1].legend(markerscale=2,frameon=False)
    fig.suptitle('Representation changes, not new measurements\nIllustration: replicate 0, first 512 normal records per source; all 50 replicates audited',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.92]);save(fig,'01_coordinates.png')
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for k,name,color in [(2,'sG','#215f91'),(3,'sP','#a55836')]:axs[0,0].plot(range(6),a['sd'][:,:,0,k].mean(0),'o-',label=name,color=color)
    axs[0,0].axhline(1,color='gray',ls=':');axs[0,0].set_ylabel('Mean within-replicate normal SD');axs[0,0].legend(frameon=False);axs[0,0].set_title('Normal source change, fixed W')
    for b,ax in enumerate([axs[0,1],axs[1,0],axs[1,1]]):
        for ci,mi,name,col,style in [(0,0,'F: W0','#627281','--'),(1,0,'C: W0','#237c68','-'),(1,1,'C: LF0 equivalent','#bc523c','-')]:
            data=a['cut'][:,:,ci,0,mi,b]
            assert np.isfinite(data).all()
            ax.plot(range(6),data.mean(0),'o'+style,color=col,label=name)
        ax.set_title(['G+ threshold','G- threshold','P threshold'][b]);ax.set_ylabel('Signed-tail score threshold');ax.legend(fontsize=8,frameon=False)
    for ax in axs.flat:ax.set_xticks(range(6),G,rotation=25);ax.grid(alpha=.15)
    fig.suptitle('Different source widths move effective branch allocation\nThreshold panels: alpha = 1%, means over 50 independent outer replicates',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.92]);save(fig,'02_source_thresholds.png')
    fig,axs=plt.subplots(1,2,figsize=(12.5,5.2))
    for gi,g in enumerate(G):
        y=a['cal_top'][:,gi,1,2]
        axs[0].scatter(gi+np.linspace(-.17,.17,50),y+.15,s=12,color=colors[gi],alpha=.7)
    axs[0].axhline(16+.15,color='#b03e42',ls='--',label='16: off at alpha 0.1%')
    axs[0].axhline(160+.15,color='#777',ls=':',label='160: off at alpha 1%')
    axs[0].set_yscale('log');axs[0].set_xticks(range(6),G,rotation=25);axs[0].set_ylabel('C records at maximum rank (+0.15 for log scale)')
    axs[0].set_title('Finite reference cap in normal calibration');axs[0].legend(fontsize=8,frameon=True,facecolor='white',framealpha=.95,loc='upper left')
    mat=a['inactive'].sum(0).reshape(6,4)
    axs[1].imshow(mat,cmap='Reds',vmin=0,vmax=50,aspect='auto')
    axs[1].set_xticks(range(4),['F 1%','F 0.1%','C 1%','C 0.1%']);axs[1].set_yticks(range(6),G)
    for (i,j),v in np.ndenumerate(mat):axs[1].text(j,i,str(v),ha='center',va='center',color='white' if v>=25 else 'black')
    axs[1].set_title('Structurally inactive LF0 models / 50')
    fig.suptitle('Rank cut = 9999 + strict greater-than comparison => no possible alarm',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.92]);save(fig,'03_rank_cap.png')
    fig,axs=plt.subplots(2,2,figsize=(11,8.6))
    for ai,alpha in enumerate(AL):
        for ci,cal in enumerate(CAL):
            ax=axs[ai,ci];mat=np.array([[100*ix[g,cal,alpha,e]['difference']['mean'] for e in EV] for g in G])
            im=ax.imshow(mat,cmap='BrBG',vmin=-90,vmax=90,aspect='auto')
            for (i,j),v in np.ndenumerate(mat):ax.text(j,i,f'{v:+.2f}',ha='center',va='center',color='white' if abs(v)>45 else '#17212c')
            ax.set_xticks(range(4),['Normal FPR','-5% gain','+5% gain','Private']);ax.set_yticks(range(6),G);ax.set_title(f'{cal}, alpha = {100*alpha:g}%')
    fig.suptitle('W0 minus LF0: same records, frozen normal calibration\nPercentage points; positive normal-column values are added false alarms, not benefits',fontsize=13)
    fig.tight_layout(rect=[0,0,.9,.92]);cb=fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.025,pad=.025);cb.set_label('W0 - LF0 (pp)')
    save(fig,'04_all_differences.png')
    fig,axs=plt.subplots(1,2,figsize=(12,5.5))
    gain=[100*ix[g,'C',.001,'P02']['W0_only']['mean'] for g in G];loss=[100*ix[g,'C',.001,'P02']['LF0_only']['mean'] for g in G]
    axs[0].barh(np.arange(6)-.15,gain,height=.3,color='#237c68',label='W0 only');axs[0].barh(np.arange(6)+.15,loss,height=.3,color='#bc523c',label='LF0 only')
    axs[0].set_yticks(range(6),G);axs[0].invert_yaxis();axs[0].set_xlabel('Private-fault evaluation records (%)');axs[0].set_title('Both directions of the detection exchange');axs[0].legend(frameon=False)
    for method,color in [('W0','#237c68'),('LF0','#bc523c')]:axs[1].plot(range(6),[100*ix[g,'C',.001,'normal'][method]['mean'] for g in G],'o-',color=color,label=method)
    axs[1].axhline(.1,color='gray',ls=':');axs[1].set_xticks(range(6),G,rotation=25);axs[1].set_ylabel('Independent normal E false alarms (%)');axs[1].set_ylim(0,.125);axs[1].set_title('The realized FPR is not equal');axs[1].legend(frameon=False)
    fig.suptitle('Clean current-source calibration C, alpha = 0.1%',fontsize=13);fig.tight_layout(rect=[0,0,1,.93]);save(fig,'05_private_tradeoffs.png')
    focus=[('K2_M',.001,'P02'),('K200_L',.001,'P02'),('V05_L',.01,'GP05')]
    fig,ax=plt.subplots(figsize=(11,5.2))
    for ki,(name,color) in enumerate([('Rank pair + sign','#277b69'),('Rank pair','#327fa5'),('Maximum rank','#b8683b')]):
        values=[ix[g,'C',al,e]['collision_sums'][ki][3] for g,al,e in focus]
        ax.bar(np.arange(3)+(ki-1)*.24,values,width=.22,color=color,label=name)
        for j,v in enumerate(values):ax.text(j+(ki-1)*.24,v*1.12,str(v),ha='center',fontsize=9)
    ax.set_yscale('log');ax.set_ylim(.7,40000);ax.set_xticks(range(3),['K2_M private / 0.1%','K200_L private / 0.1%','V05_L +gain / 1%'])
    ax.set_ylabel('Minimum W0-label mismatches from equal keys\n(sum within condition and replicate; 250,000 records each)')
    ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('Observed collisions: same retained representation, different W0 decision\nA finite-sample representational witness, not optimal fault-detection error',fontsize=12)
    fig.tight_layout();save(fig,'06_representation_collisions.png')
    fig,axs=plt.subplots(2,2,figsize=(12,8.5));x=np.arange(6)
    shift=a['mean'][:,:,1:3]-a['mean'][:,:,0,None,:]
    axs[0,0].bar(x-.17,-shift[:,:,0,2].mean(0),.34,label='Magnitude of mean -gain shift',color='#426d9c')
    axs[0,0].bar(x+.17,shift[:,:,1,2].mean(0),.34,label='Mean +gain shift',color='#bc6c39')
    axs[0,0].set_title('Asymmetry already present in signed sG');axs[0,0].set_ylabel('Shift from normal mean');axs[0,0].legend(fontsize=8,frameon=False)
    for i,label,color in [(0,'-gain','#426d9c'),(1,'+gain','#bc6c39')]:axs[0,1].bar(x+(i-.5)*.34,shift[:,:,i,3].mean(0),.34,label=label,color=color)
    axs[0,1].set_title('Private-direction mean shift sP');axs[0,1].legend(frameon=False)
    for ax,method in zip(axs[1],['W0','LF0']):
        dg=np.array([100*axix[g,'C',.01,method]['G_component']['mean'] for g in G]);dp=np.array([100*axix[g,'C',.01,method]['P_only_component']['mean'] for g in G])
        ax.bar(x,dg,color='#37759b',label='G branch contribution')
        ax.bar(x,dp,bottom=np.where(dp>=0,dg,0),color='#ba8542',label='P-only contribution')
        ax.plot(x,dg+dp,'ko',ms=4,label='Total +gain minus -gain');ax.axhline(0,color='gray',lw=.5)
        ax.set_title(method+' / C / alpha 1%');ax.set_ylabel('Detection difference (pp)')
    handles,labels=axs[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8,frameon=False)
    for ax in axs.flat:ax.set_xticks(x,G,rotation=25);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    fig.suptitle('Signed-score asymmetry and exact OR accounting\nOuter-replicate contrasts; +gain and -gain are different evaluation records',fontsize=13)
    fig.tight_layout(rect=[0,.04,1,.92]);save(fig,'07_gain_asymmetry.png')
    lines=['# 全条件表：情報圧縮・源変化の追加分析C','',
           '各率は50反復の平均（%）。SDは外側反復の標準偏差で信頼区間ではない。正常の「拾得」は追加誤警報であり利益ではない。F/Cは旧校正／独立清浄更新校正。両方法とも同じ二座標・同じLF射影。','',
           '| 群 | 校正 | α | 評価 | W0 % | LF0 % | 差 pp | 差SD pp | W0のみ % | LF0のみ % | 両方なし % | OR同じ・枝違い % | LF0非発火 /50 |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        p=r['pattern_joint'];neither=p[0][0]/(50*r['n_per_rep'])
        vals=[r['W0']['mean'],r['LF0']['mean'],r['difference']['mean'],r['difference']['sd'],r['W0_only']['mean'],r['LF0_only']['mean'],neither,r['OR_masks_pattern_difference']['mean']]
        lines.append('| '+' | '.join([r['group'],r['calibration'],str(r['alpha']),r['evaluation']]+[f'{100*v:.4f}' for v in vals]+[str(r['LF0_inactive'])])+' |')
    lines+=['','## 正負ゲイン差の全48分解','', '| 群 | 校正 | α | 方式 | 正−負 pp | G寄与 pp | P-only寄与 pp |','|---|---|---|---|---|---|---|']
    for r in asym:lines.append('| '+' | '.join([r['group'],r['calibration'],str(r['alpha']),r['method']]+[f"{100*r[k]['mean']:.4f}" for k in ['GP_minus_GM','G_component','P_only_component']])+' |')
    lines+=['','## 同一表現内のW0判定衝突','',
            '各三つ組は「符号付き順位対／順位対／最大順位」。同一条件・同一反復内でのみ集計し、その総和を掲載。異なる故障条件間の衝突は未合算で、全分布での最適誤差や学習済み分類器の性能ではない。最小不一致数は各同値群の警報／非警報の少数側の和。','',
            '| 群 | 校正 | α | 評価 | 衝突した表現の数 | 衝突群の記録数 | 最小W0ラベル不一致数 |','|---|---|---|---|---|---|---|']
    for r in rows:lines.append('| '+' | '.join([r['group'],r['calibration'],str(r['alpha']),r['evaluation']]+['/'.join(str(q[k]) for q in r['collision_sums']) for k in [1,2,3]])+' |')
    lines+=['','## 既存6配分・方式の保存率（補助）','', '| 群 | 校正 | α | 評価 | 方式 | 平均 % | SD % |','|---|---|---|---|---|---|---|']
    for r in load('other_budgets.json'):lines.append('| '+' | '.join([r['group'],r['calibration'],str(r['alpha']),r['evaluation'],r['method'],f"{100*r['rate']['mean']:.4f}",f"{100*r['rate']['sd']:.4f}"])+' |')
    (OUT/'ALL_COMPRESSION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    (OUT/'report_facts.json').write_text(json.dumps(dict(figures=files,comparison_rows=len(rows),asymmetry_rows=len(asym),auxiliary_rows=576,collision_rows=96),indent=2)+'\n')
    print('REPORT_ASSETS',len(files),'figures, 96 comparisons, 48 asymmetries, 96 collision rows, 576 auxiliary rows')

if __name__=='__main__':main()
