# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Render audited branch counts; fixed illustrative replicate, no fitting."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS={'g_only':'#2171b5','p_only':'#e07a35','both':'#48a49b','neither':'#d9dce2'}


def render(folder):
    folder=Path(folder);s=json.loads((folder/'summary.json').read_text())
    assert json.loads((folder/'validation.json').read_text())['status']=='PASS'
    b={(r['case'],r['policy'],r['method']):r for r in s['branches']}
    a={(r['group'],r['policy'],r['method'],r['amplitude']):r for r in s['asymmetry']}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    cases=['B_L','B_L__GM02','B_L__GP02','B_L__GM05','B_L__GP05','B_M__P02']
    labels=['Normal (long)','Gain -2% (long)','Gain +2% (long)','Gain -5% (long)','Gain +5% (long)','Private 0.02 (medium)']
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),sharey=True,layout='constrained')
    for ax,method in zip(axes,['LF','LF-D']):
        left=np.zeros(len(cases))
        for state in ['g_only','p_only','both','neither']:
            values=np.array([100*b[c,'B',method]['metrics'][state]['mean'] for c in cases])
            ax.barh(np.arange(len(cases)),values,left=left,color=COLORS[state],label=state.replace('_',' '));left+=values
        ax.set(title=method,xlim=(0,100),xlabel='Share of all records (%)',yticks=np.arange(len(cases)),yticklabels=labels)
        for i,c in enumerate(cases):ax.text(99,i,f"{100*b[c,'B',method]['metrics']['total']['mean']:.2f}%",ha='right',va='center',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
    axes[0].invert_yaxis()
    handles,labels_legend=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels_legend,loc='outside lower center',ncol=4)
    fig.suptitle('Fixed-threshold branch decomposition (50-replicate means)\nLabels give union alarm rate; normal alarms are false positives')
    fig.savefig(folder/'branch_decomposition.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True,layout='constrained')
    pairs=[('B_S',.05,'0.512 s, 5%'),('B_M',.05,'2.048 s, 5%'),('B_L',.02,'8.192 s, 2%'),('B_L',.05,'8.192 s, 5%')]
    for ax,method in zip(axes,['LF','LF-D']):
        up=np.zeros(4);down=np.zeros(4)
        for state in ['g_only','p_only','both']:
            v=np.array([100*a[g,'B',method,d]['differences'][state]['mean'] for g,d,_ in pairs])
            bottom=np.where(v>=0,up,down)
            ax.bar(np.arange(4),v,bottom=bottom,color=COLORS[state],label=state.replace('_',' '))
            up+=np.maximum(v,0);down+=np.minimum(v,0)
        total=[a[g,'B',method,d]['differences']['total'] for g,d,_ in pairs]
        y=np.array([100*r['mean'] for r in total]);err=np.array([100*(r['mean']-r['lower']) for r in total])
        ax.errorbar(np.arange(4),y,yerr=err,fmt='ko',capsize=4,label='Union difference / pointwise 95% CI')
        ax.axhline(0,color='black',lw=.7);ax.set(title=method,xticks=np.arange(4),xticklabels=[x[2] for x in pairs],ylabel='Positive-gain minus negative-gain detection (pp)')
        ax.tick_params(axis='x',rotation=15)
    axes[1].legend(fontsize=8,loc='upper left');fig.suptitle('Which branch categories account for the sign asymmetry?')
    fig.savefig(folder/'gain_asymmetry.png');plt.close(fig)

    meta=json.loads((folder/'geometry_model_rep00.json').read_text());m=meta['model'];cuts=np.array(meta['cutoffs'])
    with np.load(folder/'geometry_rep00.npz',allow_pickle=False) as f: data={k:f[k] for k in f.files}
    mu=np.array(m['mean']);scale=np.sqrt(np.diag(m['covariance']))
    z={c:(data[c+'_q']-mu)/scale for c in cases[:5]}
    allz=np.concatenate([z[c] for c in ['B_L','B_L__GM05','B_L__GP05']])
    lo=np.quantile(allz,.001,axis=0)-.5;hi=np.quantile(allz,.999,axis=0)+.5
    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],400),np.linspace(lo[1],hi[1],400))
    qgrid=np.column_stack((xx.ravel(),yy.ravel()))*scale+mu
    clipping={c:int(np.sum(np.any((v<lo)|(v>hi),axis=1))) for c,v in z.items()}
    fig,axes=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True,layout='constrained')
    catcolors=['#b8bec7','#2171b5','#d55e00','#497b55']
    catlabels=['Neither','LF variant only','J only','Both']
    for row,(method,mi,wkey) in enumerate([('LF',4,'weights'),('LF-D',5,'weights_diagonal')]):
        w=np.array(m[wkey]);v=(qgrid-mu)@w.T;v[:,0]=abs(v[:,0])
        order=data['reference_'+method]
        ranks=np.column_stack([np.searchsorted(order[:,j],v[:,j],side='left') for j in range(2)])
        alarm=(ranks>cuts[mi]).any(axis=1).reshape(xx.shape)
        jv=np.column_stack((abs(qgrid[:,0]),qgrid[:,1]));jo=data['reference_J']
        jr=np.column_stack([np.searchsorted(jo[:,j],jv[:,j],side='left') for j in range(2)])
        ja=(jr>cuts[3]).any(axis=1).reshape(xx.shape)
        for col,c in enumerate(['B_L__GM05','B_L__GP05']):
            ax=axes[row,col]
            ax.hist2d(z['B_L'][:,0],z['B_L'][:,1],bins=65,range=[[lo[0],hi[0]],[lo[1],hi[1]]],cmap='Greys',cmin=1,alpha=.38)
            flags=data[c+'_bits'];cat=((flags&(1<<mi))!=0).astype(int)+2*((flags&(1<<3))!=0).astype(int)
            # Deterministic display subset, not selected by detection or agreement.
            shown=np.arange(min(1200,len(cat)))
            for k in [0,3,1,2]:
                ix=shown[cat[shown]==k]
                ax.scatter(z[c][ix,0],z[c][ix,1],s=6,c=catcolors[k],alpha=.52,rasterized=True)
            ax.contour(xx,yy,alarm.astype(float),levels=[.5],colors=['#673a99'],linewidths=1.7)
            ax.contour(xx,yy,ja.astype(float),levels=[.5],colors=['black'],linewidths=1.2,linestyles='--')
            ax.set(title=f'{method} versus J: gain {"-5%" if col==0 else "+5%"}',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),
                   xlabel=r'$(R_{full}-\mu_R)/\sigma_R$',ylabel=r'$(CR-\mu_{CR})/\sigma_{CR}$')
    legend=[Line2D([0],[0],marker='o',ls='',color=c,label=l) for c,l in zip(catcolors,catlabels)]
    legend += [Line2D([0],[0],color='#673a99',label='LF / LF-D boundary'),Line2D([0],[0],color='black',ls='--',label='J boundary')]
    fig.legend(handles=legend,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Illustrative fixed replicate 0: long-window decision regions\nGrey = normal density; first 1,200 fault records displayed; counts use all records')
    fig.savefig(folder/'decision_geometry.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    w=np.array(m['weights']);threshold=meta['reference_thresholds']['LF'][4]
    for j,ax in enumerate(axes):
        selected=['B_L','B_L__GM05','B_L__GP05']
        all_projection=np.concatenate([((data[c+'_q']-mu)@w.T)[:,j] for c in selected])
        bins=np.linspace(all_projection.min(),all_projection.max(),100)
        for c,color,label in [('B_L','#555555','Normal'),('B_L__GM05','#2171b5','Gain -5%'),('B_L__GP05','#d55e00','Gain +5%')]:
            v=(data[c+'_q']-mu)@w.T
            ax.hist(v[:,j],bins=bins,density=True,histtype='step',color=color,label=label,linewidth=1.4)
        ax.axvline(threshold[j],color='black',ls='--',label='Fixed branch threshold')
        if j==0:ax.axvline(-threshold[j],color='black',ls='--')
        ax.set(title=['Signed gain projection (two-sided alarm)','Private-input projection (upper-tail alarm)'][j],xlabel='Signed projection',ylabel='Density')
    axes[1].legend(fontsize=9);fig.suptitle('The same fixed LF: projection distributions, replicate 0, long window')
    fig.savefig(folder/'projection_distributions.png');plt.close(fig)

    lines=['# 全条件の枝別結果（事後的記述解析）','',
       '率は50反復の平均（%）。枝は元の結合校正閾値のまま。正常条件では警報＝誤警報。LF-Dは副方式。区間はsummary.json内の点別95%近似t区間で、多重比較調整した確認結果ではない。','',
       '| 条件 | 方針 | 方式 | Gのみ % | Pのみ % | 両枝 % | 両枝非発火 % | OR % | G発火 % | P発火 % |',
       '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['branches']:
        v=[100*r['metrics'][k]['mean'] for k in ['g_only','p_only','both','neither','total','g_fire','p_fire']]
        lines.append('| '+' | '.join([r['case'],r['policy'],r['method']]+[f'{x:.4f}' for x in v])+' |')
    lines+=['','## 正ゲイン−負ゲインの差','',
            '| 群 | 方針 | 方式 | 振幅 | Gのみ差 pp | Pのみ差 pp | 両枝差 pp | OR差 pp | OR点別95%区間 pp |','|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in s['asymmetry']:
        d=r['differences'];v=[100*d[k]['mean'] for k in ['g_only','p_only','both','total']];iv=d['total']
        bounds=f"[{100*iv['lower']:.4f}, {100*iv['upper']:.4f}]" if iv['lower'] is not None else '不算定'
        lines.append('| '+' | '.join([r['group'],r['policy'],r['method'],str(r['amplitude'])]+[f'{x:+.4f}' for x in v]+[bounds])+' |')
    lines+=['','## LFの追加検出・損失の枝分解','',
            '全評価記録に対する%。追加検出は比較先が非発火でLFが発火したもの。損失は比較先が発火でLF両枝が非発火したもの。','',
            '| 条件 | 方針 | 比較先 | 追加Gのみ % | 追加Pのみ % | 追加両枝 % | 損失 % |','|---|---|---|---:|---:|---:|---:|']
    for r in s['branches']:
        if r['method']!='LF':continue
        for method,table in r['cross'].items():
            x=np.array(table)/r['n_total']*100
            lines.append('| '+' | '.join([r['case'],r['policy'],method]+[f'{v:.4f}' for v in [x[1,0],x[2,0],x[3,0],x[0,1]]])+' |')
    (folder/'ALL_BRANCH_RESULTS.md').write_text('\n'.join(lines)+'\n')
    (folder/'figure_metadata.json').write_text(json.dumps(dict(replicate=0,scatter_first_n=1200,
        geometry_limits=[lo.tolist(),hi.tolist()],outside_geometry_limits=clipping,
        plot_records={c:len(v) for c,v in z.items()},all_table_rows=[len(s['branches']),len(s['asymmetry']),len(s['branches'])//2*5]),indent=2)+'\n')
    print('Rendered four figures and all-condition branch tables.')


if __name__=='__main__':render(sys.argv[1])
