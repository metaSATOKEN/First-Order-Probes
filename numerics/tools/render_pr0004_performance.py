# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Deterministic tables/figures from audited saved-data analysis, no simulation."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(folder):
    folder = Path(folder)
    s = json.loads((folder/'recomputed_summary.json').read_text())
    d = json.loads((folder/'diagnostics.json').read_text())
    assert json.loads((folder/'validation.json').read_text())['status']=='PASS'
    methods=s['methods']; main=methods[:5]
    rates={(r['case'],r['policy'],r['method']):r for r in s['rates']}
    def pct(x):return f'{100*x:.4f}' if x is not None else '不算定'
    def ci(r):return f"[{pct(r['lower'])}, {pct(r['upper'])}]"
    lines=['# PR-0004 全条件結果表','',
           '固定条件の全37故障・12正常・1構造対照を収録。値は%またはpp。LF-Dは副比較。',
           '区間は50外側反復の近似t区間。主要12比較だけfamily=12、FPR監査だけfamily=140、他は点別95%。',
           '率の標本は各正常条件・方針につき100万、故障／対照は25万。ただしF/C/Rは同じ評価記録で独立標本を増やさない。',
           'ゼロ反復SDの区間は不算定。全列のSD・MCSE・10/50/90%点はJSONを参照。','',
           '## 1. 全114条件×方針の率（正常はFPR、故障は検出率）','',
           '| 条件 | 種別 | 方針 | '+' | '.join(methods)+' |',
           '|---|---|---|'+ '|'.join(['---:']*6)+'|']
    for case in s['conditions']:
        policies=['B'] if s['conditions'][case]['group'].startswith('B_') else ['F','C','R']
        for p in policies:
            lines.append('| '+case+' | '+s['conditions'][case]['kind']+' | '+p+' | '+
                         ' | '.join(f"{pct(rates[case,p,m]['mean'])} {ci(rates[case,p,m])}" for m in methods)+' |')
    lines += ['', '## 2. 主要12比較','', '| 条件 | LF−比較先 | 純差pp | family=12区間pp | 判定 |', '|---|---|---:|---|---|']
    for r in s['primary12']:lines.append(f"| {r['case']} | {r['right']} | {pct(r['mean'])} | {ci(r)} | {r['classification']} |")
    lines += ['', '## 3. FPR140系列','', '| 正常条件 | 方針 | 方式 | FPR% | family=140区間% | 下端>1% |','|---|---|---|---:|---|---|']
    for r in s['fpr140']:lines.append(f"| {r['case']} | {r['policy']} | {r['method']} | {pct(r['mean'])} | {ci(r)} | {r['excess_flag']} |")
    lines += ['', '## 4. LFと各方式の全570対応比較','',
              '4セルは全評価標本に対する割合。正常時の「拾得／損失」は警報の増加／減少であり、故障検出の成功／失敗ではない。', '',
              '| 条件 | 方針 | 比較先 | 双方% | LFのみ% | 比較先のみ% | 双方非検出% | 純差pp | 点別95%区間pp |',
              '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in s['paired']:
        c=r['cells'];lines.append(f"| {r['case']} | {r['policy']} | {r['right']} | "+
            ' | '.join(pct(c[k]['mean']) for k in ['both','gained','lost','neither'])+f" | {pct(r['mean'])} | {ci(r)} |")
    lines += ['', '## 5. 源変化後の更新：全384対応比較','',
              'R−FとR−Cを報告。正常ではFPR差、故障では検出率差。正の差を一律に改善とは呼ばない。', '',
              '| 条件 | 方式 | 差 | Rのみ% | 比較先のみ% | 純差pp | 点別95%区間pp |', '|---|---|---|---:|---:|---:|---|']
    for r in s['policy_differences']:
        lines.append(f"| {r['case']} | {r['method']} | R−{r['right']} | {pct(r['cells']['gained']['mean'])} | {pct(r['cells']['lost']['mean'])} | {pct(r['mean'])} | {ci(r)} |")
    lines += ['', '## 6. 参照モデル診断（600モデル）','',
              '| 群 | 平均q相関 | 共分散条件数の範囲 | ゲイン方向ノルム平均 | 私的方向ノルム平均 | 無効枝数 |',
              '|---|---:|---|---:|---:|---:|']
    diag_summary=[]
    for group,c in s['conditions'].items():
        if c['kind']!='normal':continue
        a=[r for r in d['models'] if r['group']==group]
        row=dict(group=group,correlation=float(np.mean([r['correlation'] for r in a])),
                 condition_min=min(r['condition_number'] for r in a),condition_max=max(r['condition_number'] for r in a),
                 gain_norm=float(np.mean([r['direction_norms'][0] for r in a])),private_norm=float(np.mean([r['direction_norms'][1] for r in a])),
                 disabled=sum(len(r['disabled']) for r in a))
        diag_summary.append(row)
        lines.append(f"| {group} | {row['correlation']:.4f} | {row['condition_min']:.3f}–{row['condition_max']:.3f} | {row['gain_norm']:.6f} | {row['private_norm']:.6f} | {row['disabled']} |")
    lines += ['', '## 7. 校正時のLF枝順位上限率と閾値張り付き','',
              '各枝の平均上限率%を示す。閾値=9999なら厳密超過が不可能になるが、今回該当する反復数も併記する。','',
              '| 群 | 方針 | ゲイン枝の上限率% | 私的枝の上限率% | LF閾値=9999の反復 |','|---|---|---:|---:|---:|']
    cap_summary=[]
    for group,c in s['conditions'].items():
        if c['kind']!='normal':continue
        for p in ['B'] if group.startswith('B_') else ['F','C','R']:
            a=[r for r in d['calibrations'] if r['group']==group and r['policy']==p]
            cap={m:sum((r['cutoffs'][i][0]==9999 if i in [0,3,4,5] else r['cutoffs'][i][1]==9999 if i==1 else all(v==9999 for v in r['cutoffs'][i])) for r in a) for i,m in enumerate(methods)}
            val=np.mean([r['saturation']['LF'] for r in a],axis=0)
            cap_summary.append(dict(group=group,policy=p,lf_saturation=val.tolist(),disabled_threshold_repeats=cap))
            lines.append(f"| {group} | {p} | {pct(val[0])} | {pct(val[1])} | {cap['LF']} / 50 |")
    (folder/'ALL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    (folder/'report_metrics.json').write_text(json.dumps(dict(models=diag_summary,calibration=cap_summary),ensure_ascii=False,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),sharey=True)
    for ax,case,title in zip(axes,['B_L__GM05','B_L__GP05','B_M__P02'],['Long window: gain -5%','Long window: gain +5%','Medium window: private input 0.02']):
        r=[rates[case,'B',m] for m in main];y=np.array([v['mean']*100 for v in r]);err=np.array([(v['upper']-v['mean'])*100 for v in r])
        ax.bar(main,y,yerr=err,color=['#8d99ae','#6a994e','#d4a373','#457b9d','#e76f51'],capsize=3)
        for i,v in enumerate(y):ax.text(i,v+2.0,f'{v:.2f}',ha='center',fontsize=9)
        ax.set_title(title);ax.set_ylim(0,108);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Detection rate (%)');fig.suptitle('PR-0004: fixed nominal 1% calibration; pointwise 95% outer intervals')
    fig.tight_layout();fig.savefig(folder/'primary_detection.png',dpi=160);plt.close(fig)
    groups=['K2_M','K2_L','K200_M','K200_L','V05_M','V05_L','V15_M','V15_L']
    fig,axes=plt.subplots(1,3,figsize=(13,5.5),sharey=True)
    for ax,p,title in zip(axes,['F','C','R'],['F: frozen detector','C: recalibrate threshold','R: relearn normal reference']):
        a=np.array([[rates[g,p,m]['mean']*100 for m in main] for g in groups])
        im=ax.imshow(a,vmin=0,vmax=5,cmap='YlOrRd',aspect='auto')
        for i in range(8):
            for j in range(5):
                label='<0.01' if 0<a[i,j]<.01 else f'{a[i,j]:.2f}'
                ax.text(j,i,label,ha='center',va='center',color='white' if a[i,j]>3 else 'black',fontsize=9)
        ax.set_xticks(range(5),main);ax.set_yticks(range(8),groups);ax.set_title(title)
    fig.suptitle('False-alarm rate (%) on NORMAL shifted-source records; target 1%')
    fig.subplots_adjust(left=.08,right=.90,wspace=.08,top=.87,bottom=.08)
    cb=fig.colorbar(im,cax=fig.add_axes([.92,.15,.015,.65]));cb.set_label('FPR (%)')
    fig.savefig(folder/'source_shift_fpr.png',dpi=160);plt.close(fig)
    print('Wrote all-condition MD, report diagnostics, and two figures.')


if __name__=='__main__':run(sys.argv[1])
