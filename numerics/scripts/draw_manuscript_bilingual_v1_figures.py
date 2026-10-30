"""Versioned editorial diagrams and translated frozen aggregate plot; no simulation.

Copyright 2026 Kentaro Sato
SPDX-License-Identifier: Apache-2.0
"""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import draw_manuscript_ja_v1_figures as base

ROOT = Path(__file__).resolve().parents[2]
BLUE, ORANGE, INK, GRAY = base.BLUE, base.ORANGE, base.INK, base.GRAY
box, arrow, canvas = base.box, base.arrow, base.canvas


def save(fig, folder, name):
    out = ROOT / 'figures' / folder
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        dest = out / f'{name}.{ext}'
        if dest.exists() and '--replace-draft' not in sys.argv:
            raise FileExistsError(f'Preserve completed figure: {dest}')
        fig.savefig(dest, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def roles(lang):
    ja = lang == 'ja'
    fig, ax = canvas('データの役割と、候補通過／保留後の経路' if ja else
                     'Data roles and decision paths after candidate auditing', 8.9)
    box(ax, (.35, 6.95), 3.5, 1.05,
        'R：正常参照\n全経路の中心・重み・順位表を固定' if ja else
        'R: normal reference\nFix centers, weights, and ranks\nfor all paths', fs=10.5)
    box(ax, (4.4, 6.95), 3.1, 1.05,
        'O：旧正常校正 → F\n清浄回復データ → C' if ja else
        'O: old normal calibration → F\nClean recovery data → C', fs=11)
    box(ax, (8.4, 6.95), 3.7, 1.05,
        'V：独立な正常確認\n固定F/CのFPR基準を分類' if ja else
        'V: independent normal validation\nClassify F/C against the FPR limit', fs=11)
    arrow(ax, (7.63, 7.48), (8.26, 7.48))
    ax.text(.35, 6.5, '判定経路｜AとUは同組成・別記録のバッチ' if ja else
            'Decision paths | A and U are separate batches with the same composition',
            color=INK, fontsize=12)
    box(ax, (.35, 3.55), 2.35, 1.3,
        'A：候補監査\n通過／保留を決定' if ja else
        'A: candidate audit\nPass or hold', ORANGE, 12)
    box(ax, (4.0, 4.7), 4.4, 1.05,
        '通過 → Uを採用（全政策共通）\nU：別取得の候補閾値' if ja else
        'PASS → adopt U (all policies)\nU: thresholds from a separate\ncandidate batch', ORANGE, 11)
    arrow(ax, (2.83, 4.5), (3.86, 5.2), ORANGE)
    ax.text(2.9, 5.18, '通過' if ja else 'PASS', color=ORANGE, fontsize=10)
    box(ax, (4.0, 1.5), 4.4, 2.45,
        '保留 → 政策で選択\nP0：常にF\nP1：F支持ならF、それ以外は停止\nP2：F支持ならF、次にC支持ならC\nどちらも非支持なら停止' if ja else
        'HOLD → select by policy\nP0: always F\nP1: F if supported; otherwise defer\nP2: F if supported; else C if supported;\notherwise defer', ORANGE, 11)
    arrow(ax, (2.83, 3.87), (3.86, 3.35), ORANGE)
    ax.text(2.9, 3.47, '保留' if ja else 'HOLD', color=ORANGE, fontsize=10)
    # V informs only the held-candidate F/C selection, not the accepted-U route.
    ax.plot([12.23, 12.75, 12.75, 7.5], [7.48, 7.48, 1.0, 1.0], color=GRAY, ls='--', lw=1.4)
    ax.add_patch(FancyArrowPatch((7.5, 1.0), (7.5, 1.36), arrowstyle='-|>',
        mutation_scale=15, color=GRAY, linewidth=1.4, linestyle='--'))
    ax.text(10.55, 5.55, 'P1/P2の\n保留後選択\nだけに使用' if ja else
            'Used only for\nP1/P2 after HOLD', color=GRAY, fontsize=10)
    box(ax, (10.1, 1.5), 2.3, 2.2,
        'E：独立評価\n警報・非警報\n未判定・技術無効\n全経路を計数' if ja else
        'E: independent\nevaluation\nAlarm / non-alarm\nDeferred / invalid\nCount all paths', fs=10.5)
    # The accepted-U path goes around the validation path, without joining it.
    ax.plot([8.53, 8.95, 8.95], [5.22, 5.22, 3.3], color=ORANGE, lw=1.5)
    arrow(ax, (8.95, 3.3), (9.96, 3.3), ORANGE)
    arrow(ax, (8.53, 2.5), (9.96, 2.5), ORANGE)
    ax.text(.35, .65, 'C・Vの清浄性は外部資源の仮定。VでF/Cを確認し、政策を固定してからEへ進む。' if ja else
            'Clean C and V are supplied resources. Validate F/C and fix the policy before evaluating E.',
            color=GRAY, fontsize=11)
    ax.text(.35, .2, '実線：採用・評価の経路　　破線：Vから保留後の選択へ渡す根拠' if ja else
            'Solid: adoption/evaluation path. Dashed: validation evidence for selection after a hold.',
            color=GRAY, fontsize=10.5)
    save(fig, 'calibration_audit_ja_v4' if ja else 'calibration_audit_en_v1', '03_roles')


def mechanisms_en():
    fig, ax = canvas('M and CR: two measurements of common-input consistency', 7)
    ax.text(.35, 5.75, 'M | Second-moment balance', color=BLUE, fontsize=14)
    box(ax, (.4, 4.15), 2.3, 1.15, 'Records and\nnoise controls\n'+r'$u_i,u_j; c_i,c_j$', fs=10.5)
    box(ax, (3.35, 4.15), 3.05, 1.15, 'Window averages\nof squares/products\n'+r'$\widehat V_i,\widehat V_j,\widehat C_{ij}$', fs=10.5)
    box(ax, (7.05, 4.15), 5.1, 1.15, 'Covariance minus the variance-based prediction\n'+r'$q_M=\widehat C_{ij}-\lambda\widehat V_i-\nu\widehat V_j$', fs=11)
    arrow(ax, (2.85, 4.72), (3.2, 4.72)); arrow(ax, (6.55, 4.72), (6.9, 4.72))
    ax.text(.35, 3.35, 'CR | Compare paths that traverse the same two responses', color=ORANGE, fontsize=14)
    box(ax, (.4, 1.3), 1.1, 1.3, 'Common\ninput '+r'$x$', ORANGE, 10.5)
    box(ax, (2.35, 2.25), 2, .65, r'$F_i\ \longrightarrow\ F_j$', ORANGE)
    box(ax, (2.35, .9), 2, .65, r'$F_j\ \longrightarrow\ F_i$', ORANGE)
    arrow(ax, (1.65, 2.15), (2.2, 2.57)); arrow(ax, (1.65, 1.75), (2.2, 1.23))
    box(ax, (5.25, 1.3), 2.45, 1.3, 'Path difference\n'+r'$r_u=F_ju_i-F_iu_j$', ORANGE, 11)
    arrow(ax, (4.5, 2.57), (5.1, 2.15)); arrow(ax, (4.5, 1.23), (5.1, 1.75))
    box(ax, (8.45, 1.3), 3.7, 1.3, 'Noise-corrected energy\n'+r'$q_C=\langle r_u^2\rangle-\langle r_c^2\rangle$', ORANGE)
    arrow(ax, (7.85, 1.95), (8.3, 1.95))
    ax.text(.4, .25, 'Apply the same operations to controls. Retain signed finite-window residuals, including negative values.', color=GRAY, fontsize=11)
    save(fig, 'calibration_audit_en_v1', '01_moment_cr')


def readouts_en():
    fig, ax = canvas('Two alarm rules from the same fixed LF projections', 7)
    box(ax, (.4, 2.65), 2.2, 1.5, 'Two signed\ncoordinates\n'+r'$q=(q_M,q_C)^{\mathsf{T}}$', fs=11)
    box(ax, (3.35, 2.65), 2.25, 1.5, 'LF fixed using\nnormal R\n'+r'$s_g,\ s_p$', fs=11)
    arrow(ax, (2.75, 3.4), (3.2, 3.4))
    box(ax, (6.5, 4.4), 2.75, 1.35, 'LF0\nRanks of '+r'$|s_g|,s_p$'+'\nTake the maximum rank', fs=11)
    box(ax, (10.1, 4.4), 2.3, 1.35, 'One integer threshold\n'+r'$L>t_L$', fs=11)
    box(ax, (6.5, 1.15), 2.75, 1.35, 'W0: direct calibration\n'+r'$s_g,\ -s_g,\ s_p$'+'\nKeep three branches', ORANGE, 11)
    box(ax, (10.1, 1.15), 2.3, 1.35, 'OR of exceedances\n'+r'$\alpha/4,\ \alpha/4,\ \alpha/2$', ORANGE, 11)
    arrow(ax, (5.75, 3.75), (6.35, 4.95)); arrow(ax, (5.75, 3.05), (6.35, 1.95), ORANGE)
    arrow(ax, (9.4, 5.08), (9.95, 5.08)); arrow(ax, (9.4, 1.83), (9.95, 1.83), ORANGE)
    ax.text(6.5, 3.7, 'The rank cap and strict inequality can suppress all alarms.', fontsize=10, color=BLUE)
    ax.text(6.5, .48, 'Budgets and decision regions also differ between the rules.', fontsize=10, color=ORANGE)
    ax.text(.4, .1, 'Fit thresholds with normal calibration data separate from R; keep them fixed for evaluation E.', color=GRAY, fontsize=11)
    save(fig, 'calibration_audit_en_v1', '02_readouts')


def readout_data_en():
    source = ROOT/'numerics/analysis/pr0012_64bf0f2c7697_confirmation_v1/calibration_summary.json'
    rows = json.loads(source.read_text())
    def rate(group, method, evaluation):
        selected = [r for r in rows if r['group']==group and r['method']==method
                    and r['evaluation']==evaluation and r['context']=='C' and r['alpha']==.001]
        assert len(selected)==1
        return selected[0]['alarm']['mean']*100
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), gridspec_kw={'height_ratios':[1, 2]})
    colors = {'W0':ORANGE,'LF0':BLUE}
    for col, group in enumerate(('K2_M','K200_L')):
        ax=axes[0,col]
        for j, method in enumerate(('LF0','W0')):
            v=rate(group,method,'normal'); ax.bar(j,v,color=colors[method],width=.55)
            ax.text(j,v+.004,f'{v:.4f}%',ha='center',fontsize=11)
        ax.axhline(.1,color=GRAY,ls='--',lw=1,label='Nominal α = 0.1%')
        ax.set(xticks=[0,1],xticklabels=['LF0','W0'],ylim=(0,.13),ylabel='Normal FPR (%)',title=group+' | independent C')
        ax.legend(loc='upper left',fontsize=9)
        ax=axes[1,col]; x=np.arange(3); width=.36
        for j,method in enumerate(('LF0','W0')):
            rates=[rate(group,method,e) for e in ('GM05','GP05','P02')]
            positions=x+(j-.5)*width
            ax.bar(positions,rates,width,color=colors[method],label=method)
            for xx,v in zip(positions,rates):ax.text(xx,v+1.2,f'{v:.2f}',ha='center',fontsize=10)
        ax.set(xticks=x,xticklabels=['Negative gain\n−5%','Positive gain\n+5%','Private input\n2%'],ylim=(0,107),ylabel='Detection rate (%)')
        ax.legend(loc='upper left')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Fixed-LF readouts: measured FPR and sensitivity at the same nominal budget',fontsize=15,y=.99)
    fig.text(.5,.01,'PR-0012 | α = 0.1% | Means over 50 replicates | Frozen aggregates; operating points are not FPR-matched',ha='center',fontsize=10.5,color=GRAY)
    fig.tight_layout(rect=(0,.045,1,.96))
    save(fig,'calibration_audit_en_v1','04_readout_tradeoffs')


if __name__ == '__main__':
    roles('ja')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mechanisms_en();readouts_en();roles('en');readout_data_en()
    print('Wrote one Japanese and four English PNG/SVG pairs. No scientific recomputation.')
