"""Japanese manuscript diagrams and redraw of frozen aggregates; no simulation.

SPDX-License-Identifier: Apache-2.0
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'figures/calibration_audit_ja_v1'
FONT = next(Path('/System/Library/Fonts').glob('*W3.ttc'), None)
if FONT:
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.size': 12, 'axes.unicode_minus': False, 'svg.fonttype': 'path'})
BLUE, ORANGE, INK, GRAY = '#245c89', '#bd602c', '#203544', '#596772'


def canvas(title, height=6.5):
    fig, ax = plt.subplots(figsize=(13, height))
    ax.set(xlim=(0, 13), ylim=(0, height))
    ax.axis('off')
    ax.text(.25, height-.25, title, fontsize=18, color=INK, weight='bold', va='top')
    return fig, ax


def box(ax, xy, w, h, label, color=BLUE, fs=12):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.09',
                              facecolor='#f5f8fa', edgecolor=color, linewidth=1.5))
    ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.55)


def arrow(ax, p1, p2, color=GRAY):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=15,
                               color=color, linewidth=1.5))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def mechanisms():
    fig, ax = canvas('MとCR：同じ入力への整合性を、異なる量で測る', 7)
    ax.text(.35, 5.75, 'M｜二次モーメントの釣り合い', color=BLUE, fontsize=14)
    box(ax, (.4, 4.15), 2.3, 1.15, '記録と雑音対照\n'+r'$u_i,u_j; c_i,c_j$')
    box(ax, (3.35, 4.15), 3.05, 1.15, '二乗・積の窓平均\n'+r'$\widehat V_i,\widehat V_j,\widehat C_{ij}$')
    box(ax, (7.05, 4.15), 5.1, 1.15, '分散から予想される共分散との差\n'+r'$q_M=\widehat C_{ij}-\lambda\widehat V_i-\nu\widehat V_j$')
    arrow(ax, (2.85, 4.72), (3.2, 4.72)); arrow(ax, (6.55, 4.72), (6.9, 4.72))
    ax.text(.35, 3.35, 'CR｜二つの応答を同じだけ通した時系列の差', color=ORANGE, fontsize=14)
    box(ax, (.4, 1.3), 1.1, 1.3, '共通入力\n'+r'$x$', ORANGE)
    box(ax, (2.35, 2.25), 2, .65, r'$F_i\ \longrightarrow\ F_j$', ORANGE)
    box(ax, (2.35, .9), 2, .65, r'$F_j\ \longrightarrow\ F_i$', ORANGE)
    arrow(ax, (1.65, 2.15), (2.2, 2.57)); arrow(ax, (1.65, 1.75), (2.2, 1.23))
    box(ax, (5.25, 1.3), 2.45, 1.3, '経路差を取る\n'+r'$r_u=F_ju_i-F_iu_j$', ORANGE, 11)
    arrow(ax, (4.5, 2.57), (5.1, 2.15)); arrow(ax, (4.5, 1.23), (5.1, 1.75))
    box(ax, (8.45, 1.3), 3.7, 1.3, '補正エネルギー\n'+r'$q_C=\langle r_u^2\rangle-\langle r_c^2\rangle$', ORANGE)
    arrow(ax, (7.85, 1.95), (8.3, 1.95))
    ax.text(.4, .25, '対照にも同じ操作を適用する。有限窓の残差は符号付きで保存し、ゼロへ丸めない。', color=GRAY, fontsize=11)
    save(fig, '01_moment_cr')


def readouts():
    fig, ax = canvas('固定したLF射影から、二つの警報規則へ', 7)
    box(ax, (.4, 2.65), 2.2, 1.5, '符号付き二座標\n'+r'$q=(q_M,q_C)^{\mathsf{T}}$')
    box(ax, (3.35, 2.65), 2.25, 1.5, '正常Rで固定したLF\n'+r'$s_g,\ s_p$')
    arrow(ax, (2.75, 3.4), (3.2, 3.4))
    box(ax, (6.5, 4.4), 2.75, 1.35, 'LF0\n'+r'$|s_g|,s_p$'+' の順位\n最大順位を選ぶ')
    box(ax, (10.1, 4.4), 2.3, 1.35, '共通の整数閾値\n'+r'$L>t_L$')
    box(ax, (6.5, 1.15), 2.75, 1.35, 'W0：直接校正\n'+r'$s_g,\ -s_g,\ s_p$'+'\n3枝を残す', ORANGE)
    box(ax, (10.1, 1.15), 2.3, 1.35, '各閾値超過のOR\n'+r'$\alpha/4,\ \alpha/4,\ \alpha/2$', ORANGE)
    arrow(ax, (5.75, 3.75), (6.35, 4.95)); arrow(ax, (5.75, 3.05), (6.35, 1.95), ORANGE)
    arrow(ax, (9.4, 5.08), (9.95, 5.08)); arrow(ax, (9.4, 1.83), (9.95, 1.83), ORANGE)
    ax.text(6.5, 3.75, '順位上端と厳密不等号が非発火を生む場合', fontsize=11, color=BLUE)
    ax.text(6.5, .48, '順位の置換だけではなく、予算と判定領域も異なる', fontsize=11, color=ORANGE)
    ax.text(.4, .1, '閾値はRとは別の正常校正で固定。評価Eで再選択しない。', color=GRAY, fontsize=11)
    save(fig, '02_readouts')


def roles():
    fig, ax = canvas('データの役割を分けて、政策を固定してから評価する', 7.3)
    box(ax, (.4, 4.65), 2.55, 1.15, 'R：正常参照\n中心・重み・順位表')
    box(ax, (3.75, 4.65), 2.55, 1.15, 'O：旧正常校正\n旧閾値F')
    box(ax, (7.1, 4.65), 2.55, 1.15, '清浄回復データ\n独立な閾値C')
    ax.text(.4, 6.15, '固定するもの', color=BLUE, fontsize=14)
    box(ax, (.4, 2.25), 2.55, 1.4, 'A：候補監査\n通過／保留を決める', ORANGE)
    box(ax, (3.75, 2.25), 2.55, 1.4, 'U：候補校正\nAと同組成・別記録', ORANGE)
    box(ax, (7.1, 2.25), 2.55, 1.4, 'V：正常確認\nF/CのFPRを確認', ORANGE)
    box(ax, (10.35, 2.25), 2.1, 1.4, '政策を固定\nU／F／C\n／DEFER', ORANGE, 11)
    arrow(ax, (5.03, 4.5), (8.2, 3.8)); arrow(ax, (8.38, 4.5), (8.38, 3.8))
    arrow(ax, (9.8, 2.95), (10.2, 2.95), ORANGE)
    ax.text(.4, 1.68, 'Aの通過時はUを採用。保留時はVの根拠に応じてF／C／停止を選ぶ。', fontsize=11, color=INK)
    box(ax, (3.75, .25), 5.9, .85, 'E：独立評価 → 誤警報・検出・見逃し・未判定')
    arrow(ax, (11.4, 2.1), (9.8, .75))
    ax.text(.4, .38, 'C・Vの清浄性は\n外部資源の仮定', color=GRAY, fontsize=10)
    save(fig, '03_roles')


def readout_data():
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
        ax.axhline(.1,color=GRAY,ls='--',lw=1,label='公称 α = 0.1%')
        ax.set(xticks=[0,1],xticklabels=['LF0','W0'],ylim=(0,.13),ylabel='正常FPR (%)',title=group+'｜独立C')
        ax.legend(loc='upper left',fontsize=9)
        ax=axes[1,col]; x=np.arange(3); width=.36
        for j,method in enumerate(('LF0','W0')):
            rates=[rate(group,method,e) for e in ('GM05','GP05','P02')]
            positions=x+(j-.5)*width
            ax.bar(positions,rates,width,color=colors[method],label=method)
            for xx,v in zip(positions,rates):ax.text(xx,v+1.2,f'{v:.2f}',ha='center',fontsize=10)
        ax.set(xticks=x,xticklabels=['負ゲイン −5%','正ゲイン +5%','私的入力 2%'],ylim=(0,107),ylabel='検出率 (%)')
        ax.legend(loc='upper left')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('固定LFの読出し比較：同じ公称予算でも、実際のFPRと感度は異なる',fontsize=16,y=.99)
    fig.text(.5,.01,'PR-0012｜α = 0.1%｜50反復平均｜保存集計の再描画（等FPR比較ではない）',ha='center',fontsize=11,color=GRAY)
    fig.tight_layout(rect=(0,.045,1,.96))
    save(fig,'04_readout_tradeoffs')


if __name__=='__main__':
    mechanisms();readouts();roles();readout_data()
    print(f'Wrote four PNG/SVG pairs to {OUT}; source aggregates unchanged.')
