# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Read-only result input; add two descriptive figures without replacing old figures."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/pr0012_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'numerics/analysis/pr0012_64bf0f2c7697_exploration_v2'

def pick(rows, **kw):
    found = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    assert len(found) == 1
    return found[0]

def main():
    data = json.loads((OUT / 'results.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, name):
        path = OUT / 'figures' / name
        assert not path.exists(), 'Preserve completed figures'
        fig.tight_layout()
        fig.savefig(path, dpi=170, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, en in zip(axes, ['GM05', 'P02']):
        q = pick(data['decomposition'], group='K2_M', candidate='TRI800', alpha=.01, evaluation=en)
        for label, key, color in [('C recovery (45)', 7, '#238d9f'), ('DEFER (5)', -1, '#d39a32')]:
            ids = [i for i, s in enumerate(q['selection']) if s == key]
            ax.scatter(ids, [q['per_repeat'][i] * 100 for i in ids], label=label, color=color, s=28)
        ax.set_xlabel('Outer repeat')
        ax.set_ylabel('P2 - P0 detection (pp)')
        ax.set_title(f"{en}: between-path share = {100*q['between_fraction']:.4f}%")
        ax.legend(fontsize=9)
    fig.suptitle('Observed variation is dominated by recovery versus defer\nK2_M / TRI800 / alpha = 0.01; descriptive decomposition, not a new test')
    save(fig, '02b_repeat_mixture_qualified.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    names = ['Normal FPR', 'Negative gain', 'Positive gain', 'Private input']
    for ax, alpha in zip(axes, [.01, .001]):
        rows = [pick(data['passrisk'], group='V05_L', candidate='BAL400', alpha=alpha, evaluation=e)
                for e in ['normal', 'GM05', 'GP05', 'P02']]
        assert {q['passed'] for q in rows} == {21}
        values = [q['delta'] * 100 for q in rows]
        bars = ax.barh(range(4), values, color=['#707e89', '#238d9f', '#c45143', '#9a79ae'])
        ax.set_yticks(range(4), names)
        ax.invert_yaxis()
        ax.set_xlim(-23, 0.5)
        ax.axvline(0, color='black', lw=.7)
        ax.set_xlabel('Contaminated U - clean U (pp)')
        ax.set_title(f'alpha = {alpha}: A-pass repeats only (21/50)')
        for bar, v in zip(bars, values):
            ax.text(v-.25, bar.get_y()+bar.get_height()/2, f'{v:.3f}', ha='right', va='center')
    fig.suptitle('Accepted updates: fewer false alarms, but lower gain sensitivity\nV05_L / BAL400; same selected repeats and evaluation records')
    save(fig, '06_accepted_V05_sensitivity.png')
    print('PASS: two additional figures; existing five figures preserved')

if __name__ == '__main__':
    main()
