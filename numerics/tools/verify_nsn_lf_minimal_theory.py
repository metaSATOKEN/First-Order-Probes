# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Deterministic algebra/edge-case audit only; no detector fit or simulation."""
from pathlib import Path
from fractions import Fraction
from itertools import combinations_with_replacement
from bisect import bisect_left
import argparse
import hashlib
import json
import math
import re
from urllib.parse import unquote
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'numerics/analysis/pr0004_e0af6fcba596_update_decomposition_v1'
MEMO = ROOT / 'numerics/theory/NSN_LF_TASK_INFORMATION_MINIMAL_DESIGN_v1_JA.md'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify():
    C, Vi, Vj, Pi, Pj, Pij, nu, delta, h = sp.symbols('C Vi Vj Pi Pj Pij nu delta h', real=True)
    qm = (1+delta)*C-(1-nu)*Vi-nu*((1+delta)**2+h)*Vj
    qc = Pi-2*(1+delta)*Pij+((1+delta)**2+h)*Pj
    base = {delta: 0, h: 0}
    change = sp.Matrix([qm-qm.subs(base), qc-qc.subs(base)])
    a, b, c, p = C-2*nu*Vj, nu*Vj, 2*(Pj-Pij), Pj
    D = sp.Matrix([[a, -b], [c, p]])
    target = sp.Matrix([delta, delta**2+h])
    assert sp.simplify(change-D*target) == sp.zeros(2, 1)
    assert sp.simplify(D.det()-(C*Pj-2*nu*Vj*Pij)) == 0
    assert sp.simplify(D.inv()*change-target) == sp.zeros(2, 1)
    rho = sp.symbols('rho', real=True)
    w = sp.Matrix([[1, -rho]])
    cov = sp.Matrix([[1, rho], [rho, 1]])
    assert sp.expand((w*cov*w.T)[0]) == 1-rho**2

    # Left-insertion ranks, including ties, out-of-range scores and disabled tails.
    ranks = 0
    for n in range(1, 6):
        for ordered in combinations_with_replacement(range(3), n):
            for cut in range(n+1):
                threshold = ordered[cut] if cut < n else math.inf
                for s in [-1, 0, .5, 1, 1.5, 2, 3]:
                    assert (bisect_left(ordered, s) > cut) == (s > threshold)
                    ranks += 1

    # Conditional on an exchangeable multiset, the new index is uniform.
    # Enumerating all index choices gives exact small-n marginal bounds, with ties.
    calibration = 0
    for total in range(2, 7):
        for multiset in combinations_with_replacement(range(3), total):
            for alpha in [Fraction(0), Fraction(1, 20), Fraction(1, 5), Fraction(1, 2)]:
                k = math.ceil(total*(1-alpha))
                count = 0
                for i, s in enumerate(multiset):
                    cal = multiset[:i]+multiset[i+1:]
                    threshold = cal[k-1] if k <= len(cal) else math.inf
                    count += s > threshold
                assert Fraction(count, total) <= alpha
                calibration += 1
    tails = {}
    for denominator, expected in [(100, [25, 25, 50]), (1000, [2, 2, 5])]:
        counts = [10001-math.ceil(10001*(1-Fraction(1, denominator*weight))) for weight in [4, 4, 2]]
        assert counts == expected
        tails[str(Fraction(1, denominator))] = counts

    previous = json.loads((OUT/'validation.json').read_text())
    preserved = {}
    for section, files in previous['preserved'].items():
        for path, expected in files.items():
            assert sha(ROOT/path) == expected, path
        preserved[section] = len(files)
    delivery = json.loads((OUT/'delivery_validation.json').read_text())
    for path, expected in delivery['files'].items():
        assert sha(OUT/path) == expected, path
    for path, expected in delivery['sources'].items():
        assert sha(ROOT/path) == expected, path
    assert sha(ROOT/'NSN.md') == 'fe743154c1c93e10a2eb8f79e7af79fad7a24e11020f3b2e7ffaf9c3e892def0'
    ledger = (ROOT/'audits/ledger_2026-10.md').read_bytes()
    assert hashlib.sha256(ledger[:117864]).hexdigest() == 'd20aef6eab493b0c43a8953c3caf88f188e3d7fbdb66a3bfb491aac116db8a23'
    assert sha(ROOT/'audits/fp_corpus.md') == '73b6dd2dd67cbee0c6accd5cc99813a56571572a0d58db0037a23ab60ad8431f'
    equipment = ['docs/THROUGHLINE.md', 'audits/ledger_2026-10.md', 'HANDOFF/LATEST.md', 'audits/fp_corpus.md', 'audits/prereg/README.md']
    assert all((ROOT/p).is_file() for p in equipment)

    docs = [MEMO, ROOT/'audits/verification/2026-10-17_NSN_LF_minimal_theory.md',
            ROOT/'audits/README.md', ROOT/'HANDOFF/LATEST.md', ROOT/'audits/prereg/README.md']
    links = 0
    for doc in docs:
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', doc.read_text()):
            target = unquote(link.strip('<>').split('#')[0])
            if not target or target.startswith(('https:', 'http:', 'mailto:')):
                continue
            path = (doc.parent/target).resolve()
            if path == ROOT/'audits/verification/2026-10-17_NSN_LF_minimal_theory.json':
                continue  # this run's optional new output
            assert path.exists(), (doc, target)
            links += 1
    return dict(status='PASS', same_agent=True, scientific_performance_evaluation=False,
                algebra=['fault_mean_expansion', 'determinant', 'inverse_mean_coordinates', 'covariance_counterexample'],
                rank_cases=ranks, exchangeable_multiset_budget_cases=calibration,
                finite_calibration_tail_counts=tails, denominator=10001,
                preserved=preserved, current_c_artifacts=len(delivery['files']),
                current_c_sources=len(delivery['sources']), ledger_prefix_bytes=117864,
                nsn_unchanged=True, corpus_unchanged=True, equipment=5, local_links=links,
                memo_sha256=sha(MEMO), verifier_sha256=sha(Path(__file__)), sympy_version=sp.__version__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(verify(), ensure_ascii=False, indent=2)+'\n'
    if args.output:
        with args.output.open('x') as f:
            f.write(result)
    print(result)
