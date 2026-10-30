# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Recheck saved PR-0012 aggregates only; never run simulations or fit models."""
from pathlib import Path
import json
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.dont_write_bytecode = True
    with tempfile.TemporaryDirectory(prefix='probe-aggregate-check-') as temp:
        os.environ['MPLCONFIGDIR'] = temp
        sys.path.insert(0, str(ROOT / 'numerics/scripts'))
        import numpy as np
        from summarize_pr0012_confirmation import verify_primary
        directory = ROOT / 'numerics/analysis/pr0012_64bf0f2c7697_confirmation_v1'
        with np.load(directory / 'outer_arrays.npz', allow_pickle=False) as saved:
            arrays = dict(saved)
        primary = json.loads((directory / 'primary13.json').read_text())
        assert len(primary) == 13
        classification = verify_primary(arrays, primary)
        assert classification == {'SUPPORTED': 12, 'UNRESOLVED': 1, 'OPPOSITE': 0,
                                  'TECHNICAL_UNESTABLISHED': 0}
        assert [p['id'] for p in primary if p['classification'] == 'UNRESOLVED'] == ['R05']
        for e, n in enumerate((20000, 5000, 5000, 5000)):
            for g in range(4):
                for c in range(6):
                    for ai in range(2):
                        for j, (left, right) in enumerate(((1, 0), (2, 0), (2, 1))):
                            h = arrays['pairs'][:, g, c, e, ai, j]
                            p = arrays['policy'][:, g, c, e, ai]
                            np.testing.assert_array_equal(h.sum(2), p[:, left])
                            np.testing.assert_array_equal(h.sum(1), p[:, right])
                            assert np.all(h.sum((1, 2)) == n)
                            np.testing.assert_array_equal(h[:, 0, 1:].sum(1) - h[:, 1:, 0].sum(1),
                                                          p[:, left, 0] - p[:, right, 0])
        invalid = {k: int(arrays[k][..., -1].sum())
                   for k in ('wcounts', 'lcounts', 'policy', 'lf_pairs')}
        assert not any(invalid.values())
        assert not np.any(arrays['selection'] == -2)
        print(json.dumps({'status': 'PASS_SAVED_AGGREGATES', 'primary': classification,
                          'unresolved': 'R05', 'policy_pair_tables_checked': 50 * 4 * 6 * 4 * 2 * 3,
                          'invalid': invalid,
                          'scope': 'Included outer aggregates; no raw-score replay, simulation, or model fitting.'},
                         indent=2))


if __name__ == '__main__':
    main()
