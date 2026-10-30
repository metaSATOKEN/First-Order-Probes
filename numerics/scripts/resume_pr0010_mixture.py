# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Idempotent entry point: never rewrites a completed PR-0010 validation record."""
import analyze_pr0010_mixture as m

if __name__=='__main__':
    if (m.OUT/'validation.json').exists():
        v=m.load(m.OUT/'validation.json')
        assert v['status']=='PASS_SAVED_DATA_RECONSTRUCTION'
        for p,h in v['inputs'].items():assert m.sha(m.ROOT/p)==h,p
        for r in range(50):
            p=m.OUT/'recomputed'/f'rep_{r:02d}.npz';receipt=m.load(p.with_suffix('.json'))
            assert receipt['inputs']==v['inputs'] and m.sha(p)==receipt['output_sha256']
        print('COMPLETE_VERIFIED_NOOP: 50 receipts verified; completed evidence not rewritten')
    else:m.main()
