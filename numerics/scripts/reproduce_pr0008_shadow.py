# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Reproduce immutable PR-0008 science after legitimate append-only navigation updates."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import analyze_pr0008_shadow as analysis


def reproduction_snapshot(out):
    original = json.loads((analysis.DEFAULT/'start_snapshot.json').read_text())
    for name, expected in {**original['protected'], **original['sources']}.items():
        assert analysis.h.digest(analysis.ROOT/name) == expected, name
    ledger = (analysis.ROOT/'audits/ledger_2026-10.md').read_bytes()
    assert hashlib.sha256(ledger[:original['ledger_bytes']]).hexdigest() == original['ledger_sha256']
    snapshot = dict(original, python=platform.python_version(), numpy=np.__version__,
        reproduction_of=str(analysis.DEFAULT.relative_to(analysis.ROOT)),
        current_navigation_hashes={name:analysis.h.digest(analysis.ROOT/name) for name in
            ('audits/README.md','audits/prereg/README.md','HANDOFF/LATEST.md','audits/ledger_2026-10.md')},
        reproduction_driver_sha256=analysis.h.digest(__file__),
        note='Frozen science/source hashes and original ledger prefix enforced; current navigation recorded separately.')
    analysis.write(out,'start_snapshot.json',snapshot)
    return snapshot


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    analysis.snapshot=reproduction_snapshot
    try: analysis.run(args.out)
    except BaseException as exc:
        analysis.write(args.out,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
