#!/usr/bin/env python3
# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Run engineering tests and save their exact scope; never run the Colab pilot/main."""
import ast
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import unittest
import zlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"numerics/scripts"))
import fusion_confirmation_pr0002_v1 as f

def protected():
    lines=(ROOT/"audits/verification/2026-10-17_setup_baseline.sha256").read_text().splitlines()
    entries=[re.fullmatch(r"([0-9a-f]{64})\s+(.+)",line) for line in lines]
    pairs=[m for m in entries if m]
    assert len(pairs)==124
    for m in pairs:
        assert f.sha_file(ROOT/m[2])==m[1],m[2]
    assert f.sha_file(ROOT/"audits/prereg/PR-0002_fusion_independent_confirmation.md")==f.CARD_SHA
    return 124

def main():
    protected()
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="output/runs/pr0002_implementation_validation")
    out=ROOT/parser.parse_args().output
    if out.exists():
        raise FileExistsError("Preserve the previous validation; choose a new explicit version")
    out.mkdir(parents=True)
    begin=time.monotonic()
    log=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/"numerics/tests"),pattern="test_pr0002.py")
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    f.atomic(out/"unittest.log",log.getvalue().encode())
    build=ROOT/"numerics/colab/pr0002_confirmation_build_manifest.json"
    meta=json.loads(build.read_text())
    py=ROOT/"numerics/colab/pr0002_confirmation_one_cell.py"
    tree=ast.parse(py.read_text())
    literals={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
              and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
    raw=zlib.decompress(base64.b64decode(literals["PAYLOAD"]))
    assert hashlib.sha256(raw).hexdigest()==meta["payload_sha256"]==literals["PAYLOAD_SHA256"]
    payload=json.loads(raw)
    for name,sha in meta["sources"].items():assert hashlib.sha256(payload[name].encode()).hexdigest()==sha
    for name,sha in meta["artifacts"].items():assert f.sha_file(py.parent/name)==sha
    assert payload["fusion_confirmation_pr0002_v1.py"]==Path(f.__file__).read_text()
    assert literals["MODE"]=="pilot" and literals["CONFIRM_APPROVAL"]==""
    nb=json.loads(py.with_suffix(".ipynb").read_text())
    assert "".join(nb["cells"][1]["source"])==py.read_text()
    assert len([c for c in nb["cells"] if c["cell_type"]=="code"])==1
    validation=dict(status="PASS" if result.wasSuccessful() else "FAIL",tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.monotonic()-begin,
        environment=f.environment(),tolerance=f.TOLERANCE,protected_assets_unchanged=protected(),
        bundle_verified=True,cell_id=meta["cell_id"],main_executed=False,colab_pilot_executed=False,
        browser_download_verified=False,
        scope="Local small engineering fixtures, deterministic scores for orchestration/aggregation; no scientific performance estimate.",
        source_hashes={str(p.relative_to(ROOT)):f.sha_file(p) for p in (
            Path(f.__file__),ROOT/"numerics/scripts/pr0002_checks.py",ROOT/"numerics/tests/test_pr0002.py",
            ROOT/"numerics/tools/build_pr0002_colab.py",Path(__file__),py,py.with_suffix(".ipynb"),build)})
    f.write_json(out/"validation.json",validation)
    print(json.dumps(validation,indent=2))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=="__main__":main()
