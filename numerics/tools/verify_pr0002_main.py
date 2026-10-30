#!/usr/bin/env python3
# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Check main delivery without executing any Colab cell or generating records."""
import ast
import base64
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "numerics/scripts"))
import fusion_confirmation_pr0002_v1 as f
from build_pr0002_main_colab import literals
from verify_pr0002 import protected


def main():
    out = ROOT / "output/runs/pr0002_pilot_acceptance_v1"
    destination = out / "main_delivery_validation.json"
    if destination.exists():
        raise FileExistsError("Preserve previous verification")
    audit = f.read_json(out / "audit.json")
    manifest = f.read_json(out / "accepted_confirmation_manifest.json")
    build = f.read_json(ROOT / "numerics/colab/pr0002_main_build_manifest.json")
    old_build = f.read_json(ROOT / "numerics/colab/pr0002_confirmation_build_manifest.json")
    pilot = (ROOT / "numerics/colab/pr0002_confirmation_one_cell.py").read_text()
    main_cell = (ROOT / "numerics/colab/pr0002_main_one_cell.py").read_text()
    compile(main_cell, "pr0002_main_one_cell.py", "exec")  # Syntax only, never exec.
    a, b = literals(main_cell), literals(pilot)
    assert main_cell.split("\nPAYLOAD_SHA256 =", 1)[1] == pilot.split("\nPAYLOAD_SHA256 =", 1)[1]
    assert {k for k in a if a[k] != b[k]} == {"MODE", "PILOT_ZIP", "CONFIRM_APPROVAL"}
    assert (a["MODE"], a["PILOT_ZIP"], a["CONFIRM_APPROVAL"], a["RESTORE_ZIP"], a["AUTO_DOWNLOAD"]) == (
        "confirm", "UPLOAD", "RUN PR-0002 CONFIRM", "", True)
    raw = zlib.decompress(base64.b64decode(a["PAYLOAD"]))
    assert f.sha_bytes(raw) == a["PAYLOAD_SHA256"] == build["payload_sha256"]
    sources = json.loads(raw)
    assert {k: f.sha_bytes(v.encode()) for k, v in sources.items()} == build["source_hashes"] == manifest["source_hashes"]
    assert a["CELL_ID"] == build["cell_id"] == old_build["cell_id"] == manifest["cell_payload_sha256"]
    for meta in (build, old_build):
        for name, sha in meta["artifacts"].items():
            assert f.sha_file(ROOT / "numerics/colab" / name) == sha
    nb = f.read_json(ROOT / "numerics/colab/pr0002_main_one_cell.ipynb")
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 1 and "".join(cells[0]["source"]) == main_cell
    assert cells[0]["execution_count"] is None and cells[0]["outputs"] == []
    assert manifest["specification"] == f.specification("confirm")
    assert manifest["environment"] == audit["accepted_colab_environment"] == build["accepted_environment"]
    assert f.sha_bytes(f.canonical(manifest).encode()) == audit["confirmation_manifest_sha256"] == build["accepted_confirmation_manifest_sha256"]
    assert f.sha_file(out / "audit.json") == build["pilot_audit_sha256"]
    assert f.sha_file(ROOT / audit["pilot_zip"]) == audit["pilot_zip_sha256"] == build["pilot_zip_sha256"]
    previous = f.read_json(ROOT / "output/runs/pr0002_implementation_validation_v2/validation.json")
    for name, sha in previous["source_hashes"].items():
        assert f.sha_file(ROOT / name) == sha
    assert f.sha_bytes((ROOT / "audits/ledger_2026-10.md").read_bytes()[:27905]) == "b5770d7ab18bdb859ff4d67b8be9407c3b3a469b3ada0155653da284a411aabd"
    docs = [ROOT / p for p in (
        "audits/verification/2026-10-17_PR-0002_pilot_acceptance.md", "numerics/PR0002_MAIN_COLAB_JA.md",
        "audits/README.md", "audits/prereg/README.md", "HANDOFF/LATEST.md", "audits/ledger_2026-10.md")]
    checked = 0
    for p in docs:
        for target in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            if re.match(r"[a-zA-Z]+://|#", target):
                continue
            link = (p.parent / unquote(target.split("#")[0].strip("<>"))).resolve()
            if link != destination:  # This verification's output is written after all checks.
                assert link.exists(), (p, target)
            checked += 1
    changed = docs[:-1] + [ROOT / p for p in (
        "numerics/tools/audit_pr0002_pilot.py", "numerics/tools/build_pr0002_main_colab.py",
        "numerics/tools/verify_pr0002_main.py", "numerics/colab/pr0002_main_one_cell.py",
        "numerics/colab/pr0002_main_one_cell.ipynb", "numerics/colab/pr0002_main_build_manifest.json")]
    result = dict(status="PASS", main_executed=False, new_simulation_records=0,
        syntax=True, one_notebook_code_cell=True, notebook_equals_python=True,
        payload_and_bootstrap_unchanged=True, main_settings_correct=True,
        accepted_manifest_matches=True, original_implementation_8_hashes_unchanged=True,
        pilot_archive_unchanged=True, ledger_prefix_preserved_bytes=27905,
        protected_assets_unchanged=protected(), resolved_local_links=checked,
        source_hashes={str(p.relative_to(ROOT)): f.sha_file(p) for p in changed})
    f.write_json(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != "source_hashes"}, indent=2))


if __name__ == "__main__":
    main()
