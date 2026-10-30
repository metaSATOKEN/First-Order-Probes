#!/usr/bin/env python3
# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Prepare main settings for the accepted, byte-identical pilot code payload."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def literals(text):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(text).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Constant)}


def main():
    out = ROOT / "numerics/colab"
    acceptance = ROOT / "output/runs/pr0002_pilot_acceptance_v1/audit.json"
    audit = json.loads(acceptance.read_text())
    build = json.loads((out / "pr0002_confirmation_build_manifest.json").read_text())
    pilot = (out / "pr0002_confirmation_one_cell.py").read_text()
    assert audit["status"] == "PASS" and audit["budget_ok"] is True
    assert audit["cell_id"] == build["cell_id"] == literals(pilot)["CELL_ID"]
    assert sha(pilot.encode()) == build["artifacts"]["pr0002_confirmation_one_cell.py"]
    assert sha((ROOT / audit["pilot_zip"]).read_bytes()) == audit["pilot_zip_sha256"]
    # All code from PAYLOAD_SHA256 onward remains byte-identical. Only the five
    # user-facing settings/comments change, as allowed by the original cell ID.
    settings = '''# Copyright 2026 Kentaro Sato -- Apache-2.0 (code); CC BY 4.0 (paper/card)
# PR-0002 MAIN RUN: paste this WHOLE file into ONE Colab code cell.
# Running this cell starts/resumes the fixed 50-repeat, 10,999,800-record run.
# The provided PILOT_COMPLETE ZIP passed saved-data acceptance on 2026-10-17.
MODE = "confirm"
RESTORE_ZIP = ""                # fresh runtime restore: "UPLOAD" or /content/latest_PARTIAL.zip
PILOT_ZIP = "UPLOAD"            # first run: choose the accepted PILOT_COMPLETE ZIP
CONFIRM_APPROVAL = "RUN PR-0002 CONFIRM"  # execute this cell only when ready to start
AUTO_DOWNLOAD = True
# No Drive, no pip install, CPU implementation; do not run two copies.
# Same runtime: keep RESTORE_ZIP blank; choose the same pilot ZIP again, or set
# PILOT_ZIP to its existing /content/...zip path to avoid another upload prompt.
# New runtime: set RESTORE_ZIP to the latest PARTIAL ZIP; PILOT_ZIP is then ignored.
# For >1 GB, upload via Colab Files and use its absolute path (less upload RAM).
# After restore, clear RESTORE_ZIP before the next SAME-runtime continuation.
# Each cell pauses safely after 45 min; save every PARTIAL ZIP locally.
# Four-hour cumulative budget and all scientific settings remain unchanged.
# Environment mismatches stop; do not bypass the gate or silently upgrade libraries.
'''
    text = settings + "\nPAYLOAD_SHA256 =" + pilot.split("\nPAYLOAD_SHA256 =", 1)[1]
    assert text.split("\nPAYLOAD_SHA256 =", 1)[1] == pilot.split("\nPAYLOAD_SHA256 =", 1)[1]
    a, b = literals(text), literals(pilot)
    assert {k for k in a if a[k] != b[k]} == {"MODE", "PILOT_ZIP", "CONFIRM_APPROVAL"}
    assert a["MODE"] == "confirm" and a["CONFIRM_APPROVAL"] == "RUN PR-0002 CONFIRM"
    assert a["RESTORE_ZIP"] == "" and a["AUTO_DOWNLOAD"] is True
    notebook = {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [{"cell_type": "markdown", "id": "read-first", "metadata": {}, "source": [
            "# PR-0002 本番版 CPU Driveなし\n",
            "下の1セルを実行すると、本番50反復・10,999,800記録を開始します。\n",
            "初回のファイル選択では、点検済みPILOT_COMPLETE ZIPを1つ選んでください。\n",
            "45分ごとのPARTIAL ZIPを保存し、同じランタイムでは同じセルを再実行してください。\n",
            "新ランタイムではRESTORE_ZIPに最新PARTIAL ZIPのパス（またはUPLOAD）を設定します。\n",
            "推定処理約2時間、余裕2〜3時間。DL・アップロード・操作待ちは別です。\n"]},
            {"cell_type": "code", "id": "pr0002-main", "metadata": {}, "execution_count": None,
             "outputs": [], "source": text.splitlines(keepends=True)}]}
    artifacts = {"pr0002_main_one_cell.py": text.encode(),
                 "pr0002_main_one_cell.ipynb": (json.dumps(notebook, ensure_ascii=False, indent=1)+"\n").encode()}
    manifest = dict(default="confirm", main_executed=False, cell_id=a["CELL_ID"],
        pilot_zip=audit["pilot_zip"], pilot_zip_sha256=audit["pilot_zip_sha256"],
        pilot_audit_sha256=sha(acceptance.read_bytes()),
        accepted_environment=audit["accepted_colab_environment"],
        accepted_confirmation_manifest_sha256=audit["confirmation_manifest_sha256"],
        original_build_manifest_sha256=sha((out / "pr0002_confirmation_build_manifest.json").read_bytes()),
        payload_sha256=a["PAYLOAD_SHA256"], source_hashes=build["sources"],
        bootstrap_and_payload_byte_identical=True,
        changed_settings=["MODE", "PILOT_ZIP", "CONFIRM_APPROVAL"],
        artifacts={name: sha(content) for name, content in artifacts.items()})
    artifacts["pr0002_main_build_manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2)+"\n").encode()
    for name, content in artifacts.items():
        p = out / name
        if p.exists() and p.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite a different build: {p}")
    for name, content in artifacts.items():
        (out / name).write_bytes(content)
        print(out / name)


if __name__ == "__main__":
    main()
