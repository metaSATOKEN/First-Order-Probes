#!/usr/bin/env python3
# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Mechanically build a self-contained one-cell script and notebook."""
import base64
import ast
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "numerics/scripts/covariance_consistency_colab_v1.py"
destination = ROOT / "numerics/colab"
destination.mkdir(exist_ok=True)
text = source.read_text(encoding="utf-8")
payload = {name: (ROOT/name).read_text(encoding="utf-8")
           for name in ("LICENSE", "LICENSE-CC-BY-4.0.txt")}
encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()
text = text.replace('LICENSE_PAYLOAD = ""  # populated in the generated self-contained Colab artifact',
                    'LICENSE_PAYLOAD = ' + repr(encoded) + '  # embedded license texts')
ast.parse(text)  # Static syntax check only: does not import or execute the suite.
(destination / "covariance_consistency_one_cell.py").write_text(text, encoding="utf-8")
notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "colab": {"name": "Covariance consistency experiments"}},
    "cells": [
        {"cell_type": "markdown", "id": "instructions", "metadata": {}, "source": [
            "# Covariance consistency: three synthetic experiments\n",
            "Run the single code cell below. Default: **main**, **no Google Drive connection**, runtime-local checkpoints, a safe pause after 45 minutes.\n",
            "A COMPLETE/PARTIAL ZIP downloads on completion or a safe pause. Keep each downloaded ZIP on your computer.\n",
            "In the same runtime, rerun the same cell to resume. If the runtime is deleted, its local files are lost: in a fresh runtime, set RESTORE_ZIP = 'UPLOAD' and select the latest saved ZIP.\n",
            "After restoring, reset RESTORE_ZIP = '' before the next rerun. Progress after the last downloaded ZIP cannot be recovered if the runtime is deleted.\n",
            "Do not run two copies against the same run folder simultaneously.\n",
            "Set PROFILE = 'smoke' for a short implementation check (not scientific evidence).\n",
            "See the repository numerics/COLAB_EXPERIMENTS_JA.md for the full protocol and limitations.\n"]},
        {"cell_type": "code", "id": "run-suite", "metadata": {}, "execution_count": None,
         "outputs": [], "source": text.splitlines(keepends=True)},
    ],
}
(destination / "covariance_consistency_one_cell.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(destination / "covariance_consistency_one_cell.py")
print(destination / "covariance_consistency_one_cell.ipynb")
