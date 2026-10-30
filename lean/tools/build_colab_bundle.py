"""Mechanically embed the project in a self-contained Colab cell and notebook."""

import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "lean" / "first_order_probe"
NAMES = ["FirstOrderProbe.lean", "lean-toolchain", "lakefile.toml", "README.md"]


def build():
    files = {name: (PROJECT / name).read_text(encoding="utf-8") for name in NAMES}
    payload = json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")
    encoded = base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")
    template = (ROOT / "lean" / "tools" / "colab_runner.py").read_text(encoding="utf-8")
    result = template.replace("__PROJECT_PAYLOAD__", encoded).replace(
        "__PROJECT_PAYLOAD_SHA256__", hashlib.sha256(payload).hexdigest()
    )
    result = result.replace(
        '"""Runner template. build_colab_bundle.py embeds the checked project below."""',
        '"""Paste this ENTIRE file into ONE Colab Python cell, then run it."""',
    )
    target = ROOT / "lean" / "colab" / "first_order_probe_lean_one_cell.py"
    target.write_text(result, encoding="utf-8")
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"},
                     "colab": {"name": "Paper 1 — Lean algebraic supplement"}},
        "cells": [
            {"cell_type": "markdown", "id": "instructions", "metadata": {},
             "source": ["# Paper 1 — Lean algebraic supplement\n",
                        "Run the single code cell below in a CPU runtime. No upload, GPU, or local Lean installation is needed.\n",
                        "A PASS/FAIL ZIP containing sources, logs, and verification.json is downloaded at completion.\n",
                        "The supplement covers real rational identities and scalar bounds, not the full stochastic-process theory.\n"]},
            {"cell_type": "code", "id": "verify", "metadata": {}, "execution_count": None,
             "outputs": [], "source": result.splitlines(keepends=True)},
        ],
    }
    target.with_suffix(".ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    compile(result, str(target), "exec")
    print(f"Generated {target.name} and {target.with_suffix('.ipynb').name}")
    print(f"Embedded source SHA-256: {hashlib.sha256(files['FirstOrderProbe.lean'].encode()).hexdigest()}")


if __name__ == "__main__":
    build()
