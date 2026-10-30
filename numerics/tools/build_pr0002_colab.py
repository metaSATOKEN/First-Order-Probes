#!/usr/bin/env python3
# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Mechanical self-contained Colab packaging; never imports the research runner."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[2]
FILES={
    "fusion_confirmation_pr0002_v1.py":"numerics/scripts/fusion_confirmation_pr0002_v1.py",
    "pr0002_checks.py":"numerics/scripts/pr0002_checks.py",
    "covariance_consistency_colab_v1.py":"numerics/scripts/covariance_consistency_colab_v1.py",
    "PR-0002.md":"audits/prereg/PR-0002_fusion_independent_confirmation.md",
    "LICENSE":"LICENSE",
    "LICENSE-CC-BY-4.0.txt":"LICENSE-CC-BY-4.0.txt",
}
bundle={name:(ROOT/path).read_text() for name,path in FILES.items()}
raw=json.dumps(bundle,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
payload=base64.b64encode(zlib.compress(raw,9)).decode()
payload_sha=hashlib.sha256(raw).hexdigest()
settings='''# Copyright 2026 Kentaro Sato -- Apache-2.0 (code); CC BY 4.0 (paper/card)
# PR-0002: paste this WHOLE file into ONE Colab code cell. Default = PILOT ONLY.
MODE = "pilot"                 # keep pilot until separate confirmation-run approval
RESTORE_ZIP = ""                # "UPLOAD", or an uploaded ZIP path; blank = same-runtime resume
PILOT_ZIP = ""                  # only for future fresh confirmation: completed pilot ZIP
CONFIRM_APPROVAL = ""           # never set without separate human main-run approval
AUTO_DOWNLOAD = True
# No Drive connection. Normal CPU runtime is sufficient. Do not run two copies.
# For >1 GB restores, Colab Files sidebar upload + absolute path avoids files.upload's RAM copy.
# Safe pause after 45 minutes; keep the downloaded PARTIAL ZIP on your computer.
# After a restore, set RESTORE_ZIP = "" for the next rerun in the SAME runtime.
'''
bootstrap='''import base64, hashlib, importlib, json, os, sys, zipfile, zlib
from pathlib import Path
if "COLAB_RELEASE_TAG" not in os.environ:
    try:
        import google.colab
    except ImportError as exc:
        raise RuntimeError("Run this cell in Colab. Local pilot/main execution is disabled.") from exc
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
raw = zlib.decompress(base64.b64decode(PAYLOAD))
if hashlib.sha256(raw).hexdigest() != PAYLOAD_SHA256:
    raise ValueError("Embedded source payload is corrupted")
bundle = json.loads(raw)
package = Path("/content") / ("pr0002_package_" + PAYLOAD_SHA256[:12])
package.mkdir(exist_ok=True)
for name, content in bundle.items():
    target = package / name
    if target.exists() and target.read_text() != content:
        raise ValueError("Existing package differs; use a clean runtime")
    target.write_text(content)
sys.path.insert(0, str(package))
for name in ("pr0002_checks", "fusion_confirmation_pr0002_v1", "covariance_consistency_colab_v1"):
    sys.modules.pop(name, None)
import fusion_confirmation_pr0002_v1 as runner
def select_zip(value):
    if value != "UPLOAD":
        return value or None
    from google.colab import files
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Select exactly one ZIP")
    filename = next(iter(uploaded))
    del uploaded
    return str(Path(filename).resolve())
restore_zip = select_zip(RESTORE_ZIP)
pilot_receipt = None
if MODE == "confirm":
    if CONFIRM_APPROVAL != "RUN PR-0002 CONFIRM":
        raise PermissionError("Main run is not authorized. Leave MODE='pilot' for now.")
    if restore_zip:
        with zipfile.ZipFile(restore_zip) as z:
            pilot_receipt = json.loads(z.read("manifest.json"))["pilot_receipt"]
    else:
        pilot_zip = select_zip(PILOT_ZIP)
        if pilot_zip:
            with zipfile.ZipFile(pilot_zip) as z:
                pilot_receipt = json.loads(z.read("pilot_receipt.json"))
        else:
            paths = list(Path("/content/pr0002_runs").glob("*_pilot_*/pilot_receipt.json"))
            matching = [json.loads(p.read_text()) for p in paths
                        if json.loads(p.read_text()).get("cell_payload_sha256") == CELL_ID]
            if len(matching) != 1:
                raise ValueError("Choose the matching completed pilot ZIP via PILOT_ZIP")
            pilot_receipt = matching[0]
RUN_RESULT = runner.run(MODE, "/content/pr0002_runs", bundle, CELL_ID,
    approval=CONFIRM_APPROVAL, pilot_receipt=pilot_receipt,
    restore_zip=restore_zip, auto_download=AUTO_DOWNLOAD)
if MODE == "pilot" and RUN_RESULT["status"] == "PILOT_COMPLETE":
    print("Pilot only completed. Share this ZIP for budget/acceptance review. No confirmation run was started.")
'''
bootstrap_sha=hashlib.sha256(bootstrap.encode()).hexdigest()
cell_id=hashlib.sha256((payload_sha+bootstrap_sha).encode()).hexdigest()
text=settings+f"\nPAYLOAD_SHA256 = {payload_sha!r}\nCELL_ID = {cell_id!r}\nPAYLOAD = {payload!r}\n"+bootstrap
ast.parse(text)
out=ROOT/"numerics/colab"
out.mkdir(exist_ok=True)
py=out/"pr0002_confirmation_one_cell.py"
py.write_text(text)
notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
          "cells":[{"cell_type":"markdown","id":"read-first","metadata":{},"source":[
              "# PR-0002 — pilot by default, CPU, no Drive\n",
              "Run the single code cell. It runs only 8,192 pilot records and downloads a ZIP. Main confirmation needs a separate approval.\n",
              "Safe pauses download PARTIAL ZIPs. Upload the latest ZIP in a new runtime with RESTORE_ZIP = 'UPLOAD', or give its path.\n",
              "Keep backups locally: runtime loss before a completed download can lose new progress.\n"]},
              {"cell_type":"code","id":"pr0002","metadata":{},"execution_count":None,"outputs":[],"source":text.splitlines(keepends=True)}]}
nb=out/"pr0002_confirmation_one_cell.ipynb"
nb.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
manifest={"cell_id":cell_id,"payload_sha256":payload_sha,"bootstrap_sha256":bootstrap_sha,
          "settings_excluded_from_cell_id":["MODE","RESTORE_ZIP","PILOT_ZIP","CONFIRM_APPROVAL","AUTO_DOWNLOAD"],
          "sources":{name:hashlib.sha256(content.encode()).hexdigest() for name,content in bundle.items()},
          "artifacts":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (py,nb)},
          "default":"pilot","main_executed":False}
(out/"pr0002_confirmation_build_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(py)
print(nb)
print("cell_id",cell_id)
