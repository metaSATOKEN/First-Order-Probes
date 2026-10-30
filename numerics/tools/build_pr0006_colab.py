# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Build trusted standalone sources; never run scientific experiments."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib
ROOT=Path(__file__).resolve().parents[2]
PATHS={n:'numerics/scripts/'+n for n in ('probe_check_io.py','probe_check_launcher.py','probe_backend_check_v1.py',
    'pr0004_spec.py','pr0004_methods.py','pr0006_spec.py','pr0006_methods.py','pr0006_io.py','pr0006_runner.py','pr0006_launcher.py')}
PATHS.update({'PR-0006.md':'audits/prereg/PR-0006_lfdc_alarm_budget_confirmation.md',
              'PR-0006-freeze.json':'audits/verification/2026-10-17_PR-0006_design_freeze.json',
              'LICENSE':'LICENSE','LICENSE-CC-BY-4.0.txt':'LICENSE-CC-BY-4.0.txt'})
PREAMBLE='''# PR-0006 B25: LF-DC five fixed alarm budgets + LF0. GPU float64.
# Paste this WHOLE file into ONE Google Colab GPU code cell, then run once.
# 25 replicates; 13,749,700 records. Estimate 50-90 min, hardware-dependent.
# No Google Drive. No 45-minute stop. Current runtime resumes automatically.
# Keep ALL FIVE non-cumulative data-shard ZIPs and the small COMPLETE_INDEX ZIP.
# Every downloadable ZIP is strictly below decimal 1 GB. Allow multiple downloads.
# Intermediate PARTIAL ZIPs contain only the CURRENT shard plus small carry/index.
# Lost runtime: upload the LATEST ZIP to Colab Files, then run this identical cell;
# or set RESTORE_ZIP="UPLOAD". Old raw data are not repeated in later ZIPs.
# Sources Apache-2.0; research card/documentation CC BY 4.0. Kentaro Sato, 2026.
RESTORE_ZIP = ""  # Blank for fresh/automatic resume; optional exact path or "UPLOAD".
AUTO_DOWNLOAD = True
'''
BOOTSTRAP='''
import base64 as _b64, hashlib as _hh, importlib as _il, json as _jj
import os as _os, sys as _sys, zlib as _zz
from pathlib import Path as _Path
if _sys.platform != "linux" or not _Path("/content").is_dir():
    raise RuntimeError("PR-0006 science runs in Colab GPU only; local main disabled.")
_il.import_module("google.colab")
_raw = _zz.decompress(_b64.b64decode(_PAYLOAD))
if _hh.sha256(_raw).hexdigest() != _PAYLOAD_SHA256:
    raise RuntimeError("Embedded source checksum mismatch")
_sources = _jj.loads(_raw)["sources"]
_package = _Path("/content/lfdc_budget_pr0006_v1") / ("package_" + _CELL_ID)
_package.mkdir(parents=True, exist_ok=True)
for _name, _body in _sources.items():
    if _Path(_name).name != _name:
        raise RuntimeError("Unsafe capsule path")
    _p = _package / _name
    _tmp = _p.with_name(_p.name + ".tmp")
    _tmp.write_text(_body, encoding="utf-8")
    _os.replace(_tmp, _p)
_sys.path.insert(0, str(_package))
for _mod in ("pr0006_launcher", "pr0006_io", "probe_check_launcher", "probe_check_io"):
    _sys.modules.pop(_mod, None)
_launcher = _il.import_module("pr0006_launcher")
_result = _launcher.run(_sources, _CELL_ID, _package, RESTORE_ZIP, AUTO_DOWNLOAD)
print(_result)
'''


def build():
    sources={n:(ROOT/p).read_text() for n,p in PATHS.items()}
    freeze=json.loads(sources['PR-0006-freeze.json'])
    if hashlib.sha256(sources['PR-0006.md'].encode()).hexdigest()!=freeze['card_sha256']:raise ValueError('Frozen card changed')
    for n,body in sources.items():
        if n.endswith('.py'):ast.parse(body,filename=n)
    raw=json.dumps(dict(schema=1,sources=sources),sort_keys=True,ensure_ascii=False).encode()
    rawsha=hashlib.sha256(raw).hexdigest();cell_id=hashlib.sha256(raw+PREAMBLE.encode()+BOOTSTRAP.encode()).hexdigest()[:16]
    cell=PREAMBLE+f'\n_CELL_ID = {cell_id!r}\n_PAYLOAD_SHA256 = {rawsha!r}\n_PAYLOAD = {base64.b64encode(zlib.compress(raw,9)).decode()!r}\n'+BOOTSTRAP
    ast.parse(cell);out=ROOT/'numerics/colab';out.mkdir(exist_ok=True)
    py=out/'pr0006_lfdc_budget_b25_one_cell.py';nb=out/'pr0006_lfdc_budget_b25_one_cell.ipynb'
    py.write_text(cell);nb.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,
        metadata=dict(accelerator='GPU',colab=dict(name=nb.name),kernelspec=dict(display_name='Python 3',name='python3')),
        cells=[dict(cell_type='code',id='pr0006-one-cell',metadata={},execution_count=None,outputs=[],source=cell.splitlines(True))]),ensure_ascii=False,indent=1)+'\n')
    m=dict(cell_id=cell_id,payload_sha256=rawsha,sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in PATHS.values()},
           artifacts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (py,nb)})
    (out/'pr0006_lfdc_budget_b25_build.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(dict(cell_id=cell_id,bytes=len(cell.encode()))));return m


if __name__=='__main__':build()
