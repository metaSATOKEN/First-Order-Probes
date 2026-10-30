# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Deterministic one-cell package builder. Does not run the scientific worker."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib
ROOT=Path(__file__).resolve().parents[2]
PATHS={n:'numerics/scripts/'+n for n in (
    'probe_check_io.py','probe_check_launcher.py','probe_backend_check_v1.py',
    'pr0004_spec.py','pr0004_methods.py','pr0006_spec.py','pr0006_methods.py','pr0006_io.py','pr0006_runner.py',
    'pr0007_spec.py','pr0007_methods.py','pr0007_io.py','pr0007_runner.py','pr0007_download.py','pr0007_launcher.py')}
PATHS.update({'PR-0007.md':'audits/prereg/PR-0007_lfdc_calibration_contamination.md',
    'PR-0007-A01.md':'audits/prereg/PR-0007_A01_five_replicates_per_zip.md',
    'PR-0007-freeze.json':'audits/verification/2026-10-18_PR-0007_design_freeze.json',
    'PR-0007-A01-freeze.json':'audits/verification/2026-10-18_PR-0007_A01.json',
    'LICENSE':'LICENSE','LICENSE-CC-BY-4.0.txt':'LICENSE-CC-BY-4.0.txt',
    'PR0007_THIRD_PARTY_NOTICE.txt':'numerics/PR0007_THIRD_PARTY_NOTICE.txt'})
PREAMBLE='''# PR-0007 + A01: calibration size/contamination, fixed LF-DC budgets + LF0.
# Paste this WHOLE file into ONE Google Colab GPU code cell. Run once.
# 25 replicates / 10,909,800 unique records. Provisional 1-2 hours; not timed for PR-0007.
# No Google Drive; no runtime restart required; no artificial 45-minute stop.
# Same live runtime: automatic resume. New runtime, fresh experiment: leave settings blank.
# Lost runtime: upload the relevant ZIPs using Colab Files before RUN,
# or set RESTORE_ZIPS="UPLOAD" to select them. A delta may need earlier PARTIAL ZIPs.
# Keep ALL FIVE non-cumulative data shards + COMPLETE_INDEX; partial/split parts may add files.
# Automatic downloads are serial and strictly <1,000,000,000 bytes each.
# Allow multiple downloads in your browser. Calculation completion != disk-save confirmation.
# Sources Apache-2.0; cards/documentation CC BY 4.0. Kentaro Sato, 2026.
RESTORE_ZIPS = ""  # Blank, "UPLOAD", one exact path, or a list of exact paths.
AUTO_DOWNLOAD = True
'''
BOOTSTRAP='''
import base64 as _b64, hashlib as _hh, importlib as _il, json as _jj
import os as _os, sys as _sys, zlib as _zz
from pathlib import Path as _Path
if _sys.platform != "linux" or not _Path("/content").is_dir():
    raise RuntimeError("PR-0007 science is Colab GPU only; local main disabled.")
_il.import_module("google.colab")
_raw = _zz.decompress(_b64.b64decode(_PAYLOAD))
if _hh.sha256(_raw).hexdigest() != _PAYLOAD_SHA256:
    raise RuntimeError("Embedded source checksum mismatch")
_sources = _jj.loads(_raw)["sources"]
_package = _Path("/content/lfdc_calibration_pr0007_v1") / ("package_" + _CELL_ID)
_package.mkdir(parents=True, exist_ok=True)
for _name, _body in _sources.items():
    if _Path(_name).name != _name: raise RuntimeError("Unsafe capsule path")
    _p = _package / _name; _tmp = _p.with_name(_p.name + ".tmp")
    _tmp.write_text(_body, encoding="utf-8"); _os.replace(_tmp, _p)
_sys.path.insert(0, str(_package))
for _mod in ("pr0007_launcher", "pr0007_download", "pr0007_io", "probe_check_launcher", "probe_check_io"):
    _sys.modules.pop(_mod, None)
_launcher = _il.import_module("pr0007_launcher")
_result = _launcher.run(_sources, _CELL_ID, _package, RESTORE_ZIPS, AUTO_DOWNLOAD)
print(_result)
'''


def build():
    sources={n:(ROOT/p).read_text() for n,p in PATHS.items()}
    expected={'PR-0007.md':'8309b2143879fef69a5020fc2945fb2159786f3de7c1da62b787b6907d98be37',
              'PR-0007-A01.md':'8c87378bb87920c985f4e94477799cf382a2519c7db3678a85a63ebde05e4a7f'}
    for n,h in expected.items():
        if hashlib.sha256(sources[n].encode()).hexdigest()!=h:raise ValueError('Frozen card changed: '+n)
    for n,b in sources.items():
        if n.endswith('.py'):ast.parse(b,filename=n)
    raw=json.dumps(dict(schema=1,sources=sources),sort_keys=True,ensure_ascii=False).encode()
    rawsha=hashlib.sha256(raw).hexdigest();cell_id=hashlib.sha256(raw+PREAMBLE.encode()+BOOTSTRAP.encode()).hexdigest()[:16]
    cell=PREAMBLE+f'\n_CELL_ID = {cell_id!r}\n_PAYLOAD_SHA256 = {rawsha!r}\n_PAYLOAD = {base64.b64encode(zlib.compress(raw,9)).decode()!r}\n'+BOOTSTRAP
    ast.parse(cell);out=ROOT/'numerics/colab';out.mkdir(exist_ok=True)
    py=out/'pr0007_lfdc_calibration_a25_one_cell.py';nb=py.with_suffix('.ipynb')
    py.write_text(cell);nb.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,
        metadata=dict(accelerator='GPU',colab=dict(name=nb.name),kernelspec=dict(display_name='Python 3',name='python3')),
        cells=[dict(cell_type='code',id='pr0007-one-cell',metadata={},execution_count=None,outputs=[],source=cell.splitlines(True))]),ensure_ascii=False,indent=1)+'\n')
    manifest=dict(cell_id=cell_id,payload_sha256=rawsha,
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in PATHS.values()},
        artifacts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (py,nb)})
    (out/'pr0007_lfdc_calibration_a25_build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(cell_id=cell_id,bytes=len(cell.encode()))));return manifest


if __name__=='__main__':build()
