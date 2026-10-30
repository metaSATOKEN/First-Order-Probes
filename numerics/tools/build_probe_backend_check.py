# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Mechanical packaging only: does not run the pilot."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    'probe_check_io.py': 'numerics/scripts/probe_check_io.py',
    'probe_check_launcher.py': 'numerics/scripts/probe_check_launcher.py',
    'probe_backend_check_v1.py': 'numerics/scripts/probe_backend_check_v1.py',
    'PR-0003.md': 'audits/prereg/PR-0003_cpu_gpu_backend_pilot.md',
    'LICENSE': 'LICENSE',
    'LICENSE-CC-BY-4.0.txt': 'LICENSE-CC-BY-4.0.txt',
}

PREAMBLE = '''# PR-0003 CPU/GPU float64 pilot — paste the ENTIRE file in one Colab cell.
# Use a GPU runtime (A100 if available). No Drive, HF token, or previous ZIP needed.
# Same-runtime rerun: automatic resume. Lost runtime: upload latest result ZIP
# into Colab Files, then rerun this same cell. No 45-minute scheduled stop.
# Copyright 2026 Kentaro Sato; code Apache-2.0; paper/card CC BY 4.0.
RESTORE_ZIP = ""  # Normally leave blank. Optional exact path, or "UPLOAD".
AUTO_DOWNLOAD = True
'''
BOOTSTRAP = '''
import base64 as _b64, hashlib as _hh, importlib as _il, json as _jj
import os as _os, sys as _sys, zlib as _zz
from pathlib import Path as _Path
if _sys.platform != "linux" or not _Path("/content").is_dir():
    raise RuntimeError("This pilot runs in Colab GPU; local execution is disabled.")
_il.import_module("google.colab")
_raw = _zz.decompress(_b64.b64decode(_PAYLOAD))
if _hh.sha256(_raw).hexdigest() != _PAYLOAD_SHA256:
    raise RuntimeError("Embedded source checksum mismatch")
_capsule = _jj.loads(_raw)
_sources = _capsule["sources"]
_package = _Path("/content/probe_cpu_gpu_check_v1") / ("package_" + _CELL_ID)
_package.mkdir(parents=True, exist_ok=True)
for _name, _body in _sources.items():
    if _Path(_name).name != _name:
        raise RuntimeError("Unsafe embedded filename")
    _p = _package / _name
    _tmp = _p.with_name(_p.name + ".tmp")
    _tmp.write_text(_body, encoding="utf-8")
    _os.replace(_tmp, _p)
_sys.path.insert(0, str(_package))
for _mod in ("probe_check_launcher", "probe_check_io"):
    _sys.modules.pop(_mod, None)
_launcher = _il.import_module("probe_check_launcher")
_result = _launcher.run(_sources, _CELL_ID, _package, RESTORE_ZIP, AUTO_DOWNLOAD)
print(_result)
'''


def build():
    sources = {name:(ROOT/path).read_text() for name,path in PATHS.items()}
    for name, body in sources.items():
        if name.endswith('.py'):
            ast.parse(body,filename=name)
    raw = json.dumps(dict(schema=1,sources=sources),sort_keys=True,ensure_ascii=False).encode()
    rawsha = hashlib.sha256(raw).hexdigest()
    cell_id = hashlib.sha256(raw+PREAMBLE.encode()+BOOTSTRAP.encode()).hexdigest()[:16]
    payload = base64.b64encode(zlib.compress(raw,9)).decode()
    cell = PREAMBLE + f'\n_CELL_ID = {cell_id!r}\n_PAYLOAD_SHA256 = {rawsha!r}\n_PAYLOAD = {payload!r}\n' + BOOTSTRAP
    ast.parse(cell)
    out = ROOT/'numerics/colab'
    out.mkdir(exist_ok=True)
    py = out/'probe_cpu_gpu_check_one_cell.py'
    nb = out/'probe_cpu_gpu_check_one_cell.ipynb'
    py.write_text(cell)
    notebook = dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',
        colab=dict(name=nb.name), kernelspec=dict(display_name='Python 3',name='python3')),
        cells=[dict(cell_type='code',id='pr0003-one-cell',metadata={},execution_count=None,outputs=[],source=cell.splitlines(True))])
    nb.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    manifest = dict(cell_id=cell_id,payload_sha256=rawsha,
        sources={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in PATHS.values()},
        artifacts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (py,nb)})
    (out/'probe_cpu_gpu_check_build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(cell_id=cell_id,bytes=len(cell.encode()),artifacts=list(manifest['artifacts'])),ensure_ascii=False))
    return manifest


if __name__=='__main__':
    build()
