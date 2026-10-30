# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Package frozen sources. Does not execute scientific experiments."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib
ROOT=Path(__file__).resolve().parents[2]
PATHS={name:'numerics/scripts/'+name for name in (
    'probe_check_io.py','probe_check_launcher.py','probe_backend_check_v1.py',
    'pr0004_spec.py','pr0004_methods.py','pr0004_io.py','pr0004_runner.py','pr0004_launcher.py')}
PATHS.update({'PR-0004.md':'audits/prereg/PR-0004_linear_fusion_source_shift_gpu.md',
              'PR-0004-freeze.json':'audits/verification/2026-10-17_PR-0004_design_freeze.json',
              'PR-0003-audit.json':'numerics/analysis/pr0003_cpu_gpu_23538c370862_v1/delivery_validation.json',
              'LICENSE':'LICENSE','LICENSE-CC-BY-4.0.txt':'LICENSE-CC-BY-4.0.txt'})
PREAMBLE='''# PR-0004 MAIN: LF + M/CR/B/J, 50 replicates, 33,499,400 records.
# Paste this ENTIRE file into ONE Colab GPU code cell and run once.
# Drive-free. Fresh run needs no ZIP. About 2.5-4 hours, hardware dependent.
# Existing runtime: automatic resume. Lost runtime: upload the latest PR-0004
# ZIP into Colab Files before running this same cell. No periodic manual restart.
# Intermediate recovery ZIPs at 10/20/30/40 replicates; final COMPLETE ZIP.
# Allow multiple downloads. Keep the latest ZIP locally until the final one arrives.
# Code Apache-2.0; card/documentation CC BY 4.0. Copyright 2026 Kentaro Sato.
RESTORE_ZIP = ""  # Usually blank; optional exact path or "UPLOAD" for recovery only.
AUTO_DOWNLOAD = True
'''
BOOTSTRAP='''
import base64 as _b64, hashlib as _hh, importlib as _il, json as _jj
import os as _os, sys as _sys, zlib as _zz
from pathlib import Path as _Path
if _sys.platform != "linux" or not _Path("/content").is_dir():
    raise RuntimeError("PR-0004 main runs in Colab GPU only; local execution disabled.")
_il.import_module("google.colab")
_raw = _zz.decompress(_b64.b64decode(_PAYLOAD))
if _hh.sha256(_raw).hexdigest() != _PAYLOAD_SHA256:
    raise RuntimeError("Embedded source checksum mismatch")
_sources = _jj.loads(_raw)["sources"]
_package = _Path("/content/linear_fusion_pr0004_v1") / ("package_" + _CELL_ID)
_package.mkdir(parents=True, exist_ok=True)
for _name, _body in _sources.items():
    if _Path(_name).name != _name:
        raise RuntimeError("Unsafe capsule path")
    _p = _package / _name
    _tmp = _p.with_name(_p.name + ".tmp")
    _tmp.write_text(_body, encoding="utf-8")
    _os.replace(_tmp, _p)
_sys.path.insert(0, str(_package))
for _mod in ("pr0004_launcher", "pr0004_io", "probe_check_launcher", "probe_check_io"):
    _sys.modules.pop(_mod, None)
_launcher = _il.import_module("pr0004_launcher")
_result = _launcher.run(_sources, _CELL_ID, _package, RESTORE_ZIP, AUTO_DOWNLOAD)
print(_result)
'''


def build():
    sources={name:(ROOT/path).read_text() for name,path in PATHS.items()}
    freeze=json.loads(sources['PR-0004-freeze.json'])
    if hashlib.sha256(sources['PR-0004.md'].encode()).hexdigest()!=freeze['card_sha256']:
        raise ValueError('Frozen card changed')
    for name,body in sources.items():
        if name.endswith('.py'): ast.parse(body,filename=name)
    raw=json.dumps(dict(schema=1,sources=sources),sort_keys=True,ensure_ascii=False).encode()
    rawsha=hashlib.sha256(raw).hexdigest()
    cell_id=hashlib.sha256(raw+PREAMBLE.encode()+BOOTSTRAP.encode()).hexdigest()[:16]
    payload=base64.b64encode(zlib.compress(raw,9)).decode()
    cell=PREAMBLE+f'\n_CELL_ID = {cell_id!r}\n_PAYLOAD_SHA256 = {rawsha!r}\n_PAYLOAD = {payload!r}\n'+BOOTSTRAP
    ast.parse(cell); out=ROOT/'numerics/colab'; out.mkdir(exist_ok=True)
    py=out/'pr0004_gpu_main_one_cell.py'; nb=out/'pr0004_gpu_main_one_cell.ipynb'
    py.write_text(cell)
    nb.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,
        metadata=dict(accelerator='GPU',colab=dict(name=nb.name),kernelspec=dict(display_name='Python 3',name='python3')),
        cells=[dict(cell_type='code',id='pr0004-one-cell',metadata={},execution_count=None,outputs=[],source=cell.splitlines(True))]),
        ensure_ascii=False,indent=1)+'\n')
    m=dict(cell_id=cell_id,payload_sha256=rawsha,
           sources={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in PATHS.values()},
           artifacts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (py,nb)})
    (out/'pr0004_gpu_main_build.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(dict(cell_id=cell_id,bytes=len(cell.encode())),ensure_ascii=False))
    return m


if __name__=='__main__': build()
