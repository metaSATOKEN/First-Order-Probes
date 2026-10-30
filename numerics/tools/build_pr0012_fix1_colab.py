# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Deterministic source capsule builder. No scientific execution."""
import ast,base64,hashlib,json,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PATHS={n:'numerics/scripts/'+n for n in (
    'probe_check_io.py','probe_check_launcher.py','probe_backend_check_v1.py',
    'pr0004_spec.py','pr0004_methods.py','pr0006_spec.py','pr0006_methods.py','pr0006_io.py','pr0006_runner.py',
    'pr0007_spec.py','pr0008_shadow_methods.py',
    'pr0012_spec.py','pr0012_input.py','pr0012_methods.py','pr0012_storage.py',
    'pr0012_io.py','pr0012_runner.py','pr0012_download.py','pr0012_launcher.py')}
PATHS.update({'PR-0012.md':'audits/prereg/PR-0012_mixture_recovery_confirmation.md',
    'PR0012-development.json':'audits/verification/2026-10-18_PR-0009_development.json',
    'PR0012-freeze.json':'audits/verification/2026-10-18_PR-0012_design_freeze.json',
    'PR0012-design.md':'numerics/design/MIXTURE_RECOVERY_INDEPENDENT_CONFIRMATION_DESIGN_v1_JA.md',
    'LICENSE':'LICENSE','LICENSE-CC-BY-4.0.txt':'LICENSE-CC-BY-4.0.txt',
    'PR0012_THIRD_PARTY_NOTICE.txt':'numerics/PR0012_THIRD_PARTY_NOTICE.txt'})
PATHS.update({n:'numerics/repairs/pr0012_preflight_fix1/'+n for n in
    ('pr0012_runner.py','pr0012_launcher.py','pr0012_preflight.py')})
PATHS['PR0012-A01.md']='audits/prereg/PR-0012_A01_preflight_boundary_fix.md'
PREAMBLE='''# PR-0012 B50: independent mixture cancellation and conditional recovery.
# FIX1: CPU/GPU preflight boundary audit repaired; scientific rules unchanged.
# Old failed run 606fc449db00 saved ZERO science records. Keep its failure ZIP.
# Paste this corrected cell and RUN; no runtime disconnect or ZIP picker needed.
# Separate fix1 runtime folder preserves the old run. Only restore FIX1 ZIPs here.
# Paste THIS WHOLE FILE into ONE Google Colab GPU code cell, then RUN.
# 50 replicates / 23,999,200 records. Planning estimate 2-3 hours; reserve 4 hours.
# No Drive; no runtime restart; no artificial 45-minute exit. GPU float64.
# Live runtime: automatic resume without picker. Fresh run: leave settings below blank.
# Lost runtime: upload saved ZIPs into /content first, or set RESTORE_ZIPS="UPLOAD".
# Restore the latest complete shard/index and all ongoing PARTIAL dependencies.
# Keep all TEN non-cumulative data shards and COMPLETE_INDEX; partial/parts may add ZIPs.
# Every ZIP is strictly below decimal 1 GB. Serial browser download queue.
# Allow multiple downloads; COMPLETE means science done, NOT all browser saves finished.
# Code Apache-2.0; documents CC BY 4.0. Kentaro Sato, 2026.
RESTORE_ZIPS = ""  # Blank, "UPLOAD", one exact path, or list of exact paths.
AUTO_DOWNLOAD = True
'''
BOOTSTRAP='''
import base64 as _b64, hashlib as _hh, importlib as _il, json as _jj
import os as _os, sys as _sys, zlib as _zz
from pathlib import Path as _Path
if _sys.platform != "linux" or not _Path("/content").is_dir():
    raise RuntimeError("PR-0012 scientific main is Colab GPU only.")
_il.import_module("google.colab")
_raw = _zz.decompress(_b64.b64decode(_PAYLOAD))
if _hh.sha256(_raw).hexdigest() != _PAYLOAD_SHA256: raise RuntimeError("Payload checksum")
_sources = _jj.loads(_raw)["sources"]
_package = _Path("/content/mixture_recovery_pr0012_v1_preflightfix1") / ("package_" + _CELL_ID)
_package.mkdir(parents=True, exist_ok=True)
for _name, _body in _sources.items():
    if _Path(_name).name != _name: raise RuntimeError("Unsafe source path")
    _p = _package / _name; _tmp = _p.with_name(_p.name + ".tmp")
    _tmp.write_text(_body, encoding="utf-8"); _os.replace(_tmp, _p)
_sys.path.insert(0, str(_package))
for _mod in ("pr0012_launcher", "pr0012_download", "pr0012_io", "probe_check_launcher", "probe_check_io"):
    _sys.modules.pop(_mod, None)
_launcher = _il.import_module("pr0012_launcher")
_result = _launcher.run(_sources, _CELL_ID, _package, RESTORE_ZIPS, AUTO_DOWNLOAD)
print(_result)
'''

def build():
    sources={n:(ROOT/p).read_text() for n,p in PATHS.items()}
    freeze=json.loads(sources['PR0012-freeze.json'])
    for record in [freeze['card'],freeze['design']]:
        if hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Frozen document changed')
    for p,h in freeze['inherited_sources'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Inherited source changed: '+p)
    for n,b in sources.items():
        if n.endswith('.py'):ast.parse(b,filename=n)
    raw=json.dumps(dict(schema=1,sources=sources),sort_keys=True,ensure_ascii=False).encode()
    rawsha=hashlib.sha256(raw).hexdigest();cellid=hashlib.sha256(raw+PREAMBLE.encode()+BOOTSTRAP.encode()).hexdigest()[:16]
    cell=PREAMBLE+f'\n_CELL_ID = {cellid!r}\n_PAYLOAD_SHA256 = {rawsha!r}\n_PAYLOAD = {base64.b64encode(zlib.compress(raw,9)).decode()!r}\n'+BOOTSTRAP
    ast.parse(cell);out=ROOT/'numerics/colab';out.mkdir(exist_ok=True)
    py=out/'pr0012_mixture_recovery_b50_fix1_one_cell.py';nb=py.with_suffix('.ipynb')
    py.write_text(cell);nb.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,
        metadata=dict(accelerator='GPU',colab=dict(name=nb.name),kernelspec=dict(display_name='Python 3',name='python3')),
        cells=[dict(cell_type='code',id='pr0012-one-cell',metadata={},execution_count=None,outputs=[],source=cell.splitlines(True))]),ensure_ascii=False,indent=1)+'\n')
    manifest=dict(cell_id=cellid,payload_sha256=rawsha,sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in PATHS.values()},
        artifacts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (py,nb)})
    (out/'pr0012_mixture_recovery_b50_fix1_build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(cell_id=cellid,bytes=len(cell.encode()))));return manifest

if __name__=='__main__':build()
