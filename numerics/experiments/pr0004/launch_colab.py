# PR-0004 MAIN: LF + M/CR/B/J, 50 replicates, 33,499,400 records.
# Drive-free. Fresh run needs no ZIP. About 2.5-4 hours, hardware dependent.
# Existing runtime: automatic resume. Lost runtime: upload the latest PR-0004
# ZIP into Colab Files before running this same cell. No periodic manual restart.
# Intermediate recovery ZIPs at 10/20/30/40 replicates; final COMPLETE ZIP.
# Allow multiple downloads. Keep the latest ZIP locally until the final one arrives.
# Code Apache-2.0; card/documentation CC BY 4.0. Copyright 2026 Kentaro Sato.
RESTORE_ZIP = _READABLE_SETTINGS['RESTORE_ZIP']
AUTO_DOWNLOAD = _READABLE_SETTINGS['AUTO_DOWNLOAD']

_CELL_ID = '0e4b2ef31e02eacd'
_PAYLOAD_SHA256 = 'e79e19c73a3ae0c83d5a6434d2f7f60d78074c792a17a608590b47a31a85eb56'

import base64 as _b64, hashlib as _hh, importlib as _il, json as _jj
import os as _os, sys as _sys, zlib as _zz
from pathlib import Path as _Path
if _sys.platform != "linux" or not _Path("/content").is_dir():
    raise RuntimeError("PR-0004 main runs in Colab GPU only; local execution disabled.")
_il.import_module("google.colab")
_raw = _READABLE_RAW
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
