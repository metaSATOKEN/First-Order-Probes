# Use a GPU runtime (A100 if available). No Drive, HF token, or previous ZIP needed.
# Same-runtime rerun: automatic resume. Lost runtime: upload latest result ZIP
# into Colab Files, then rerun this same cell. No 45-minute scheduled stop.
# Copyright 2026 Kentaro Sato; code Apache-2.0; paper/card CC BY 4.0.
RESTORE_ZIP = _READABLE_SETTINGS['RESTORE_ZIP']
AUTO_DOWNLOAD = _READABLE_SETTINGS['AUTO_DOWNLOAD']

_CELL_ID = '8178b6de2ae1584b'
_PAYLOAD_SHA256 = 'bb41051d0de3710bd8a4f6a85a53b2388a6d3456a9ddc7d8901ebf2b4c3a442b'

import base64 as _b64, hashlib as _hh, importlib as _il, json as _jj
import os as _os, sys as _sys, zlib as _zz
from pathlib import Path as _Path
if _sys.platform != "linux" or not _Path("/content").is_dir():
    raise RuntimeError("This pilot runs in Colab GPU; local execution is disabled.")
_il.import_module("google.colab")
_raw = _READABLE_RAW
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
