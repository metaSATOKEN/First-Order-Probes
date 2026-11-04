# PR-0009 A50: candidate-contamination gate and old-calibration FPR evidence.
# 50 replicates / 31,039,200 records. Planning estimate 2.5-4 hours, not yet timed.
# No Drive; no runtime restart; no artificial 45-minute exit. GPU float64.
# Live runtime: automatic resume without picker. Fresh run: leave settings below blank.
# Lost runtime: upload saved ZIPs into /content first, or set RESTORE_ZIPS="UPLOAD".
# Restore the latest complete shard/index and all ongoing PARTIAL dependencies.
# Keep all TEN non-cumulative data shards and COMPLETE_INDEX; partial/parts may add ZIPs.
# Every ZIP is strictly below decimal 1 GB. Serial browser download queue.
# Allow multiple downloads; COMPLETE means science done, NOT all browser saves finished.
# Code Apache-2.0; documents CC BY 4.0. Kentaro Sato, 2026.
RESTORE_ZIPS = _READABLE_SETTINGS['RESTORE_ZIPS']
AUTO_DOWNLOAD = _READABLE_SETTINGS['AUTO_DOWNLOAD']

_CELL_ID = '51187a0316767e6a'
_PAYLOAD_SHA256 = '92810c7853eab2d733bf49c9f3c3e753c6c42f65990e8b52dac4a8617c56d16f'

import base64 as _b64, hashlib as _hh, importlib as _il, json as _jj
import os as _os, sys as _sys, zlib as _zz
from pathlib import Path as _Path
if _sys.platform != "linux" or not _Path("/content").is_dir():
    raise RuntimeError("PR-0009 scientific main is Colab GPU only.")
_il.import_module("google.colab")
_raw = _READABLE_RAW
if _hh.sha256(_raw).hexdigest() != _PAYLOAD_SHA256: raise RuntimeError("Payload checksum")
_sources = _jj.loads(_raw)["sources"]
_package = _Path("/content/two_axis_pr0009_v1") / ("package_" + _CELL_ID)
_package.mkdir(parents=True, exist_ok=True)
for _name, _body in _sources.items():
    if _Path(_name).name != _name: raise RuntimeError("Unsafe source path")
    _p = _package / _name; _tmp = _p.with_name(_p.name + ".tmp")
    _tmp.write_text(_body, encoding="utf-8"); _os.replace(_tmp, _p)
_sys.path.insert(0, str(_package))
for _mod in ("pr0009_launcher", "pr0009_download", "pr0009_io", "probe_check_launcher", "probe_check_io"):
    _sys.modules.pop(_mod, None)
_launcher = _il.import_module("pr0009_launcher")
_result = _launcher.run(_sources, _CELL_ID, _package, RESTORE_ZIPS, AUTO_DOWNLOAD)
print(_result)
