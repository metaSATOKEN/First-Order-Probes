# PR-0007 + A01: calibration size/contamination, fixed LF-DC budgets + LF0.
# 25 replicates / 10,909,800 unique records. Provisional 1-2 hours; not timed for PR-0007.
# No Google Drive; no runtime restart required; no artificial 45-minute stop.
# Same live runtime: automatic resume. New runtime, fresh experiment: leave settings blank.
# Lost runtime: upload the relevant ZIPs using Colab Files before RUN,
# or set RESTORE_ZIPS="UPLOAD" to select them. A delta may need earlier PARTIAL ZIPs.
# Keep ALL FIVE non-cumulative data shards + COMPLETE_INDEX; partial/split parts may add files.
# Automatic downloads are serial and strictly <1,000,000,000 bytes each.
# Allow multiple downloads in your browser. Calculation completion != disk-save confirmation.
# Sources Apache-2.0; cards/documentation CC BY 4.0. Kentaro Sato, 2026.
RESTORE_ZIPS = _READABLE_SETTINGS['RESTORE_ZIPS']
AUTO_DOWNLOAD = _READABLE_SETTINGS['AUTO_DOWNLOAD']

_CELL_ID = 'af385f6b95739a08'
_PAYLOAD_SHA256 = 'eaa2dd2e093c16e05ae1623d5aec13c76002fb643c211fd73243eddb56dbcd17'

import base64 as _b64, hashlib as _hh, importlib as _il, json as _jj
import os as _os, sys as _sys, zlib as _zz
from pathlib import Path as _Path
if _sys.platform != "linux" or not _Path("/content").is_dir():
    raise RuntimeError("PR-0007 science is Colab GPU only; local main disabled.")
_il.import_module("google.colab")
_raw = _READABLE_RAW
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
