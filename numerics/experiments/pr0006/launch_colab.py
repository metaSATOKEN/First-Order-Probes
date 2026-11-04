# PR-0006 B25: LF-DC five fixed alarm budgets + LF0. GPU float64.
# 25 replicates; 13,749,700 records. Estimate 50-90 min, hardware-dependent.
# No Google Drive. No 45-minute stop. Current runtime resumes automatically.
# Keep ALL FIVE non-cumulative data-shard ZIPs and the small COMPLETE_INDEX ZIP.
# Every downloadable ZIP is strictly below decimal 1 GB. Allow multiple downloads.
# Intermediate PARTIAL ZIPs contain only the CURRENT shard plus small carry/index.
# Lost runtime: upload the LATEST ZIP to Colab Files, then run this identical cell;
# or set RESTORE_ZIP="UPLOAD". Old raw data are not repeated in later ZIPs.
# Sources Apache-2.0; research card/documentation CC BY 4.0. Kentaro Sato, 2026.
RESTORE_ZIP = _READABLE_SETTINGS['RESTORE_ZIP']
AUTO_DOWNLOAD = _READABLE_SETTINGS['AUTO_DOWNLOAD']

_CELL_ID = 'f98f8d0167e97f69'
_PAYLOAD_SHA256 = '6704f9d502fd5bb0f1978303b23f7c8c7cdcca097a5734465ca37bb1901f9c18'

import base64 as _b64, hashlib as _hh, importlib as _il, json as _jj
import os as _os, sys as _sys, zlib as _zz
from pathlib import Path as _Path
if _sys.platform != "linux" or not _Path("/content").is_dir():
    raise RuntimeError("PR-0006 science runs in Colab GPU only; local main disabled.")
_il.import_module("google.colab")
_raw = _READABLE_RAW
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
