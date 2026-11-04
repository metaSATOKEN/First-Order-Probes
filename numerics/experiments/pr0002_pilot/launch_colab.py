# Copyright 2026 Kentaro Sato -- Apache-2.0 (code); CC BY 4.0 (paper/card)
MODE = _READABLE_SETTINGS['MODE']
RESTORE_ZIP = _READABLE_SETTINGS['RESTORE_ZIP']
PILOT_ZIP = _READABLE_SETTINGS['PILOT_ZIP']
CONFIRM_APPROVAL = _READABLE_SETTINGS['CONFIRM_APPROVAL']
AUTO_DOWNLOAD = _READABLE_SETTINGS['AUTO_DOWNLOAD']
# No Drive connection. Normal CPU runtime is sufficient. Do not run two copies.
# For >1 GB restores, Colab Files sidebar upload + absolute path avoids files.upload's RAM copy.
# Safe pause after 45 minutes; keep the downloaded PARTIAL ZIP on your computer.
# After a restore, set RESTORE_ZIP = "" for the next rerun in the SAME runtime.

PAYLOAD_SHA256 = '3f8eaf10ccbc0a961d32c6173ac39173ced377e9a128fdc90f9b1445e24a90bb'
CELL_ID = '568b5189bb5d58d5b1f4ba6f7497ebf234783d01e3bd3ebc90c543ca7ebcf80c'
import base64, hashlib, importlib, json, os, sys, zipfile, zlib
from pathlib import Path
if "COLAB_RELEASE_TAG" not in os.environ:
    try:
        import google.colab
    except ImportError as exc:
        raise RuntimeError("Run this cell in Colab. Local pilot/main execution is disabled.") from exc
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
raw = _READABLE_RAW
if hashlib.sha256(raw).hexdigest() != PAYLOAD_SHA256:
    raise ValueError("Embedded source payload is corrupted")
bundle = json.loads(raw)
package = Path("/content") / ("pr0002_package_" + PAYLOAD_SHA256[:12])
package.mkdir(exist_ok=True)
for name, content in bundle.items():
    target = package / name
    if target.exists() and target.read_text() != content:
        raise ValueError("Existing package differs; use a clean runtime")
    target.write_text(content)
sys.path.insert(0, str(package))
for name in ("pr0002_checks", "fusion_confirmation_pr0002_v1", "covariance_consistency_colab_v1"):
    sys.modules.pop(name, None)
import fusion_confirmation_pr0002_v1 as runner
def select_zip(value):
    if value != "UPLOAD":
        return value or None
    from google.colab import files
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Select exactly one ZIP")
    filename = next(iter(uploaded))
    del uploaded
    return str(Path(filename).resolve())
restore_zip = select_zip(RESTORE_ZIP)
pilot_receipt = None
if MODE == "confirm":
    if CONFIRM_APPROVAL != "RUN PR-0002 CONFIRM":
        raise PermissionError("Main run is not authorized. Leave MODE='pilot' for now.")
    if restore_zip:
        with zipfile.ZipFile(restore_zip) as z:
            pilot_receipt = json.loads(z.read("manifest.json"))["pilot_receipt"]
    else:
        pilot_zip = select_zip(PILOT_ZIP)
        if pilot_zip:
            with zipfile.ZipFile(pilot_zip) as z:
                pilot_receipt = json.loads(z.read("pilot_receipt.json"))
        else:
            paths = list(Path("/content/pr0002_runs").glob("*_pilot_*/pilot_receipt.json"))
            matching = [json.loads(p.read_text()) for p in paths
                        if json.loads(p.read_text()).get("cell_payload_sha256") == CELL_ID]
            if len(matching) != 1:
                raise ValueError("Choose the matching completed pilot ZIP via PILOT_ZIP")
            pilot_receipt = matching[0]
RUN_RESULT = runner.run(MODE, "/content/pr0002_runs", bundle, CELL_ID,
    approval=CONFIRM_APPROVAL, pilot_receipt=pilot_receipt,
    restore_zip=restore_zip, auto_download=AUTO_DOWNLOAD)
if MODE == "pilot" and RUN_RESULT["status"] == "PILOT_COMPLETE":
    print("Pilot only completed. Share this ZIP for budget/acceptance review. No confirmation run was started.")
