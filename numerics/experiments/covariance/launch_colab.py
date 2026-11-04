# Copyright 2026 Kentaro Sato; Apache-2.0
import base64, importlib.util, json, zlib
from pathlib import Path
source = Path(__file__).parent / 'sources/covariance_consistency_colab_v1.py'
spec = importlib.util.spec_from_file_location('readable_covariance_engine', source)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
for name, value in _READABLE_SETTINGS.items():
    if name not in {"PROFILE", "EXPERIMENTS", "SAVE_TO_DRIVE", "DRIVE_FOLDER", "LOCAL_FOLDER", "MAX_MINUTES", "AUTO_DOWNLOAD", "RESTORE_ZIP", "MASTER_SEED", "BATCH_SIZE", "MEMORY_BATCH_SIZE", "ALLOW_ENVIRONMENT_CHANGE"}:
        raise ValueError('Unexpected benchmark setting: ' + name)
    setattr(engine, name, tuple(value) if name == 'EXPERIMENTS' else value)
if engine.SAVE_TO_DRIVE:
    raise ValueError('This public entry keeps Drive disabled.')
notices = {name: body for name, body in _READABLE_SOURCES.items() if name.startswith('LICENSE')}
engine.LICENSE_PAYLOAD = base64.b64encode(zlib.compress(json.dumps(notices).encode())).decode()
RUN_RESULT = engine.run_suite()
