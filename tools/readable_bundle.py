# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Authenticate plain-text experiment bundles without importing scientific code."""
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import sys


def check(directory, expected_sha256):
    root = Path(directory).resolve()
    manifest_path = root / 'bundle_manifest.json'
    if manifest_path.is_symlink():
        raise ValueError('Symlinked manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['schema'] != 1 or manifest['payload_sha256'] != expected_sha256:
        raise ValueError('Unexpected frozen bundle identity')
    sources, physical = {}, set()
    for name, info in manifest['members'].items():
        if not name or Path(name).name != name or '\\' in name:
            raise ValueError('Unsafe logical name')
        relative = Path(info['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in {'sources','metadata'}:
            raise ValueError('Unsafe physical path')
        path = root / relative
        if (not path.is_file() or not path.resolve().is_relative_to(root)
                or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))):
            raise ValueError('Missing or symlinked member: ' + name)
        if info['path'] in physical:
            raise ValueError('Duplicate physical member')
        physical.add(info['path'])
        data = path.read_bytes()
        if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
            raise ValueError('Member integrity mismatch: ' + name)
        sources[name] = data.decode('utf-8')
    actual = {str(p.relative_to(root)) for sub in ('sources','metadata')
              for p in (root/sub).rglob('*') if p.is_file()}
    if actual != physical:
        raise ValueError('Extra or missing bundle member')
    form = manifest['serialization']
    if form not in {'direct','schema_sources'} or type(manifest['compact']) is not bool:
        raise ValueError('Unknown serialization')
    value = dict(schema=1, sources=sources) if form == 'schema_sources' else sources
    kwargs = {'separators': (',', ':')} if manifest['compact'] else {}
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, **kwargs).encode('utf-8')
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Frozen logical source identity mismatch')
    return sources, raw


def execute(directory, sources, raw):
    """Called only by explicit --run, inside the Colab notebook kernel."""
    if sys.platform != 'linux' or not Path('/content').is_dir():
        raise RuntimeError('Execution is supported in Colab only; use --check elsewhere.')
    importlib.import_module('google.colab')
    root = Path(directory).resolve()
    settings = json.loads((root/'settings.json').read_text(encoding='utf-8'))
    if settings.get('SAVE_TO_DRIVE', False):
        raise ValueError('This entry keeps Google Drive disabled.')
    result = runpy.run_path(str(root/'launch_colab.py'), run_name='__main__', init_globals={
        '_READABLE_SOURCES': sources, '_READABLE_RAW': raw, '_READABLE_SETTINGS': settings})
    return result.get('_result', result.get('RUN_RESULT', result.get('LEAN_RESULT_ZIP')))
