# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Verify the readable FIX1 bundle, or explicitly launch it in a Colab kernel.

Default: standard-library-only, read-only integrity check. No installation,
simulation, upload, download, or Drive access. See README.md before --run.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

BUNDLE_DIR = Path(__file__).resolve().parent
CELL_ID = 'd41a11833edd6d21'
PAYLOAD_SHA256 = '4045cf2857875c9d0978a19961e3daea38a382303418f2267572023eb1c3789f'


def load_sources(bundle_dir=BUNDLE_DIR):
    """Read and authenticate all logical members, preserving original names."""
    root = Path(bundle_dir).resolve()
    manifest_path = root / 'bundle_manifest.json'
    if manifest_path.is_symlink():
        raise ValueError('The bundle manifest must be an ordinary file.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['schema'] != 1 or manifest['cell_id'] != CELL_ID
            or manifest['payload_sha256'] != PAYLOAD_SHA256):
        raise ValueError('This entry requires the frozen PR-0012 FIX1 bundle.')
    sources = {}
    paths = set()
    for name, entry in manifest['members'].items():
        if Path(name).name != name or not name or '\\' in name:
            raise ValueError('Invalid logical member name.')
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid member path.')
        path = root / relative
        if any(p.is_symlink() for p in (path, *path.parents) if p != root and p.is_relative_to(root)):
            raise ValueError('Symlinked members are not accepted.')
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or out-of-bundle member: ' + name)
        if entry['path'] in paths:
            raise ValueError('Duplicate physical member path.')
        paths.add(entry['path'])
        data = path.read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Member integrity mismatch: ' + name)
        sources[name] = data.decode('utf-8')
    raw = json.dumps(dict(schema=1, sources=sources), sort_keys=True, ensure_ascii=False).encode('utf-8')
    if hashlib.sha256(raw).hexdigest() != PAYLOAD_SHA256:
        raise ValueError('The logical payload differs from the completed FIX1 capsule.')
    expected_python = {name for name in sources if name.endswith('.py')}
    if len(sources) != 28 or len(expected_python) != 20:
        raise ValueError('Unexpected bundle membership.')
    actual_python = {p.name for p in (root / 'sources').glob('*.py')}
    if actual_python != expected_python or any(
            manifest['members'][name]['path'] != 'sources/' + name for name in expected_python):
        raise ValueError('Extra, missing, or misplaced executable sources.')
    return sources


def check_import_origins(sources, source_dir):
    """Reject stale same-name modules; never silently mix historical versions."""
    for name in sources:
        if not name.endswith('.py'):
            continue
        module = sys.modules.get(name[:-3])
        if module is None:
            continue
        origin = getattr(module, '__file__', None)
        if origin is None or Path(origin).resolve() != source_dir / name:
            raise RuntimeError('Conflicting loaded module: ' + name
                               + '. Use a clean Colab kernel for this bundle.')


def require_colab():
    if sys.platform != 'linux' or not Path('/content').is_dir():
        raise RuntimeError('Scientific execution is Colab GPU only; use --check elsewhere.')
    importlib.import_module('google.colab')


def launch(sources, restore_zips='', auto_download=True, bundle_dir=BUNDLE_DIR):
    """Delegate unchanged science and checkpoint handling to the frozen launcher."""
    require_colab()
    source_dir = (Path(bundle_dir) / 'sources').resolve()
    check_import_origins(sources, source_dir)
    if str(source_dir) in sys.path:
        sys.path.remove(str(source_dir))
    sys.path.insert(0, str(source_dir))
    importlib.invalidate_caches()
    launcher = importlib.import_module('pr0012_launcher')
    check_import_origins(sources, source_dir)
    return launcher.run(sources, CELL_ID, source_dir, restore_zips, auto_download)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='read-only integrity check (default)')
    mode.add_argument('--run', action='store_true', help='explicit full/resumed Colab GPU execution')
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument('--upload', action='store_true', help='select matching restore ZIPs after runtime loss')
    restore.add_argument('--restore-zip', action='append', default=[], help='already uploaded ZIP path; repeatable')
    parser.add_argument('--no-download', action='store_true', help='retain runtime archives without browser delivery')
    args = parser.parse_args(argv)
    if not args.run and (args.upload or args.restore_zip or args.no_download):
        parser.error('Restore/download options require explicit --run.')
    sources = load_sources()
    if not args.run:
        result = dict(status='PASS', mode='read_only_check', members=len(sources), python_files=20,
                      cell_id=CELL_ID, payload_sha256=PAYLOAD_SHA256,
                      simulation_started=False, environment_installed=False)
    else:
        restore_zips = 'UPLOAD' if args.upload else (args.restore_zip or '')
        result = launch(sources, restore_zips, not args.no_download)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
