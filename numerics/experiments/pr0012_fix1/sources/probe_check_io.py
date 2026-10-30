# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only checkpoint/ZIP transport. Never executes restored source files."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import uuid
import zipfile

LIMIT = 1024**3


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write(path, obj):
    atomic(path, (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2,
                            allow_nan=False) + '\n').encode())


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError('Another copy is running. Do not run two cells.') from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def evidence(root):
    return [p for p in sorted(Path(root).rglob('*')) if p.is_file()
            and not p.name.endswith(('.tmp', '.lock')) and '__pycache__' not in p.parts]


def safe_name(name):
    p = PurePosixPath(name)
    if (not name or not p.parts or p.is_absolute() or '..' in p.parts or '\\' in name
            or str(p) != name):
        raise ValueError('Unsafe archive path: ' + name)
    return p


def snapshot(root, destination, status):
    root, destination = Path(root), Path(destination)
    manifest = read(root / 'manifest.json')
    paths = evidence(root)
    entries = {str(p.relative_to(root)): {'size': p.stat().st_size, 'sha256': digest(p)}
               for p in paths}
    if sum(x['size'] for x in entries.values()) > LIMIT:
        raise RuntimeError('Evidence exceeds the 1 GiB archive limit; no files dropped')
    index = dict(schema=1, experiment='probe_cpu_gpu_check_v1', status=status,
                 cell_id=manifest['cell_id'], run_id=manifest['run_id'],
                 created=stamp(), entries=entries)
    destination.mkdir(parents=True, exist_ok=True)
    name = 'probe_cpu_gpu_check_v1_' + manifest['run_id'] + '_' + status + '_' + uuid.uuid4().hex[:8] + '.zip'
    target = destination / name
    tmp = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in paths:
            z.write(p, str(p.relative_to(root)))
        z.writestr('ARCHIVE_INDEX.json', json.dumps(index, sort_keys=True))
    if tmp.stat().st_size > LIMIT:
        raise RuntimeError('ZIP size exceeds fixed limit')
    os.replace(tmp, target)
    return target


def inspect_zip(path, cell_id):
    if Path(path).stat().st_size > LIMIT:
        raise ValueError('ZIP too large')
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)) or len(names) > 10000:
            raise ValueError('Duplicate/excess archive entries')
        if sum(i.file_size for i in infos) > LIMIT:
            raise ValueError('Expanded archive too large')
        for info in infos:
            safe_name(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Directory/symlink member rejected')
        index = json.loads(z.read('ARCHIVE_INDEX.json'))
        if index.get('schema') != 1 or index.get('experiment') != 'probe_cpu_gpu_check_v1' or index.get('cell_id') != cell_id:
            raise ValueError('Wrong experiment/code version')
        if set(names) != set(index['entries']) | {'ARCHIVE_INDEX.json'}:
            raise ValueError('Archive inventory mismatch')
        for name, item in index['entries'].items():
            data = z.read(name)
            if len(data) != item['size'] or sha(data) != item['sha256']:
                raise ValueError('Archive checksum mismatch: ' + name)
        m = json.loads(z.read('manifest.json'))
        if m['run_id'] != index['run_id'] or m['cell_id'] != cell_id:
            raise ValueError('Archive manifest identity mismatch')
        return index


def choose_archive(paths, cell_id):
    """Only a single run lineage is auto-selected; newest snapshot in it wins."""
    if not paths:
        return None
    candidates = [(p, inspect_zip(p, cell_id)) for p in paths]
    if len({i['run_id'] for _, i in candidates}) != 1:
        raise ValueError('Multiple run IDs: specify the desired restore ZIP explicitly')
    return max(candidates, key=lambda pi: pi[1]['created'])[0]


def restore(path, root, cell_id):
    root = Path(root)
    if root.exists():
        raise FileExistsError('Refuse to overwrite an existing run')
    index = inspect_zip(path, cell_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent, prefix='restore-') as td:
        stage = Path(td) / 'run'
        stage.mkdir()
        with zipfile.ZipFile(path) as z:
            for name in index['entries']:
                p = stage / safe_name(name)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(z.read(name))
        os.replace(stage, root)


def new_run(root, cell_id, sources):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'manifest.json', dict(experiment='probe_cpu_gpu_check_v1', schema=1,
          run_id=uuid.uuid4().hex[:12], cell_id=cell_id, created=stamp(),
          sources={name: sha(body.encode()) for name, body in sources.items()}))
    for name, body in sources.items():
        atomic(root / 'sources' / safe_name(name), body.encode())


def check_run(root, cell_id, sources):
    m = read(Path(root) / 'manifest.json')
    expected = {name: sha(body.encode()) for name, body in sources.items()}
    if m['cell_id'] != cell_id or m['sources'] != expected:
        raise ValueError('Existing run differs from this source capsule')
    for name, value in expected.items():
        if digest(Path(root) / 'sources' / name) != value:
            raise ValueError('Saved source mismatch: ' + name)
    return m
