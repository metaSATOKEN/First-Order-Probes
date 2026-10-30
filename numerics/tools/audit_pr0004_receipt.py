# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Read-only PR-0004 archive receipt audit; never generates scientific records.

Uses trusted local frozen design/check functions, not code from the archive.
Does not extract or modify the original archive. New JSON evidence only.
"""
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'numerics/scripts'))
sys.path.insert(0, str(ROOT / 'numerics/tools'))
import pr0004_spec as design
import pr0004_runner as checks
import probe_backend_check_v1 as core
from build_pr0004_colab import PATHS


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def load_arrays(data):
    """Bound allocations, reject object arrays and malformed inner archives."""
    stream = io.BytesIO(data)
    with zipfile.ZipFile(stream) as z:
        items = z.infolist()
        assert len(items) <= 64 and len({i.filename for i in items}) == len(items)
        assert sum(i.file_size for i in items) <= 32 * 1024**2
        for item in items:
            assert '/' not in item.filename and item.filename.endswith('.npy')
            with z.open(item) as f:
                version = np.lib.format.read_magic(f)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}[version]
                shape, _, dtype = reader(f)
                assert not dtype.hasobject and dtype.kind in 'fiuU' and len(shape) <= 3
                assert all(n >= 0 for n in shape)
                assert math.prod(shape) * dtype.itemsize == item.file_size - f.tell()
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def audit(archive, output):
    start = time.monotonic()
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise FileExistsError('New audit output required')
    output.mkdir(parents=True)
    result = dict(status='IN_PROGRESS', archive=str(archive), scope='receipt/integrity only',
                  simulation_generated=0, manuscript_changed=False)
    def save():
        (output / 'validation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    save()
    try:
        before = archive.stat()
        result['archive_bytes'] = before.st_size
        print('Hashing original ZIP...', flush=True)
        result['archive_sha256'] = file_hash(archive)
        manifest_local = json.loads((ROOT / 'numerics/colab/pr0004_gpu_main_build.json').read_text())
        for name, h in (manifest_local['sources'] | manifest_local['artifacts']).items():
            assert file_hash(ROOT / name) == h, name
        result['local_frozen_hashes'] = len(manifest_local['sources']) + len(manifest_local['artifacts'])
        anchors = re.findall(r'\| `([^`]+)` \| `([0-9a-f]{64})` \|',
                             (ROOT / 'HANDOFF/2026-10-17_session-01.md').read_text())
        for name, h in anchors:
            assert file_hash(ROOT / name) == h, name
        result['handoff_anchors'] = len(anchors)
        ledger = (ROOT / 'audits/ledger_2026-10.md').read_bytes()
        result['ledger_before'] = dict(bytes=len(ledger), sha256=sha(ledger))
        source_expected = {k: file_hash(ROOT / v) for k, v in PATHS.items()}
        expected = {'chunks/' + t['id'] + f'/{s:09d}.npz': (t, s)
                    for t in design.tasks() for s in range(0, t['count'], design.CHUNK)}
        spec = design.specification()
        spec_hash = sha(canonical(spec).encode())
        count, done, preflight, waveforms = 0, {}, {}, set()
        hashes, jsons, validated = {}, {}, set()
        roles, reps = {}, {}
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist(); names = [i.filename for i in infos]
            assert len(names) == len(set(names)) and len(names) <= 50000
            assert sum(i.file_size for i in infos) <= 16 * 1024**3
            assert before.st_size <= 6 * 1024**3
            for i in infos:
                p = PurePosixPath(i.filename)
                assert not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename
                assert str(p) == i.filename and not i.is_dir() and not stat.S_ISLNK(i.external_attr >> 16)
            assert z.getinfo('ARCHIVE_INDEX.json').file_size < 16 * 1024**2
            index = json.loads(z.read('ARCHIVE_INDEX.json'))
            assert index['schema'] == 1 and index['experiment'] == design.EXPERIMENT
            assert index['status'] == 'COMPLETE' and index['cell_id'] == manifest_local['cell_id']
            assert set(names) == set(index['entries']) | {'ARCHIVE_INDEX.json'}
            assert {n for n in names if n.startswith('chunks/') and n.endswith('.npz')} == set(expected)
            result.update(run_id=index['run_id'], cell_id=index['cell_id'], archive_created=index['created'],
                          members=len(names), indexed_files=len(index['entries']),
                          uncompressed_bytes=sum(i.file_size for i in infos),
                          archive_comment=json.loads(z.comment))
            for number, (name, item) in enumerate(index['entries'].items(), 1):
                assert z.getinfo(name).file_size == item['size'], name
                assert item['size'] <= 64 * 1024**2, ('unexpectedly large member', name)
                data = z.read(name)  # also verifies ZIP CRC; one bounded file at a time
                h = sha(data)
                assert h == item['sha256'], name
                hashes[name] = h
                if name.endswith('.json'):
                    obj = json.loads(data)
                    if not name.endswith('.timing.json'):
                        jsons[name] = obj
                    if name.endswith('/done.json') and name.startswith('analysis/repeats/'):
                        rep = int(name.split('/')[2]); assert rep not in done
                        assert obj['replicate'] == rep and obj['specification_sha256'] == spec_hash
                        done[rep] = obj
                if name in expected:
                    task, offset = expected[name]
                    a = load_arrays(data); assert set(a) == {'values', 'metadata'}
                    values = a['values']; meta = json.loads(str(a['metadata']))
                    n = min(design.CHUNK, task['count'] - offset)
                    assert values.shape == (n, 17) and values.dtype == np.dtype('float64')
                    assert np.isfinite(values).all()
                    assert meta['task'] == task and meta['start'] == offset and meta['count'] == n
                    assert meta['specification_sha256'] == spec_hash and meta['values_sha256'] == sha(values.tobytes())
                    checks.verify_values(values, task['case'])  # algebra only; no RNG/trajectory generation
                    count += n; validated.add(name)
                    roles[task['role']] = roles.get(task['role'], 0) + n
                    reps[task['replicate']] = reps.get(task['replicate'], 0) + n
                elif name.startswith('preflight/') and name.endswith('.npz'):
                    a = load_arrays(data)
                    pairs = [('cpu', 'gpu')] + [(k, 'gpu_' + k[4:]) for k in a if k.startswith('cpu_')]
                    diffs = [core.compare(a[x], a[y]) for x, y in pairs]
                    assert all(d['passed'] for d in diffs), name
                    preflight[name] = dict(pairs=len(pairs), checks=diffs)
                elif name.startswith('waveforms/') and name.endswith('.npz'):
                    waveforms.add(name)
                if number % 3000 == 0 or number == len(index['entries']):
                    print(f'Validated {number:,}/{len(index["entries"]):,} files; {count:,} records', flush=True)
            assert jsons['manifest.json']['sources'] == source_expected
            for name, h in source_expected.items():
                assert hashes['sources/' + name] == h
            assert jsons['manifest.json']['run_id'] == index['run_id']
            assert canonical(jsons['specification.json']) == canonical(spec)
            assert count == 33499400 and len(validated) == 10400 and set(done) == set(range(50))
            assert set(waveforms) == {'waveforms/r00_' + c + '.npz' for c in design.conditions()}
            assert len(preflight) == 50
            for rep, d in done.items():
                assert '/' not in d['attempt'] and d['attempt'].startswith('attempt_')
                prefix = f'analysis/repeats/{rep:02d}/' + d['attempt'] + '/'
                assert {n[len(prefix):] for n in names if n.startswith(prefix)} == set(d['files'])
                for n, h in d['files'].items():
                    assert hashes[prefix + n] == h
                r = jsons[prefix + 'results.json']
                assert r['replicate'] == rep and len(r['rates']) == 684 and len(r['paired']) == 570
                assert len([n for n in d['files'] if n.startswith('decisions/')]) == 114
            summary = jsons['summary.json']
            for k, v in dict(status='COMPLETE', records=33499400, chunks=10400, repeats=50,
                             primary_comparisons=12, fpr_series=140, scientific_exclusions=0).items():
                assert summary[k] == v
            science_summary = jsons['analysis/summary.json']
            assert len(science_summary['primary12']) == 12 and len(science_summary['fpr140']) == 140
            assert len(science_summary['rates']) == 684
            errors = [n for n in names if n.endswith('/error.json')]
            sessions = {n: o for n, o in jsons.items() if n.startswith('sessions/') and n.endswith('/session.json')}
            launches = {n: o for n, o in jsons.items() if n.startswith('launches/') and n.endswith('/launch.json')}
            environments = {n: o for n, o in jsons.items() if n.startswith('sessions/') and n.endswith('/environment.json')}
            assert len(sessions) == len(launches) == len(environments) == 1
            assert all(o['status'] == 'COMPLETE' for o in sessions.values())
            assert all(o['status'] == 'COMPLETE' and o['returncode'] == 0 for o in launches.values())
            assert not errors and summary['previous_error_records'] == 0
            assert summary['active_seconds'] < design.LIMITS['active_seconds']
            assert summary['rss_high_water_bytes'] <= design.LIMITS['rss_bytes']
            assert summary['gpu_reserved_bytes'] <= design.LIMITS['gpu_reserved_bytes']
            assert all(e['packages'][k] == v for e in environments.values() for k, v in core.PINS.items())
            result.update(status='PASS_RECEIPT', records=count, chunks=len(validated), tasks=len(design.tasks()),
                          records_by_role=roles, records_by_replicate=reps, repeats=len(done),
                          analysis_files_in_done=sum(len(d['files']) for d in done.values()),
                          decision_files=114 * len(done), source_files=len(source_expected),
                          saved_summary=summary, active_time=jsons['active_time.json'],
                          sessions=sessions, launches=launches, environments=environments,
                          error_files=errors, cpu_gpu_saved_array_comparisons=preflight,
                          waveform_files=len(waveforms),
                          analysis_summary_shapes={k: len(v) for k, v in science_summary.items() if isinstance(v, list)},
                          missing_timing_chunks=jsons['analysis/costs.json']['missing_chunk_save_timings'],
                          full_statistical_recomputation=False,
                          limitations=['All file hashes/CRC, every saved chunk identity/count/algebra checked.',
                                       'Analysis completion-file hashes checked; detector refits, decisions, confidence intervals not independently recomputed.',
                                       'No fresh trajectories, no GPU execution, no manuscripts changed.',
                                       'Download delay cause unconfirmed; transferred file is intact.',
                                       'Archive remains at original Downloads path; separate backup not created.',
                                       'Same agent receipt audit, not independent scientific replication.'])
        after = archive.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        result['audit_seconds'] = time.monotonic() - start
        result['auditor_sha256'] = file_hash(__file__)
        save()
        print(json.dumps({k: result[k] for k in ('status', 'archive_bytes', 'archive_sha256', 'records', 'chunks', 'repeats', 'audit_seconds')}, indent=2), flush=True)
    except BaseException as exc:
        result.update(status='AUDIT_FAILED', error=type(exc).__name__ + ': ' + str(exc), audit_seconds=time.monotonic()-start)
        save()
        raise
    return result


if __name__ == '__main__':
    audit(sys.argv[1], sys.argv[2])
