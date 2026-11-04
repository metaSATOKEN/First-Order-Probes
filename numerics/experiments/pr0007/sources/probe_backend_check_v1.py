# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""PR-0003 backend pilot. Import is inert; main requires Colab/Linux/CUDA.

CPU and GPU share explicit innovations, NOT separate random generators.
Only the first-order recurrence is custom CUDA; no dense covariance per record.
"""
from functools import lru_cache
import csv
import gc
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback
import uuid
import zipfile

import numpy as np
import scipy
from scipy import linalg, signal as ss
import probe_check_io as disk

VERSION = 'probe-backend-check-1.0.0'
SEED = 2026101706
TOL = {'atol': 2e-12, 'rtol': 2e-12}
BATCHES = (64, 512, 4096)
ROUNDS, RECORDS = 3, 4096
COLUMNS = [s + '_' + q for s in ('none', 'diag', 'full') for q in ('vi', 'vj', 'c')] + [
    'R_none', 'R_diag', 'R_full', 'CR', 'reversal_moment_error', 'P_i', 'P_j', 'P_ij']
PINS = {'numpy': '2.1.3', 'scipy': '1.16.3', 'cupy-cuda12x': '14.2.0'}
BENCH = ('B_S', 'B_M', 'B_L', 'B_R10', 'K2_L', 'K200_L', 'PRIVATE_M', 'DELAY_PLUS')
CUDA_SOURCE = r'''
extern "C" __global__ void first_order(const double* x, const double* initial,
    double* y, const int B, const int T, const double a, const double b) {
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= B) return;
    double previous = initial[j];
    for (int t = 0; t < T; ++t) {
        const long long k = (long long)t * B + j;
        previous = b * x[k] + a * previous;
        y[k] = previous;
    }
}
'''


def conditions():
    base = dict(dt=.002, tau_i=.04, tau_j=.01, kappa=20., variance=1.,
                T=2.048, fault='none', strength=0.)
    out = {k: dict(base, T=t, tau_i=ti) for k, t, ti in (
        ('B_S', .512, .04), ('B_M', 2.048, .04), ('B_L', 8.192, .04), ('B_R10', 2.048, .10))}
    for tag, changes in [('K2', {'kappa': 2.}), ('K200', {'kappa': 200.}),
                         ('V05', {'variance': .5}), ('V15', {'variance': 1.5})]:
        for win in ('M', 'L'):
            out[tag + '_' + win] = dict(out['B_' + win], **changes)
    for name, group, fault, strength in (
        ('GAIN_MINUS', 'B_L', 'gain', -.05), ('GAIN_PLUS', 'B_L', 'gain', .05),
        ('GAIN_WEAK_MINUS', 'B_L', 'gain', -.02), ('GAIN_WEAK_PLUS', 'B_L', 'gain', .02),
        ('PRIVATE_M', 'B_M', 'private', .02), ('PRIVATE_WEAK', 'B_M', 'private', .005),
        ('DELAY_MINUS', 'B_M', 'delay', -2), ('DELAY_PLUS', 'B_M', 'delay', 2),
        ('REVERSE', 'B_M', 'time_reverse', 0), ('PRIVATE_SLOW', 'K2_L', 'private', .02)):
        out[name] = dict(out[group], fault=fault, strength=strength)
    return out


def dimensions(p):
    n = round(p['T'] / p['dt'])
    w = math.ceil(20 * max(p['tau_i'], p['tau_j']) / p['dt'])
    guard = abs(int(p['strength'])) if p['fault'] == 'delay' else 0
    return n, w, guard


@lru_cache(None)
def stationary(kappa, ti, tj, dt):
    phi, ai, aj = np.exp(-dt * np.array([kappa, 1 / ti, 1 / tj]))
    A = np.array([[phi, 0, 0], [(1-ai)*phi, ai, 0], [(1-aj)*phi, 0, aj]])
    b = np.sqrt(1-phi*phi) * np.array([1., 1-ai, 1-aj])
    P = linalg.solve_discrete_lyapunov(A, np.outer(b, b))
    v, U = np.linalg.eigh(P)
    if v.min() < -1e-12:
        raise ValueError('Non-PSD stationary covariance')
    return P, U @ np.diag(np.sqrt(np.maximum(v, 0)))


def coefficients(p):
    phi, ai, aj = np.exp(-p['dt'] * np.array([p['kappa'], 1/p['tau_i'], 1/p['tau_j']]))
    lam = (1+ai)*(1-aj)/(2*(1-ai*aj))
    nu = (1+aj)*(1-ai)/(2*(1-ai*aj))
    return phi, ai, aj, lam, nu


def make_inputs(p, case, repeat, start, count):
    """64-record RNG blocks make input identity independent of compute batch."""
    n, w, g = dimensions(p)
    L, C = n+2*w, n+w
    _, factor = stationary(p['kappa'], p['tau_i'], p['tau_j'], p['dt'])
    baseline, _ = stationary(20., p['tau_i'], p['tau_j'], p['dt'])
    noise = np.sqrt(.1*np.diag(baseline)[1:])
    parts = {}
    case_key = int.from_bytes(bytes.fromhex(disk.sha(case.encode()))[:4], 'little')
    first, last = start//64, (start+count-1)//64
    for block in range(first, last+1):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, case_key, repeat, block])))
        z = {'initial': np.sqrt(p['variance']) * (rng.normal(size=(64, 3)) @ factor.T),
             'source': rng.normal(size=(64, L+2*g))}
        if p['fault'] == 'private':
            z['private_initial'] = np.sqrt(p['variance']) * (rng.normal(size=(64, 3)) @ factor.T)
            z['private_source'] = rng.normal(size=(64, L))
        z.update(ni=noise[0]*rng.normal(size=(64, L)), nj=noise[1]*rng.normal(size=(64, L)),
                 ci=noise[0]*rng.normal(size=(64, C)), cj=noise[1]*rng.normal(size=(64, C)))
        lo, hi = max(start-block*64, 0), min(start+count-block*64, 64)
        for key, a in z.items():
            parts.setdefault(key, []).append(a[lo:hi])
    return {k: np.ascontiguousarray(np.concatenate(v, axis=0), dtype=np.float64) for k, v in parts.items()}


def cpu_filter(x, a, b, initial):
    return ss.lfilter([b], [1., -a], np.ascontiguousarray(x), axis=1,
                      zi=(a*initial)[:, None])[0]


def loop_filter(x, a, b, initial):
    """Tiny fixed-input oracle; not a performance baseline."""
    y = np.empty_like(x)
    old = initial.copy()
    for i in range(x.shape[1]):
        old = b*x[:, i] + a*old
        y[:, i] = old
    return y


class GPU:
    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.kernel = cp.RawKernel(CUDA_SOURCE, 'first_order', options=('--std=c++11', '--fmad=false'))

    def filt(self, x, a, b, initial):
        cp = self.cp
        if x.dtype != cp.float64 or initial.dtype != cp.float64:
            raise TypeError('GPU input must be float64')
        B, T = x.shape
        xt = cp.ascontiguousarray(x.T)
        out = cp.empty_like(xt)
        initial = cp.ascontiguousarray(initial)
        self.kernel(((B+127)//128,), (128,), (xt, initial, out, np.int32(B), np.int32(T), np.float64(a), np.float64(b)))
        return out.T


def calculate(p, z, xp=np, filt=cpu_filter, traces=False):
    n, w, g = dimensions(p)
    phi, ai, aj, lam, nu = coefficients(p)
    gain = math.sqrt(p['variance']*(1-phi*phi))
    x = filt(z['source'], phi, gain, z['initial'][:, 0])
    yi = filt(x, ai, 1-ai, z['initial'][:, 1])
    yj = filt(x, aj, 1-aj, z['initial'][:, 2])
    L = n+2*w
    delay = int(p['strength']) if p['fault']=='delay' else 0
    yi, yj = yi[:, g:g+L], yj[:, g-delay:g-delay+L]
    if p['fault']=='gain':
        yj = (1+p['strength'])*yj
    if p['fault']=='private':
        extra = filt(z['private_source'], phi, gain, z['private_initial'][:, 0])
        extra = filt(extra, aj, 1-aj, z['private_initial'][:, 2])
        yj = yj + math.sqrt(p['strength'])*extra
    yi, yj = yi+z['ni'], yj+z['nj']
    def moments(a, b):
        a, b = a[:, w:w+n], b[:, w:w+n]
        return xp.column_stack((xp.mean(a*a, axis=1), xp.mean(b*b, axis=1), xp.mean(a*b, axis=1)))
    original = moments(yi, yj) if p['fault']=='time_reverse' else None
    if original is not None:
        yi, yj = yi[:, ::-1], yj[:, ::-1]
    O, N = moments(yi, yj), moments(z['ci'], z['cj'])
    full = O-N
    diag = O-N*xp.asarray([1., 1., 0.], dtype=xp.float64)
    def residual(a):
        return a[:, 2]-lam*a[:, 0]-nu*a[:, 1]
    zero = xp.zeros(len(yi), dtype=xp.float64)
    A, B = filt(yi, aj, 1-aj, zero), filt(yj, ai, 1-ai, zero)
    Ac, Bc = filt(z['ci'], aj, 1-aj, zero), filt(z['cj'], ai, 1-ai, zero)
    P = moments(A, B)-moments(Ac, Bc)
    # Direct CR energy is independent of the moment expansion, and can be signed.
    CR = xp.mean((A[:, w:w+n]-B[:, w:w+n])**2, axis=1)-xp.mean((Ac[:, w:w+n]-Bc[:, w:w+n])**2, axis=1)
    rev = xp.max(xp.abs(O-original), axis=1) if original is not None else zero
    result = xp.column_stack((O, diag, full, residual(O), residual(diag), residual(full), CR, rev, P))
    trace = {'yi': yi, 'yj': yj, 'A': A, 'B': B, 'Ac': Ac, 'Bc': Bc} if traces else {}
    return result, trace


def compare(a, b):
    if a.shape != b.shape or a.dtype != np.float64 or b.dtype != np.float64:
        return dict(passed=False, reason='shape/dtype mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(passed=False, reason='nonfinite')
    diff = np.abs(a-b)
    scaled = diff/(TOL['atol']+TOL['rtol']*np.abs(a))
    return dict(passed=bool((scaled<=1).all()), max_abs=float(diff.max()),
                max_scaled=float(scaled.max()), exceed=int((scaled>1).sum()),
                max_abs_by_column=np.max(diff, axis=0).tolist() if a.ndim==2 and a.shape[1]==17 else None,
                max_scaled_by_column=np.max(scaled, axis=0).tolist() if a.ndim==2 and a.shape[1]==17 else None,
                exceed_by_column=np.sum(scaled>1, axis=0).tolist() if a.ndim==2 and a.shape[1]==17 else None)


def identity(a):
    return compare(a[:, 12:13], (a[:, 14]+a[:, 15]-2*a[:, 16])[:, None])


def tasks():
    out = [dict(id='check-'+c, phase='check', case=c, repeat=100, batch=17, count=17) for c in conditions()]
    for r in range(ROUNDS):
        for c in BENCH:
            # Rotate batch order, separately from alternating CPU/GPU order.
            for b in BATCHES[r:]+BATCHES[:r]:
                out.append(dict(id=f'time-{c}-r{r}-b{b}', phase='time', case=c, repeat=r, batch=b, count=RECORDS))
    return out


def specification():
    return dict(version=VERSION, seed=SEED, tolerance=TOL, conditions=conditions(),
                columns=COLUMNS, tasks=tasks(), pins=PINS, random_block=64,
                active_limit_seconds=3600, task_limit_seconds=600,
                backend='float64 CUDA recurrence; CPU PCG64 inputs',
                scientific_main=False, lf_detector_implemented=False)


def environment(gpu):
    import importlib.metadata as md
    cp = gpu.cp
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    cpu = next((x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), 'unknown')
    config = io.StringIO()
    from contextlib import redirect_stdout
    with redirect_stdout(config):
        cp.show_config()
    boot = Path('/proc/sys/kernel/random/boot_id')
    return dict(python=sys.version, platform=platform.platform(), cpu=cpu,
                cpu_count=os.cpu_count(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                runtime_boot_id=boot.read_text().strip() if boot.exists() else 'unresolved-'+uuid.uuid4().hex,
                gpu=name, gpu_memory=int(props['totalGlobalMem']),
                driver=cp.cuda.runtime.driverGetVersion(), runtime=cp.cuda.runtime.runtimeGetVersion(),
                packages={d.metadata['Name']: d.version for d in md.distributions()},
                cupy_config=config.getvalue(), threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})


def save_values(path, values, trace=None):
    b = io.BytesIO()
    np.savez(b, values=values, **(trace or {}))
    disk.atomic(path, b.getvalue())


def open_values(path):
    """Bound nested NPZ expansion/header sizes before NumPy allocates arrays."""
    allowed = {'values.npy','yi.npy','yj.npy','A.npy','B.npy','Ac.npy','Bc.npy'}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [x.filename for x in infos]
        if len(names)!=len(set(names)) or not set(names)<=allowed or 'values.npy' not in names:
            raise ValueError('Invalid NPZ inventory')
        if sum(x.file_size for x in infos)>128*1024**2:
            raise ValueError('Oversized NPZ evidence')
        for item in infos:
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                readers = {(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}
                if version not in readers:
                    raise ValueError('Unsupported NPY header')
                shape, _, dtype = readers[version](stream)
                if dtype!=np.dtype('float64') or len(shape)!=2 or min(shape)<=0:
                    raise ValueError('Invalid evidence shape/type')
                if item.file_size-stream.tell()!=math.prod(shape)*8:
                    raise ValueError('NPY header/payload size mismatch')
    return np.load(path,allow_pickle=False)


def backend_task(task, gpu, backend, folder):
    p = conditions()[task['case']]
    total = dict(input_seconds=0., h2d_seconds=0., compute_seconds=0., gpu_event_seconds=0., d2h_seconds=0.)
    values, trace = [], {}
    begin = time.perf_counter()
    for start in range(0, task['count'], task['batch']):
        t = time.perf_counter()
        z = make_inputs(p, task['case'], task['repeat'], start, min(task['batch'], task['count']-start))
        total['input_seconds'] += time.perf_counter()-t
        if backend=='cpu':
            t = time.perf_counter()
            out, tr = calculate(p, z, traces=task['phase']=='check')
            total['compute_seconds'] += time.perf_counter()-t
        else:
            cp = gpu.cp
            t = time.perf_counter()
            device = {k: cp.asarray(v) for k,v in z.items()}
            cp.cuda.get_current_stream().synchronize()
            total['h2d_seconds'] += time.perf_counter()-t
            a, b = cp.cuda.Event(), cp.cuda.Event()
            t = time.perf_counter(); a.record()
            out, tr = calculate(p, device, cp, gpu.filt, traces=task['phase']=='check')
            b.record(); b.synchronize()
            total['compute_seconds'] += time.perf_counter()-t
            total['gpu_event_seconds'] += cp.cuda.get_elapsed_time(a,b)/1000
            t = time.perf_counter()
            out = cp.asnumpy(out)
            tr = {k: cp.asnumpy(v) for k,v in tr.items()}
            total['d2h_seconds'] += time.perf_counter()-t
            del device
        values.append(out)
        trace.update(tr)
        del z, out, tr
    out = np.concatenate(values)
    before_save = time.perf_counter()
    path = folder/(backend+'.npz')
    save_values(path, out, trace)
    total['save_seconds'] = time.perf_counter()-before_save
    total['end_to_end_seconds'] = time.perf_counter()-begin
    total['records_per_second'] = task['count']/total['end_to_end_seconds']
    total['rss_high_water_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    if backend=='gpu':
        total['gpu_pool_reserved_bytes_after'] = int(gpu.cp.get_default_memory_pool().total_bytes())
        total['gpu_free_bytes_after'] = int(gpu.cp.cuda.runtime.memGetInfo()[0])
    return total


def verify_pair(folder):
    with open_values(folder/'cpu.npz') as ca, open_values(folder/'gpu.npz') as ga:
        if set(ca.files) != set(ga.files):
            raise ValueError('CPU/GPU evidence keys differ')
        comparisons = {k: compare(ca[k], ga[k]) for k in ca.files}
        identities = {name: identity(data['values']) for name,data in [('cpu',ca),('gpu',ga)]}
    return dict(passed=all(x['passed'] for x in list(comparisons.values())+list(identities.values())),
                arrays=comparisons, cr_identity=identities)


def load_done(root, task):
    path = root/'tasks'/task['id']/'done.json'
    if not path.exists():
        return None
    row = disk.read(path)
    if row['task'] != task:
        raise ValueError('Task specification differs')
    if set(row['files']) != {'cpu.npz','gpu.npz'}:
        raise ValueError('Checkpoint file inventory differs')
    folder = path.parent/disk.safe_name(row['attempt'])
    for name, h in row['files'].items():
        if disk.digest(folder/disk.safe_name(name)) != h:
            raise ValueError('Checkpoint checksum mismatch')
    if verify_pair(folder) != row['comparison']:
        raise ValueError('Checkpoint numerical audit differs')
    return row


def summarize(root):
    rows = [load_done(root, t) for t in tasks()]
    complete = all(r is not None for r in rows)
    rows = [r for r in rows if r is not None]
    passed = complete and all(r['comparison']['passed'] for r in rows)
    groups = {}
    for r in rows:
        t = r['task']
        if t['phase']=='time':
            key = f"{r['environment_id']} / {t['case']} / batch={t['batch']}"
            groups.setdefault(key, []).append(r)
    summary = {}
    cpu64 = {(r['environment_id'], r['task']['case'], r['task']['repeat']): r
             for r in rows if r['task']['phase']=='time' and r['task']['batch']==64}
    for key, group in groups.items():
        summary[key] = dict(rounds=len(group), full_three_rounds=len(group)==ROUNDS)
        for metric in ('compute_seconds','end_to_end_seconds'):
            ratios = [r['timing']['gpu'][metric]/r['timing']['cpu'][metric] for r in group]
            summary[key][metric] = dict(gpu_over_cpu=ratios, median_ratio=float(np.median(ratios)))
        pairs = [(r, cpu64.get((r['environment_id'],r['task']['case'],r['task']['repeat']))) for r in group]
        ratios = [r['timing']['gpu']['end_to_end_seconds']/c['timing']['cpu']['end_to_end_seconds']
                  for r,c in pairs if c is not None]
        summary[key]['end_to_end_vs_cpu64'] = dict(paired_rounds=len(ratios), gpu_over_cpu64=ratios,
                                                   median_ratio=float(np.median(ratios)) if ratios else None)
    # Batch invariance is a separate test: no cross-environment timing pooling.
    batch_checks = []
    references = { (r['task']['case'],r['task']['repeat']):r for r in rows
                   if r['task']['phase']=='time' and r['task']['batch']==64 }
    for r in rows:
        t = r['task']
        ref = references.get((t['case'],t['repeat']))
        if t['phase']!='time' or t['batch']==64 or ref is None:
            continue
        for backend in ('cpu','gpu'):
            path = root/'tasks'/t['id']/r['attempt']/(backend+'.npz')
            refpath = root/'tasks'/ref['task']['id']/ref['attempt']/(backend+'.npz')
            with open_values(path) as a, open_values(refpath) as b:
                batch_checks.append(dict(task=t['id'],backend=backend,comparison=compare(b['values'],a['values'])))
    passed = passed and all(x['comparison']['passed'] for x in batch_checks)
    all_arrays = [v for r in rows for v in r['comparison']['arrays'].values()]
    numeric = dict(max_abs=max((x['max_abs'] for x in all_arrays if 'max_abs' in x),default=None),
                   max_scaled=max((x['max_scaled'] for x in all_arrays if 'max_scaled' in x),default=None),
                   unscorable_arrays=sum('max_abs' not in x for x in all_arrays),
                   tolerance=TOL, dtype='float64', includes_check_waveforms=True)
    result = dict(status='PILOT_COMPLETE' if passed else 'NUMERICAL_FAIL' if complete else 'PARTIAL',
                  planned_tasks=len(tasks()), completed_tasks=len(rows), numeric_pass=passed,
                  mismatched_tasks=[r['task']['id'] for r in rows if not r['comparison']['passed']],
                  timing_by_environment=summary, numerical_summary=numeric,
                  batch_comparisons=batch_checks, scientific_main=False,
                  note='Ratios below 1 favour GPU. No pooling across different environments. No LF detection/FPR study.')
    disk.write(root/'summary.json',result)
    timing_rows = [dict(task=r['task']['id'], phase=r['task']['phase'], case=r['task']['case'],
                       repeat=r['task']['repeat'], batch=r['task']['batch'], records=r['task']['count'],
                       environment_id=r['environment_id'], backend=backend, **timing)
                   for r in rows for backend,timing in r['timing'].items()]
    if timing_rows:
        stream = io.StringIO()
        keys = list(dict.fromkeys(k for row in timing_rows for k in row))
        writer = csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader(); writer.writerows(timing_rows)
        disk.atomic(root/'timings.csv',stream.getvalue().encode())
    lines = ['# CPU/GPU float64 pilot', '', f"Status: {result['status']}",
             f"Completed: {len(rows)}/{len(tasks())}; numerical pass: {passed}", '',
             f"Finite comparisons: maximum CPU/GPU absolute difference {numeric['max_abs']}; maximum tolerance-scaled difference {numeric['max_scaled']} (must be <= 1). Unscorable arrays: {numeric['unscorable_arrays']} (must be 0).", '',
             'GPU/CPU ratios below 1 favour GPU. End-to-end includes CPU RNG, transfers and NPZ save; excludes setup, warmup, audit and ZIP.', '',
             '| Environment / condition / batch | Rounds | Compute ratio | End-to-end ratio | vs CPU batch 64 |',
             '|---|---:|---:|---:|---:|']
    for key, row in summary.items():
        c64 = row['end_to_end_vs_cpu64']['median_ratio']
        c64text = f'{c64:.4f}' if c64 is not None else 'not paired'
        lines.append(f"| {key} | {row['rounds']} | {row['compute_seconds']['median_ratio']:.4f} | {row['end_to_end_seconds']['median_ratio']:.4f} | {c64text} |")
    lines += ['', 'Three rounds are technical repetitions, not a scientific confirmation. A GPU pass is not approval for the LF main study.',
              'Resumed hardware/environment changes are separated. Incomplete three-round groups remain provisional.',
              'CPU and GPU inputs are generated on CPU with identical logical seeds. GPU-native RNG is not benchmarked.',
              'Memory fields are process RSS high-water / allocator reserved snapshots, not per-task total GPU peak.',
              'Bootstrap, session, errors, numerical differences, and per-stage timings are retained in JSON/NPZ.']
    disk.atomic(root/'REPORT.md', ('\n'.join(lines)+'\n').encode())
    return result


def warmup(gpu):
    start = time.perf_counter()
    # All reduction shapes/batches and both time layouts; use separate seeds.
    for name in BENCH + ('REVERSE',):
        for size in (17,) + BATCHES:
            z = make_inputs(conditions()[name], name, 999, 0, size)
            calculate(conditions()[name], z)
            dz = {k: gpu.cp.asarray(v) for k,v in z.items()}
            calculate(conditions()[name], dz, gpu.cp, gpu.filt)
            gpu.cp.cuda.get_current_stream().synchronize()
            del z,dz
            gpu.cp.get_default_memory_pool().free_all_blocks()
    return time.perf_counter()-start


def main(root):
    if sys.platform!='linux' or not Path('/content').is_dir():
        raise RuntimeError('Worker is for Colab GPU only; no local benchmark')
    root = Path(root)
    session = uuid.uuid4().hex[:12]
    folder = root/'sessions'/session
    folder.mkdir(parents=True)
    begin = time.monotonic()
    previous = sum(disk.read(p).get('active_seconds',0.) for p in (root/'sessions').glob('*/session.json'))
    disk.write(folder/'session.json',dict(active_seconds=0.,status='STARTING',started=disk.stamp()))
    def end_signal(*args):
        raise TimeoutError('Pilot safety deadline / termination')
    signal.signal(signal.SIGTERM, end_signal)
    signal.signal(signal.SIGALRM, end_signal)
    status = 'PARTIAL'
    last_snapshot = time.monotonic()
    try:
        spec = specification()
        if (root/'specification.json').exists() and disk.read(root/'specification.json') != spec:
            raise ValueError('Specification changed')
        disk.write(root/'specification.json', spec)
        if all(load_done(root,t) is not None for t in tasks()):
            status = summarize(root)['status']
            print('All tasks already committed; verified without rerunning GPU work.',flush=True)
            return 0 if status=='PILOT_COMPLETE' else 2
        remaining = 3600-previous-(time.monotonic()-begin)
        if remaining <= 0:
            raise TimeoutError('Cumulative pilot budget exhausted; no automatic extension')
        signal.alarm(min(600,max(1,math.ceil(remaining))))
        gpu = GPU()
        env = environment(gpu)
        env_id = disk.sha(json.dumps(env,sort_keys=True).encode())[:12]
        disk.write(folder/'environment.json',env)
        disk.write(folder/'warmup.json',dict(seconds=warmup(gpu)))
        disk.write(folder/'session.json',dict(active_seconds=time.monotonic()-begin,status='RUNNING',environment_id=env_id))
        for number, task in enumerate(tasks()):
            if previous+time.monotonic()-begin >= 3600:
                raise TimeoutError('Cumulative pilot budget exhausted; no automatic extension')
            if load_done(root, task) is not None:
                continue
            signal.alarm(min(600, max(1,math.ceil(3600-previous-(time.monotonic()-begin)))))
            attempt = uuid.uuid4().hex[:12]
            target = root/'tasks'/task['id']/attempt
            target.mkdir(parents=True)
            disk.write(folder/'active_task.json',dict(task=task,attempt=attempt,status='STARTED'))
            timing = {}
            order = ('cpu','gpu') if number%2==0 else ('gpu','cpu')
            for backend in order:
                gc.collect()
                gpu.cp.get_default_memory_pool().free_all_blocks()
                timing[backend] = backend_task(task,gpu,backend,target)
            comparison = verify_pair(target)
            row = dict(task=task, attempt=attempt, environment_id=env_id, session=session,
                       order=order, timing=timing, comparison=comparison,
                       files={name:disk.digest(target/name) for name in ('cpu.npz','gpu.npz')})
            disk.write(target.parent/'done.json',row)
            disk.write(folder/'active_task.json',dict(task=task,attempt=attempt,status='COMPLETE'))
            disk.write(folder/'session.json',dict(active_seconds=time.monotonic()-begin,status='RUNNING',environment_id=env_id))
            print(f"[{number+1}/{len(tasks())}] {task['id']} numerical={'PASS' if comparison['passed'] else 'FAIL'} CPU={timing['cpu']['end_to_end_seconds']:.3f}s GPU={timing['gpu']['end_to_end_seconds']:.3f}s",flush=True)
            disk.write(root/'progress.json',dict(completed_tasks=number+1,checkpoint_requested=number+1==len(conditions())))
            if sum(p.stat().st_size for p in disk.evidence(root)) > 2*1024**3:
                raise RuntimeError('Result directory exceeds 2 GiB')
            # The worker is the sole evidence writer. Snapshot only between tasks.
            signal.alarm(0)
            if time.monotonic()-last_snapshot >= 300 or number+1==len(conditions()):
                archive = disk.snapshot(root,root.parent.parent,'CHECKPOINT')
                print('CHECKPOINT_ZIP='+str(archive),flush=True)
                last_snapshot = time.monotonic()
        status = summarize(root)['status']
        return 0 if status=='PILOT_COMPLETE' else 2
    except BaseException as exc:
        disk.write(folder/'error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        print(traceback.format_exc(),flush=True)
        try:
            summarize(root)
            status = 'TECHNICAL_FAILURE'
        except Exception as audit:
            disk.write(folder/'audit_error.json',dict(error=str(audit)))
            status = 'AUDIT_FAILED'
        return 1
    finally:
        signal.alarm(0)
        disk.write(folder/'session.json',dict(active_seconds=time.monotonic()-begin,status=status,
                                            rss_high_water_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024))


if __name__=='__main__':
    sys.exit(main(sys.argv[1]))
