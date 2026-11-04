# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Resumable synthetic covariance-consistency experiments (one Colab cell).

The generated copy in numerics/colab/ also embeds the repository licenses.
No hardware data, superiority claim, or manuscript update is produced by running it.
"""

# ================= USER SETTINGS =================
PROFILE = "main"                 # "smoke", "pilot", or "main"
EXPERIMENTS = (1, 2, 3)           # select a subset if desired
SAVE_TO_DRIVE = False             # default: no Drive connection or authorization
DRIVE_FOLDER = "CovarianceConsistencyExperiments"
LOCAL_FOLDER = "covariance_consistency_runs"  # used outside Colab / without Drive
MAX_MINUTES = 45                  # safe pause after a batch; 0 = no time budget
AUTO_DOWNLOAD = True              # COMPLETE or clearly labelled PARTIAL ZIP
RESTORE_ZIP = ""                  # "UPLOAD" to restore a saved ZIP; then reset to ""
MASTER_SEED = 20261017
BATCH_SIZE = 256
MEMORY_BATCH_SIZE = 64            # CPU/RAM compute batch; independent of checkpoint size
ALLOW_ENVIRONMENT_CHANGE = False  # normally keep False on resume
# Same settings + same cell -> resume. A changed scientific configuration -> new run.
# Default checkpoints are runtime-local. Download each PARTIAL ZIP as a backup.
# If the runtime is deleted, restore the latest downloaded ZIP in a fresh runtime.
# Optional SAVE_TO_DRIVE = True requests Drive authorization; default never mounts it.

CODE_VERSION = "covariance-suite-1.0.1"
LICENSE_PAYLOAD = ""  # populated in the generated self-contained Colab artifact

# === ENGINE START ===
import base64
import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
import traceback
import zipfile
import zlib
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath

# Respect explicit user thread limits; keep fresh runtimes conservative.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
import scipy
from scipy import linalg, signal, special, integrate, optimize, stats


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n").encode())


def source_text():
    try:
        if "__file__" in globals():
            return Path(__file__).read_text(encoding="utf-8")
        ip = get_ipython()  # noqa: F821 -- available in a Colab cell
        for text in reversed(ip.history_manager.input_hist_raw):
            if "# === ENGINE START ===" in text and CODE_VERSION in text:
                return text
    except (NameError, OSError, AttributeError):
        pass
    return ""


def scientific_config(profile=PROFILE, experiments=EXPERIMENTS, seed=MASTER_SEED,
                      batch_size=BATCH_SIZE, memory_batch_size=MEMORY_BATCH_SIZE):
    sizes = {
        "smoke": dict(reference=39, null=48, alternative=32, endpoint=64),
        "pilot": dict(reference=499, null=500, alternative=500, endpoint=2000),
        "main": dict(reference=9999, null=20000, alternative=5000, endpoint=200000),
    }
    if profile not in sizes or not set(experiments) <= {1, 2, 3} or not experiments:
        raise ValueError("PROFILE / EXPERIMENTS is invalid")
    if int(batch_size) < 1 or int(memory_batch_size) < 1:
        raise ValueError("Batch sizes must be positive")
    src = source_text()
    engine = src.split("# === ENGINE START ===", 1)[-1] if src else CODE_VERSION
    return dict(code_version=CODE_VERSION, engine_sha256=hashlib.sha256(engine.encode()).hexdigest(),
                profile=profile, experiments=sorted(set(experiments)), seed=int(seed),
                batch_size=int(batch_size), memory_batch_size=int(memory_batch_size),
                repetitions=sizes[profile], alpha=[0.01, 0.05],
                protocol="fixed-v1; nonperiodic; independent reference/test; known-zero primary")


def rng_for(config, name, phase, start):
    words = np.frombuffer(hashlib.sha256((name + "/" + phase).encode()).digest(),
                          dtype="<u4").astype(np.uint32).tolist()
    return np.random.default_rng(np.random.SeedSequence([config["seed"], *words, int(start)]))


def psd_factor(matrix):
    matrix = (matrix + matrix.T) / 2
    eig, vectors = np.linalg.eigh(matrix)
    tolerance = 1e-10 * max(float(np.max(np.abs(eig))), np.finfo(float).tiny)
    if float(np.min(eig)) < -tolerance:
        raise ValueError("Covariance is not PSD; refusing to hide an invalid model")
    return vectors @ np.diag(np.sqrt(np.maximum(eig, 0.0)))


def discrete_weights(tau_i, tau_j, dt):
    ai, aj = np.exp(-dt / np.array([tau_i, tau_j]))
    den = 2 * (1 - ai * aj)
    return (1 + ai) * (1 - aj) / den, (1 + aj) * (1 - ai) / den


@lru_cache(maxsize=256)
def digital_model(kappa, tau_i, tau_j, dt, pole=0.0):
    phi = math.exp(-kappa * dt)
    ai, aj = np.exp(-dt / np.array([tau_i, tau_j]))
    bi, bj = 1 - ai, 1 - aj
    dim = 4 if pole > 0 else 3
    A = np.zeros((dim, dim))
    A[:3, :3] = [[phi, 0, 0], [bi * phi, ai, 0], [bj * phi, 0, aj]]
    B = np.zeros(dim)
    B[:3] = np.sqrt(1 - phi**2) * np.array([1, bi, bj])
    if pole > 0:
        ap = math.exp(-dt / pole)
        A[3] = (1 - ap) * A[2]
        A[3, 3] = ap
        B[3] = (1 - ap) * B[2]
    P = linalg.solve_discrete_lyapunov(A, np.outer(B, B))
    return A, P, psd_factor(P)


def digital_pair(rng, count, length, kappa, tau_i, tau_j, dt, pole=0.0):
    A, P, L = digital_model(kappa, tau_i, tau_j, dt, pole)
    initial = rng.normal(size=(count, P.shape[0])) @ L.T
    phi, ai, aj = A[0, 0], A[1, 1], A[2, 2]
    x, _ = signal.lfilter([np.sqrt(1 - phi**2)], [1, -phi],
                         rng.normal(size=(count, length)), axis=1,
                         zi=(phi * initial[:, 0])[:, None])
    yi, _ = signal.lfilter([1 - ai], [1, -ai], x, axis=1,
                          zi=(ai * initial[:, 1])[:, None])
    yj, _ = signal.lfilter([1 - aj], [1, -aj], x, axis=1,
                          zi=(aj * initial[:, 2])[:, None])
    if pole > 0:
        ap = A[3, 3]
        yj, _ = signal.lfilter([1 - ap], [1, -ap], yj, axis=1,
                              zi=(ap * initial[:, 3])[:, None])
    return yi, yj


@lru_cache(maxsize=256)
def analog_model(kappa, tau_i, tau_j, dt):
    vi = 1 / (1 + kappa * tau_i)
    vj = 1 / (1 + kappa * tau_j)
    cij = (tau_i * vi + tau_j * vj) / (tau_i + tau_j)
    P = np.array([[1, vi, vj], [vi, vi, cij], [vj, cij, vj]])
    A = np.array([[-kappa, 0, 0], [1 / tau_i, -1 / tau_i, 0],
                  [1 / tau_j, 0, -1 / tau_j]])
    F = linalg.expm(A * dt)
    return F, P, psd_factor(P), psd_factor(P - F @ P @ F.T)


def endpoint_scale(p):
    ti, tj, k, T = p["tau_i"], p["tau_j"], p["kappa"], p["T"]
    F, P, _, _ = analog_model(k, ti, tj, T)
    z = np.array([0, ti, -tj])
    var = float(z @ P @ z)
    cov = float(z @ F @ P @ z)
    return math.sqrt(max(0, var**2 - cov**2)) / T


def simulate_endpoint(p, rng, count):
    F, P, L, QL = analog_model(p["kappa"], p["tau_i"], p["tau_j"], p["T"])
    initial = rng.normal(size=(count, 3)) @ L.T
    terminal = initial @ F.T + rng.normal(size=(count, 3)) @ QL.T
    z = np.array([0, p["tau_i"], -p["tau_j"]])
    z0, zT = initial @ z, terminal @ z
    return dict(endpoint=(zT**2 - z0**2) / (2 * p["T"]), z0=z0, zT=zT)


def simulate_analog(p, rng, count):
    ti, tj, dt, T = p["tau_i"], p["tau_j"], p["dt"], p["T"]
    n = int(round(T / dt))
    F, P, L, QL = analog_model(p["kappa"], ti, tj, dt)
    initial = rng.normal(size=(count, 3)) @ L.T
    # Explicit tiny covariance contraction avoids a batched BLAS dispatch.
    innovations = np.einsum("bnt,kt->bnk", rng.normal(size=(count, n, 3)), QL, optimize=False)
    x, _ = signal.lfilter([1], [1, -F[0, 0]], innovations[:, :, 0], axis=1,
                         zi=(F[0, 0] * initial[:, 0])[:, None])
    xprev = np.concatenate((initial[:, 0, None], x[:, :-1]), axis=1)
    ys = []
    for col in (1, 2):
        y, _ = signal.lfilter([1], [1, -F[col, col]],
                             F[col, 0] * xprev + innovations[:, :, col], axis=1,
                             zi=(F[col, col] * initial[:, col])[:, None])
        ys.append(y)
    z0 = ti * initial[:, 1] - tj * initial[:, 2]
    zT = ti * ys[0][:, -1] - tj * ys[1][:, -1]
    yi = np.concatenate((initial[:, 1, None], ys[0][:, :-1]), axis=1)
    yj = np.concatenate((initial[:, 2, None], ys[1][:, :-1]), axis=1)
    raw = (ti + tj) * np.mean(yi * yj, axis=1) - ti * np.mean(yi**2, axis=1) - tj * np.mean(yj**2, axis=1)
    mi, mj = yi.mean(axis=1), yj.mean(axis=1)
    centered = raw + ti * mi**2 + tj * mj**2 - (ti + tj) * mi * mj
    return dict(raw=raw, centered=centered, endpoint=(zT**2 - z0**2) / (2 * T),
                z0=z0, zT=zT)


def baseline_covariance(p):
    k = p.get("noise_kappa", p["kappa"])
    P = digital_model(k, p["tau_i"], p["tau_j"], p["dt"])[1]
    return P[1:3, 1:3].copy()


def make_noise(p, rng, count, length, factor=1.0):
    C = baseline_covariance(p)
    noise_i = np.zeros((count, length))
    noise_j = np.zeros((count, length))
    eta = p.get("eta_common", 0.0)
    if eta > 0:
        if p.get("topology") == "upstream":
            ni, nj = digital_pair(rng, count, length, p.get("noise_kappa", p["kappa"]),
                                  p["tau_i"], p["tau_j"], p["dt"])
            noise_i += math.sqrt(eta) * ni
            noise_j += math.sqrt(eta) * nj
        else:
            common = rng.normal(size=(count, length))
            noise_i += math.sqrt(eta * C[0, 0]) * common
            noise_j += p.get("common_sign", 1) * math.sqrt(eta * C[1, 1]) * common
    private = p.get("eta_readout", 0.0)
    if private > 0:
        noise_i += math.sqrt(private * C[0, 0]) * rng.normal(size=(count, length))
        noise_j += math.sqrt(private * C[1, 1]) * rng.normal(size=(count, length))
    return math.sqrt(factor) * noise_i, math.sqrt(factor) * noise_j


def moments(yi, yj):
    return np.column_stack((np.mean(yi**2, axis=1), np.mean(yj**2, axis=1),
                            np.mean(yi * yj, axis=1)))


def cr_energy(yi, yj, p, start, count):
    ai, aj = np.exp(-p["dt"] / np.array([p["tau_i"], p["tau_j"]]))
    r = signal.lfilter([1 - aj], [1, -aj], yi, axis=1)
    r -= signal.lfilter([1 - ai], [1, -ai], yj, axis=1)
    return np.mean(r[:, start:start + count]**2, axis=1)


def simulate_digital(p, rng, count, timings=None):
    dt, ti, tj = p["dt"], p["tau_i"], p["tau_j"]
    n = int(round(p["T"] / dt))
    n0 = max(2, int(round(p.get("control_ratio", 1.0) * n)))
    warm = int(math.ceil(20 * max(ti, tj) / dt))
    length = n + 2 * warm
    fault, strength = p.get("fault", "none"), p.get("strength", 0.0)
    actual_tj = tj * (1 + strength) if fault == "tau" else tj
    pole = tj * strength if fault == "pole" else 0.0
    delay = int(strength) if fault == "delay" else 0
    guard = abs(delay)
    yi, yj = digital_pair(rng, count, length + 2 * guard, p["kappa"], ti, actual_tj, dt, pole)
    yi = yi[:, guard:guard + length]
    yj = yj[:, guard - delay:guard - delay + length]
    if fault == "gain":
        yj = (1 + strength) * yj
    if fault == "private":
        _, extra = digital_pair(rng, count, length, p["kappa"], ti, tj, dt)
        yj = yj + math.sqrt(strength) * extra
    if fault == "upstream":
        ui, uj = digital_pair(rng, count, length, p["kappa"], ti, tj, dt)
        yi, yj = yi + math.sqrt(strength) * ui, yj + math.sqrt(strength) * uj
    ni, nj = make_noise(p, rng, count, length)
    yi, yj = yi + ni, yj + nj
    reverse = fault == "time_reverse"
    original = moments(yi[:, warm:warm+n], yj[:, warm:warm+n]) if reverse else None
    if reverse:
        yi, yj = yi[:, ::-1], yj[:, ::-1]
    ci, cj = make_noise(p, rng, count, n0 + warm, p.get("control_factor", 1.0))
    moment_start = time.perf_counter()
    O = moments(yi[:, warm:warm+n], yj[:, warm:warm+n])
    N = moments(ci[:, warm:warm+n0], cj[:, warm:warm+n0])
    lam, nu = discrete_weights(ti, tj, dt)
    full = O - N
    result = {"R_full": full[:,2] - lam*full[:,0] - nu*full[:,1]}
    moment_elapsed = time.perf_counter() - moment_start
    for name, corrected in (("none", O), ("diag", O - N * [1, 1, 0]), ("full", O - N)):
        result[name + "_vi"], result[name + "_vj"], result[name + "_c"] = corrected.T
        result["R_" + name] = corrected[:, 2] - lam * corrected[:, 0] - nu * corrected[:, 1]
    cr_start = time.perf_counter()
    result["CR"] = cr_energy(yi, yj, p, warm, n) - cr_energy(ci, cj, p, warm, n0)
    if timings is not None:
        timings.update(moment_seconds=moment_elapsed, cr_seconds=time.perf_counter()-cr_start)
    result["reversal_moment_error"] = (np.max(np.abs(O - original), axis=1)
                                          if reverse else np.zeros(count))
    return result


def build_design(config):
    """All cases are fixed before any performance result is observed."""
    cases = []
    base = dict(tau_i=.04, tau_j=.01, dt=.002, kappa=20.0, T=2.048)

    def add(exp, name, generator, params, kind="null", reference=None, group="primary"):
        p = dict(base, **params)
        ref = dict(p if reference is None else dict(base, **reference))
        key = digest(dict(generator=generator, params=ref))[:16]
        cases.append(dict(id=name, experiment=exp, generator=generator, params=p,
                          reference_params=ref, reference_id="ref_" + key, kind=kind, group=group))

    if 1 in config["experiments"]:
        for k in (2., 20., 200.):
            for T in (.032, .128, .512, 2.048, 8.192):
                add(1, f"e1_endpoint_k{k:g}_T{T:g}", "endpoint", dict(kappa=k, T=T))
        for dt in (.004, .002, .001):
            for T in (.512, 2.048, 8.192):
                add(1, f"e1_sample_dt{dt:g}_T{T:g}", "analog", dict(dt=dt, T=T))
    if 2 in config["experiments"]:
        seen = set()
        def noise_case(topology, **updates):
            p = dict(base, topology=topology, eta_common=.1, eta_readout=0.,
                     control_ratio=1., control_factor=1., common_sign=1)
            p.update(updates)
            identity = digest(p)
            if identity in seen:
                return
            seen.add(identity)
            ref = dict(p, control_factor=1.)
            add(2, f"e2_{topology}_{identity[:10]}", "digital", p, reference=ref,
                group="stress" if updates.get("control_factor", 1) != 1 else "primary")
        for topology in ("upstream", "downstream"):
            for eta in (0., .1, .5):
                noise_case(topology, eta_common=eta)
            for T in (.512, 8.192):
                noise_case(topology, T=T)
            for ratio in (.25, 4.):
                noise_case(topology, control_ratio=ratio)
            noise_case(topology, eta_readout=.1)
            for factor in (.5, 1.5):
                noise_case(topology, control_factor=factor)
        noise_case("downstream", common_sign=-1)
    if 3 in config["experiments"]:
        normal = dict(base, topology="downstream", eta_common=0., eta_readout=.1,
                      control_ratio=1., control_factor=1.)
        levels = dict(gain=(-.10, -.05, -.02, .02, .05, .10),
                      tau=(-.20, -.10, -.05, -.02, .02, .05, .10, .20),
                      delay=(-4, -2, -1, 1, 2, 4), pole=(.1, .25, .5),
                      private=(.02, .1, .5))
        for T in (.512, 2.048, 8.192):
            ref = dict(normal, T=T)
            add(3, f"e3_null_T{T:g}", "digital", ref)
            for fault, values in levels.items():
                for value in values:
                    add(3, f"e3_{fault}_{value:g}_T{T:g}", "digital",
                        dict(ref, fault=fault, strength=value), "alternative", ref)
            for fault, value in (("upstream", .5), ("time_reverse", 1.)):
                add(3, f"e3_blind_{fault}_T{T:g}", "digital",
                    dict(ref, fault=fault, strength=value), "alternative", ref, "structural")
        for name, change in (("r2", dict(tau_i=.02)), ("r10", dict(tau_i=.10)),
                             ("k2", dict(kappa=2.)), ("k200", dict(kappa=200.)),
                             ("noise001", dict(eta_readout=.01)),
                             ("noise05", dict(eta_readout=.5))):
            ref = dict(normal, **change)
            add(3, "e3_robust_null_" + name, "digital", ref, group="robustness")
            add(3, "e3_robust_gain_" + name, "digital", dict(ref, fault="gain", strength=.05),
                "alternative", ref, "robustness")
        for k in (2., 200.):
            add(3, f"e3_source_shift_{k:g}", "digital", dict(normal, kappa=k, noise_kappa=20.),
                "null_shift", dict(normal, noise_kappa=20.), "source_shift")
    if config["profile"] == "smoke":
        # Representative coverage of every implementation branch, not performance evidence.
        selected = []
        for exp in config["experiments"]:
            pool = [c for c in cases if c["experiment"] == exp]
            if exp == 1:
                selected += [next(c for c in pool if c["generator"] == g) for g in ("endpoint", "analog")]
            elif exp == 2:
                selected += [pool[1], next(c for c in pool if c["params"]["control_factor"] != 1), pool[-1]]
            else:
                selected.append(pool[0])
                for fault in ("gain", "tau", "delay", "pole", "private", "upstream", "time_reverse"):
                    selected.append(next(c for c in pool if c["params"].get("fault") == fault))
                selected.append(pool[-1])
        cases = selected
    return cases


def tasks_for(cases, config):
    tasks, known = [], set()
    sizes = config["repetitions"]
    for case in cases:
        if case["reference_id"] not in known:
            known.add(case["reference_id"])
            tasks.append(dict(id=case["reference_id"], phase="reference", generator=case["generator"],
                              params=case["reference_params"], total=sizes["reference"]))
        total = (sizes["endpoint"] if case["generator"] == "endpoint" else
                 sizes["alternative"] if case["kind"] == "alternative" else sizes["null"])
        tasks.append(dict(id=case["id"], phase="evaluation", generator=case["generator"],
                          params=case["params"], total=total))
    return tasks


def chunk_size(task, config):
    if task["generator"] == "endpoint":
        return max(config["batch_size"], 2048) if config["profile"] != "smoke" else config["batch_size"]
    return config["batch_size"]


def simulate_checkpoint(task, config, start, count):
    """Bound temporary time-series memory without multiplying Drive writes."""
    p = task["params"]
    if task["generator"] == "endpoint":
        step = count  # Only three state variables per endpoint, no long time series.
    else:
        n = int(round(p["T"] / p["dt"]))
        length = max(n * p.get("control_ratio", 1), n) + int(40 * max(p["tau_i"], p["tau_j"]) / p["dt"])
        cap = max(1, int(96 * 1024**2 / (8 * 16 * length)))
        step = min(config["memory_batch_size"], cap)
    pieces, totals = {}, {}
    for offset in range(0, count, step):
        size = min(step, count-offset)
        rng = rng_for(config, task["id"], task["phase"], start+offset)
        timings = {}
        if task["generator"] == "digital":
            data = simulate_digital(p, rng, size, timings)
        else:
            data = {"endpoint": simulate_endpoint, "analog": simulate_analog}[task["generator"]](p, rng, size)
        for key, values in data.items():
            pieces.setdefault(key, []).append(values)
        for key, value in timings.items():
            totals[key] = totals.get(key, 0.) + value
    return {key:np.concatenate(values) for key,values in pieces.items()}, totals


def colab_self_check():
    """Very small runtime checks, executed on Colab before the experiment."""
    lam, nu = discrete_weights(.04,.01,.002)
    P = digital_model(20.,.04,.01,.002)[1]
    balance = abs(P[1,2]-lam*P[1,1]-nu*P[2,2])
    if balance > 1e-11:
        raise AssertionError("Discrete covariance self-check failed")
    q = product_threshold(.01)
    if abs(q-3.60421320225167) > 1e-8:
        raise AssertionError("Product-normal quantile self-check failed")
    p = dict(kappa=20.,tau_i=.04,tau_j=.01,dt=.002,T=.512,
             eta_readout=.1,topology="downstream",fault="time_reverse",strength=1.)
    values = simulate_digital(p,np.random.default_rng(2026101799),8)
    reversal = float(values["reversal_moment_error"].max())
    if reversal > 1e-11:
        raise AssertionError("Time-reversal moment self-check failed")
    return dict(status="PASS", at=utc_now(), covariance_balance_error=float(balance),
                product_1pct_threshold=q, reversal_moment_error=reversal,
                scope="small implementation check, not scientific performance evidence")


def chunk_path(root, task, start):
    return Path(root) / "chunks" / task["id"] / f"{start:09d}.npz"


def read_chunk(path, expected_hash=None):
    path = Path(path)
    check = json.loads(path.with_suffix(".json").read_text())
    if expected_hash is not None and check["config_hash"] != expected_hash:
        raise ValueError("Checkpoint belongs to another configuration")
    if file_sha(path) != check["sha256"]:
        raise ValueError("Checkpoint checksum mismatch: " + str(path))
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    if not data or any(v.ndim != 1 or len(v) != check["count"] or not np.all(np.isfinite(v))
                       for v in data.values()):
        raise ValueError("Invalid or non-finite checkpoint arrays")
    return data, check


def commit_chunk(path, data, config_hash, task, start, count, timings=None):
    if not data or any(v.ndim != 1 or len(v) != count or not np.all(np.isfinite(v)) for v in data.values()):
        raise ValueError("Non-finite / invalid results; no checkpoint was committed")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **data)
    atomic_bytes(path, buffer.getvalue())
    write_json(Path(path).with_suffix(".json"), dict(config_hash=config_hash, task=task["id"],
               start=start, count=count, sha256=file_sha(path), committed_at=utc_now(), timings=timings or {}))


def load_task(root, task, config):
    data = {}
    for start in range(0, task["total"], chunk_size(task, config)):
        path = chunk_path(root, task, start)
        if not path.exists() or not path.with_suffix(".json").exists():
            return None
        values, check = read_chunk(path, digest(config))
        expected = min(chunk_size(task, config), task["total"] - start)
        if check["start"] != start or check["count"] != expected or check["task"] != task["id"]:
            raise ValueError("Checkpoint range/task mismatch")
        for key, value in values.items():
            data.setdefault(key, []).append(value)
    return {key: np.concatenate(values) for key, values in data.items()}


def wilson(k, n):
    z = stats.norm.ppf(.975)
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return center - half, center + half


def empirical_p(reference, observed):
    ordered = np.sort(reference)
    return (1 + len(ordered) - np.searchsorted(ordered, observed, side="left")) / (len(ordered) + 1)


@lru_cache(maxsize=8)
def product_threshold(alpha):
    def tail(a):
        return 2 / np.pi * integrate.quad(special.k0, a, np.inf, epsabs=1e-11)[0]
    return optimize.brentq(lambda a: tail(a) - alpha, .01, 20.)


def fitted_endpoint_scale(reference, T, limit=None):
    a, b = reference["z0"], reference["zT"]
    if limit is not None:
        a, b = a[:limit], b[:limit]
    var = float(np.mean((a*a + b*b) / 2))
    cov = float(np.mean(a*b))
    return math.sqrt(max(0, var*var - cov*cov)) / T


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode())


def rate_row(case, method, alpha, flags, **extras):
    k, n = int(np.sum(flags)), len(flags)
    low, high = wilson(k, n)
    p = case["params"]
    warm = math.ceil(20*max(p["tau_i"],p["tau_j"])/p["dt"])*p["dt"] if case["generator"] == "digital" else 0.
    return dict(case=case["id"], experiment=case["experiment"], kind=case["kind"], group=case["group"],
                method=method, alpha=alpha, trials=n, rejections=k, rate=k/n,
                wilson_low=low, wilson_high=high, T=p["T"], dt=p["dt"], kappa=p["kappa"],
                fault=p.get("fault", "none"), strength=p.get("strength", 0.),
                topology=p.get("topology", ""), eta_common=p.get("eta_common", 0.),
                control_ratio=p.get("control_ratio", 1.), control_factor=p.get("control_factor", 1.),
                filter_warmup_T=warm,
                target_record_duration=p["T"]+warm*(2 if p.get("fault")=="time_reverse" else 1),
                control_record_duration=p["T"]*p.get("control_ratio",1.)+warm if case["generator"]=="digital" else 0.,
                **extras)


def paired_interval(a, b, rng):
    d = a.astype(int) - b.astype(int)
    counts = np.array([np.sum(d == -1), np.sum(d == 0), np.sum(d == 1)])
    boot = rng.multinomial(len(d), counts / len(d), size=2000)
    interval = np.quantile((boot[:, 2] - boot[:, 0]) / len(d), [.025, .975])
    return float(d.mean()), float(interval[0]), float(interval[1])


def calibration_sensitivity(reference, observed, alpha, rng, draws=100):
    """Resample independent *whole-record statistics*, never time blocks."""
    ordered = np.sort(observed)
    B = len(reference)
    rank = int(math.ceil((1 - alpha) * (B + 1))) - 1
    if rank >= B:
        return None
    qs = []
    for _ in range(draws):
        sample = reference[rng.integers(0, B, B)]
        qs.append(np.partition(sample, rank)[rank])
    rates = (len(ordered) - np.searchsorted(ordered, qs, side="right")) / len(ordered)
    lo, hi = np.quantile(rates, [.025, .975])
    return float(lo), float(hi)


def summarize(root, cases, tasks, config):
    root = Path(root)
    lookup = {t["id"]: t for t in tasks}
    rates, moments_rows, pairs, scales, coverage, calibration_rows = [], [], [], [], [], []
    refs = {}
    for case_index, case in enumerate(cases):
        if case_index % 20 == 0:
            print(f"Summary: inspecting case {case_index+1}/{len(cases)}", flush=True)
        data = load_task(root, lookup[case["id"]], config)
        rid = case["reference_id"]
        if rid not in refs:
            refs[rid] = load_task(root, lookup[rid], config)
        reference = refs[rid]
        complete = data is not None and reference is not None
        coverage.append(dict(case=case["id"], reference=rid, complete=complete,
                             planned=lookup[case["id"]]["total"]))
        if not complete:
            continue
        p = case["params"]
        if case["experiment"] == 1:
            true_scale = endpoint_scale(p)
            limits = sorted(set([min(100, len(reference["z0"])),
                                 min(500, len(reference["z0"])), len(reference["z0"])]))
            for name in ("endpoint", "raw", "centered"):
                if name not in data:
                    continue
                v = data[name]
                scales.append(dict(case=case["id"], statistic=name, mean=float(v.mean()),
                                   empirical_sd=float(v.std(ddof=1)), endpoint_sd=true_scale,
                                   sd_ratio=float(v.std(ddof=1) / true_scale),
                                   endpoint_difference_rms=float(np.sqrt(np.mean((v-data["endpoint"])**2)))))
                for alpha in config["alpha"]:
                    decisions = {
                        name + "/normal_oracle_scale": np.abs(v) > stats.norm.isf(alpha/2) * true_scale,
                        name + "/product_oracle_scale": np.abs(v) > product_threshold(alpha) * true_scale,
                        name + "/empirical_matched": empirical_p(np.abs(reference[name]), np.abs(v)) <= alpha,
                    }
                    for limit in limits:
                        scale = fitted_endpoint_scale(reference, p["T"], limit)
                        decisions[name + f"/product_fitted_B{limit}"] = np.abs(v) > product_threshold(alpha) * scale
                    for method, flag in decisions.items():
                        rates.append(rate_row(case, method, alpha, flag))
                rr = rng_for(config, case["id"], "calibration_report_"+name, 0)
                interval = calibration_sensitivity(np.abs(reference[name]), np.abs(v), .01, rr,
                                                   20 if config["profile"]=="smoke" else 100)
                if interval:
                    calibration_rows.append(dict(case=case["id"], method=name+"/empirical_matched", alpha=.01,
                                                  reference_resampling_rate_low=interval[0],
                                                  reference_resampling_rate_high=interval[1]))
            continue
        lam, nu = discrete_weights(p["tau_i"], p["tau_j"], p["dt"])
        if case["experiment"] == 2:
            truth = digital_model(p["kappa"], p["tau_i"], p["tau_j"], p["dt"])[1][1:3, 1:3]
            gamma_true = abs(truth[0, 1]) / math.sqrt(truth[0, 0] * truth[1, 1])
            for rule in ("none", "diag", "full"):
                vi, vj, c = data[rule+"_vi"], data[rule+"_vj"], data[rule+"_c"]
                err = (vi-truth[0,0])**2 + (vj-truth[1,1])**2 + 2*(c-truth[0,1])**2
                valid = (vi > 0) & (vj > 0)
                gamma = np.abs(c[valid]) / np.sqrt(vi[valid]*vj[valid])
                mineig = (vi+vj-np.sqrt((vi-vj)**2+4*c*c))/2
                moments_rows.append(dict(case=case["id"], rule=rule, trials=len(vi),
                    covariance_relative_rmse=float(np.sqrt(err.mean()) / np.linalg.norm(truth)),
                    bias_vi=float(vi.mean()-truth[0,0]), bias_vj=float(vj.mean()-truth[1,1]),
                    bias_c=float(c.mean()-truth[0,1]), residual_mean=float(data["R_"+rule].mean()),
                    invalid_gamma_rate=float(1-valid.mean()), negative_eigenvalue_rate=float((mineig < 0).mean()),
                    gamma_above_one_count=int(np.sum(gamma > 1)), valid_gamma_count=int(valid.sum()),
                    gamma_rmse_valid_only=float(np.sqrt(np.mean((gamma-gamma_true)**2))) if len(gamma) else ""))
                for alpha in config["alpha"]:
                    flags = empirical_p(np.abs(reference["R_full"]), np.abs(data["R_"+rule])) <= alpha
                    rates.append(rate_row(case, "correction/"+rule, alpha, flags,
                                          threshold_basis="matched FULL correction; not re-centered by rule"))
        else:
            for alpha in config["alpha"]:
                flags_r = empirical_p(np.abs(reference["R_full"]), np.abs(data["R_full"])) <= alpha
                flags_cr = empirical_p(reference["CR"], data["CR"]) <= alpha
                rates.append(rate_row(case, "moment/two_sided", alpha, flags_r))
                rates.append(rate_row(case, "CR/upper", alpha, flags_cr))
                rr = rng_for(config, case["id"], "paired_report", int(alpha*10000))
                diff, low, high = paired_interval(flags_r, flags_cr, rr)
                pairs.append(dict(case=case["id"], alpha=alpha, difference_moment_minus_CR=diff,
                                  paired_bootstrap_low=low, paired_bootstrap_high=high,
                                  reversal_moment_error_max=float(data["reversal_moment_error"].max())))
        # Separate finite-reference uncertainty from conditional binomial intervals.
        for name, refscore, score in (("moment", np.abs(reference["R_full"]), np.abs(data["R_full"])),
                                      ("CR", reference["CR"], data["CR"])):
            rr = rng_for(config, case["id"], "calibration_report_"+name, 0)
            interval = calibration_sensitivity(refscore, score, .01, rr,
                                                20 if config["profile"] == "smoke" else 100)
            if interval:
                calibration_rows.append(dict(case=case["id"], method=name, alpha=.01,
                                              reference_resampling_rate_low=interval[0],
                                              reference_resampling_rate_high=interval[1]))
    tables = root / "tables"
    write_csv(tables / "rejection_rates.csv", rates)
    write_csv(tables / "covariance_errors.csv", moments_rows)
    write_csv(tables / "paired_comparisons.csv", pairs)
    write_csv(tables / "endpoint_scaling.csv", scales)
    write_csv(tables / "calibration_uncertainty.csv", calibration_rows)
    write_csv(tables / "coverage.csv", coverage)
    timing_rows = []
    for task in tasks:
        elapsed = dict(generation_seconds=0., moment_seconds=0., cr_seconds=0.)
        count = 0
        for checkfile in sorted((root / "chunks" / task["id"]).glob("*.json")):
            check = json.loads(checkfile.read_text())
            count += check["count"]
            for key in elapsed:
                elapsed[key] += check.get("timings", {}).get(key, 0.)
        if count:
            timing_rows.append(dict(task=task["id"], trials=count,
                                   **{key+"_per_record": value/count for key,value in elapsed.items()}))
    write_csv(tables / "method_timings.csv", timing_rows)
    # Descriptive grid crossing, not a guaranteed continuous-time sample-size bound.
    lengths = []
    groups = {}
    for row in rates:
        if row["experiment"] == 3 and row["kind"] == "alternative" and row["group"] == "primary" and row["alpha"] == .01:
            key = row["fault"], row["strength"], row["method"]
            groups.setdefault(key, []).append(row)
    for (fault, strength, method), rows in groups.items():
        reached = [r["T"] for r in rows if r["rate"] >= .8]
        lengths.append(dict(fault=fault, strength=strength, method=method,
                            first_tested_T_with_power_ge_80=min(reached) if reached else "not_reached",
                            scope="observed grid crossing; inspect paired null FPR and intervals"))
    write_csv(tables / "record_length_80.csv", lengths)
    plot_results(root, rates, moments_rows, cases)
    return sum(c["complete"] for c in coverage), len(coverage)


def plot_results(root, rates, errors, cases):
    if not rates:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(root) / "figures"
    folder.mkdir(exist_ok=True)
    chosen = [r for r in rates if r["experiment"] == 1 and r["kappa"] == 20 and r["alpha"] == .01
              and r["method"] in ("endpoint/normal_oracle_scale", "endpoint/product_oracle_scale")
              and "e1_endpoint" in r["case"]]
    if chosen:
        fig, ax = plt.subplots(figsize=(7,4))
        for method in sorted(set(r["method"] for r in chosen)):
            rows = sorted((r for r in chosen if r["method"] == method), key=lambda r:r["T"])
            ax.plot([r["T"] for r in rows], [r["rate"] for r in rows], "o-", label=method)
        ax.axhline(.01, color="black", linestyle=":", label="nominal 1%")
        ax.set(xscale="log", xlabel="Window T (s)", ylabel="False-alarm fraction", title="Ideal endpoint: independent null evaluation")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(folder/"experiment1_false_alarm.png", dpi=160); plt.close(fig)
    if errors:
        ids = list(dict.fromkeys(r["case"] for r in errors))
        fig, ax = plt.subplots(figsize=(max(8, len(ids)*.3),4))
        for k, rule in enumerate(("none", "diag", "full")):
            values = {r["case"]:r["covariance_relative_rmse"] for r in errors if r["rule"] == rule}
            ax.bar(np.arange(len(ids))+(k-1)*.25, [values.get(i,np.nan) for i in ids], .25, label=rule)
        ax.set(xticks=np.arange(len(ids)), xticklabels=[str(i+1) for i in range(len(ids))],
               xlabel="Case index (see case_index.csv)", ylabel="Relative covariance RMSE", title="Correction errors: all trials retained")
        write_csv(Path(root)/"tables"/"case_index.csv", [dict(index=i+1,case=name) for i,name in enumerate(ids)])
        ax.legend(); fig.tight_layout(); fig.savefig(folder/"experiment2_covariance_error.png", dpi=160); plt.close(fig)
    chosen = [r for r in rates if r["experiment"] == 3 and r["group"] == "primary"
              and r["kind"] == "alternative" and r["alpha"] == .01 and r["T"] == 2.048]
    if chosen:
        fig, axes = plt.subplots(2,3,figsize=(12,7))
        for ax, fault in zip(axes.flat, ("gain","tau","delay","pole","private")):
            for method in ("moment/two_sided", "CR/upper"):
                rows=sorted((r for r in chosen if r["fault"]==fault and r["method"]==method), key=lambda r:r["strength"])
                ax.plot([r["strength"] for r in rows], [r["rate"] for r in rows], "o-", label=method)
            ax.set(title=fault, xlabel="Signed / nonnegative strength (see protocol)", ylabel="Detection fraction", ylim=(0,1))
        axes.flat[-1].axis("off"); axes.flat[0].legend(fontsize=7)
        fig.suptitle("Independent thresholds; verify matched-null FPR before comparing power")
        fig.tight_layout(); fig.savefig(folder/"experiment3_power.png", dpi=160); plt.close(fig)


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform())


def archive_run(root, status):
    root = Path(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root.parent / f"{root.name}_{status}_{stamp}.zip"
    tmp = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for index, path in enumerate(sorted(root.rglob("*"))):
            if index and index % 2000 == 0:
                print(f"ZIP packaging: {index} entries inspected", flush=True)
            if path.is_file() and not path.name.endswith(".tmp") and ".quarantine." not in path.name:
                z.write(path, path.relative_to(root).as_posix())
    os.replace(tmp, destination)
    return destination


def restore_archive(archive, root, config):
    root = Path(root)
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if len(members) > 50000 or sum(m.file_size for m in members) > 4 * 1024**3:
            raise ValueError("Resume archive exceeds safety limits")
        if json.loads(z.read("config.json")) != config:
            raise ValueError("Resume ZIP configuration/code does not match this cell")
        names = set()
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive entry")
            if member.filename in names:
                raise ValueError("Duplicate archive entry")
            names.add(member.filename)
            target = root.joinpath(*path.parts)
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("Archive path escapes destination through a symlink")
        for member in members:
            if member.is_dir():
                continue
            path = PurePosixPath(member.filename)
            target = root.joinpath(*path.parts)
            content = z.read(member)
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError("Restore would overwrite a different existing file: " + str(target))
            else:
                atomic_bytes(target, content)


def write_artifact_notes(root, config, cases):
    root = Path(root)
    src = source_text()
    if src:
        atomic_bytes(root / "runner_source.py", src.encode())
    notices = {}
    if LICENSE_PAYLOAD:
        notices = json.loads(zlib.decompress(base64.b64decode(LICENSE_PAYLOAD)))
    elif "__file__" in globals():
        repo = Path(__file__).resolve().parents[2]
        for name in ("LICENSE", "LICENSE-CC-BY-4.0.txt"):
            if (repo/name).exists():
                notices[name] = (repo/name).read_text()
    for name, text in notices.items():
        atomic_bytes(root/name, text.encode())
    note = f"""# Synthetic covariance-consistency experiment archive

Code: {CODE_VERSION}; profile: {config['profile']}; configuration hash: {digest(config)}.
Read run_status.json first. PARTIAL/FAILED is not a completed experiment.
smoke/pilot runs are implementation/timing checks, not publication-level evidence.

## Reproduction and resume
Copy runner_source.py into one Colab cell. Use the same scientific settings.
Use the persistent Drive directory, or set RESTORE_ZIP to UPLOAD with an empty
destination. A committed batch has both NPZ and a matching JSON SHA-256 sidecar.
Only complete, verified batches are reused; at most the in-flight batch is lost.
Individual arrays are per-independent-record statistics, not full time series.
config.json, design.json and environment_history.json specify the execution.

## Interpretation
Experiment 1 uses a continuous OU model and exact sampled transitions. Rectangular
sample moments and mean-subtracted moments are NOT equated to endpoint differences.
Product-normal critical values are an ideal known-scale benchmark. Fitted-scale
values use independent reference records; finite-reference uncertainty remains.
Experiment 2 uses exact digital recursions. All three corrections share data and
the matched FULL-correction reference threshold; wrong-rule bias is not re-centered.
Its wrong-rule rejection rates are not labelled false-positive rates.
Experiment 3 compares fixed-calibration moments (two-sided) with cross-filter
energy (upper-tail), both corrected using the same independent noise control.
This is a detection adaptation, not a reproduction of a parameter optimizer.
The reference/test source law is matched except explicitly labelled source shifts.
Known simulated truth is used for error scoring, never supplied to the detectors.
The time-reversal case preserves the central-window moments and is a structural
counterexample, not a claim about real-world fault prevalence.
Upstream common fluctuations may pass both tests while changing the declared target.

Rates have conditional 95% Wilson intervals. calibration_uncertainty.csv separately
resamples independent reference-record statistics; it is not a time-block bootstrap
and not a combined confidence interval. Paired differences resample paired binary
outcomes. These are pointwise, not simultaneous, intervals. No winner is selected
automatically; inspect matched-null rates before power comparisons. No PSD
projection, gamma clipping, or silent trial deletion is performed.

tables/coverage.csv lists completed cases. design.json maps all parameters.
tables/record_length_80.csv reports a descriptive crossing on the tested T grid.
batch_timings.csv contains batch timings including generation and warmup.
method_timings.csv additionally separates the moments/full-correction residual
computation and the CR filtering/energy calculation. These are single-call wall
times, exclude reference-threshold construction and I/O, and are not an optimized
implementation benchmark. All claims concern synthetic models.

Original code: Apache-2.0. Original explanatory text/results to the extent covered
by copyright: CC BY 4.0. Copyright 2026 Kentaro Sato. Cite the associated working
manuscripts when using the methods; this request is not an extra license condition.

## References
- Gillespie et al. (2016), https://doi.org/10.1016/j.ifacol.2016.07.087
- Gaunt (2019), https://doi.org/10.1111/stan.12152
- Phipson and Smyth (2010), https://doi.org/10.2202/1544-6115.1585
"""
    atomic_bytes(root/"README.md", note.encode())
    write_json(root/"design.json", cases)


def run_suite(config=None, base_directory=None, max_minutes=MAX_MINUTES,
              auto_download=AUTO_DOWNLOAD, restore_zip=RESTORE_ZIP, max_new_batches=None):
    config = config or scientific_config()
    in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
    if base_directory is None:
        if in_colab and SAVE_TO_DRIVE:
            from google.colab import drive
            drive.mount("/content/drive")  # Never silently fall back to ephemeral storage.
            base_directory = Path("/content/drive/MyDrive") / DRIVE_FOLDER
        else:
            base_directory = Path(LOCAL_FOLDER).resolve()
            if in_colab:
                print("WARNING: local /content storage is lost when the runtime is deleted. Download the PARTIAL ZIP.", flush=True)
    root = Path(base_directory) / (config["profile"] + "_" + digest(config)[:16])
    if restore_zip:
        if restore_zip == "UPLOAD":
            if not in_colab:
                raise ValueError("UPLOAD is available only in Colab; pass a ZIP path locally")
            from google.colab import files
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError("Choose exactly one resume ZIP")
            restore_zip = io.BytesIO(next(iter(uploaded.values())))
        restore_archive(restore_zip, root, config)
    root.mkdir(parents=True, exist_ok=True)
    config_file = root / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError("Existing run configuration differs")
    write_json(config_file, config)
    env = environment()
    envfile = root / "environment_history.json"
    history = json.loads(envfile.read_text()) if envfile.exists() else []
    if history:
        old = history[-1]["environment"]
        changed = any(old[k] != env[k] for k in ("python", "numpy", "scipy"))
        if changed and not ALLOW_ENVIRONMENT_CHANGE:
            raise ValueError("Python/NumPy/SciPy changed. Use the same environment or explicitly allow and document the change.")
    history.append(dict(at=utc_now(), environment=env))
    write_json(envfile, history)
    cases = build_design(config)
    tasks = tasks_for(cases, config)
    write_artifact_notes(root, config, cases)
    print(f"Run: {root}\nProfile: {config['profile']}; {len(cases)} cases, {len(tasks)} shared-reference/evaluation tasks", flush=True)
    if config["profile"] != "main":
        print("SMOKE/PILOT: timing and implementation checks only; not final scientific evidence.", flush=True)
    start_time = time.monotonic()
    log_path = root / "run.log"
    timings_path = root / "batch_timings.csv"
    new_batches, reused, error = 0, 0, None
    status = "PARTIAL"
    write_json(root/"run_status.json", dict(status="RUNNING", started_at=utc_now(), config_hash=digest(config)))
    try:
        if in_colab:
            print("Running small Colab-side self-checks...", flush=True)
            write_json(root/"colab_self_check.json", colab_self_check())
        for task in tasks:
            step = chunk_size(task, config)
            for start in range(0, task["total"], step):
                count = min(step, task["total"] - start)
                path = chunk_path(root, task, start)
                if path.exists() and path.with_suffix(".json").exists():
                    _, check = read_chunk(path, digest(config))
                    if check["start"] != start or check["count"] != count or check["task"] != task["id"]:
                        raise ValueError("Checkpoint batch identity mismatch")
                    reused += 1
                    continue
                if ((max_minutes and time.monotonic()-start_time >= max_minutes*60)
                        or (max_new_batches is not None and new_batches >= max_new_batches)):
                    raise TimeoutError("Safe time/batch limit: rerun the same cell to resume")
                t0 = time.monotonic()
                data, timings = simulate_checkpoint(task, config, start, count)
                elapsed = time.monotonic()-t0
                timings["generation_seconds"] = elapsed
                commit_chunk(path, data, digest(config), task, start, count, timings)
                new_batches += 1
                new_timing_file = not timings_path.exists()
                with timings_path.open("a", newline="") as handle:
                    writer = csv.writer(handle)
                    if new_timing_file:
                        writer.writerow(["task", "start", "count", "generation_seconds", "utc"])
                    writer.writerow([task["id"], start, count, elapsed, utc_now()])
                if start == 0 or start + count == task["total"] or new_batches % 20 == 0:
                    message = f"{task['id']}: {start+count}/{task['total']} | elapsed {(time.monotonic()-start_time)/60:.1f} min | reused {reused} batches"
                    print(message, flush=True)
                    with log_path.open("a") as handle:
                        handle.write(utc_now()+" "+message+"\n")
        status = "COMPLETE"
    except (KeyboardInterrupt, TimeoutError) as exc:
        print("PAUSED:", str(exc) or "interrupted", flush=True)
    except Exception:
        status, error = "FAILED", traceback.format_exc()
        atomic_bytes(root/"error.txt", error.encode())
        print(error, flush=True)
    finally:
        completed = 0
        print("Auditing completed checkpoints and writing tables/figures...", flush=True)
        try:
            completed, total = summarize(root, cases, tasks, config)
        except Exception:
            status = "FAILED"
            error = (error or "") + "\nSummary failure:\n" + traceback.format_exc()
            atomic_bytes(root/"error.txt", error.encode())
            total = len(cases)
        if status == "COMPLETE" and completed != total:
            status = "FAILED"
            error = "Completion audit did not find every planned case"
        result = dict(status=status, finished_at=utc_now(), completed_cases=completed,
                      planned_cases=total, new_batches=new_batches, reused_batches=reused,
                      elapsed_seconds=time.monotonic()-start_time, config_hash=digest(config),
                      profile=config["profile"], main_profile_complete=(status=="COMPLETE" and config["profile"]=="main"),
                      scientific_interpretation="completion is not a statistical validation or a superiority claim")
        write_json(root/"run_status.json", result)
        print("Packaging the resumable ZIP (may take a few minutes on Drive)...", flush=True)
        archive = archive_run(root, status)
        print(f"{status}: {completed}/{total} cases\nZIP: {archive}\nResume: rerun with the SAME scientific settings.", flush=True)
        if in_colab and auto_download:
            try:
                from google.colab import files
                files.download(str(archive))
            except Exception as exc:
                print("Browser download could not start:", exc, "\nThe ZIP remains saved at", archive, flush=True)
    if error:
        raise RuntimeError("Run failed; checkpoints were retained. See error.txt in " + str(root))
    return dict(root=str(root), archive=str(archive), **result)


if __name__ == "__main__":
    RUN_RESULT = run_suite()
