# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""PR-0002 CPU implementation. Importing this module never starts a run.

The scientific kernel is the unchanged archived simulator. Pilot and confirmation
have separate namespaces, manifests and outputs. No Drive API is used here.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath

import numpy as np
import scipy
from scipy import stats

import covariance_consistency_colab_v1 as kernel

VERSION = "pr0002-cpu-1.0.0"
CARD_SHA = "ac2725d797cb6cdfe85bc0a3f389fa3dbf2bd8a9cd303acafe95a63a381e3e07"
FROZEN_SHA = "5cbeda59663cc91c789d0ce57d163c9972d76966a88604f0150596edc17f9585"
KERNEL_SHA = "0124e988e1a2927afd034ed5fbcc827f4eccb4653583c22d9c2066484836678c"
METHODS = ("M", "CR", "B", "J")
PAIRS = ((2, 0), (2, 1), (3, 0), (3, 1), (3, 2))
COLUMNS = tuple(f"{s}_{q}" for s in ("none", "diag", "full") for q in ("vi", "vj", "c")) + (
    "R_none", "R_diag", "R_full", "CR", "reversal_moment_error")
MICRO, CHUNK, REPEATS = 64, 2048, 50
GIB = 1024**3
TOLERANCE = dict(rtol=2e-12, atol=2e-12, exact_resume=True)
SEEDS = dict(confirm=2026101704, pilot=2026101705)
PRIMARY = (
    ("G_L_MINUS", "B", "CR", 1), ("G_L_MINUS", "J", "CR", 1),
    ("G_L_MINUS", "B", "M", -1), ("G_L_MINUS", "J", "M", -1),
    ("G_L_PLUS", "B", "CR", 1), ("G_L_PLUS", "J", "CR", 1),
    ("G_L_PLUS", "B", "M", 0), ("G_L_PLUS", "J", "M", 0),
    ("PRIVATE_M", "B", "M", 1), ("PRIVATE_M", "J", "M", 1),
    ("PRIVATE_M", "B", "CR", -1), ("PRIVATE_M", "J", "CR", -1),
    ("G_L_MINUS", "J", "B", 0), ("G_L_PLUS", "J", "B", 0))


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def write_json(p, data):
    atomic(p, (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def read_json(p):
    return json.loads(Path(p).read_text())


def conditions():
    base = dict(tau_i=.04, tau_j=.01, dt=.002, kappa=20., T=2.048,
                topology="downstream", eta_common=0., eta_readout=.1,
                control_ratio=1., control_factor=1.)
    result = {}
    for key, T, ti in (("N_S", .512, .04), ("N_M", 2.048, .04),
                       ("N_L", 8.192, .04), ("N_R10", 2.048, .10)):
        result[key] = dict(group=key, kind="normal", params=dict(base, T=T, tau_i=ti))
    for tag in ("S", "M", "L"):
        for sign, strength in (("MINUS", -.05), ("PLUS", .05)):
            key, group = f"G_{tag}_{sign}", f"N_{tag}"
            result[key] = dict(group=group, kind="fault", params=dict(
                result[group]["params"], fault="gain", strength=strength))
    for key, group, fault, strength, kind in (
        ("PRIVATE_M", "N_M", "private", .02, "fault"),
        ("DELAY_M_MINUS", "N_M", "delay", -2, "fault"),
        ("DELAY_M_PLUS", "N_M", "delay", 2, "fault"),
        ("G_R10_PLUS", "N_R10", "gain", .05, "fault"),
        ("REVERSE_M", "N_M", "time_reverse", 1., "structural"),
        ("UPSTREAM_M", "N_M", "upstream", .5, "structural")):
        result[key] = dict(group=group, kind=kind, params=dict(
            result[group]["params"], fault=fault, strength=strength))
    return result


def tasks(mode):
    if mode not in SEEDS:
        raise ValueError("mode must be pilot or confirm")
    result = []
    for rep in range(REPEATS if mode == "confirm" else 2):
        for cid, c in conditions().items():
            role = {"normal": "normal_eval", "fault": "fault_eval", "structural": "structural_eval"}[c["kind"]]
            sizes = ([("reference", 9999), ("calibration", 10000), (role, 20000)]
                     if c["kind"] == "normal" else [(role, 5000)])
            if mode == "pilot":
                sizes = [(role, 256)]
            for role, total in sizes:
                result.append(dict(id=f"{rep:02d}/{cid}/{role}", replicate=rep,
                                   condition=cid, role=role, total=total))
    return result


def namespace(mode, task, start):
    return f"PR-0002/v1/{mode}/{task['replicate']:02d}/{task['condition']}/{task['role']}/{start}"


def rng_for(mode, task, start):
    words = np.frombuffer(hashlib.sha256(namespace(mode, task, start).encode("ascii")).digest(), dtype="<u4")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEEDS[mode], *map(int, words)])))


def generate(mode, task, start, count, trace_path=None):
    if start % MICRO or count < 1 or start + count > task["total"]:
        raise ValueError("Invalid deterministic microbatch range")
    pieces = []
    p = conditions()[task["condition"]]["params"]
    for offset in range(0, count, MICRO):
        size = min(MICRO, count - offset)
        if trace_path is not None and offset==0:
            # Preselected pilot N_M / rep00 / record0 only. Copies consume no RNG.
            captured=[]
            original=kernel.cr_energy
            def capture(yi,yj,params,w,n):
                captured.append((yi[0].copy(),yj[0].copy(),w,n))
                return original(yi,yj,params,w,n)
            kernel.cr_energy=capture
            try:
                data=kernel.simulate_digital(p,rng_for(mode,task,start+offset),size)
            finally:
                kernel.cr_energy=original
            b=io.BytesIO()
            np.savez(b,observed_i=captured[0][0],observed_j=captured[0][1],
                     control_i=captured[1][0],control_j=captured[1][1],
                     warm=np.array(captured[0][2]),n=np.array(captured[0][3]),
                     scalars=np.array([data[c][0] for c in COLUMNS]))
            atomic(trace_path,b.getvalue())
        else:
            data = kernel.simulate_digital(p, rng_for(mode, task, start + offset), size)
        pieces.append(np.column_stack([data[c] for c in COLUMNS]))
    a = np.concatenate(pieces)
    if a.shape != (count, 14) or not np.isfinite(a).all():
        raise ValueError("Non-finite / malformed scores: no records may be excluded")
    return a


def scores(values):
    return np.column_stack((np.abs(values[:, 11]), values[:, 12]))


def cutoff_order(n, alpha):
    q = (n + 1) * (1 - Fraction(str(alpha)))
    return (q.numerator + q.denominator - 1) // q.denominator


def ranks(reference, observed):
    ordered = np.sort(reference, axis=0)
    return np.column_stack([np.searchsorted(ordered[:, j], observed[:, j], side="left")
                            for j in range(2)]).astype(np.uint16)


def fit(reference, calibration):
    n = len(reference)
    r = ranks(reference, calibration)
    def cut(x, alpha):
        k = cutoff_order(len(x), alpha)
        return n if k > len(x) else int(np.partition(x, k-1)[k-1])
    q = np.full((4, 2), n, dtype=np.int64)
    q[0, 0], q[1, 1] = cut(r[:, 0], .01), cut(r[:, 1], .01)
    q[2] = [cut(r[:, j], .005) for j in range(2)]
    q[3] = cut(r.max(axis=1), .01)
    ties = [[int(np.sum(r[:, j] == q[i, j])) for j in range(2)] for i in range(4)]
    return q, dict(rank_cutoffs=q.tolist(), component_ties=ties,
                   joint_max_ties=int(np.sum(r.max(axis=1) == q[3, 0])))


def decide(r, q):
    flags = np.any(r[:, None, :] > q[None, :, :], axis=2)
    bits = np.sum(flags.astype(np.uint8) * (1 << np.arange(4, dtype=np.uint8)), axis=1).astype(np.uint8)
    return flags, bits


def wilson(k, n):
    z = float(stats.norm.ppf(.975))
    p, d = k/n, 1 + z*z/n
    center = (p + z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return center-half, center+half


def decomposition(f, a, b):
    both = int(np.sum(f[:, a] & f[:, b]))
    gain = int(np.sum(f[:, a] & ~f[:, b]))
    loss = int(np.sum(~f[:, a] & f[:, b]))
    neither = len(f)-both-gain-loss
    if gain-loss != int(f[:, a].sum())-int(f[:, b].sum()):
        raise AssertionError("Paired count conservation failed")
    return dict(both=both, gained=gain, lost=loss, neither=neither, difference=(gain-loss)/len(f))


def interval(x, critical, bounds=(-1., 1.)):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or len(x) < 2:
        raise ValueError("Outer-repeat series incomplete")
    sd = float(x.std(ddof=1))
    mean, se = float(x.mean()), sd/math.sqrt(len(x))
    lo, hi = (mean-critical*se, mean+critical*se) if sd else (None, None)
    return dict(mean=mean, sd=sd, mcse=se, lower=lo, upper=hi,
                display_lower=None if lo is None else max(bounds[0], lo),
                display_upper=None if hi is None else min(bounds[1], hi),
                q10=float(np.quantile(x, .1, method="linear")),
                q50=float(np.quantile(x, .5, method="linear")),
                q90=float(np.quantile(x, .9, method="linear")),
                positive=int(np.sum(x > 0)), zero=int(np.sum(x == 0)), negative=int(np.sum(x < 0)))


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                system=platform.system(), machine=platform.machine(), byteorder=sys.byteorder)


def specification(mode):
    return dict(version=VERSION, mode=mode, seed=SEEDS[mode], micro=MICRO, chunk=CHUNK,
                card_sha256=CARD_SHA, frozen_sha256=FROZEN_SHA, columns=list(COLUMNS),
                conditions=conditions(), tasks=tasks(mode), primary=list(map(list, PRIMARY)),
                tolerance=TOLERANCE, fixed_limits=dict(cell_seconds=2700, active_seconds=14400,
                                                       rss_bytes=4*GIB, workspace_bytes=8*GIB, zip_bytes=2*GIB))


def chunk_path(root, task, start):
    return Path(root)/"chunks"/task["id"]/f"{start:09d}.npz"


def save_chunk(root, task, start, values, manifest_hash, seconds):
    p = chunk_path(root, task, start)
    if p.exists():
        raise FileExistsError("Refusing to overwrite completed evidence")
    values = np.ascontiguousarray(values, dtype="<f8")
    meta = dict(task=task, start=start, count=len(values), manifest_hash=manifest_hash,
                values_sha256=sha_bytes(values.tobytes()), generation_seconds=seconds, at=stamp())
    b = io.BytesIO()
    np.savez(b, values=values, metadata=np.array(canonical(meta)))
    atomic(p, b.getvalue())


def read_chunk(p, task, start, manifest_hash):
    with np.load(p, allow_pickle=False) as z:
        if set(z.files) != {"values", "metadata"}:
            raise ValueError("Unexpected checkpoint fields")
        a, m = z["values"], json.loads(str(z["metadata"]))
    count = min(CHUNK, task["total"]-start)
    if (m["task"] != task or m["start"] != start or m["count"] != count or
        m["manifest_hash"] != manifest_hash or a.shape != (count, 14) or a.dtype != np.dtype("<f8") or
        not np.isfinite(a).all() or sha_bytes(a.tobytes()) != m["values_sha256"]):
        raise ValueError(f"Corrupt or mismatched checkpoint: {p}")
    return a, m


def audit_chunks(root, spec, mh):
    expected = {chunk_path(root, t, s): (t, s) for t in spec["tasks"] for s in range(0, t["total"], CHUNK)}
    actual = set((Path(root)/"chunks").rglob("*.npz"))
    if actual - set(expected):
        raise ValueError("Unexpected / duplicate record IDs")
    count, timings = 0, []
    for p in sorted(actual):
        a, m = read_chunk(p, *expected[p], mh)
        count += len(a)
        timings.append(m)
    return count, timings


def load_task(root, t, mh):
    return np.concatenate([read_chunk(chunk_path(root, t, s), t, s, mh)[0]
                           for s in range(0, t["total"], CHUNK)])


def write_csv(path, rows):
    if not rows:
        return
    out = io.StringIO(newline="")
    w = csv.DictWriter(out, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
    atomic(path, out.getvalue().encode())


def summarize(root, spec, mh, budget_check=lambda: None):
    """No performance analysis until every planned confirmation record exists."""
    count, timing = audit_chunks(root, spec, mh)
    if spec["mode"] != "confirm" or count != 10999800:
        raise ValueError("Confirmation analysis requires all 50 repeats, never a pilot/partial run")
    out = Path(root)/"analysis"
    lookup = {t["id"]: t for t in spec["tasks"]}
    rates, diffs, pats, cutrows, overlap = [], [], [], [], []
    for rep in range(REPEATS):
        for group in ("N_S", "N_M", "N_L", "N_R10"):
            budget_check()
            def load(cid, role):
                return scores(load_task(root, lookup[f"{rep:02d}/{cid}/{role}"], mh))
            ref, cal = load(group, "reference"), load(group, "calibration")
            q, model = fit(ref, cal)
            write_json(out/"calibration"/f"{rep:02d}_{group}.json", model)
            for mi, method in enumerate(METHODS):
                for j in range(2):
                    cutrows.append(dict(replicate=rep, group=group, method=method, component=j,
                                        cutoff=int(q[mi, j]), ties=model["component_ties"][mi][j]))
            for cid, c in conditions().items():
                if c["group"] != group:
                    continue
                role = {"normal": "normal_eval", "fault": "fault_eval", "structural": "structural_eval"}[c["kind"]]
                r = ranks(ref, load(cid, role))
                f, bits = decide(r, q)
                b = io.BytesIO(); np.savez(b, ranks=r, flags=bits)
                atomic(out/"decisions"/f"{rep:02d}_{cid}.npz", b.getvalue())
                hist = np.bincount(bits, minlength=16)
                pats.append(dict(replicate=rep, condition=cid, n=len(f),
                                 **{f"pattern_{j:02d}": int(hist[j]) for j in range(16)}))
                if c["kind"]=="normal":
                    overlap.append(dict(replicate=rep,group=group,n=len(f),
                        M_CR_both=int(np.sum(f[:,0]&f[:,1])),M_CR_union=int(np.sum(f[:,0]|f[:,1])),
                        bonferroni_component_overlap=int(np.sum((r[:,0]>q[2,0])&(r[:,1]>q[2,1])))))
                for mi, method in enumerate(METHODS):
                    k, n = int(f[:, mi].sum()), len(f)
                    lo, hi = wilson(k, n)
                    rates.append(dict(replicate=rep, condition=cid, group=group, kind=c["kind"],
                                      method=method, n=n, alarms=k, rate=k/n, wilson_lower=lo, wilson_upper=hi))
                for a, d in PAIRS:
                    diffs.append(dict(replicate=rep, condition=cid, group=group, kind=c["kind"],
                                      A=METHODS[a], D=METHODS[d], n=len(f), **decomposition(f, a, d)))
    rate_summary, difference_summary, primary, fpr = [], [], [], []
    critical = float(stats.t.ppf(.975, 49))
    for cid, c in conditions().items():
        for m in METHODS:
            values = [r["rate"] for r in rates if r["condition"] == cid and r["method"] == m]
            rec = dict(condition=cid, group=c["group"], kind=c["kind"], method=m,
                       **interval(values, critical, (0., 1.)))
            rate_summary.append(rec)
            if c["kind"] == "normal":
                lower = None if rec["sd"] == 0 else rec["mean"]-float(stats.t.ppf(1-.05/16, 49))*rec["mcse"]
                fpr.append(dict(rec, audit_lower=lower, above_nominal=None if lower is None else lower > .01))
        for a, d in PAIRS:
            rows = [r for r in diffs if r["condition"] == cid and r["A"] == METHODS[a] and r["D"] == METHODS[d]]
            difference_summary.append(dict(condition=cid, A=METHODS[a], D=METHODS[d],
                                           gained_mean=float(np.mean([r["gained"]/r["n"] for r in rows])),
                                           lost_mean=float(np.mean([r["lost"]/r["n"] for r in rows])),
                                           **interval([r["difference"] for r in rows], critical)))
    for i, (cid, a, d, direction) in enumerate(PRIMARY):
        vals = [r["difference"] for r in diffs if (r["condition"], r["A"], r["D"]) == (cid, a, d)]
        rec = interval(vals, float(stats.t.ppf(1-.05/28, 49)))
        outcome = "unresolved"
        if rec["lower"] is not None:
            sign = 1 if rec["lower"] > 0 else -1 if rec["upper"] < 0 else 0
            if sign:
                outcome = ("supported" if sign == direction else "refuted") if direction else ("A_higher" if sign > 0 else "D_higher")
        primary.append(dict(id=f"C{i+1:02d}", condition=cid, A=a, D=d, expected=direction,
                            outcome=outcome, **rec))
    threshold_summary = []
    for g in ("N_S", "N_M", "N_L", "N_R10"):
        for m in METHODS:
            for j in range(2):
                v = [r["cutoff"] for r in cutrows if (r["group"], r["method"], r["component"]) == (g,m,j)]
                threshold_summary.append(dict(group=g, method=m, component=j,
                    q10=float(np.quantile(v,.1)), q50=float(np.quantile(v,.5)), q90=float(np.quantile(v,.9))))
    for name, rows in (("rates_by_repeat", rates), ("paired_by_repeat", diffs), ("patterns", pats),
                       ("thresholds", cutrows), ("threshold_summary", threshold_summary),
                       ("rates_summary", rate_summary), ("differences_summary", difference_summary),
                       ("primary_14", primary), ("fpr_audit_16", fpr)):
        write_csv(out/f"{name}.csv", rows)
    h1 = [primary[i-1]["outcome"] for i in (1,2,5,6,9,10)]
    write_csv(out/"normal_tail_overlap.csv",overlap)
    failure_log=read_json(Path(root)/"state.json").get("failures",[])
    unique_failed={(x["task"]["id"],x["start"]):x.get("attempted_records",0)
                   for x in failure_log if x.get("task") is not None and x.get("start") is not None}
    write_json(out/"inference.json", dict(H1_all_six_supported=all(x=="supported" for x in h1),
        primary=primary, fpr=fpr, scientific_exclusions=0, planned_records=count,
        technical_failure_events=len(failure_log),unique_records_in_failed_attempts=sum(unique_failed.values()),
        failed_attempt_record_fraction=sum(unique_failed.values())/count,
        warning="Separate approximate t-interval families; nominal, not matched empirical FPR; no automatic adoption."))
    budget_check()
    plot_results(out, rate_summary, difference_summary, timing)
    atomic(out/"REPORT.md", ("# PR-0002 confirmation output\n\nAll 50 repeats completed. "
        "See primary_14.csv and fpr_audit_16.csv together. Gains and losses are in paired_by_repeat.csv. "
        "Intervals average over independent calibration repetitions; zero-SD intervals are unavailable. "
        "Structural-control alarms are not fault-detection successes. No manuscript has been changed.\n").encode())


def plot_results(out, rates, diffs, timing):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for sign in ("MINUS", "PLUS"):
        fig, ax = plt.subplots(figsize=(7,4))
        for m in METHODS:
            rows = [next(r for r in rates if r["condition"] == f"G_{s}_{sign}" and r["method"] == m) for s in ("S","M","L")]
            yerr=[0. if r["lower"] is None else (r["upper"]-r["lower"])/2 for r in rows]
            ax.errorbar([.512,2.048,8.192], [r["mean"] for r in rows],yerr=yerr,fmt="o-",label=m,capsize=3)
        ax.set(xlabel="Window (s)", ylabel="Calibration-averaged detection rate", ylim=(0,1),
               title=f"Gain {sign.lower()}; nominal system FPR 1%")
        ax.legend(); fig.tight_layout(); fig.savefig(out/f"gain_{sign.lower()}.png", dpi=150); plt.close(fig)
    chosen=[r for r in diffs if r["condition"] in ("G_L_MINUS","G_L_PLUS","PRIVATE_M") and r["A"]=="J"]
    fig, ax=plt.subplots(figsize=(9,4))
    x=np.arange(len(chosen))
    ax.bar(x-.18,[r["gained_mean"] for r in chosen],.36,label="Gained")
    ax.bar(x+.18,[-r["lost_mean"] for r in chosen],.36,label="Lost")
    ax.set_xticks(x,[r["condition"]+" vs "+r["D"] for r in chosen],rotation=35,ha="right")
    ax.set_ylabel("Fraction of paired records"); ax.legend(); fig.tight_layout()
    fig.savefig(out/"rescue_loss.png",dpi=150);plt.close(fig)
    fig, ax=plt.subplots(figsize=(7,4))
    ids=list(conditions()); vals=[sum(t["generation_seconds"] for t in timing if t["task"]["condition"]==c) for c in ids]
    ax.barh(ids,vals);ax.set_xlabel("Generation + scoring CPU wall time (s)")
    fig.tight_layout();fig.savefig(out/"cost.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for mi,m in enumerate(METHODS):
        rows=[r for r in rates if r["kind"]=="normal" and r["method"]==m]
        ax.errorbar(np.arange(4)+(mi-1.5)*.14,[r["mean"] for r in rows],
                    yerr=[0. if r["lower"] is None else (r["upper"]-r["lower"])/2 for r in rows],
                    fmt="o",label=m,capsize=3)
    ax.axhline(.01,color="black",linestyle="--")
    ax.set_xticks(range(4),["N_S","N_M","N_L","N_R10"])
    ax.set_ylabel("Mean FPR (pointwise 95% intervals)");ax.legend();fig.tight_layout()
    fig.savefig(out/"fpr.png",dpi=150);plt.close(fig)


def rss_bytes():
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def archive(root, status, finalize=None):
    root = Path(root)
    dest = root.parent/f"{root.name}_{status}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}.zip"
    index = {}
    with zipfile.ZipFile(dest.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for p in sorted(root.rglob("*")):
            if p.is_symlink():
                raise ValueError("Symlink in run directory")
            if p.is_file() and not p.name.endswith(".tmp") and p.name not in (".lock", "state.json"):
                name = p.relative_to(root).as_posix()
                index[name] = dict(size=p.stat().st_size, sha256=sha_file(p))
                z.write(p,name)
        if finalize is not None:
            finalize()  # Charge bulk packaging before freezing the archived state.
        p=root/"state.json"
        if p.exists():
            index["state.json"]=dict(size=p.stat().st_size,sha256=sha_file(p))
            z.write(p,"state.json")
        z.writestr("_archive_index.json",canonical(index))
    os.replace(dest.with_suffix(".tmp"),dest)
    return dest


def restore(archive_path, destination):
    """Validate every byte first; extract to staging; never overwrite an old run."""
    dest=Path(destination)
    if dest.exists():
        raise FileExistsError("Restore destination exists; use same-runtime resume or a fresh directory")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive_path) as z:
        members=z.infolist(); names=[m.filename for m in members]
        if len(names)!=len(set(names)) or len(names)>40000 or sum(m.file_size for m in members)>8*GIB:
            raise ValueError("Duplicate members / archive size limit")
        for m in members:
            p=PurePosixPath(m.filename)
            if (p.is_absolute() or ".." in p.parts or "\\" in m.filename or
                str(p)!=m.filename or (m.external_attr>>16)&0o170000==0o120000 or m.is_dir()):
                raise ValueError("Unsafe ZIP path or symlink")
        idx=json.loads(z.read("_archive_index.json"))
        if set(idx)!=set(names)-{"_archive_index.json"}:
            raise ValueError("Archive inventory mismatch")
        with tempfile.TemporaryDirectory(prefix="pr0002_restore_",dir=dest.parent) as tmp:
            stage=Path(tmp)/"run";stage.mkdir()
            for name, expected in idx.items():
                target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
                h=hashlib.sha256(); size=0
                with z.open(name) as src, target.open("wb") as out:
                    for block in iter(lambda:src.read(1024*1024),b""):
                        h.update(block);size+=len(block);out.write(block)
                if h.hexdigest()!=expected["sha256"] or size!=expected["size"]:
                    raise ValueError("Archive checksum mismatch")
            os.replace(stage,dest)
    return dest


@contextmanager
def run_lock(root):
    # OS lock is released even after a hard-killed runtime; no stale PID override.
    import fcntl
    with (Path(root)/".lock").open("a") as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A second runner is using this output") from exc
        try:
            yield
        finally:
            fcntl.flock(f,fcntl.LOCK_UN)


def notify_download(path, enabled=True):
    print("Saved ZIP:",path,flush=True)
    if "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ:
        from IPython.display import display, FileLink
        display(FileLink(str(path)))
        if enabled:
            try:
                from google.colab import files
                files.download(str(path))
            except Exception as exc:
                print("Automatic download did not start:",exc,". Use Colab Files > Download.")


def check_sources(bundle):
    if sha_file(kernel.__file__)!=KERNEL_SHA:
        raise ValueError("Frozen scientific kernel changed")
    if sha_bytes(bundle["PR-0002.md"].encode())!=CARD_SHA:
        raise ValueError("Frozen research card changed")
    if bundle["fusion_confirmation_pr0002_v1.py"]!=Path(__file__).read_text():
        raise ValueError("Running source does not match bundled source")


def pilot_report(root, spec, mh, cell_hash):
    count,timing=audit_chunks(root,spec,mh)
    if count!=8192:
        raise ValueError("Incomplete pilot")
    from pr0002_checks import check_trace
    trace_check=check_trace(Path(root)/"traces"/"pilot_00_N_M_record0.npz")
    totals={c:sum(t["total"] for t in tasks("confirm") if t["condition"]==c) for c in conditions()}
    measured=[]
    for cid in conditions():
        ts=[t for t in timing if t["task"]["condition"]==cid and t["task"]["replicate"]==1]
        measured.append(dict(condition=cid,count=sum(t["count"] for t in ts),
                             seconds=sum(t["generation_seconds"] for t in ts)))
    estimate=sum(r["seconds"]/r["count"]*totals[r["condition"]] for r in measured)
    raw_bytes=sum(p.stat().st_size for p in (Path(root)/"chunks").rglob("*.npz"))
    # Conservative uncompressed projection plus decisions/metadata, not a compression guarantee.
    projected_zip=raw_bytes/count*10999800+200*1024**2
    report=dict(status="PILOT_COMPLETE",records=count,performance_estimates=False,
        environment=environment(),cell_payload_sha256=cell_hash,source_hashes=read_json(Path(root)/"manifest.json")["source_hashes"],
        generation_estimate_seconds=estimate,planning_seconds=estimate*1.5+600,
        projected_zip_bytes=projected_zip,peak_rss_bytes=rss_bytes(),timings=measured,
        limits_ok=estimate*1.5+600<=14400 and projected_zip<=2*GIB and rss_bytes()<=4*GIB,
        trace_check=trace_check,main_authorized=False,acceptance="implementation checks only; no power/FPR conclusions")
    write_json(Path(root)/"pilot_receipt.json",report)
    return report


def run(mode, base, bundle, cell_hash, approval="", pilot_receipt=None, restore_zip=None,
        auto_download=True, cell_seconds=2700, max_new_chunks=None):
    """Colab entry. Main has two gates: pilot receipt and explicit human run token."""
    if "google.colab" not in sys.modules and "COLAB_RELEASE_TAG" not in os.environ:
        raise RuntimeError("Pilot/confirmation are Colab-only. Use the separate small local unit tests.")
    check_sources(bundle)
    spec=specification(mode)
    env=environment()
    sources={name:sha_bytes(text.encode()) for name,text in bundle.items()}
    if mode=="confirm":
        if approval!="RUN PR-0002 CONFIRM":
            raise PermissionError("Confirmation needs separate human approval; default is pilot only")
        if not pilot_receipt or pilot_receipt.get("status")!="PILOT_COMPLETE" or not pilot_receipt.get("limits_ok"):
            raise ValueError("A completed, within-budget Colab pilot receipt is required")
        if (pilot_receipt["environment"]!=env or pilot_receipt["source_hashes"]!=sources or
            pilot_receipt["cell_payload_sha256"]!=cell_hash):
            raise ValueError("Pilot code/environment differs: reaccept before confirmation")
    manifest=dict(specification=spec,environment=env,source_hashes=sources,
                  cell_payload_sha256=cell_hash,pilot_receipt=pilot_receipt if mode=="confirm" else None)
    mh=sha_bytes(canonical(manifest).encode())
    root=Path(base)/f"fusion_confirmation_pr0002_v1_{mode}_{mh[:12]}"
    if restore_zip:
        with zipfile.ZipFile(restore_zip) as z:
            if json.loads(z.read("manifest.json"))!=manifest:
                raise ValueError("Resume ZIP code/settings/environment mismatch; nothing extracted")
        restore(restore_zip,root)
    root.mkdir(parents=True,exist_ok=True)
    with run_lock(root):
        mf=root/"manifest.json"
        if mf.exists() and read_json(mf)!=manifest:
            raise ValueError("Run manifest/environment mismatch; no implicit migration")
        if not mf.exists():
            write_json(mf,manifest)
            for name,txt in bundle.items():atomic(root/"sources"/name,txt.encode())
        statepath=root/"state.json"
        state=read_json(statepath) if statepath.exists() else dict(active_seconds=0.,sessions=0,status="NEW",failures=[])
        state["sessions"]+=1
        start=time.monotonic();previous=float(state["active_seconds"])
        status="PARTIAL";new=0;result=None;failure=None
        def elapsed():return previous+time.monotonic()-start
        def check():
            if elapsed()>=14400:raise TimeoutError("Cumulative 4-hour budget: human review required")
            if time.monotonic()-start>=cell_seconds:raise TimeoutError("Safe cell time limit")
            if rss_bytes()>4*GIB:raise TimeoutError("RSS budget exceeded: human review required")
            if shutil.disk_usage(root).free<2*GIB:raise TimeoutError("Insufficient disk for safe ZIP")
            if sum(p.stat().st_size for p in Path(base).rglob("*") if p.is_file())>8*GIB:
                raise TimeoutError("8 GiB workspace budget exceeded; preserve backups and review")
        try:
            if state["status"]=="NEW" and shutil.disk_usage(root).free<8*GIB:
                raise TimeoutError("Need 8 GiB free workspace before starting")
            if not (root/"checks.json").exists():
                from pr0002_checks import deterministic_checks
                write_json(root/"checks.json",deterministic_checks())
            if read_json(root/"checks.json").get("status")!="PASS":
                raise ValueError("Implementation check receipt is not PASS")
            count,_=audit_chunks(root,spec,mh)
            state.update(committed_records=count,planned_records=sum(t["total"] for t in spec["tasks"]))
            print(f"{mode.upper()}: {count}/{sum(t['total'] for t in spec['tasks'])} committed records; CPU; no Drive",flush=True)
            for t in spec["tasks"]:
                for s in range(0,t["total"],CHUNK):
                    p=chunk_path(root,t,s)
                    if p.exists():continue
                    check()
                    if max_new_chunks is not None and new>=max_new_chunks:raise TimeoutError("Requested checkpoint pause")
                    t0=time.monotonic()
                    trace_path=(root/"traces"/"pilot_00_N_M_record0.npz" if
                                mode=="pilot" and t["replicate"]==0 and t["condition"]=="N_M" and s==0 else None)
                    values=generate(mode,t,s,min(CHUNK,t["total"]-s),trace_path)
                    save_chunk(root,t,s,values,mh,time.monotonic()-t0)
                    new+=1;count+=len(values)
                    state.update(active_seconds=elapsed(),status="RUNNING",committed_records=count)
                    write_json(statepath,state)
                    if new%20==0:print(f"Committed {count} records; active {elapsed()/60:.1f} min",flush=True)
            check()
            if mode=="pilot":
                result=pilot_report(root,spec,mh,cell_hash);status="PILOT_COMPLETE"
            else:
                summarize(root,spec,mh,check);status="COMPLETE"
        except (KeyboardInterrupt,TimeoutError) as exc:
            state["pause_reason"]=str(exc) or "Keyboard interrupt"
            print("PAUSED:",state["pause_reason"],flush=True)
        except Exception:
            failure=traceback.format_exc();status="FAILED"
            failed_task,failed_start=locals().get("t"),locals().get("s")
            state["failures"].append(dict(at=stamp(),error=failure,task=failed_task,start=failed_start,
                attempted_records=min(CHUNK,failed_task["total"]-failed_start) if failed_task is not None and failed_start is not None else 0))
            print(failure,flush=True)
        finally:
            state.update(status=status,active_seconds=elapsed(),updated_at=stamp(),new_chunks=new)
            write_json(statepath,state)
            try:
                def finalize():
                    # Five seconds conservatively budgeted for the tiny ZIP footer/state write.
                    # This charged value travels inside the ZIP, so a fresh runtime cannot reset it.
                    state.update(active_seconds=elapsed()+5.,packaging_tail_charge_seconds=5.)
                    write_json(statepath,state)
                dest=archive(root,status,finalize)
                if elapsed()>state["active_seconds"]:
                    print("Packaging tail exceeded 5 s reserve. Review elapsed budget before resuming.")
                if dest.stat().st_size>2*GIB:print("ZIP exceeds 2 GiB: review operating budget before continuing.")
                notify_download(dest,auto_download)
            except Exception:
                print("ZIP/save failed. Raw checkpoints remain at",root,flush=True)
                raise
        if failure:raise RuntimeError("Technical failure retained in state.json; do not discard failed records")
        return dict(root=str(root),archive=str(dest),status=status,result=result)
