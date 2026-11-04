"""Paste this ENTIRE file into ONE Colab Python cell, then run it."""

import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import traceback
import zipfile
import zlib

# Replaced mechanically by build_colab_bundle.py; no repository access is needed.
PAYLOAD_SHA256 = "14c40f61942ee407ef417154b7cb0b8cfbc054788d2eae9b75a176b38b394c07"

MATHLIB_REV = "c44e0c8ee63ca166450922a373c7409c5d26b00b"
LEAN_VERSION = "leanprover/lean4:v4.19.0"
ELAN_VERSION = "v4.1.2"
THEOREMS = [
    "continuous_pointwise_identity", "variance_identity_of_balance",
    "compatibility_of_identity", "numerator_lower_bound", "sharp_lower_bound",
    "sharp_lower_bound_eq_iff", "discrete_den_pos", "one_sub_mul_pos",
    "discrete_weights_sum", "discrete_pointwise_identity",
]
CACHE_MODULES = [
    "Mathlib.Data.Real.Sqrt", "Mathlib.Tactic.FieldSimp",
    "Mathlib.Tactic.Linarith", "Mathlib.Tactic.Ring", "Mathlib.Tactic.Positivity",
]


def main():
    try:
        from google.colab import files as colab_files
    except ImportError:
        colab_files = None

    # All installs, caches, and build products stay in this run's private directory.
    base = Path("/content") if colab_files is not None else Path(tempfile.gettempdir())
    root = Path(tempfile.mkdtemp(prefix="first_order_probe_lean_", dir=base))
    project = root / "project"
    project.mkdir()
    logs = root / "logs"
    logs.mkdir()
    env = os.environ.copy()
    env["ELAN_HOME"] = str(root / "elan")
    env["XDG_CACHE_HOME"] = str(root / "cache")
    env["MATHLIB_NO_CACHE_ON_UPDATE"] = "1"
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env["PATH"] = str(root / "elan" / "bin") + os.pathsep + env.get("PATH", "")
    status = {
        "status": "RUNNING", "all_proofs_checked": False,
        "lean_toolchain": LEAN_VERSION, "mathlib_commit": MATHLIB_REV,
        "platform": platform.platform(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commands": [], "theorem_axioms": {},
        "scope": "Real rational frequency kernels and scalar covariance consequences; not a formalization of stochastic processes or spectral integration.",
    }
    project_files = {}
    error = None

    def run(args, label, timeout=1800):
        number = len(status["commands"]) + 1
        logfile = logs / f"{number:02d}_{label}.log"
        print(f"[{number}] {label} ...", flush=True)
        start = time.monotonic()
        entry = {"args": [str(a) for a in args], "log": logfile.name}
        status["commands"].append(entry)
        with logfile.open("w", encoding="utf-8") as out:
            proc = subprocess.Popen(args, cwd=project, env=env, stdout=out, stderr=subprocess.STDOUT)
            next_update = 20
            try:
                while proc.poll() is None:
                    elapsed = time.monotonic() - start
                    if elapsed > timeout:
                        raise TimeoutError(f"{label} exceeded {timeout} seconds")
                    if elapsed >= next_update:
                        print(f"    {label}: {int(elapsed)} s elapsed", flush=True)
                        next_update += 20
                    time.sleep(1)
            except BaseException:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
            finally:
                entry["returncode"] = proc.poll()
                entry["seconds"] = round(time.monotonic() - start, 2)
        output = logfile.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            print(output[-6000:], flush=True)
            raise RuntimeError(f"{label} failed with exit code {proc.returncode}")
        print(f"    OK ({entry['seconds']:.1f} s)", flush=True)
        return output

    try:
        print("Paper 1 — Lean algebraic supplement | CPU runtime is sufficient", flush=True)
        print(f"Private working directory: {root}", flush=True)
        raw = _READABLE_RAW
        if hashlib.sha256(raw).hexdigest() != PAYLOAD_SHA256:
            raise RuntimeError("Embedded project payload checksum mismatch")
        project_files = json.loads(raw)
        for name, content in project_files.items():
            if Path(name).name != name:
                raise RuntimeError(f"Unexpected payload path: {name}")
            (project / name).write_text(content, encoding="utf-8")
        source = project_files["FirstOrderProbe.lean"]
        if re.search(r"\b(sorry|admit|axiom|unsafe|native_decide)\b", source):
            raise RuntimeError("Unapproved proof shortcut found in the supplement source")
        status["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        if shutil.disk_usage(root).free < 6 * 1024**3:
            raise RuntimeError("At least 6 GiB of free temporary disk space is required")
        for program in ("git", "curl"):
            if shutil.which(program) is None:
                raise RuntimeError(f"Required program is missing: {program}")
        machine, system = platform.machine().lower(), platform.system()
        if system == "Linux" and machine in ("x86_64", "amd64"):
            arch = "x86_64-unknown-linux-gnu"
        elif system == "Darwin" and machine in ("arm64", "aarch64"):
            arch = "aarch64-apple-darwin"
        else:
            raise RuntimeError(f"Unsupported runtime: {system}/{machine}; use a standard Colab CPU runtime")
        url = f"https://github.com/leanprover/elan/releases/download/{ELAN_VERSION}/elan-{arch}.tar.gz"
        archive = root / "elan.tar.gz"
        run(["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30", url, "--output", str(archive)], "download_elan")
        status["elan_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        installer = root / "elan-init"
        with tarfile.open(archive, "r:gz") as tar:
            member = tar.getmember("elan-init")
            if not member.isfile():
                raise RuntimeError("Unexpected elan installer archive")
            with tar.extractfile(member) as src, installer.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        installer.chmod(0o755)
        run([str(installer), "-y", "--no-modify-path", "--default-toolchain", "none"], "install_private_elan")
        lake = str(root / "elan" / "bin" / "lake")
        run([lake, "update"], "resolve_pinned_dependencies")
        actual_rev = run(["git", "-C", str(project / ".lake/packages/mathlib"), "rev-parse", "HEAD"], "check_mathlib_commit").strip()
        if actual_rev != MATHLIB_REV:
            raise RuntimeError(f"Unexpected mathlib commit: {actual_rev}")
        status["lean_version_output"] = run([lake, "env", "lean", "--version"], "lean_version").strip()
        run([lake, "exe", "cache", "get", *CACHE_MODULES], "download_mathlib_cache")
        result = run([lake, "env", "lean", "-DwarningAsError=true", "-o", "FirstOrderProbe.olean", "FirstOrderProbe.lean"], "kernel_check")
        allowed = {"propext", "Classical.choice", "Quot.sound"}
        for short_name in THEOREMS:
            name = "FirstOrderProbe." + short_name
            match = re.search(re.escape("'" + name + "'") + r" depends on axioms:\s*\[([^\]]*)\]", result, re.S)
            if match:
                axioms = {x.strip() for x in match.group(1).split(",") if x.strip()}
            elif f"'{name}' does not depend on any axioms" in result:
                axioms = set()
            else:
                raise RuntimeError(f"Missing axiom audit: {name}")
            if not axioms <= allowed:
                raise RuntimeError(f"Unapproved axioms for {name}: {sorted(axioms - allowed)}")
            status["theorem_axioms"][name] = sorted(axioms)
        status["status"] = "PASS"
        status["all_proofs_checked"] = True
        print(f"PASS: {len(THEOREMS)} declarations checked by Lean; axiom audit passed.", flush=True)
    except BaseException as exc:
        error = exc
        status["status"] = "FAIL"
        status["error"] = f"{type(exc).__name__}: {exc}"
        (logs / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"FAIL: {status['error']}\nPackaging sources and diagnostic logs.", flush=True)
    finally:
        status["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (root / "verification.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        bundle = root / f"first_order_probe_lean_{status['status']}_{stamp}.zip"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(root / "verification.json", "verification.json")
            for name in [*project_files, "lake-manifest.json", "FirstOrderProbe.olean"]:
                path = project / name
                if path.is_file():
                    z.write(path, "project/" + name)
            for path in sorted(logs.glob("*.log")):
                z.write(path, "logs/" + path.name)
        print(f"Result archive: {bundle}", flush=True)
        if colab_files is not None:
            try:
                colab_files.download(str(bundle))
                print("Download requested. If the browser blocks it, use the link below.", flush=True)
            except Exception as exc:
                print(f"Automatic download could not start: {exc}", flush=True)
            from IPython.display import display, HTML
            import html
            rel = bundle.relative_to(Path("/content"))
            display(HTML(f'<a download href="/files/{html.escape(str(rel))}">Download result ZIP</a>'))
        else:
            print("Outside Colab: the archive is saved locally; no browser download was requested.", flush=True)
    if error is not None:
        raise RuntimeError("Lean verification did not pass. Inspect verification.json and logs in the downloaded ZIP.") from error
    return bundle


if __name__ == "__main__":
    LEAN_RESULT_ZIP = main()
