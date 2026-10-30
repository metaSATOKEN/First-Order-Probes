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
PAYLOAD = "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"
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
        raw = zlib.decompress(base64.b64decode(PAYLOAD))
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
