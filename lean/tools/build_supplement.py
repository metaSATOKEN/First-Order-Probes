"""Validate saved real Lean runs and assemble the formal supplement ZIP."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "lean/first_order_probe"
ARCHIVES = {
    "macos": ROOT / "lean/verification/first_order_probe_lean_local_PASS.zip",
    "colab": ROOT / "lean/verification/colab/first_order_probe_lean_PASS_20261015T035501Z.zip",
}
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
TOOLCHAIN = "leanprover/lean4:v4.19.0"
MATHLIB = "c44e0c8ee63ca166450922a373c7409c5d26b00b"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build():
    source = (PROJECT / "FirstOrderProbe.lean").read_bytes()
    text = source.decode()
    declarations = re.findall(r"^#print axioms (FirstOrderProbe\.\w+)$", text, re.M)
    assert len(declarations) == 10 and len(set(declarations)) == 10
    assert not re.search(r"\b(sorry|admit|axiom|unsafe|native_decide)\b", text)
    files = {
        "README.md": (ROOT / "lean/SUPPLEMENT.md").read_bytes(),
        "LICENSE": (ROOT / "LICENSE").read_bytes(),
        "LICENSE-CC-BY-4.0.txt": (ROOT / "LICENSE-CC-BY-4.0.txt").read_bytes(),
    }
    runs = {}
    manifests = []
    for platform, archive in ARCHIVES.items():
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            record = json.loads(z.read("verification.json"))
            assert record["status"] == "PASS" and record["all_proofs_checked"] is True
            assert record["source_sha256"] == digest(source)
            assert record["lean_toolchain"] == TOOLCHAIN
            assert record["mathlib_commit"] == MATHLIB
            assert len(record["commands"]) == 7
            assert all(c["returncode"] == 0 for c in record["commands"])
            for name in ("FirstOrderProbe.lean", "README.md", "lakefile.toml", "lean-toolchain"):
                data = z.read("project/" + name)
                assert data == (PROJECT / name).read_bytes(), (platform, name)
                files["project/" + name] = data
            manifests.append(z.read("project/lake-manifest.json"))
            log = z.read("logs/07_kernel_check.log").decode()
            reports = re.findall(r"'(FirstOrderProbe\.\w+)' depends on axioms: \[([^]]*)\]", log)
            assert len(reports) == 10
            assert {name for name, _ in reports} == set(declarations)
            assert set(record["theorem_axioms"]) == set(declarations)
            assert not re.search(r"sorryAx|\berror\b|\bwarning\b", log)
            for name, axioms in reports:
                actual = {s.strip() for s in axioms.split(",") if s.strip()}
                assert actual == set(record["theorem_axioms"][name]) == ALLOWED
            assert z.getinfo("project/FirstOrderProbe.olean").file_size > 0
            prefix = f"verification/{platform}/"
            files[prefix + archive.name] = archive.read_bytes()
            files[prefix + "verification.json"] = z.read("verification.json")
            for command in record["commands"]:
                name = "logs/" + command["log"]
                files[prefix + name] = z.read(name)
            start = datetime.fromisoformat(record["started_utc"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(record["finished_utc"].replace("Z", "+00:00"))
            runs[platform] = {key: record[key] for key in (
                "status", "all_proofs_checked", "platform", "started_utc", "finished_utc",
                "lean_version_output", "theorem_axioms",
            )}
            runs[platform].update(elapsed_seconds=(end - start).total_seconds(),
                                  original_zip_sha256=digest(archive.read_bytes()))
    assert manifests[0] == manifests[1]
    files["project/lake-manifest.json"] = manifests[0]
    for suffix in ("py", "ipynb"):
        name = f"colab/first_order_probe_lean_one_cell.{suffix}"
        files[name] = (ROOT / "lean" / name).read_bytes()
    manifest = dict(lean_toolchain=TOOLCHAIN, mathlib_commit=MATHLIB,
                    source_sha256=digest(source), dependency_manifest_sha256=digest(manifests[0]),
                    checked_declarations=declarations, runs=runs,
                    scope="Algebraic core only; see README.md and project/README.md.")
    files["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    files["CHECKSUMS.sha256"] = "".join(
        f"{digest(data)}  {name}\n" for name, data in sorted(files.items())
    ).encode()
    target = ROOT / "output/supplements/First-Order_Probe_Lean_Supplement.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for line in z.read("CHECKSUMS.sha256").decode().splitlines():
            expected, name = line.split("  ", 1)
            assert digest(z.read(name)) == expected
    print(f"Validated two real PASS runs, 10 declarations each; packaged {len(files)} files.")
    print(target)


if __name__ == "__main__":
    build()
