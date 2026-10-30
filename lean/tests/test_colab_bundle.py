"""Runner/packaging smoke tests. Mocked runs are NOT Lean proof verification."""

import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
CELL = ROOT / "lean" / "colab" / "first_order_probe_lean_one_cell.py"


class ColabBundleTests(unittest.TestCase):
    def test_payload_and_notebook_match_project(self):
        ns = runpy.run_path(str(CELL), run_name="test_import")
        raw = zlib.decompress(base64.b64decode(ns["PAYLOAD"]))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), ns["PAYLOAD_SHA256"])
        for name, text in json.loads(raw).items():
            self.assertEqual(text, (ROOT / "lean/first_order_probe" / name).read_text())
        notebook = json.loads(CELL.with_suffix(".ipynb").read_text())
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(cells), 1)
        self.assertEqual("".join(cells[0]["source"]), CELL.read_text())
        compile(CELL.read_text(), str(CELL), "exec")

    def run_mocked(self, mode):
        ns = runpy.run_path(str(CELL), run_name="test_import")
        downloads = []
        displayed = []
        with tempfile.TemporaryDirectory(prefix="lean_runner_test_") as tmp:
            run_root = Path(tmp) / "run"
            run_root.mkdir()

            class FakePopen:
                def __init__(self, args, cwd, env, stdout, stderr):
                    self.returncode = 0
                    args = list(map(str, args))
                    text = "MOCK RUNNER TEST — NOT A REAL PROOF CHECK\n"
                    if args[0] == "curl":
                        if mode == "download_failure":
                            self.returncode = 22
                        else:
                            dest = Path(args[args.index("--output") + 1])
                            with tarfile.open(dest, "w:gz") as archive:
                                data = b"mock installer"
                                info = tarfile.TarInfo("elan-init")
                                info.size = len(data)
                                archive.addfile(info, io.BytesIO(data))
                    elif args[-1] == "update":
                        (Path(cwd) / "lake-manifest.json").write_text('{"mock": true}')
                    elif args[0] == "git":
                        text = ns["MATHLIB_REV"] + "\n"
                    elif args[-1] == "--version":
                        text = "MOCK Lean 4.19.0\n"
                    elif args[-1] == "FirstOrderProbe.lean":
                        if mode == "kernel_failure":
                            self.returncode = 1
                            text += "mock kernel error\n"
                        else:
                            ax = "sorryAx" if mode == "axiom_failure" else "propext, Classical.choice, Quot.sound"
                            text = "\n".join(
                                f"'FirstOrderProbe.{n}' depends on axioms: [{ax}]" for n in ns["THEOREMS"]
                            )
                            (Path(cwd) / "FirstOrderProbe.olean").write_bytes(b"MOCK NOT A PROOF")
                    stdout.write(text)
                    stdout.flush()

                def poll(self):
                    return self.returncode

            colab = types.ModuleType("google.colab")
            colab.files = types.SimpleNamespace(download=lambda p: downloads.append(p))
            display = types.ModuleType("IPython.display")
            display.display = lambda x: displayed.append(x)
            display.HTML = lambda x: x
            with patch.dict(sys.modules, {"google.colab": colab, "IPython.display": display}), \
                 patch("tempfile.mkdtemp", return_value=str(run_root)), \
                 patch("subprocess.Popen", FakePopen), \
                 patch("platform.system", return_value="Linux"), \
                 patch("platform.machine", return_value="x86_64"), \
                 patch("platform.platform", return_value="Mock Linux (runner test only)"), \
                 patch("shutil.which", return_value="/mock/bin"), \
                 contextlib.redirect_stdout(io.StringIO()):
                # The manual link uses /content-relative paths in real Colab.
                # Remap only this path operation for the temporary test location.
                original_relative_to = Path.relative_to

                def relative_to(path, other, *args, **kwargs):
                    if str(other) == "/content":
                        return original_relative_to(path, run_root.parent)
                    return original_relative_to(path, other, *args, **kwargs)

                with patch.object(Path, "relative_to", relative_to):
                    if mode == "success":
                        ns["main"]()
                    else:
                        with self.assertRaises(RuntimeError):
                            ns["main"]()
            self.assertEqual(len(downloads), 1)
            self.assertEqual(len(displayed), 1)
            with zipfile.ZipFile(downloads[0]) as archive:
                report = json.loads(archive.read("verification.json"))
                self.assertEqual(report["status"], "PASS" if mode == "success" else "FAIL")
                self.assertEqual(report["all_proofs_checked"], mode == "success")
                self.assertIn("project/FirstOrderProbe.lean", archive.namelist())
                self.assertTrue(any(n.startswith("logs/") for n in archive.namelist()))
                self.assertFalse(any(".lake/" in n or "/elan/" in n for n in archive.namelist()))

    def test_success_packages_and_requests_download(self):
        self.run_mocked("success")

    def test_download_failure_still_requests_diagnostic_download(self):
        self.run_mocked("download_failure")

    def test_kernel_failure_cannot_report_pass(self):
        self.run_mocked("kernel_failure")

    def test_unapproved_axiom_cannot_report_pass(self):
        self.run_mocked("axiom_failure")


if __name__ == "__main__":
    unittest.main()
