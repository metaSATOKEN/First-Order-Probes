#!/usr/bin/env python3
# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Audit saved PR-0002 pilot evidence. Never generate records or execute ZIP code."""
import argparse
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "numerics/scripts"))
import fusion_confirmation_pr0002_v1 as f
from pr0002_checks import check_trace, deterministic_checks
from verify_pr0002 import protected


def audit(path):
    build = f.read_json(ROOT / "numerics/colab/pr0002_confirmation_build_manifest.json")
    protected_count = protected()
    for name, sha in build["artifacts"].items():
        assert f.sha_file(ROOT / "numerics/colab" / name) == sha
    checks = deterministic_checks()  # Algebra and fixed fixtures, no random generation.
    with tempfile.TemporaryDirectory(prefix="pr0002_saved_pilot_") as tmp:
        root = f.restore(path, Path(tmp) / "restored")
        manifest = f.read_json(root / "manifest.json")
        receipt = f.read_json(root / "pilot_receipt.json")
        state = f.read_json(root / "state.json")
        saved_checks = f.read_json(root / "checks.json")
        assert manifest["specification"] == f.specification("pilot")
        assert manifest["pilot_receipt"] is None
        assert manifest["source_hashes"] == receipt["source_hashes"] == build["sources"]
        assert manifest["cell_payload_sha256"] == receipt["cell_payload_sha256"] == build["cell_id"]
        for name, sha in build["sources"].items():
            assert f.sha_file(root / "sources" / name) == sha
        assert (root / "sources/fusion_confirmation_pr0002_v1.py").read_bytes() == Path(f.__file__).read_bytes()
        assert (root / "sources/pr0002_checks.py").read_bytes() == (ROOT / "numerics/scripts/pr0002_checks.py").read_bytes()
        assert manifest["environment"] == receipt["environment"] == saved_checks["environment"]
        assert saved_checks["status"] == "PASS" and saved_checks["checks"] == checks["checks"]
        assert saved_checks["tolerance"] == f.TOLERANCE
        assert saved_checks["generated_probe_records"] == 0
        mh = f.sha_bytes(f.canonical(manifest).encode())
        assert f"_pilot_{mh[:12]}_PILOT_COMPLETE_" in path.name
        expected = {f.chunk_path(root, t, 0) for t in f.tasks("pilot")}
        assert set((root / "chunks").rglob("*.npz")) == expected
        count, timing = f.audit_chunks(root, manifest["specification"], mh)
        assert count == receipt["records"] == state["committed_records"] == state["planned_records"] == 8192
        assert state["status"] == receipt["status"] == "PILOT_COMPLETE"
        assert state["failures"] == [] and state["sessions"] == 1 and state["new_chunks"] == 32
        assert receipt["performance_estimates"] is False and receipt["main_authorized"] is False
        assert not (root / "analysis").exists()
        assert all(math.isfinite(t["generation_seconds"]) and t["generation_seconds"] > 0 for t in timing)
        trace = check_trace(root / "traces/pilot_00_N_M_record0.npz")
        with np.load(root / "traces/pilot_00_N_M_record0.npz", allow_pickle=False) as z:
            trace_values = z["scalars"]
        t = next(t for t in f.tasks("pilot") if t["id"] == "00/N_M/normal_eval")
        values, _ = f.read_chunk(f.chunk_path(root, t, 0), t, 0, mh)
        np.testing.assert_array_equal(values[0], trace_values)
        moment_error = 0.
        reverse_error = 0.
        for t in f.tasks("pilot"):
            values, _ = f.read_chunk(f.chunk_path(root, t, 0), t, 0, mh)
            p = f.conditions()[t["condition"]]["params"]
            ai, aj = np.exp(-p["dt"] / np.array([p["tau_i"], p["tau_j"]]))
            lam, nu = (1+ai)*(1-aj)/(2*(1-ai*aj)), (1+aj)*(1-ai)/(2*(1-ai*aj))
            for j in range(3):
                calculated = values[:, 3*j+2] - lam*values[:, 3*j] - nu*values[:, 3*j+1]
                np.testing.assert_allclose(calculated, values[:, 9+j], rtol=2e-12, atol=2e-12)
                moment_error = max(moment_error, float(np.max(np.abs(calculated-values[:, 9+j]))))
            reverse_error = max(reverse_error, float(np.max(np.abs(values[:, 13]))))
        assert reverse_error <= 2e-12
        totals = {cid: sum(t["total"] for t in f.tasks("confirm") if t["condition"] == cid)
                  for cid in f.conditions()}
        per_condition = []
        pass_seconds = []
        projections = []
        for rep in (0, 1):
            rows = [t for t in timing if t["task"]["replicate"] == rep]
            pass_seconds.append(sum(t["generation_seconds"] for t in rows))
            projections.append(sum(t["generation_seconds"] / t["count"] * totals[t["task"]["condition"]]
                                   for t in rows))
        for cid in f.conditions():
            t = next(t for t in timing if t["task"]["replicate"] == 1 and t["task"]["condition"] == cid)
            saved = next(t for t in receipt["timings"] if t["condition"] == cid)
            assert saved == dict(condition=cid, count=t["count"], seconds=t["generation_seconds"])
            per_condition.append(dict(condition=cid, pilot_count=t["count"], pilot_seconds=t["generation_seconds"],
                                      main_count=totals[cid], projected_seconds=t["generation_seconds"]/t["count"]*totals[cid]))
        planning = projections[1]*1.5+600
        chunk_bytes = sum(p.stat().st_size for p in expected)
        projected_bytes = chunk_bytes/count*10999800+200*1024**2
        assert math.isclose(projections[1], receipt["generation_estimate_seconds"], rel_tol=1e-14)
        assert math.isclose(planning, receipt["planning_seconds"], rel_tol=1e-14)
        assert projected_bytes == receipt["projected_zip_bytes"]
        assert receipt["limits_ok"] is True
        assert planning <= 14400 and projected_bytes <= 2*f.GIB and receipt["peak_rss_bytes"] <= 4*f.GIB
        assert state["active_seconds"] >= sum(pass_seconds)
        # Freeze the exact future manifest from accepted pilot evidence, without starting it.
        confirmation_manifest = dict(manifest, specification=f.specification("confirm"), pilot_receipt=receipt)
        confirmation_hash = f.sha_bytes(f.canonical(confirmation_manifest).encode())
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            inventory = dict(members=len(z.infolist()), indexed_files=len(json.loads(z.read("_archive_index.json"))),
                             uncompressed_bytes=sum(i.file_size for i in z.infolist()), zip_bytes=path.stat().st_size)
        return dict(status="PASS", scope="Saved-data implementation and budget acceptance; no main run or power/FPR evaluation",
                    pilot_zip=str(path.relative_to(ROOT)), pilot_zip_sha256=f.sha_file(path),
                    manifest_sha256=mh, cell_id=build["cell_id"], source_hashes=build["sources"],
                    accepted_colab_environment=receipt["environment"], audit_environment=f.environment(),
                    inventory=inventory, records=count, scalar_values=count*14, chunks=len(expected),
                    conditions=16, passes=2, failures=0, all_finite=True, complete_record_ids=True,
                    restored_all_files_with_hash_validation=True, trace_check=trace,
                    trace_matches_saved_record_exactly=True, max_moment_recalculation_error=moment_error,
                    max_reversal_moment_error=reverse_error, deterministic_checks=checks,
                    state=state, generation_seconds_by_pass=pass_seconds, projected_generation_seconds_by_pass=projections,
                    per_condition_projection=per_condition, planning_seconds=planning,
                    projected_zip_bytes=projected_bytes, pilot_peak_rss_bytes=receipt["peak_rss_bytes"],
                    budget_ok=True, protected_assets_unchanged=protected_count,
                    main_executed=False, simulation_records_generated_in_audit=0,
                    browser_auto_download_observed=False,
                    confirmation_manifest_sha256=confirmation_hash), confirmation_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; preserve old evidence")
    result, manifest = audit(args.zip.resolve())
    args.output.mkdir(parents=True)
    f.write_json(args.output / "audit.json", result)
    f.write_json(args.output / "accepted_confirmation_manifest.json", manifest)
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_condition_projection", "source_hashes", "deterministic_checks")}, indent=2))


if __name__ == "__main__":
    main()
