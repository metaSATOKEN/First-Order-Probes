# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Reuse frozen plot blocks, change labels only, and verify numeric artist data.

No generator, calibration fit, or performance inference is run. Historical
scripts and figures stay untouched; only versioned manuscript copies are saved.
"""
from pathlib import Path
import ast
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/odug_manuscript_labels_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figure"
CHECK = ROOT / "audits/verification/2026-10-19_manuscript_ja_v5_en_v2_figures.json"
CONF = ROOT / "numerics/analysis/pr0012_64bf0f2c7697_confirmation_v1"
EXPL = ROOT / "numerics/analysis/pr0012_64bf0f2c7697_exploration_v2"
COMP = ROOT / "numerics/analysis/pr0009_9adfa7a8c6e0_compression_v1"
SOURCES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track(path):
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return path


def pick(rows, **kw):
    values = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    assert len(values) == 1
    return values[0]


def plot_block(script, function, target, env):
    """Execute only the inspected original subplot-to-save block, without I/O."""
    source = track(ROOT / "numerics/scripts" / script)
    tree = ast.parse(source.read_text())
    body = next(n.body for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    block = None
    captured = []
    def capture(fig, name):
        assert name == target
        captured.append(fig)
    for node in body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "plt.subplots":
            block = []
        if block is not None:
            block.append(node)
        if (block and isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "save"
                and ast.literal_eval(node.value.args[-1]) == target):
            local = dict(env, np=np, plt=plt, pick=pick, save=capture)
            exec(compile(ast.Module(body=block, type_ignores=[]), str(source), "exec"), local)
            assert len(captured) == 1
            return captured[0]
    raise AssertionError(target)


def numeric_payload(fig):
    """All plotted numeric primitives, data limits and scales; exclude labels/layout."""
    out = []
    for ax in fig.axes:
        item = {"limits": [list(ax.get_xlim()), list(ax.get_ylim())],
                "scales": [ax.get_xscale(), ax.get_yscale()],
                "patches": [], "lines": [], "collections": [], "images": []}
        for patch in ax.patches:
            item["patches"].append([float(patch.get_x()), float(patch.get_y()),
                                    float(patch.get_width()), float(patch.get_height())])
        for line in ax.lines:
            item["lines"].append([np.asarray(line.get_xdata()).tolist(), np.asarray(line.get_ydata()).tolist()])
        for col in ax.collections:
            item["collections"].append(np.asarray(col.get_offsets()).tolist())
        for image in ax.images:
            item["images"].append({"array": np.asarray(image.get_array()).tolist(),
                                    "clim": list(image.get_clim())})
        out.append(item)
    return out


def finish(fig, name, replacements, title):
    before = numeric_payload(fig)
    labels = []
    for text in fig.findobj(match=Text):
        old = text.get_text()
        if old in replacements:
            text.set_text(replacements[old])
            labels.append([old, replacements[old]])
    old_title = fig._suptitle.get_text()
    fig.suptitle(title)
    labels.append([old_title, title])
    fig.tight_layout()
    fig.canvas.draw()
    after = numeric_payload(fig)
    assert before == after, "Label changes altered plotted numeric data"
    paths = []
    for ext in ("png", "svg"):
        dest = OUT / f"{name}.{ext}"
        assert not dest.exists(), f"Preserve completed figure: {dest}"
        fig.savefig(dest, dpi=170, bbox_inches="tight", facecolor="white")
        paths.append({"path": str(dest.relative_to(ROOT)), "sha256": sha(dest)})
    plt.close(fig)
    encoded = json.dumps(before, sort_keys=True, allow_nan=False).encode()
    return {"name": name, "numeric_artist_sha256_before": hashlib.sha256(encoded).hexdigest(),
            "numeric_artist_sha256_after": hashlib.sha256(json.dumps(after, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "axes": len(before), "numeric_values_unchanged": True,
            "label_changes": labels, "outputs": paths}


def main():
    assert not CHECK.exists(), "Preserve completed verification"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans", "svg.fonttype": "none"})
    common = dict(G=["B_L", "K2_M", "K200_L", "V05_L"], E=["normal", "GM05", "GP05", "P02"],
                  N=[20000, 5000, 5000, 5000], LABEL=["SUPPORTED", "UNRESOLVED", "EXCEEDED"],
                  COL=["#238d9f", "#707e89", "#c45143", "#e4b65b"])
    support = json.loads(track(CONF / "support_summary.json").read_text())
    with np.load(track(CONF / "outer_arrays.npz"), allow_pickle=False) as f:
        a = {"policy": f["policy"]}
    rows = []
    fig = plot_block("summarize_pr0012_confirmation.py", "figures", "04_V_support.png", dict(common, support=support))
    rows.append(finish(fig, "fig07_normal_validation_v2", {"Outer repeats / 50": "Outer replicates / 50"},
                       "Independent normal validation of fixed W0 calibrations\nFPR classification against the 1.25α tolerance"))
    fig = plot_block("summarize_pr0012_confirmation.py", "figures", "06_recovery_tradeoffs.png", dict(common, a=a))
    rows.append(finish(fig, "fig08_recovery_tradeoffs_v2", {"alarm": "Alarm", "provided quiet / miss": "Non-alarm\n(correct rejection or miss)",
                       "DEFER": "Deferred", "INVALID": "Invalid"},
                       "K2_M / TRI800: alarms, non-alarms, and deferred decisions"))
    data = json.loads(track(EXPL / "results.json").read_text())
    fig = plot_block("plot_pr0012_exploration_addendum.py", "main", "02b_repeat_mixture_qualified.png", {"data": data})
    rows.append(finish(fig, "fig09_replicate_differences_v2", {"Outer repeat": "Outer replicate", "DEFER (5)": "Deferred (5)"},
                       "Recovery and deferral account for observed between-replicate variation\nK2_M / TRI800 / α = 0.01; post-hoc descriptive decomposition"))
    arrays = {k: [] for k in ("cal_top", "inactive")}
    for rep in range(50):
        with np.load(track(COMP / "recomputed" / f"rep_{rep:02d}.npz"), allow_pickle=False) as f:
            for k in arrays:
                arrays[k].append(f[k])
    env = {"a": {k: np.stack(v) for k, v in arrays.items()},
           "G": ["B_M", "B_L", "K2_M", "K200_L", "V05_L", "V15_L"],
           "colors": ["#173f70", "#347fc4", "#bf582f", "#a57eb5", "#218a69", "#bc9034"]}
    fig = plot_block("report_pr0009_compression.py", "main", "03_rank_cap.png", env)
    rows.append(finish(fig, "figS1_rank_cap_v2", {"16: off at alpha 0.1%": "16: no alarms possible at α = 0.1%",
                       "160: off at alpha 1%": "160: no alarms possible at α = 1%"},
                       "Rank threshold = 9999 with strict exceedance: no possible alarm"))
    for path, expected in SOURCES.items():
        assert sha(ROOT / path) == expected
    CHECK.write_text(json.dumps({"status": "PASS_LABEL_ONLY_NUMERIC_ARTISTS", "sources": SOURCES,
                                "figures": rows, "new_scientific_records": 0, "new_performance_inference": False},
                               ensure_ascii=False, indent=2) + "\n")
    print("PASS: four PNG/SVG pairs; original plotting blocks; numeric artist values unchanged")


if __name__ == "__main__":
    main()
