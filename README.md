# Readout and Calibration-Update Audits for First-Order Probes

Research manuscripts, scientific code, figures, and saved analysis results by **Kentaro Sato**.

This is a local distribution package for an **unpublished working paper**. The current manuscript versions are English v2 and Japanese v5, dated 19 October 2026. No journal acceptance, archival DOI, or public data deposit is implied. The package was assembled on 27 October 2026; its scientific results are unchanged.

## Start here

| Goal | Entry point |
|---|---|
| Read the paper | [English v2](Calibration_Update_Audits_for_First_Order_Probes_EN_v2.md) · [日本語 v5](Calibration_Update_Audits_for_First_Order_Probes_JA_v5.md) |
| Inspect the figures | [Figure index](figure/README.md) |
| Check the saved primary results without simulation | [Reproduction guide](docs/REPRODUCIBILITY.md#check-the-included-results) |
| Read the confirmation code and match it to the paper | [Readable PR-0012 FIX1 source, reading order, and equation map](numerics/experiments/pr0012_fix1/README.md) |
| Run the confirmation experiment in Colab | [Thin, check-first notebook](numerics/experiments/pr0012_fix1/run_colab.ipynb) · [Python entry](numerics/experiments/pr0012_fix1/run_colab.py) |
| Find experiments, reports, and code | [Numerical evidence index](numerics/README.md) |
| Understand what data are included | [Data availability](docs/DATA_AVAILABILITY.md) |
| Inspect the limited formal supplement | [Lean scope and saved checks](lean/SUPPLEMENT.md) |

日本語で読む場合は、まず日本語 v5 と数値資料の索引を参照してください。公開コードの主入口は、展開済みPython20本と読み順・式対応のREADMEです。実行は薄いColab用Notebookから明示的に開始します。従来の[Onecell](numerics/colab/pr0012_mixture_recovery_b50_fix1_one_cell.py)は補助として保持しています。既存の実行ガイドは当時の記録として収録しており、現在の完了状況は本 README と再現手順にまとめています。

## Research question

Two calibrated sensors can produce different normal records because their response times differ. This study separates the physical consistency of those records from threshold calibration and from the decision to adopt, withhold, or replace a calibration after source statistics change.

The paper develops three linked audits:

1. **Readout:** trace how signed moment (M) and cross-relation (CR) statistics become linear-fusion (LF) scores and alarms, including finite-rank suppression and sensitivity tradeoffs.
2. **Candidate versus adopted calibration:** separate detection of a suspicious update batch from the false alarms and sensitivity of thresholds fitted to it. Balanced gain contamination and added private input expose different failure modes.
3. **False-alarm evidence, sensitivity, and coverage:** account separately for continued decisions, deferral, and recovery with independent normal data.

The PR-0012 confirmation experiment contains 50 independent outer replicates and 13 fixed primary contrasts: 12 support their specified directions and R05 remains unresolved after the planned multiplicity adjustment. The accompanying exploratory analyses retain their exploratory status. The paper reports lost detections and deferred decisions alongside gains; it does not establish uniform superiority of a detector or policy.

## Package layout

```text
.
├── Calibration_Update_Audits_for_First_Order_Probes_EN_v2.md
├── Calibration_Update_Audits_for_First_Order_Probes_JA_v5.md
├── figure/                 # figures used by the current manuscripts
├── numerics/
│   ├── experiments/pr0012_fix1/  # readable frozen sources + thin Colab entry
│   ├── colab/              # auxiliary frozen one-cell programs and notebooks
│   ├── scripts/            # historical simulation, scoring, and analysis code
│   ├── tests/              # historical technical tests
│   ├── tools/              # historical bundle builders
│   ├── theory/             # instrument and fusion notes
│   └── analysis/           # selected complete analysis sets and summaries
├── audits/prereg/          # historical fixed protocols, portable links
├── lean/                   # limited algebraic supplement
├── tools/                  # package and saved-primary-result checks
├── docs/                   # reproduction and availability notes
└── provenance/             # hashes, frozen protocol originals, checks
```

Historical directory names are retained so code and evidence remain traceable. The parent research archive, old manuscripts, PDFs, raw return ZIPs, and private workflow logs are outside this distribution. References to omitted historical materials lead to an explicit [availability catalog](docs/OMITTED_MATERIALS.md), not to a purported public copy.

## Quick checks

From this directory, with Python 3.11 or newer:

```bash
python3 tools/verify_package.py
python3 -B numerics/experiments/pr0012_fix1/run_colab.py --check
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_primary_results.py
python -B numerics/experiments/pr0012_fix1/tests/test_readable_fix1.py
```

The package and bundle integrity checks use only the standard library. The primary-result check recomputes the 13 primary intervals/classifications and paired-count identities from included outer-replicate aggregates. The readable-source tests replay saved technical arrays and artificial statistics. None of these commands runs a time-series simulation. The dependency ranges are not a lockfile; historical environment records remain in the evidence directories.

## Citation and license

If you use this work, please cite the manuscript and identify the manuscript/code version and any experiment ID used. Suggested citation:

> Sato, Kentaro. *Readout and Calibration-Update Audits for First-Order Probes: False Alarms, Detection Sensitivity, and Decision Coverage under Source Shifts and Calibration Contamination*. Unpublished manuscript, English v2 / Japanese v5, 19 October 2026.

Machine-readable metadata are provided in [CITATION.cff](CITATION.cff). Add a persistent archive identifier to your citation when one becomes available.

- Original manuscripts, explanatory documents, and figures: [CC BY 4.0](LICENSE-CC-BY-4.0.txt).
- Original software, including Python, notebook code, and Lean sources: [Apache License 2.0](LICENSE).
- Included original numerical result files are distributed under CC BY 4.0 to the extent copyright applies. Third-party components retain their own terms; see [NOTICE](NOTICE).

The manuscript's acknowledgments describe LLM assistance and the author's responsibility for the work.
