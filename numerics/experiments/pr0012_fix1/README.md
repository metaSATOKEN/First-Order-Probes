# PR-0012 FIX1: readable source and Colab entry

This is the primary public code entry for the paper's **50-replicate mixture–recovery confirmation**. The Python is ordinary source: open the files in `sources/` to inspect the physical model, statistics, thresholds, policies, and storage. [run_colab.py](run_colab.py) only verifies the bundle and calls its frozen launcher.

日本語の入口：まず「Reading order」と式の対応表から、物理モデル → M・CR → LF → 閾値 → 校正更新の順に読めます。圧縮セルを解読する必要はありません。20本の科学・運転コードは完成版 FIX1 とバイト同一で、以下の説明と薄い入口だけを追加しています。

[Paper, English v2](../../../Calibration_Update_Audits_for_First_Order_Probes_EN_v2.md) · [論文、日本語 v5](../../../Calibration_Update_Audits_for_First_Order_Probes_JA_v5.md) · [Numerical evidence](../../README.md)

## Version and scope

The completed experiment is run `64bf0f2c7697`. Its frozen capsule has cell ID `d41a11833edd6d21` and payload SHA-256 `4045cf2857875c9d0978a19961e3daea38a382303418f2267572023eb1c3789f`.

All **28 logical members**, including **20 Python files**, match that capsule exactly. [bundle_manifest.json](bundle_manifest.json) maps each logical member to its readable file and hash. The FIX1 launcher, runner, and added preflight checker are present together. The older copies in `numerics/scripts/` remain historical sources; do not add that directory to the import path when running this experiment.

The [one-cell Python](../../colab/pr0012_mixture_recovery_b50_fix1_one_cell.py) and its [notebook](../../colab/pr0012_mixture_recovery_b50_fix1_one_cell.ipynb) remain unmodified auxiliary launch formats. Their numerical kernels, seeds, role separation, thresholds, strict comparisons, checkpoint identities, and download logic are the same as this readable bundle. The new wrapper adds verification and an explicit start switch. This packaging preserves the implementation; it makes no promise of bitwise-identical results across different hardware or numerical libraries.

Execution remains **Google Colab GPU only**. Reading, hashing, and the saved-data technical tests can be done locally. A general CPU, cluster, or other-notebook port would require a separate change to the Colab runtime and delivery layer.

## Reading order

Each row answers a different question. Read the numerical path first; the browser and archive code can wait.

| Order | Question | File and useful entry points |
|---|---|---|
| 1 | What is fixed, and which data may inform which decision? | [pr0012_spec.py](sources/pr0012_spec.py): `COUNTS`, `ROLES`, `conditions`, `recipe`, `primary`, `specification`. The selected physical conditions originate in [pr0007_spec.py](sources/pr0007_spec.py) and [pr0006_spec.py](sources/pr0006_spec.py). |
| 2 | How do source and sensor records become M and CR? | [probe_backend_check_v1.py](sources/probe_backend_check_v1.py): `coefficients`, `stationary`, `cpu_filter`, `GPU`, `calculate`; [pr0012_input.py](sources/pr0012_input.py): `make_inputs`. |
| 3 | Where do the LF directions and weights come from? | [pr0004_methods.py](sources/pr0004_methods.py): `fit_reference`; [pr0006_methods.py](sources/pr0006_methods.py): `fit`, `signed`, `ranks`, `order_index`. |
| 4 | How do scores become alarms and update decisions? | [pr0012_methods.py](sources/pr0012_methods.py): `calibrate`, `decide`, `gates`, `evidence`, `select`, `policy_states`, `accumulate`, `aggregate`. |
| 5 | How are independent roles assembled and evaluated? | [pr0012_runner.py](sources/pr0012_runner.py): `mixture`, `readout_audit`, and the worker flow. Read the calls to the methods above rather than starting with its runtime loop. |
| 6 | What did FIX1 change? | [pr0012_preflight.py](sources/pr0012_preflight.py): `boundaries`, `compare_readouts`, `audit`; [technical amendment A01](../../../audits/prereg/PR-0012_A01_preflight_boundary_fix.md). |
| 7 | How do resumption, archives, and downloads work? | [pr0012_launcher.py](sources/pr0012_launcher.py), [pr0012_io.py](sources/pr0012_io.py), [pr0012_storage.py](sources/pr0012_storage.py), [pr0012_download.py](sources/pr0012_download.py). |

The remaining inherited modules provide configuration and runtime utilities: [pr0004_spec.py](sources/pr0004_spec.py), [pr0006_io.py](sources/pr0006_io.py), [pr0006_runner.py](sources/pr0006_runner.py), [pr0008_shadow_methods.py](sources/pr0008_shadow_methods.py), [probe_check_io.py](sources/probe_check_io.py), and [probe_check_launcher.py](sources/probe_check_launcher.py). They are retained under their original names to keep imports and the frozen source identity intact. Some original functions use compact formatting; this edition exposes their source without rewriting the arithmetic.

## Paper equations → implementation

Equation numbers refer to the current English v2 / Japanese v5 manuscripts. “Explanatory” means an identity or interpretation used to explain the algorithm, not an additional runtime estimation step.

| Paper equation | Meaning | Implementation / saved quantity |
|---|---|---|
| (1) | Discrete first-order probe response | `probe_backend_check_v1.coefficients`, `cpu_filter`, `GPU.filt`; filters inside `calculate`. |
| (2) | Observation-minus-control second moments | Nested `moments` in `calculate`; `full_vi`, `full_vj`, `full_c`. Known zero means are used, without sample recentering. |
| (3) | Stationary population balance | Explanatory derivation of (4)–(5); checked through the resulting residual, not estimated as a separate regression. |
| (4) | Discrete-time weights λ and ν | `coefficients`; the `lam` and `nu` values consumed by `calculate`. |
| (5) | Signed moment statistic q_M | Nested `residual` in `calculate`; `R_full` (column 11). |
| (6) | Cross-filter commutation under common input | Explanatory relation; `calculate` forms the two cross-filtered paths. |
| (7) | Noise-corrected CR energy q_C | `calculate`; `CR` (column 12), with `P_i`, `P_j`, `P_ij` retaining its quadratic components. |
| (8) | Gain and private-input fault model | `calculate` applies gain/private input before acquisition noise; conditions and amplitudes come from the fixed specification. |
| (9)–(11) | Physical response directions d_g and d_p | `pr0004_methods.fit_reference`, using reference estimates of `full_vj`, `full_c`, `P_j`, `P_ij`. |
| (12) | Signed LF projections | `pr0006_methods.signed`: centered `(q_M, q_C)` multiplied by fixed reference weights. |
| (13) | Population deflection inequality | Explanatory justification for the linear directions. No evaluation-data optimization is performed by this code. |
| (14) | Regularized, normalized LF weights | `pr0004_methods.fit_reference`; covariance shrinkage ε = 10⁻⁶, followed by a linear solve and normalization. |
| (15) | LF0 finite-reference ranks and alarm | `pr0006_methods.ranks` uses left insertion counts; `pr0012_methods.calibrate` / `decide(method=1)` use the maximum rank and strict threshold. |
| (16) | W0 direct three-branch alarm | `calibrate` / `decide(method=0)` use `(s_g, −s_g, s_p)` with budgets `(α/4, α/4, α/2)`. |
| (17) | Finite-calibration order statistic | `pr0006_methods.order_index` uses exact rational index arithmetic; `pr0012_methods.calibrate` selects the cut. An unreachable order gives an inactive branch. |
| (18) | Candidate-batch gates | `pr0012_methods.gates`; absolute signed-M mean for G-M, combined with one-sided CR mean for G-MC. Frozen cuts are in `pr0012_spec.GATE_CUTS`. |
| (19) | Discrete OU source | `stationary` and `calculate`, with normal innovations from `pr0012_input.make_inputs`. |
| (20) | Alarm, quiet, deferral, invalid accounting | `policy_states`, `accumulate`, `aggregate`; separate all-record and provided-decision denominators. |
| (21) | Rank decision as an effective half-plane | `pr0012_runner.readout_audit`; `pr0012_preflight.boundaries` also uses this relation for technical checks. |
| (22) | Empirical within-/between-path variance | For implementation inspection: [explore_pr0012_saved.py](../../scripts/explore_pr0012_saved.py) and its historical [cross-check](../../scripts/verify_pr0012_exploration.py). Rerunning the cross-check requires the earlier `exploration_v1/results.json`, which is not bundled. These are separate from the frozen experiment launcher and the runnable checks below. |

The half-plane tolerance in FIX1 classifies CPU/GPU **preflight** discrepancies close to an unresolved floating-point boundary. Scientific alarms still use strict `>`; the tolerance is never inserted into `decide`.

## Arrays and decision codes

The backend returns a float64 matrix of shape `(number_of_records, 17)`. Each row describes one finite-window record. This row count is different from the number of time samples within a record.

| Columns, zero-based | Names | Role |
|---|---|---|
| 0–2 | `none_vi`, `none_vj`, `none_c` | Uncorrected moments retained for audit. |
| 3–5 | `diag_vi`, `diag_vj`, `diag_c` | Diagonal-only correction retained for audit. |
| 6–8 | `full_vi`, `full_vj`, `full_c` | Fully noise-corrected moments. |
| 9–11 | `R_none`, `R_diag`, `R_full` | M residuals under the three correction choices; current LF uses column 11. |
| 12 | `CR` | Signed noise-corrected CR statistic; current LF's second coordinate. |
| 13 | `reversal_moment_error` | Retained reversal diagnostic. |
| 14–16 | `P_i`, `P_j`, `P_ij` | Cross-filter moments used in CR and in physical direction estimates. |

The moment/energy columns have signal-squared units. LF scores are normalized projections, with dimensionless scale set by the reference covariance. Corrected CR can be negative after subtraction of the independent noise control; retain that sign when tracing the implementation.

`decide` returns uint8 codes with branch bits 0, 1, 2 for positive gain, negative gain, and private-input directions, and bit 3 for an available decision. Thus `8` is a valid non-alarm and `0` is invalid. A branch name denotes the physical direction used to build a score, not an identified fault label. `histogram` stores eight branch patterns followed by invalid counts. Policy states instead use four categories: `0 = alarm`, `1 = provided quiet`, `2 = defer`, `3 = invalid`.

`aggregate` has a historical field named `provided_miss`; its denominator is **all planned records** (`50 × n`), not only provided decisions. Use its formula, the `coverage` field, and paper equation (20) when interpreting it.

## Data separation and fixed scale

Reference R fixes LF weights and reference ranks. Old calibration F fixes baseline thresholds. Candidate audit A decides whether an update is suspicious; a separate batch U fits its thresholds. Independent recovery C fits replacement thresholds. Normal validation V supports/exceeds/leaves unresolved the false-alarm cap for F and C. Evaluation E measures outcomes and never selects the rule.

The configuration fixes 50 outer replicates, 5 per noncumulative shard; 9,999 reference records; 15,999 records per old calibration, normal audit, update, or recovery batch; 20,000 normal validation and normal evaluation records; and 5,000 evaluation records per fault. Candidate recipes replace fixed numbers in a fixed-size batch. A/U replacement permutations and source streams have distinct role addresses. One replicate contains 479,984 physical records; a full run contains 23,999,200.

These settings are documented in [PR-0012](../../../audits/prereg/PR-0012_mixture_recovery_confirmation.md). Frozen metadata are also preserved as exact bytes under `metadata/`; the three Markdown originals have `.md.txt` suffixes so their archival relative links are not presented as navigable package documentation. Their logical names remain unchanged when passed to the launcher.

## Check without running an experiment

From the public repository root:

```bash
python3 -B numerics/experiments/pr0012_fix1/run_colab.py --check
python3 -B numerics/experiments/pr0012_fix1/tests/test_readable_fix1.py
python3 -B tools/check_primary_results.py
python3 -B tools/verify_package.py
```

The first and last commands use the standard library. The numerical tests and saved-primary checker require the numerical dependencies in [requirements.txt](../../../requirements.txt). Tests replay 1,088 saved CPU/GPU comparison rows and small artificial statistics; they do not create new physical records. The [fixture provenance](tests/fixtures/saved_preflight.json) identifies the original failed preflight archive and its members. Its two rank-boundary flips are deliberately retained for regression coverage.

## Explicit Colab execution

Use [run_colab.ipynb](run_colab.ipynb) after placing the **whole public package** in the Colab runtime, for example by uploading and unpacking a repository archive. Set `EXPERIMENT_DIR` to this directory. The notebook first checks the sources. Its execution switch defaults to `START_FULL_RUN = False`.

A fresh run requires at least **24 GiB of free runtime disk space**, checked at startup; this is disk storage, not RAM or GPU memory. The fresh-run check is distinct from resuming compatible saved state.

To launch later, set that switch to `True` in an authorized GPU runtime. The notebook calls the entry **inside the notebook kernel**, preserving the original browser callbacks. With default settings, an existing compatible runtime resumes automatically; otherwise the frozen launcher discovers matching archives or starts fresh. After runtime loss, set `UPLOAD_RESTORE_ZIPS = True` to choose matching FIX1 ZIPs, or fill `RESTORE_ZIPS` with paths to already uploaded archives. Leave Google Drive unmounted.

The worker retains the original environment pins and float64 preflight, 8-hour active-time safety limit, periodic checkpoints, and automatic noncumulative ZIP delivery. Normal completion produces ten five-replicate shards plus a final index; partial or size-fallback parts can add files. Each ZIP stays below decimal 1 GB. Browser delivery requests and successful saving on your computer are separate events: keep all required shards and the index.

Starting an actual GPU run, testing live browser downloads or runtime-loss recovery, and adapting the code to another execution environment are separate validation steps. The current [technical check record](../../../provenance/READABLE_FIX1_CHECK.json) states exactly which checks were run for this readable package.
