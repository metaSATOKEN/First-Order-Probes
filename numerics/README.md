# Numerical code and evidence

[Repository overview](../README.md) · [Reproduction guide](../docs/REPRODUCIBILITY.md) · [Data availability](../docs/DATA_AVAILABILITY.md)

## Main entry points

The confirmation experiment used in the current manuscript is **PR-0012 FIX1**, run ID `64bf0f2c7697`. Its 50 outer replicates independently separate reference, candidate audit, threshold update, normal validation, and evaluation roles. FIX1 repairs a CPU/GPU rank-boundary preflight check; it preserves the scientific rules and strict comparisons.

| Resource | Link |
|---|---|
| Scientific conditions | [PR-0012](../audits/prereg/PR-0012_mixture_recovery_confirmation.md) |
| Technical repair | [PR-0012 A01](../audits/prereg/PR-0012_A01_preflight_boundary_fix.md) |
| Primary code entry | [Readable FIX1 source and equation map](experiments/pr0012_fix1/README.md) · [Thin Colab notebook](experiments/pr0012_fix1/run_colab.ipynb) |
| Auxiliary one-cell format | [Frozen FIX1 Python](colab/pr0012_mixture_recovery_b50_fix1_one_cell.py) · [Notebook](colab/pr0012_mixture_recovery_b50_fix1_one_cell.ipynb) |
| Historical execution instructions | [Japanese FIX1 guide](PR0012_MIXTURE_RECOVERY_FIX1_COLAB_JA.md) |
| Main report | [Confirmation report](analysis/pr0012_64bf0f2c7697_confirmation_v1/PR0012_CONFIRMATION_REPORT_JA.md) |
| Primary results | [13 contrasts](analysis/pr0012_64bf0f2c7697_confirmation_v1/primary13.json) · [Outer arrays](analysis/pr0012_64bf0f2c7697_confirmation_v1/outer_arrays.npz) |
| Scope and roles | [Scope](analysis/pr0012_64bf0f2c7697_confirmation_v1/scope.json) · [Axes](analysis/pr0012_64bf0f2c7697_confirmation_v1/axes.json) |

## Included analysis sets

All selected analysis sets retain their available condition tables and machine-readable results, including adverse and unresolved outcomes. These sets serve different inferential roles; combining them in a distribution does not turn exploratory analyses into independent confirmation.

| Directory | Role in the evidence chain |
|---|---|
| [main_2ae37613a10478c8](analysis/main_2ae37613a10478c8) | Earlier M/CR comparison and saved-score ROC analysis |
| [pr0006…performance_v2](analysis/pr0006_6f23fc7946f8_performance_v2) | Alarm-budget allocation study |
| [pr0009…two_axis_v1](analysis/pr0009_9adfa7a8c6e0_two_axis_v1) | Independent candidate/calibration two-axis experiment |
| [pr0009…compression_v1](analysis/pr0009_9adfa7a8c6e0_compression_v1) | Exploratory readout and rank-cap decomposition |
| [pr0009…v_budget_v1](analysis/pr0009_9adfa7a8c6e0_v_budget_v1) | Exploratory normal-validation sample-size analysis |
| [pr0010…mixture_v1](analysis/pr0010_9adfa7a8c6e0_mixture_v1) | Saved-data mixture/cancellation analysis |
| [pr0011…recovery_v2](analysis/pr0011_9adfa7a8c6e0_recovery_v2) | Saved-data stopping/recovery policy analysis |
| [pr0012…confirmation_v1](analysis/pr0012_64bf0f2c7697_confirmation_v1) | Fixed 50-replicate mixture/recovery confirmation |
| [pr0012…exploration_v2](analysis/pr0012_64bf0f2c7697_exploration_v2) | Post-confirmation explanatory analyses |

Reports are mostly in Japanese. The English manuscript supplies the integrated scientific account. `validation.json` and similar files inside analysis directories describe historical checks at the time of those analyses. The [initial package check](../provenance/VERIFICATION.json) and [readable FIX1 check](../provenance/READABLE_FIX1_CHECK.json) have separate scopes and dates.

## Code organization

- [experiments/pr0012_fix1](experiments/pr0012_fix1/README.md): complete versioned source for the current confirmation, a reading order, equation/column map, thin check-first entry, and saved-array technical tests. All 28 capsule members (20 Python files) retain exact bytes.
- [scripts](scripts/README.md): historical physical simulation/backends, scoring and calibration methods, fixed experiment specifications, saved-data analysis, and figure generation. Sources are copied byte for byte; the PR-0012 launcher and runner here predate FIX1. Use the versioned source above for execution.
- [colab](colab): self-contained frozen bundles. Use the FIX1 bundle for PR-0012; the superseded broken launch bundle and abandoned PR-0004 rescue program are omitted.
- [tests](tests): technical tests from development. Some require original raw archives or local historical fixtures and are not standalone package checks.
- [tools](tools): historical one-cell bundle builders. Rebuilding a bundle changes its source identity; use the frozen bundle when reproducing a recorded run.
- [Instrument specification](theory/M_CR_LF_INSTRUMENT_SPECIFICATION_v1_JA.md) and [quadratic-fusion note](theory/THEORY_NOTE_M_CR_QUADRATIC_FUSION_v1_JA.md): methodological background, with original dates and scope.

For an earlier independent experiment, use the [PR-0009 one-cell program](colab/pr0009_two_axis_a50_one_cell.py) and its [guide](PR0009_TWO_AXIS_COLAB_JA.md). Running any full experiment is a new computational job, not part of the lightweight package check.
