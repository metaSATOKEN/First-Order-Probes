# Reproduction guide

[Overview](../README.md) · [Evidence index](../numerics/README.md)

There are three distinct levels of reproduction. Choose the level required for your task.

## Check the included results

Run these commands from the distribution root using Python 3.11 or newer:

```bash
python3 tools/verify_package.py
python3 -B numerics/experiments/pr0012_fix1/run_colab.py --check
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_primary_results.py
python -B numerics/experiments/pr0012_fix1/tests/test_readable_fix1.py
```

`verify_package.py` checks the distribution manifest, included local Markdown links, file-size limits, executable-source syntax, and forbidden generated/private files. It performs no network access or scientific simulation. The manifest verifies integrity relative to this package, not authorship or an external signature.

`check_primary_results.py` reads the saved PR-0012 outer arrays and primary-result JSON. It recomputes all 13 intervals/classifications using the original reporting function, checks every policy pair-table margin and alarm gain–loss identity, and checks the stored invalid-state counts. This tests consistency of the supplied aggregates and results. It neither regenerates time series nor independently validates the original generator or its scientific assumptions.

Dependencies in [requirements.txt](../requirements.txt) are supported ranges, not a historical environment lock. Environment/runtime records are retained with the saved analyses. A future environment may produce small numerical differences; the checker uses the original tolerances.

The readable-bundle integrity check uses only the standard library and starts no experiment. Its technical suite compares all 28 members to the frozen capsule, exercises the thin entry, and replays saved CPU/GPU arrays plus artificial decision fixtures. See the [reading order and equation map](../numerics/experiments/pr0012_fix1/README.md) and [check record](../provenance/READABLE_FIX1_CHECK.json).

## Generate fresh experimental data in Colab

1. Start a Google Colab GPU runtime. A fresh run requires at least **24 GiB of free runtime disk space**, checked at startup; this is disk storage, not RAM or GPU memory, and is a fresh-run requirement rather than a resumption check. The historical implementation uses float64 CPU/GPU checks and a GPU backend; performance depends on the allocated device.
2. Place the entire public package in runtime storage and open the [thin FIX1 notebook](../numerics/experiments/pr0012_fix1/run_colab.ipynb). Set `EXPERIMENT_DIR` to the unpacked experiment directory. Inspect the [ordinary Python sources](../numerics/experiments/pr0012_fix1/README.md), then run the check cell.
3. Keep the scientific settings unchanged. Set `START_FULL_RUN = True` only when ready for a full/resumed run. The default `RESTORE_ZIPS = []` and `UPLOAD_RESTORE_ZIPS = False` reuse compatible runtime data, discover matching ZIPs, or start fresh. `AUTO_DOWNLOAD = True` requests downloads. The entry is called inside the notebook kernel for browser callbacks. No Google Drive mount is required.
4. Keep all noncumulative SHARD files and the COMPLETE_INDEX. Normal completion produces ten five-replicate shards for 50 replicates; partial checkpoints or size-split parts may add files. Each ZIP is kept below decimal 1 GB by the program. Browser download permission and actual arrival on your computer must be checked separately.
5. If the runtime remains alive, rerun the notebook's execution cell to reuse compatible saved state. If runtime storage has been lost, set `UPLOAD_RESTORE_ZIPS = True` and select the matching FIX1 checkpoints, including required partial dependencies, or list already-uploaded paths in `RESTORE_ZIPS`. See the [dated operational guide](../numerics/PR0012_MIXTURE_RECOVERY_FIX1_COLAB_JA.md) for historical checkpoint details.

The unmodified [one-cell Python](../numerics/colab/pr0012_mixture_recovery_b50_fix1_one_cell.py) and [one-cell notebook](../numerics/colab/pr0012_mixture_recovery_b50_fix1_one_cell.ipynb) remain auxiliary alternatives. In that older entry format, `RESTORE_ZIPS = ""` is the default and `"UPLOAD"` selects ZIPs. Use one entry format per kernel; do not import historical `numerics/scripts` modules into the FIX1 run. Other-environment execution remains outside this package's current support scope.

The early failed preflight run `606fc449db00` contains no completed scientific replicates and is not a restore source for FIX1. The completed evidence analyzed here is `64bf0f2c7697`. A new run may differ in environment metadata, archive names, runtime, and floating-point boundary behavior. Fixed seeds do not make all hardware behavior identical.

The original execution guide includes estimates and “not yet run” statements written before return of the completed data. Those statements are preserved as history; the included confirmation report documents the later completed run. This package preparation did not execute Colab or test a fresh browser download or runtime-loss recovery.

## Reanalyze the original raw returns

Original checkpoint ZIPs are not bundled, and a public raw-data download location has not yet been assigned. See [Data availability](DATA_AVAILABILITY.md). The original analysis scripts are provided, but some expect those ZIPs at fixed root-relative locations, such as `pr0012/` and `Pr0009/`.

When raw returns become available, verify their hashes against [the archive inventory](RAW_ARCHIVES.json). Use a separate working copy for reanalysis: original analysis scripts have fixed output directories and can overwrite existing reports. Do not invoke them over the published saved-result directories. Compare freshly produced outputs to the supplied results after retaining the original copies. Acquisition of the raw archives and a full end-to-end reanalysis are outside the lightweight checks above.

## Figures and formal supplement

Manuscripts reference the included [figure directory](../figure/README.md). Plotted images and vector sources are retained as published assets. Historical drawing scripts are included for inspection; they have dated output paths and may intentionally reject overwriting existing figures. They are not a single-command clean build of all manuscript figures.

The [Lean supplement](../lean/SUPPLEMENT.md) covers a limited set of algebraic identities. Saved kernel-check logs are supplied; a fresh Lean build requires the specified Lean/mathlib environment and network access for dependencies. It is separate from statistical validation of the experiments.

## Provenance and historical links

[Source manifest](../provenance/source_manifest.json) records original and distribution hashes and the applied transformations. Scientific Python/Lean sources and numerical arrays retain their original bytes. Manuscript scientific content is unchanged; local links are made portable, and each public manuscript adds a navigation paragraph linking to the current code reading order and equation map. Both language editions now use V1.0; [version history](VERSION_HISTORY.md) maps the earlier draft labels to this shared version. JSON metadata paths can be normalized without changing numerical values.

Protocols in `audits/prereg/` have portable distribution links and a banner. Their exact frozen source text is separately included in `provenance/frozen_protocols/` so historical hashes embedded in code remain verifiable. A historical link to material outside this selected package leads to the [omission catalog](OMITTED_MATERIALS.md), which identifies the unavailable item explicitly.

For an eventual repository upload, use Git-based transfer rather than assuming every file can be uploaded through the browser: the largest preserved JSON is 26,223,128 bytes, just above the browser's 25 MiB limit. All included files are below the ordinary Git 100 MiB per-file limit. See [GitHub's file-size guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). No repository was initialized or uploaded during this package preparation.
