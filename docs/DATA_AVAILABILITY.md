# Data availability

This package includes the current manuscripts, their figures, original scientific programs, frozen protocol text, and nine selected saved-analysis sets. Machine-readable JSON/NPZ/CSV files, condition tables, and figures are included across the selected conditions, including losses, failures, and unresolved comparisons.

## Included data

- Outer-replicate aggregates and full primary comparisons for PR-0012.
- Selected complete saved-analysis outputs supporting the readout, contamination, validation-size, and recovery arguments. The [numerical index](../numerics/README.md) lists their inferential roles.
- Runtime/receipt/validation metadata and source hashes associated with these analyses, with machine-local path metadata made portable where needed.
- Scientific code and self-contained Colab programs for generating data under the fixed specifications.

The included arrays support aggregate-level consistency checks. They do not contain every original record-level score or generated time series. Some simulation workflows themselves retain scores rather than every transient time-series array.

## Download the original archives

The **42 original return/checkpoint ZIPs** listed in [RAW_ARCHIVES.json](RAW_ARCHIVES.json) are available as assets of [Release v1.0](https://github.com/metaSATOKEN/First-Order-Probes/releases/tag/v1.0), issued on 3 November 2026. They are separate from the Git repository and its automatic **Source code (zip)** / **Source code (tar.gz)** downloads.

The full data download is approximately **14.7 GB**. The Release contains **50 attached files**: 41 unchanged ZIPs, five ordered binary parts reconstructing the PR-0004 ZIP, and four helper files. Every attached file is below 1,000,000,000 bytes. The `.part001`–`.part005` files are pieces of one ZIP, not independent ZIP archives.

Start with [RAW_DATA_README.md](https://github.com/metaSATOKEN/First-Order-Probes/releases/download/v1.0/RAW_DATA_README.md). Download these helpers alongside the required data assets:

- [RAW_DATA_MANIFEST.json](https://github.com/metaSATOKEN/First-Order-Probes/releases/download/v1.0/RAW_DATA_MANIFEST.json): original paths, sizes, SHA-256 values, and ordered parts.
- [SHA256SUMS.txt](https://github.com/metaSATOKEN/First-Order-Probes/releases/download/v1.0/SHA256SUMS.txt): checksums for the data and other helpers.
- [restore_raw_archives.py](https://github.com/metaSATOKEN/First-Order-Probes/releases/download/v1.0/restore_raw_archives.py): inspectable verification and restoration utility, using Python 3.9+ and the standard library.

With the full set in one directory:

```sh
python3 restore_raw_archives.py --verify-only
python3 restore_raw_archives.py --output restored-archives
```

The helper checks transported parts and reconstructed original hashes, restores the original logical paths, and refuses to overwrite different existing files. It does not extract ZIP members or execute archived code. The attached guide also describes downloading a subset with `--archive`, disk-space requirements, and safe reanalysis in a separate working copy. The [reproduction guide](REPRODUCIBILITY.md#reanalyze-the-original-raw-returns) explains how these archives relate to the historical analysis programs.

## Scope and provenance

The deposit preserves successful runs, pilots, repeated final indexes, and the earlier PR-0012 technical-failure return. Repeated indexes are not extra independent replicates; the failed preflight is not a completed confirmation run. Follow the attached data guide and the relevant reports for their roles. The original workflows retain scores and metadata, but not necessarily every transient simulated time series.

The Release's source snapshot is commit `b14829f6fb3911fd70b23213fb9f0347b5faeacf`. Its documentation and the status field of the retained [archive inventory](RAW_ARCHIVES.json) describe the pre-deposit state. This current availability note and the Release manifest supply the download location without rewriting those historical records or moving the `v1.0` tag. Original archive names, sizes, and hashes are unchanged.

The larger working archive also holds earlier manuscripts, older analysis versions, and private workflow logs that are outside this deposit. [Omitted-material references](OMITTED_MATERIALS.md) identify those historical targets. Availability of the selected archives enables saved-score reanalysis; it does not turn every historical script into a single-command portable workflow or replace the narrower aggregate checks in the reproduction guide.

## Access and citation

Access follows repository visibility. **While the repository is private, authorized GitHub access is required** to view or download its Release assets. No DOI has been assigned. Cite the manuscript, Release `v1.0`, and the experiment/run identifiers used. Original numerical data and explanatory documents are under [CC BY 4.0](../LICENSE-CC-BY-4.0.txt) to the extent copyright applies; original code is under [Apache-2.0](../LICENSE), with third-party notices preserved.
