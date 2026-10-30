# Data availability

This package includes the current manuscripts, their figures, original scientific programs, frozen protocol text, and nine selected saved-analysis sets. Machine-readable JSON/NPZ/CSV files, condition tables, and figures are included across the selected conditions, including losses, failures, and unresolved comparisons.

## Included data

- Outer-replicate aggregates and full primary comparisons for PR-0012.
- Selected complete saved-analysis outputs supporting the readout, contamination, validation-size, and recovery arguments. The [numerical index](../numerics/README.md) lists their inferential roles.
- Runtime/receipt/validation metadata and source hashes associated with these analyses, with machine-local path metadata made portable where needed.
- Scientific code and self-contained Colab programs for generating data under the fixed specifications.

The included arrays support aggregate-level consistency checks. They do not contain every original record-level score or generated time series. Some simulation workflows themselves retain scores rather than every transient time-series array.

## Original returns outside the package

Large original return/checkpoint ZIPs remain in the author's research archive. [RAW_ARCHIVES.json](RAW_ARCHIVES.json) identifies selected original archives by logical filename, size, and SHA-256. It is an inventory, not a download service. A public repository/DOI for these archives is **pending**; no access or persistent availability is claimed here.

Consequently, complete raw-score replay of all historical analyses is not possible from this package alone. The provided Colab bundles permit new runs, and the provided saved aggregates permit the narrower checks described in the [reproduction guide](REPRODUCIBILITY.md).

The larger working archive also holds earlier manuscripts, older analysis versions, raw failure returns, and workflow logs. They are preserved outside this distribution. [Omitted-material references](OMITTED_MATERIALS.md) identify unavailable historical targets without silently replacing them with different scientific evidence.

## Before an external release

Assign a raw-data access policy and, if publishing those archives, a stable deposit and citation. Recheck this availability statement and citation metadata at that time. This local package does not constitute submission, publication, or an external upload.
