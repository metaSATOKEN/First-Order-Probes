# pr0002_pilot: readable frozen experiment

Ordinary source files replace the historical compressed launch format. Scientific members retain their original bytes. The entry defaults to a read-only check; it never starts a simulation or installs dependencies without `--run`.

1. Inspect [bundle_manifest.json](bundle_manifest.json), the sources below, and [settings.json](settings.json).
2. Run `python3 -B numerics/experiments/pr0002_pilot/run_colab.py --check` from the repository root.
3. For an explicitly authorized new/resumed run, upload/unpack the complete repository in Colab, select the original CPU/GPU runtime, and use [run_colab.ipynb](run_colab.ipynb). The start switch defaults to false. Keep Drive unmounted.

[launch_colab.py](launch_colab.py) preserves the original orchestration, reading authenticated source files instead of decompressing embedded code. Restore settings, safety time limits, environment pins, checkpoints and download behavior remain experiment-specific; consult the historical protocol before running. Actual Colab/browser execution has not been retested by this packaging change.

Payload/member identity: `3f8eaf10ccbc0a961d32c6173ac39173ced377e9a128fdc90f9b1445e24a90bb`. This is an integrity checksum, not an authorship signature. Historical cell ID: `568b5189bb5d58d5b1f4ba6f7497ebf234783d01e3bd3ebc90c543ca7ebcf80c`.

## Sources

- [covariance_consistency_colab_v1.py](sources/covariance_consistency_colab_v1.py)
- [fusion_confirmation_pr0002_v1.py](sources/fusion_confirmation_pr0002_v1.py)
- [pr0002_checks.py](sources/pr0002_checks.py)

For the current paper's mechanism, reading order and equation map, start at [PR-0012 FIX1](../../../numerics/experiments/pr0012_fix1/README.md). The [reproduction guide](../../../docs/REPRODUCIBILITY.md) separates light checks from full experiments.
