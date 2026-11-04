# pr0004: readable frozen experiment

Ordinary source files replace the historical compressed launch format. Scientific members retain their original bytes. The entry defaults to a read-only check; it never starts a simulation or installs dependencies without `--run`.

1. Inspect [bundle_manifest.json](bundle_manifest.json), the sources below, and [settings.json](settings.json).
2. Run `python3 -B numerics/experiments/pr0004/run_colab.py --check` from the repository root.
3. For an explicitly authorized new/resumed run, upload/unpack the complete repository in Colab, select the original CPU/GPU runtime, and use [run_colab.ipynb](run_colab.ipynb). The start switch defaults to false. Keep Drive unmounted.

[launch_colab.py](launch_colab.py) preserves the original orchestration, reading authenticated source files instead of decompressing embedded code. Restore settings, safety time limits, environment pins, checkpoints and download behavior remain experiment-specific; consult the historical protocol before running. Actual Colab/browser execution has not been retested by this packaging change.

Payload/member identity: `e79e19c73a3ae0c83d5a6434d2f7f60d78074c792a17a608590b47a31a85eb56`. This is an integrity checksum, not an authorship signature. Historical cell ID: `0e4b2ef31e02eacd`.

## Sources

- [pr0004_io.py](sources/pr0004_io.py)
- [pr0004_launcher.py](sources/pr0004_launcher.py)
- [pr0004_methods.py](sources/pr0004_methods.py)
- [pr0004_runner.py](sources/pr0004_runner.py)
- [pr0004_spec.py](sources/pr0004_spec.py)
- [probe_backend_check_v1.py](sources/probe_backend_check_v1.py)
- [probe_check_io.py](sources/probe_check_io.py)
- [probe_check_launcher.py](sources/probe_check_launcher.py)

For the current paper's mechanism, reading order and equation map, start at [PR-0012 FIX1](../../../numerics/experiments/pr0012_fix1/README.md). The [reproduction guide](../../../docs/REPRODUCIBILITY.md) separates light checks from full experiments.
