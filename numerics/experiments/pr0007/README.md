# pr0007: readable frozen experiment

Ordinary source files replace the historical compressed launch format. Scientific members retain their original bytes. The entry defaults to a read-only check; it never starts a simulation or installs dependencies without `--run`.

1. Inspect [bundle_manifest.json](bundle_manifest.json), the sources below, and [settings.json](settings.json).
2. Run `python3 -B numerics/experiments/pr0007/run_colab.py --check` from the repository root.
3. For an explicitly authorized new/resumed run, upload/unpack the complete repository in Colab, select the original CPU/GPU runtime, and use [run_colab.ipynb](run_colab.ipynb). The start switch defaults to false. Keep Drive unmounted.

[launch_colab.py](launch_colab.py) preserves the original orchestration, reading authenticated source files instead of decompressing embedded code. Restore settings, safety time limits, environment pins, checkpoints and download behavior remain experiment-specific; consult the historical protocol before running. Actual Colab/browser execution has not been retested by this packaging change.

Payload/member identity: `eaa2dd2e093c16e05ae1623d5aec13c76002fb643c211fd73243eddb56dbcd17`. This is an integrity checksum, not an authorship signature. Historical cell ID: `af385f6b95739a08`.

## Sources

- [pr0004_methods.py](sources/pr0004_methods.py)
- [pr0004_spec.py](sources/pr0004_spec.py)
- [pr0006_io.py](sources/pr0006_io.py)
- [pr0006_methods.py](sources/pr0006_methods.py)
- [pr0006_runner.py](sources/pr0006_runner.py)
- [pr0006_spec.py](sources/pr0006_spec.py)
- [pr0007_download.py](sources/pr0007_download.py)
- [pr0007_io.py](sources/pr0007_io.py)
- [pr0007_launcher.py](sources/pr0007_launcher.py)
- [pr0007_methods.py](sources/pr0007_methods.py)
- [pr0007_runner.py](sources/pr0007_runner.py)
- [pr0007_spec.py](sources/pr0007_spec.py)
- [probe_backend_check_v1.py](sources/probe_backend_check_v1.py)
- [probe_check_io.py](sources/probe_check_io.py)
- [probe_check_launcher.py](sources/probe_check_launcher.py)

For the current paper's mechanism, reading order and equation map, start at [PR-0012 FIX1](../../../numerics/experiments/pr0012_fix1/README.md). The [reproduction guide](../../../docs/REPRODUCIBILITY.md) separates light checks from full experiments.
