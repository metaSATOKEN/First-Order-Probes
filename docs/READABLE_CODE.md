# Readable source distribution

The current main branch distributes ordinary Python/Lean files, per-experiment metadata, and explicit check/start entry points. Compressed self-contained launch cells and their builders are omitted from current and future distributions. The historical v1.0 tag and earlier Git history are preserved and still contain that former format. Original execution archives retain their original source evidence and hashes.

The scientific members are unchanged. Historical same-name modules are not interchangeable: always use an experiment's complete frozen bundle. The active entry for the current paper is [PR-0012 FIX1](../numerics/experiments/pr0012_fix1/README.md), whose reading order maps M, CR, LF, calibration and policy equations to the functions.

| Experiment | Read first | Python | Colab |
|---|---|---|---|
| algebra | [README](../lean/experiments/algebra/README.md) | [check/start](../lean/experiments/algebra/run_colab.py) | [notebook](../lean/experiments/algebra/run_colab.ipynb) |
| covariance | [README](../numerics/experiments/covariance/README.md) | [check/start](../numerics/experiments/covariance/run_colab.py) | [notebook](../numerics/experiments/covariance/run_colab.ipynb) |
| pr0002_pilot | [README](../numerics/experiments/pr0002_pilot/README.md) | [check/start](../numerics/experiments/pr0002_pilot/run_colab.py) | [notebook](../numerics/experiments/pr0002_pilot/run_colab.ipynb) |
| pr0002_main | [README](../numerics/experiments/pr0002_main/README.md) | [check/start](../numerics/experiments/pr0002_main/run_colab.py) | [notebook](../numerics/experiments/pr0002_main/run_colab.ipynb) |
| backend_check | [README](../numerics/experiments/backend_check/README.md) | [check/start](../numerics/experiments/backend_check/run_colab.py) | [notebook](../numerics/experiments/backend_check/run_colab.ipynb) |
| pr0004 | [README](../numerics/experiments/pr0004/README.md) | [check/start](../numerics/experiments/pr0004/run_colab.py) | [notebook](../numerics/experiments/pr0004/run_colab.ipynb) |
| pr0006 | [README](../numerics/experiments/pr0006/README.md) | [check/start](../numerics/experiments/pr0006/run_colab.py) | [notebook](../numerics/experiments/pr0006/run_colab.ipynb) |
| pr0007 | [README](../numerics/experiments/pr0007/README.md) | [check/start](../numerics/experiments/pr0007/run_colab.py) | [notebook](../numerics/experiments/pr0007/run_colab.ipynb) |
| pr0009 | [README](../numerics/experiments/pr0009/README.md) | [check/start](../numerics/experiments/pr0009/run_colab.py) | [notebook](../numerics/experiments/pr0009/run_colab.ipynb) |
| pr0012_fix1 | [README](../numerics/experiments/pr0012_fix1/README.md) | [check/start](../numerics/experiments/pr0012_fix1/run_colab.py) | [notebook](../numerics/experiments/pr0012_fix1/run_colab.ipynb) |

## Checks and execution

From the repository root, use Python 3.11+:

```bash
python3 -B tools/test_readable_bundles.py
python3 -B numerics/experiments/pr0012_fix1/tests/test_readable_fix1.py
python3 -B tools/check_primary_results.py
python3 -B tools/verify_package.py
```

The first command uses the standard library only. The latter scientific-aggregate checks require `requirements.txt`; none launches a time-series experiment. Every run entry defaults to a read-only integrity check. Full runs require explicit `--run` or the notebook start switch, in the original Colab environment. No Drive mount is needed. Legacy experiments retain their original pause/checkpoint policies; they are not silently upgraded to later policies.

For older bundles, edit the separate `settings.json` before an explicit start. FIX1 exposes restore/download controls through its documented entry and notebook instead. The orchestration in `launch_colab.py` reads authenticated plain source instead of decompressing code. The original source identity and checkpoint cell ID are retained. Runtime-side temporary packages and downloads remain managed by the frozen launcher.

The covariance benchmark and PR-0002 use CPU runtimes; the backend comparison and PR-0004/0006/0007/0009/0012 use GPU runtimes. The Lean supplement uses a CPU runtime and downloads its pinned Lean/mathlib dependencies. PR-0002 retains its historical 45-minute pause policy. PR-0004 retains its historical cumulative archive format, whose completed ZIP can exceed 1 GB; later experiments use noncumulative shards. Changing these historical I/O policies is separate from exposing their source.

## Provenance and archived checks

[READABLE_MIGRATION.json](../provenance/READABLE_MIGRATION.json) lists source-format exclusions and the per-experiment identities. Historical packaging tests that depended on omitted launch cells are preserved as `.py.txt` inspection files; the current executable packaging tests above replace that dependency. Scientific modules and stored outcomes are not rewritten to make historical packaging tests pass. Original protocols and older guides are also retained as explicitly dated evidence.

Integrity hashes detect mismatched files relative to the pinned bundle. They are not signatures, cross-hardware reproducibility guarantees, or evidence of a new scientific run. Actual Colab execution, browser delivery, and runtime-loss recovery were not rerun during this packaging migration.
