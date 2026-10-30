# Verification records: macOS and Google Colab

Both real end-to-end runs passed all ten Lean declarations and their axiom
audits. The Lean source, project configuration, and resolved dependency
manifest are identical across the two runs.

| Run | Platform | Total elapsed time | Kernel check | Result |
|---|---|---:|---:|---|
| Local verification | macOS arm64 | 173 s | 13.06 s | PASS, 10 declarations |
| Google Colab | Linux x86-64, glibc 2.39 | 99 s | 13.00 s | PASS, 10 declarations |

Total elapsed times are differences between the recorded UTC timestamps.
They describe these individual runs, including setup; they are not
performance benchmarks. Both runs used Lean 4.19.0 and mathlib commit
`c44e0c8ee63ca166450922a373c7409c5d26b00b`. The common Lean-source SHA-256 is
`6c88044eb777d6ac07cf8d467fc421ecec6fb30de6a88b80a2dc2d8432dd2e23`.

## Local run

- [`first_order_probe_lean_local_PASS.zip`](../../docs/OMITTED_MATERIALS.md#item-ea0c25f4ab45)
  contains the project, resolved dependency manifest, compiled output, and all
  seven execution logs.
- [`logs/07_kernel_check.log`](logs/07_kernel_check.log) is the compiler's axiom
  report, extracted for convenient inspection.
- [`verification.json`](verification.json) records the platform, toolchain,
  source hash, exit codes, durations, and per-declaration axiom lists.

## Google Colab run

- [`colab/first_order_probe_lean_PASS_20261015T035501Z.zip`](../../docs/OMITTED_MATERIALS.md#item-0076b82da9e6)
  is the unmodified archive downloaded by the author after the Colab run.
- [`colab/verification.json`](colab/verification.json) records the successful
  run from 2026-10-15 03:53:22 UTC to 03:55:01 UTC.
- [`colab/logs/07_kernel_check.log`](colab/logs/07_kernel_check.log) contains
  the compiler's axiom report; the adjacent logs preserve every setup step.

The archive integrity, source hashes, dependency manifests, successful exit
codes, and all ten axiom reports were checked against the local run. The
reports use only `propext`, `Classical.choice`, and `Quot.sound`; no `sorryAx`
or proof placeholders occur. The generated runner also has separate mocked
success/failure packaging tests in `lean/tests/test_colab_bundle.py`.

The temporary toolchains and library caches used for the local verification
were removed after its records were saved. The formalization scope and the
mapping to the paper are in the [project documentation](../first_order_probe/README.md).

The [distribution archive](../../docs/OMITTED_MATERIALS.md#item-52530dfd8816)
collects the project, Colab runner, both original result archives, readable
execution records, and checksums.
