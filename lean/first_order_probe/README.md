# Formal algebraic supplement to Paper 1

Kentaro Sato — *An Exact Variance Identity for Two First-Order Probes*

This is a small Lean 4 supplement to the manuscript's mathematical core. It
checks the real rational frequency identities and the scalar consequences of
covariance closure. The human-readable proofs remain in the paper.

## Coverage

| Lean declaration | Mathematical statement | Paper location |
|---|---|---|
| `continuous_pointwise_identity` | The continuous first-order real cross kernel is the response-time-weighted sum of the two power kernels | Theorem 1, spectral proof |
| `variance_identity_of_balance` | The given stationary covariance-balance equation implies `(t+s)c = t vi + s vj` | Theorem 1, time-domain proof |
| `compatibility_of_identity` | Positive response times and variances make the closed covariance positive and give the normalized formula with `abs c` | Theorem 1 |
| `numerator_lower_bound` | Weighted arithmetic-geometric mean inequality in the paper's variables | Corollary 1b |
| `sharp_lower_bound` | The closed-form compatibility is at least `2 sqrt(t s)/(t+s)` | Corollary 1b |
| `sharp_lower_bound_eq_iff` | Equality holds exactly when `t vi = s vj` | Corollary 1b |
| `discrete_den_pos`, `one_sub_mul_pos` | Denominator positivity in the stable discrete parameter range | Theorem 2, prerequisites |
| `discrete_weights_sum` | The discrete coefficients sum to one | Theorem 2 |
| `discrete_pointwise_identity` | The discrete real cross kernel is the stated two-power linear combination | Theorem 2, spectral proof |

Here `t,s` are the two response times, `vi,vj` the variances, and `w` angular
frequency. In the discrete statements, `a,b` are recursion coefficients in
`(0,1)` and `z = cos(theta)`. The kernel identity actually holds for every real
`z <= 1`, so its formal statement includes the physical cosine range.

The kernels are defined explicitly as real rational functions. Identifying
them with `|H|^2` and `Re(H_i conjugate(H_j))`, and integrating them against a
source spectral measure, are interfaces to the manuscript's analytic proof.
This file does not formalize complex transfer functions, spectral integration,
stationarity, stochastic differential equations, generalized white noise, or
the finite-window sampling law. The lower-bound theorem concerns positive
scalar variances in the closed formula; it does not establish existence of a
source realizing each scalar pair, the upper bound, or the U-law convexity.

## Reproducible environment

- Lean: `leanprover/lean4:v4.19.0`
- mathlib: `c44e0c8ee63ca166450922a373c7409c5d26b00b` (release `v4.19.0`)
- The downloaded `lake-manifest.json` records the transitive dependency commits.
- `FirstOrderProbe.lean` contains the statements, proofs, and `#print axioms`
  commands for all ten declarations.

From this project directory, with the matching Lean toolchain available:

```sh
MATHLIB_NO_CACHE_ON_UPDATE=1 lake update
lake exe cache get Mathlib.Data.Real.Sqrt Mathlib.Tactic.FieldSimp Mathlib.Tactic.Linarith Mathlib.Tactic.Ring Mathlib.Tactic.Positivity
lake env lean -DwarningAsError=true -o FirstOrderProbe.olean FirstOrderProbe.lean
```

The [readable Colab runner](../experiments/algebra/README.md) automates these commands
and installs its toolchain inside a private temporary directory. It does not require
a GPU, a local Lean installation, Google Drive mounting, or any account token.

## Verification record

The runner reports `PASS` only when Lean exits successfully and every listed
declaration passes its axiom audit. It rejects proof placeholders and unapproved
axioms. The allowed foundations are Lean/mathlib's standard `propext`,
`Classical.choice`, and `Quot.sound`; this is not a claim of axiom-free logic.
The source SHA-256, exact dependency commit, platform, compiler version, command
exit codes, and per-theorem axiom lists are recorded in `verification.json`.

The output ZIP contains the project sources, the dependency lock file when
available, compiled output on success, and complete execution logs. A failed
run produces a `FAIL` ZIP with diagnostics, not a successful proof certificate.
The Lean toolchain and mathlib cache are excluded from the ZIP and must be
downloaded again to reproduce it in a fresh environment.

The official installation and library sources are:

- [Lean installation](https://lean-lang.org/install/manual/)
- [Pinned mathlib release](https://github.com/leanprover-community/mathlib4/tree/c44e0c8ee63ca166450922a373c7409c5d26b00b)

## Colab usage

Place the repository in a fresh Colab CPU runtime and use the [check-first notebook](../experiments/algebra/run_colab.ipynb).
Inspect the plain project files, run the integrity check, then explicitly enable
the start switch. Setup downloads compiled tools and libraries and can take several
minutes; the runner prints progress while it waits. Internet access and at least
6 GiB of free temporary disk are required. No data upload or numerical experiment
is involved. The original README bytes are preserved with the frozen bundle metadata.

At completion the cell requests a ZIP download, including on a normal caught
failure. Allow the browser download if prompted; a manual download link is also
displayed. A disconnected or forcibly terminated runtime cannot finish the
packaging step. Check `verification.json`: success means `status = PASS` and
`all_proofs_checked = true`.
