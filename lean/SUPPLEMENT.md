# Limited algebraic formal supplement

The [Lean project](first_order_probe/README.md) formalizes ten declarations supporting the algebraic first-order-probe relations developed during this research. It is an optional supplement, not a formalization of the complete current paper.

## Covered scope

The sources verify real/rational identities for continuous- and discrete-time frequency kernels, the implication from stationary covariance-balance equations to covariance closure, the normalized correlation formula, and a lower-bound/equality calculation. The assumed covariance-balance premises are mathematical inputs to those proofs.

The supplement does not formalize spectral integration, stochastic processes, finite-window sampling distributions, experimental code, calibration validity, or the manuscript's statistical conclusions. It also does not establish every bound or convexity claim discussed in earlier research notes.

## Environment and saved evidence

- Lean: `v4.19.0`, as specified in [lean-toolchain](first_order_probe/lean-toolchain).
- mathlib revision: `c44e0c8ee63ca166450922a373c7409c5d26b00b`.
- Verified source SHA-256: `6c88044eb777d6ac07cf8d467fc421ecec6fb30de6a88b80a2dc2d8432dd2e23` for [FirstOrderProbe.lean](first_order_probe/FirstOrderProbe.lean).
- Saved local and Colab checks are indexed in [verification](verification/README.md). Their logs report the standard axioms `propext`, `Classical.choice`, and `Quot.sound`; this is not an axiom-free claim.

The verification records are historical evidence from 15 October 2026. Package assembly checks source and log integrity; it does not install Lean or rerun the kernel. A fresh build needs the pinned Lean/mathlib environment. The [readable Colab entry](experiments/algebra/README.md) loads the plain Lean project and pinned metadata. Its default action checks the files without installing Lean or running the kernel.

Source code is under [Apache-2.0](../LICENSE); explanatory prose is under [CC BY 4.0](../LICENSE-CC-BY-4.0.txt). Lean/mathlib dependencies retain their own licenses.
