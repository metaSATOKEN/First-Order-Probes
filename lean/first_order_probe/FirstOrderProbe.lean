import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity

/-!
# Algebraic core of the first-order probe variance identity

This supplement checks real rational frequency kernels and consequences of the
covariance identity. It does not formalize stochastic processes, Fourier theory,
spectral integration, or the assertion that a physical probe realizes a filter.

For H(t,w) = 1 / (1 - i*w*t), `power t w` is |H|^2 and `cross t s w`
is Re[H(t,w) * conj(H(s,w))]. Their identification with complex transfer
functions is an explicit mathematical interface, not a theorem of this file.
-/

set_option autoImplicit false

namespace FirstOrderProbe

noncomputable def power (t w : ℝ) : ℝ := 1 / (1 + (w * t)^2)

noncomputable def cross (t s w : ℝ) : ℝ :=
  (1 + w^2 * t * s) / ((1 + (w * t)^2) * (1 + (w * s)^2))

/-- The real rational integrand identity in the spectral proof of Theorem 1. -/
theorem continuous_pointwise_identity (t s w : ℝ) (ht : 0 < t) (hs : 0 < s) :
    cross t s w = (t * power t w + s * power s w) / (t + s) := by
  have h1 : 1 + (w * t)^2 ≠ 0 := ne_of_gt (by positivity)
  have h2 : 1 + (w * s)^2 ≠ 0 := ne_of_gt (by positivity)
  have h3 : t + s ≠ 0 := ne_of_gt (add_pos ht hs)
  unfold cross power
  field_simp
  ring

/-- Algebraic closure once the stationary covariance-balance equation is given. -/
theorem variance_identity_of_balance (t s vi vj c : ℝ)
    (ht : 0 < t) (hs : 0 < s)
    (hbalance : (1 / t + 1 / s) * c = vj / t + vi / s) :
    (t + s) * c = t * vi + s * vj := by
  have ht0 : t ≠ 0 := ne_of_gt ht
  have hs0 : s ≠ 0 := ne_of_gt hs
  field_simp at hbalance
  nlinarith [hbalance]

noncomputable def compatibility (t s vi vj : ℝ) : ℝ :=
  (t * vi + s * vj) / ((t + s) * Real.sqrt (vi * vj))

noncomputable def lowerBound (t s : ℝ) : ℝ :=
  2 * Real.sqrt (t * s) / (t + s)

/-- The absolute-covariance definition reduces to the positive closed form. -/
theorem compatibility_of_identity (t s vi vj c : ℝ)
    (ht : 0 < t) (hs : 0 < s) (hi : 0 < vi) (hj : 0 < vj)
    (hid : (t + s) * c = t * vi + s * vj) :
    |c| / Real.sqrt (vi * vj) = compatibility t s vi vj := by
  have hts : 0 < t + s := add_pos ht hs
  have hc : 0 < c := (mul_pos_iff_of_pos_left hts).mp (by rw [hid]; positivity)
  rw [abs_of_pos hc]
  unfold compatibility
  have hcval : c = (t * vi + s * vj) / (t + s) :=
    (eq_div_iff (ne_of_gt hts)).mpr (by nlinarith [hid])
  rw [hcval, div_div]

/-- Weighted arithmetic-geometric mean inequality, in the paper's variables. -/
theorem numerator_lower_bound (t s vi vj : ℝ)
    (ht : 0 < t) (hs : 0 < s) (hi : 0 < vi) (hj : 0 < vj) :
    2 * Real.sqrt (t * s) * Real.sqrt (vi * vj) ≤ t * vi + s * vj := by
  have hA := Real.sq_sqrt (le_of_lt (mul_pos ht hs))
  have hB := Real.sq_sqrt (le_of_lt (mul_pos hi hj))
  have hroot : (Real.sqrt (t * s) * Real.sqrt (vi * vj))^2 = t*s*(vi*vj) := by
    rw [mul_pow, hA, hB]
  have hnonneg : 0 ≤ Real.sqrt (t*s) * Real.sqrt (vi*vj) := by positivity
  have hsum : 0 < t*vi + s*vj := by positivity
  nlinarith [sq_nonneg (t*vi - s*vj)]

/-- Sharp lower bound for the closed-form compatibility. -/
theorem sharp_lower_bound (t s vi vj : ℝ)
    (ht : 0 < t) (hs : 0 < s) (hi : 0 < vi) (hj : 0 < vj) :
    lowerBound t s ≤ compatibility t s vi vj := by
  have hts : 0 < t+s := add_pos ht hs
  have hv : 0 < Real.sqrt (vi*vj) := Real.sqrt_pos.2 (mul_pos hi hj)
  unfold lowerBound compatibility
  apply (div_le_div_iff₀ hts (mul_pos hts hv)).2
  have h := numerator_lower_bound t s vi vj ht hs hi hj
  nlinarith [mul_nonneg (le_of_lt hts) (sub_nonneg.mpr h)]

/-- Complete equality condition: the lower bound is attained iff t*vi = s*vj. -/
theorem sharp_lower_bound_eq_iff (t s vi vj : ℝ)
    (ht : 0 < t) (hs : 0 < s) (hi : 0 < vi) (hj : 0 < vj) :
    compatibility t s vi vj = lowerBound t s ↔ t*vi = s*vj := by
  have hts : t+s ≠ 0 := ne_of_gt (add_pos ht hs)
  have hv : Real.sqrt (vi*vj) ≠ 0 := ne_of_gt (Real.sqrt_pos.2 (mul_pos hi hj))
  have hA := Real.sq_sqrt (le_of_lt (mul_pos ht hs))
  have hB := Real.sq_sqrt (le_of_lt (mul_pos hi hj))
  have hroot : (Real.sqrt (t*s) * Real.sqrt (vi*vj))^2 = t*s*(vi*vj) := by
    rw [mul_pow, hA, hB]
  have hn : 0 ≤ Real.sqrt (t*s) * Real.sqrt (vi*vj) := by positivity
  unfold compatibility lowerBound
  constructor
  · intro heq
    have heq' : t*vi + s*vj = 2 * Real.sqrt (t*s) * Real.sqrt (vi*vj) := by
      have hcross := (div_eq_div_iff (mul_ne_zero hts hv) hts).mp heq
      apply mul_right_cancel₀ hts
      nlinarith [hcross]
    nlinarith [sq_nonneg (t*vi - s*vj)]
  · intro hbal
    have hpos : 0 < t*vi := mul_pos ht hi
    have heq' : t*vi + s*vj = 2 * Real.sqrt (t*s) * Real.sqrt (vi*vj) := by
      nlinarith [sq_nonneg (t*vi - Real.sqrt (t*s) * Real.sqrt (vi*vj))]
    rw [heq']
    apply (div_eq_div_iff (mul_ne_zero hts hv) hts).2
    ring

/-! The discrete kernel is expressed using z = cos(theta). -/

noncomputable def discreteDen (a z : ℝ) : ℝ := 1 + a^2 - 2*a*z

noncomputable def discretePower (a z : ℝ) : ℝ := (1-a)^2 / discreteDen a z

noncomputable def discreteCross (a b z : ℝ) : ℝ :=
  (1-a)*(1-b)*(1+a*b-(a+b)*z) / (discreteDen a z * discreteDen b z)

noncomputable def weight (a b : ℝ) : ℝ := (1+a)*(1-b) / (2*(1-a*b))

theorem discrete_den_pos (a z : ℝ) (ha : 0 < a) (ha1 : a < 1) (hz : z ≤ 1) :
    0 < discreteDen a z := by
  unfold discreteDen
  have h := mul_nonneg (le_of_lt ha) (sub_nonneg.mpr hz)
  have hsq : 0 < (1-a)^2 := sq_pos_of_pos (sub_pos.mpr ha1)
  nlinarith

theorem one_sub_mul_pos (a b : ℝ) (ha : 0 < a) (ha1 : a < 1)
    (_hb : 0 < b) (hb1 : b < 1) : 0 < 1-a*b := by
  have h := mul_lt_mul_of_pos_left hb1 ha
  nlinarith

/-- The discrete weights in Theorem 2 sum to one. -/
theorem discrete_weights_sum (a b : ℝ) (ha : 0 < a) (ha1 : a < 1)
    (hb : 0 < b) (hb1 : b < 1) : weight a b + weight b a = 1 := by
  have hab : 1-a*b ≠ 0 := ne_of_gt (one_sub_mul_pos a b ha ha1 hb hb1)
  unfold weight
  rw [mul_comm b a]
  field_simp
  ring

/-- The real rational frequency identity underlying the discrete-time theorem. -/
theorem discrete_pointwise_identity (a b z : ℝ)
    (ha : 0 < a) (ha1 : a < 1) (hb : 0 < b) (hb1 : b < 1) (hz : z ≤ 1) :
    discreteCross a b z =
      weight a b * discretePower a z + weight b a * discretePower b z := by
  have hda : discreteDen a z ≠ 0 := ne_of_gt (discrete_den_pos a z ha ha1 hz)
  have hdb : discreteDen b z ≠ 0 := ne_of_gt (discrete_den_pos b z hb hb1 hz)
  have hab : 1-a*b ≠ 0 := ne_of_gt (one_sub_mul_pos a b ha ha1 hb hb1)
  unfold discreteCross discretePower weight
  rw [mul_comm b a]
  field_simp
  unfold discreteDen
  ring

end FirstOrderProbe

#print axioms FirstOrderProbe.continuous_pointwise_identity
#print axioms FirstOrderProbe.variance_identity_of_balance
#print axioms FirstOrderProbe.compatibility_of_identity
#print axioms FirstOrderProbe.numerator_lower_bound
#print axioms FirstOrderProbe.sharp_lower_bound
#print axioms FirstOrderProbe.sharp_lower_bound_eq_iff
#print axioms FirstOrderProbe.discrete_den_pos
#print axioms FirstOrderProbe.one_sub_mul_pos
#print axioms FirstOrderProbe.discrete_weights_sum
#print axioms FirstOrderProbe.discrete_pointwise_identity
