# Verification log

*Verification asks: am I solving the equations correctly?* (Validation, the
separate question of whether these are the right equations, is in
[`validation_log.md`](validation_log.md).)

Every check below is automated. The "test" column names the test that enforces
it, so nothing here can silently rot: `pytest` re-runs all 93 of them.

Values were obtained on the environment recorded in `results/environment.json`
(Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1, Windows 11, AMD64).

**Legend:** ✅ pass · ⚠️ pass with a documented discrepancy.

---

## V1–V9. Analytic results from the kinetics

| # | Check | Expected | Obtained | Status | Test |
|---|---|---|---|---|---|
| V1 | Rest potential `V*` | −1.199408 (brief) | **−1.199408035** | ✅ | `test_rest_state_matches_brief_values` |
| V2 | Rest recovery `w*` | −0.624260 (brief) | **−0.624260044** | ✅ | `test_rest_state_matches_brief_values` |
| V3 | Jacobian trace | −0.5034 (brief) | **−0.502580** | ⚠️ see note 1 | `test_jacobian_trace_is_close_to_brief_value` |
| V4 | Jacobian determinant | 0.1081 (brief) | **0.108069** | ✅ | `test_jacobian_determinant_matches_brief` |
| V5 | Rest-state classification | stable spiral | **stable spiral**, eigenvalues −0.251290 ± 0.211949 i | ✅ | `test_rest_state_is_a_stable_spiral` |
| V6 | Both RHS vanish at `(V*, w*)` | 0 | **< 1e−12** | ✅ | `test_rest_state_actually_annihilates_both_right_hand_sides` |
| V7 | Bistable roots | −1.19941, −0.78638, +1.98579 (brief) | **−1.199408, −0.786321, +1.985729** | ✅ (1e−4) | `test_bistable_roots_match_brief` |
| V8 | Roots sum to zero | 0 exactly | **< 1e−10** | ✅ | `test_bistable_roots_sum_to_zero` |
| V9 | CV prefactor `θ/√D` | 0.9634 (brief) | **0.963043** | ⚠️ see note 2 | `test_analytic_cv_prefactor_matches_brief` |

**Note 1 — trace discrepancy (0.16 %).** Recomputing `tr J = (1 − V*²) − εb`
directly from the exact rest state gives −0.502580, against the brief's
−0.5034. The test independently reconstructs the trace from the definition and
confirms −0.502580 to 1e−12, so the difference is rounding carried through the
brief's intermediate arithmetic, not an error here. The determinant, which is
less sensitive to that rounding, agrees exactly. **Sign and magnitude are
unaffected, so the physical conclusion — stable spiral, excitable medium — is
untouched.**

**Note 2 — prefactor discrepancy (0.04 %).** The brief's 0.9634 comes from its
rounded roots. Recomputing from the exact roots gives 0.963043. Confirmed
independently by the identity `θ/√D = −3V₂/√6`, which the code reproduces to
1e−12.

---

## V10–V17. Spatial operator

| # | Check | Expected | Obtained | Status | Test |
|---|---|---|---|---|---|
| V10 | Laplacian of a quadratic, interior | exact | **< 1e−10 relative** | ✅ | `test_laplacian_of_a_quadratic_is_exact_in_the_interior` |
| V11 | Ghost-node end formula on `V = x²` | exact at the left end | **agrees to 1e−12** | ✅ | `test_boundary_stencil_is_exact_for_a_quadratic_that_obeys_the_bc` |
| V12 | Boundary stencil order | 2 | **2.00, 2.00** (grid triple) | ✅ | `test_boundary_stencil_is_second_order_for_a_field_obeying_both_ends` |
| V13 | No-flux at both ends | 0 | **exactly 0.0** (bitwise) | ✅ | `test_neumann_boundary_flux_is_identically_zero` |
| V14 | Discrete charge conservation, uniform `D` | 0 | **< 1e−12 × scale** | ✅ | `test_operator_conserves_charge_to_machine_precision` |
| V15 | Charge conservation across a 100-fold `D` step | 0 | **< 1e−12 × scale** | ✅ | `test_charge_conservation_also_holds_across_a_sharp_coupling_gap` |
| V16 | Uniform `D` reproduces the textbook stencil | identical | **< 1e−12** | ✅ | `test_uniform_D_reproduces_the_plain_second_difference` |
| V17 | Constant field is annihilated | 0 | **< 1e−18** | ✅ | `test_operator_annihilates_a_constant_field` |

**Negative control.** `test_uniform_weights_would_NOT_conserve_charge` confirms
that naive uniform quadrature weights give a residual **> 1e−6**, so the
half-weight end cells in V14–V15 are load-bearing rather than decorative.

---

## V18–V23. Time integration and stability

| # | Check | Expected | Obtained | Status | Test |
|---|---|---|---|---|---|
| V18 | Explicit-Euler `dt` limit | 2/43 = 0.046512 ms | **0.046512 ms** | ✅ | `test_stability_limit_matches_the_brief` |
| V19 | Pure-diffusion limit is too optimistic | +7 % (brief) | **+7.50 %** (0.05 vs 0.046512 ms) | ✅ | `test_pure_diffusion_limit_is_seven_percent_too_optimistic` |
| V20 | Safety factor at `dt = 0.02 ms` | ≈2.3 (brief) | **2.33** | ✅ | `test_default_step_has_the_stated_safety_factor` |
| V21 | Measured vs predicted amplification `\|1 − 4D dt/dx²\|` | equal | **max error 2.22e−16** | ✅ | `test_checkerboard_amplification_matches_the_von_neumann_prediction` |
| V22 | Checkerboard mode decays below / grows above the limit | decay, growth | **confirmed both sides** | ✅ | `test_checkerboard_mode_decays_below_the_limit_and_grows_above_it` |
| V23 | Unstable `dt` raises unless `force=True` | raises | **raises** | ✅ | `test_simulation_refuses_an_unstable_step_unless_forced` |

**Empirical stability boundary (ex02).** The full nonlinear model first fails at
**1.065×** the conservative limit, not at exactly 1×. This is expected and
explained: `|f_V|max = 3` (assumption A17) is an upper bound taken at the
excited root `V₃ = 1.986`, whereas the solution actually peaks at **1.691**,
giving `|f_V| = 1.859` and a refined limit of 0.047780 ms = **1.027×** the
conservative one. The remaining gap is because that peak is attained only
briefly and locally, while instability needs sustained amplification. **The
conservative bound is therefore doing its job: it is never optimistic.**

---

## V24–V29. Orders of accuracy (ex03)

| # | Check | Design | Obtained | Status |
|---|---|---|---|---|
| V24 | Spatial order, pure diffusion vs exact | 2 | **1.9987** (asymptotic range) | ✅ |
| V25 | Spatial order, velocity vs `dx` | 2 | **1.9793** | ✅ |
| V26 | Temporal order, explicit Euler (smooth) | 1 | **0.9988** | ✅ |
| V27 | Temporal order, RK4 (smooth) | 4 | **3.9545** | ✅ |
| V28 | Temporal order, RK4 *with* the stimulus | — | **1.0197** | ⚠️ see note 3 |
| V29 | Euler is first order on `exp(−t)` | 1 | **0.998, 1.003, 1.010** | ✅ |

**Note 3 — RK4 capped at first order by the stimulus.** This was discovered, not
anticipated. The stimulus switches off discontinuously at `t = 1 ms`
(assumption A13), and a jump in the forcing destroys the Taylor expansion that
higher order depends on — so *any* one-step method is capped at order 1,
regardless of stage count. ex03 therefore measures the temporal order twice:
including the stimulus (RK4 → 1.02) and on smooth propagation from a pre-formed
wave (RK4 → 3.95). **The implementation is correct; the problem is
non-smooth.**

**Pre-asymptotic point.** The coarsest grid in V24 (`dx = 0.04 cm`) gives a
pairwise order of 5.31 because the Gaussian's `σ = 0.1 cm` spans only 2.5
nodes — the profile is not resolved and the error is not yet governed by the
leading truncation term. It is excluded from the fitted order (which would
otherwise read 2.66) and retained in the plot to show where the asymptotic
regime begins.

---

## V30–V33. Analytic-limit and conservation checks (ex04)

| # | Check | Expected | Obtained | Status |
|---|---|---|---|---|
| V30 | Pure diffusion vs exact sealed-strand solution, `L²` | small | **9.612e−06** | ✅ |
| V31 | Same, `L∞` | small | **1.788e−05** | ✅ |
| V32 | Charge conservation over 1000 steps | machine precision | **1.329e−15 relative** | ✅ |
| V33 | Peak decay follows `σ₀/σ(t)` | — | **max relative error 6.242e−05** | ✅ |

**Method-of-images reference.** Comparing against the textbook *infinite-line*
Gaussian conflates two different errors. At `t = 20 ms` the boundary reflection
contributes **2.03e−05**, as large as the discretisation error itself.
`operators.analytic_gaussian_sealed` gives the exact sealed-strand solution by
method of images, so V30–V31 measure the discretisation error alone. The
distinction is shown in ex04 panel (b).

---

## V34–V38. Measurement layer

| # | Check | Expected | Obtained | Status | Test |
|---|---|---|---|---|---|
| V34 | Velocity recovered from a synthetic linear map | exact | **1e−12 relative** | ✅ | `test_velocity_is_recovered_exactly_from_a_linear_map` |
| V35 | Fit ignores the discarded windows | unaffected | **unaffected to 1e−12** | ✅ | `test_velocity_fit_ignores_the_discarded_windows` |
| V36 | Fit refuses with fewer than 3 points | NaN, not a guess | **NaN** | ✅ | `test_velocity_fit_refuses_rather_than_guesses_with_too_few_points` |
| V37 | Observed order recovers a known `h²` law | 2 | **2.0 exactly** | ✅ | `test_observed_order_recovers_a_known_power_law` |
| V38 | `L²` norm is grid-independent | converges | **within 1e−3 across 4× refinement** | ✅ | `test_l2_error_is_grid_independent` |

**Recorded limitation of `R²`.** `test_r_squared_is_a_weak_detector_of_MILD_curvature`
is a deliberate negative result: a curvature changing the local velocity by a
few per cent still leaves `R² > 0.9999`. **`R²` near 1 is necessary but not
sufficient** evidence of steady propagation — which is precisely why ex05 does
not rely on a fitted log-log slope alone and plots `θ/√D` directly as well.

---

## V39–V42. Physics results cross-checked against theory

| # | Check | Expected | Obtained | Status |
|---|---|---|---|---|
| V39 | `θ ∝ √D` log-log exponent | 0.5 | **0.49996** (R² = 0.99999999) | ✅ |
| V40 | `θ/√D` flatness, resolution-matched grid | flat | **0.02 % spread** | ✅ |
| V41 | Elasticity of `θ` w.r.t. `D₀` (independent route) | 0.5 exactly | **0.5044** | ✅ |
| V42 | Measured vs analytic prefactor | analytic 0.9630 | **0.8074** | ⚠️ see note 4 |

**Note 4 — the −16 % velocity discrepancy is physical, not numerical.** This is
the largest single discrepancy in the project and was investigated rather than
excused:

1. **Ruled out as numerical.** Grid refinement to `dx = 0.00125 cm` shows the
   measured velocity converging at second order to 0.0256705 cm/ms; the default
   `dx = 0.01 cm` carries only **0.52 %** discretisation error (V25). The gap
   survives refinement, so it is not a grid artefact.
2. **Explained.** The analytic derivation freezes `w` at rest (assumption A7).
   The measured recovery level at the front is **−0.5906**, not `w* = −0.6243`.
   Substituting it gives a prefactor of **0.8693**, closing the gap from −16.2 %
   to −7.1 %.
3. **Residual acknowledged.** The remaining 7 % is because `w` varies *across*
   the front rather than being constant at any one value, so no single frozen
   `w` can be exactly right.

**A fixed grid is what makes `θ/√D` look imperfect.** On a fixed `dx` the spread
is 1.92 %; scaling `dx ∝ √D` to hold nodes-per-front-thickness constant reduces
it to 0.02 %. The drift was a resolution artefact, since front thickness itself
scales as `√D`.

---

## V43–V47. Reproducibility

| # | Check | Expected | Obtained | Status | Test |
|---|---|---|---|---|---|
| V43 | Same config → bitwise identical arrays | identical | **bitwise identical** | ✅ | `test_same_config_gives_bitwise_identical_output` |
| V44 | Derived measurements reproducible | identical | **bitwise identical** | ✅ | `test_derived_measurements_are_bitwise_reproducible` |
| V45 | Run order does not affect results | no hidden state | **confirmed** | ✅ | `test_run_order_does_not_affect_results` |
| V46 | Different config → different result (control) | differs | **differs** | ✅ | `test_a_different_config_gives_a_different_answer` |
| V47 | Provenance record is JSON-serialisable | serialises | **serialises** | ✅ | `test_provenance_record_is_json_serialisable` |

---

## Summary

**93 automated tests, all passing, no skips.** Full pipeline rebuilds from empty
in **roughly 6–10 minutes** (6 m 15 s and 9 m 14 s measured on two runs of the
same machine), producing 8 figures (PNG + PDF) and their results files.

Four entries are marked ⚠️. **None is an unexplained failure:**

- **V3, V9** — rounding in the brief's quoted values; the code recomputes from
  first principles and is independently cross-checked.
- **V28** — a genuine property of the problem (discontinuous forcing), measured
  and reported rather than hidden by only ever testing the smooth case.
- **V42** — a real physical effect, traced to a stated assumption (A7),
  quantified, and mostly accounted for.

### How to regenerate every number in this log

```bash
python scripts/make_all_figures.py
```

```bash
pytest -v
```
