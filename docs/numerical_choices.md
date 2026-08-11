# Numerical choices, and why each one was made

Every non-obvious numerical decision in FibroBlock, with the reasoning that
would be needed to defend it. Written for the viva: each section states the
choice, the alternative, and the evidence.

---

## 1. Conservative form, not `D(x)·∂²V/∂x²`

**Choice.** The diffusion term is discretised as fluxes on interfaces, then
differenced:

```
F_{j+1/2} = D_{j+1/2} (V_{j+1} − V_j) / Δx
L(V)_j    = (F_{j+1/2} − F_{j−1/2}) / Δx
```

**Alternative.** Expand the derivative to `D V_xx + D_x V_x` and discretise
that.

**Why the alternative is wrong here.** The expanded form has no exact discrete
divergence-theorem analogue, so the scheme creates and destroys charge wherever
`D` varies. In this project `D` varies *precisely at the interface whose effect
is being measured*. A non-conservative scheme would corrupt the answer at the
one location that matters.

**Evidence.** With the conservative form, every interior flux appears twice with
opposite signs and cancels exactly. `test_charge_conservation_also_holds_across_a_sharp_coupling_gap`
confirms charge is conserved to machine precision **across a 100-fold drop in
`D`**, and ex04 shows a relative drift of 1.33e−15 over 1000 steps.

---

## 2. Harmonic mean at half-nodes, not arithmetic

**Choice.** `D_{j+1/2} = 2 D_j D_{j+1} / (D_j + D_{j+1})`.

**Why.** The diffusion coefficient is inversely proportional to axial
resistance, and **resistances in series add**. Averaging the resistances and
inverting gives the harmonic mean of `D`. This is not a numerical preference; it
is what the physics of a series circuit requires.

**How much it matters — measured, not assumed.** ex06 runs both schemes:

| Gap length | Effect of using the arithmetic mean |
|---|---|
| 0.02–0.40 cm | Threshold shifts by up to **4.2 %** |
| **0.01 cm (single node)** | **Block becomes impossible at any coupling ratio** |

The single-node case is the one that exposes the error qualitatively. As
`ρ → 0`:

- harmonic: `2ρD₀/(1+ρ) → 0` ✅ the connection genuinely opens
- arithmetic: `(1+ρ)D₀/2 → D₀/2` ❌ **half the healthy conductance survives, no
  matter how completely the tissue is uncoupled**

**Why the shift is only a few per cent for wider gaps.** A 0.1 cm gap spans ~10
intervals, of which only the two at the edges are transitional; the other eight
carry `ρD₀` under either scheme. The averaging choice matters in proportion to
how much of the gap *is* interface — which is why the effect is dramatic at one
node and modest at ten. Worth stating, because "the arithmetic mean is
catastrophically wrong" would overclaim.

**Implementation.** Both are available via `GapParams.averaging`; harmonic is
the default. The half-node array is built once per run in `grid.build_grid` and
made read-only, so it cannot be recomputed inside the time loop by accident.

---

## 3. Explicit Euler as the primary integrator

**Choice.** Hand-coded forward Euler. RK4 available for accuracy comparison.
**No adaptive solver anywhere in the pipeline.**

**Why not `scipy.integrate.solve_ivp`.** Part (c) of the assignment requires
demonstrating the stability limit of *this* scheme. An adaptive controller would
detect the incipient instability and silently shrink the step until it went
away — hiding the exact phenomenon being asked for. SciPy is used only for root
finding (`brentq`) and least-squares fitting.

**Why Euler rather than RK4 for production runs.** ex03 shows that at the
working step the *spatial* error dominates: Euler and RK4 give conduction
velocities agreeing to well under 1 %. Paying four right-hand-side evaluations
per step for temporal accuracy that is already not the limiting factor would be
wasted. RK4's larger stability region (~2.785 vs 2 on the negative real axis)
buys only a 1.39× larger step, which does not repay 4× the cost per step.

---

## 4. `Δt = 0.02 ms`

**Choice.** `Δt = 0.02 ms`, a safety factor of **2.33** below the computed
limit.

**The limit.** Von Neumann analysis on the linearised system, worst case at the
checkerboard mode `k = π/Δx`:

```
Δt ≤ 2 / (4 D_max/Δx² + |f_V|_max) = 2/(40 + 3) = 0.046512 ms
```

**Why not the familiar `Δx²/(2D)`.** That is the pure-diffusion limit, 0.05 ms
here — **7.5 % too optimistic**, because it ignores the reaction term's
contribution to the amplification factor. Both are reported in ex02 for exactly
this reason.

**Why a safety factor of 2.3 rather than 1.1.** Three reasons:

1. `|f_V|max = 3` is a bound over the whole action potential; the true local
   value varies.
2. Sweeps vary `D` and `Δx`, and a margin means the step stays valid without
   recomputation at every point.
3. Accuracy, not just stability. Euler is first order, so running near the
   stability limit maximises temporal error precisely where the solution is
   changing fastest.

**Evidence it is the right side of the line.** ex02 finds empirical failure at
1.065× the limit — i.e. the conservative bound is never optimistic. The margin
between 1.0 and 1.065 is itself explained: the solution peaks at `V = 1.691`,
not at the excited root 1.986, so the true `|f_V|` is 1.86 rather than 3.

---

## 5. `Δx = 0.01 cm`

**Choice.** 200 intervals, 201 nodes on a 2 cm strand.

**Not an arbitrary round number.** Three independent justifications:

1. **Physical scale.** ≈ one myocyte length, the natural discretisation of
   cardiac tissue.
2. **Front resolution.** The reaction–diffusion balance length is
   `√(D/|f_V|) = √(0.001/3) = 0.0183 cm`, so `Δx = 0.01 cm` resolves it with
   about two nodes, and roughly ten nodes span the full visible upstroke. This
   is enforced by `test_front_thickness_justifies_the_default_grid_spacing`.
3. **Measured error.** ex03 refines to `Δx = 0.0025 cm` and Richardson-
   extrapolates the conduction velocity to 0.0256705 cm/ms. The default spacing
   carries **0.52 %** discretisation error — small enough that the −16 %
   discrepancy against the analytic velocity is demonstrably physical, not a
   grid artefact.

**Cost of refining.** The stability limit scales as `Δx²`, so halving `Δx`
roughly quarters `Δt` and multiplies the work by ~8. Going to
`Δx = 0.00125 cm` costs 27 s for a change in the extrapolated velocity below
0.01 %. Not worth putting in a pipeline the examiner has to run.

---

## 6. Trapezoidal quadrature weights (half-weight end cells)

**Choice.** Total charge is `Q = Σ w_j V_j` with `w = [Δx/2, Δx, …, Δx, Δx/2]`.

**Why.** In the finite-volume reading, the end nodes own control volumes of half
the width, because half of each lies outside the domain. With these weights the
no-flux operator conserves `Q` **exactly**; with uniform weights it does not.

**Evidence.** `test_uniform_weights_would_NOT_conserve_charge` is a deliberate
negative control: uniform weights leave a residual > 1e−6 where trapezoidal
weights give < 1e−12. The half weights are load-bearing, not cosmetic.

---

## 7. Activation defined by `V = 0` crossing

**Choice.** First upward crossing of `V = 0`, linearly interpolated *within* the
time step.

**Why `V = 0`.** It lies between the threshold root (−0.786) and the excited
plateau (+1.986), so it is crossed exactly once per upstroke — no ambiguity
about which crossing to take.

**Why interpolate within the step.** At `Δt = 0.02 ms` and `θ ≈ 0.026 cm/ms`,
one step is about 1.5 node spacings. Without sub-step interpolation the
activation map would be visibly stepped and the velocity fit would inherit that
quantisation as noise.

**The alternative is also computed.** Time of maximum `dV/dt` is recorded in
every run and selectable via `MeasurementParams.activation_rule`. The two differ
by a fraction of a millisecond. **The report states which was used** (`V = 0`
crossing), and either figure can be redrawn under the other definition without
re-running anything.

---

## 8. Velocity fitted only over `x ∈ [0.5, 1.8] cm`

**Choice.** Discard the first 0.5 cm and the last 0.2 cm.

**Why.** The wave is still forming out of the stimulus over roughly the first
half-centimetre (accelerating), and the sealed end reflects charge back into the
approaching front over the last 0.2 cm (also accelerating). Fitting through
either gives a velocity that is wrong in a direction that still looks plausible.

**Evidence the retained window is genuinely steady.** `R² = 1.00000000` on every
homogeneous run.

**But `R²` alone is not enough.** A deliberate negative result,
`test_r_squared_is_a_weak_detector_of_MILD_curvature`, shows that a curvature
changing the local velocity by a few per cent still leaves `R² > 0.9999`. `R²`
near 1 is **necessary but not sufficient**. This is why ex05 also plots `θ/√D`
directly, where a few per cent of drift is immediately visible.

---

## 9. Two grid families in the `√D` sweep

**Choice.** ex05 runs the sweep twice: at fixed `Δx = 0.01 cm`, and with
`Δx ∝ √D`.

**Why.** Front thickness scales as `√(D/|f_V|)`. On a fixed grid, small `D`
means a thinner front spread over fewer nodes — so the low-`D` end is
systematically less resolved than the high-`D` end, and the resulting drift is a
numerical artefact masquerading as physics.

**Evidence.** `θ/√D` spread: **1.92 %** on the fixed grid, **0.02 %** on the
resolution-matched grid. The scaled family also, conveniently, holds
`4D/Δx²` constant, so every run in it shares one time step.

---

## 10. Stimulus width scaled with `√D`

**Choice.** In the `D` sweep only, stimulus width scales as `√D`, holding it at
a fixed number of space constants.

**Why — discovered, not anticipated.** The **liminal length**, the smallest
patch that can be excited into a propagating wave, scales with the space
constant, hence as `√D`. A fixed 0.1 cm stimulus **fails to launch a wave at all
above `D ≈ 0.002 cm²/ms`** (peak `V` reaches only −0.24, well below the
activation level). Without scaling, the high-`D` end of the sweep would simply
have no data, for reasons unrelated to the scaling law being tested.

Documented in ex05 panel (c) rather than quietly worked around.

---

## 11. Block criterion: 0.3 cm margin, 200 ms window

**Choice.** Blocked if no node at `x > x_gap + L_gap + 0.3 cm` reaches `V = 0`
within 200 ms.

**Why the margin.** Charge leaks passively past a blocked gap and decays over
roughly one space constant. Judging block at the gap edge would mistake this
**electrotonic foot** for a propagating front. At 0.3 cm the passive
contribution is negligible — visible directly in ex06 panel (d), where the
blocked run's peak potential falls from +1.6 inside the gap to resting level
well before the detection point.

**Why 200 ms is comfortable.** The longest successful transit measured anywhere
in the project is 43.4 ms (ex07). The window is ~4.6× that, so it never
misclassifies slow-but-successful conduction as block.

---

## 12. Bisection rather than a root finder for thresholds

**Choice.** Plain bisection on the boolean "did it propagate?".

**Why.** The quantity being bracketed is a **boolean**, not a continuous
function with a sign change. There is no residual to interpolate, so Brent's
method has nothing to work with. Bisection's guaranteed halving is exactly the
right tool.

**Fail loudly.** `bisect_threshold` verifies both endpoints before starting and
raises if the bracket does not straddle. This is what surfaced the finding that
**arithmetic averaging cannot block a single-node gap at any `ρ`** — recorded as
NaN and plotted, rather than silently skipped.

---

## 13. Divergence checked every step, not at the snapshot cadence

**Choice.** `|V|` is checked against a threshold of 1e6 on **every** time step.

**Why — a bug found during development.** Above the stability limit, blow-up is
*hyper*-exponential rather than exponential. Once `|V|` passes `√(3/Δt)` the
cubic dominates and the update becomes `V ← −Δt V³/3`, which **cubes** the
magnitude each step: about seven steps carry `V` from 10 to double-precision
overflow. Checking every 25 steps routinely missed that window and left the
arrays full of `inf` and `NaN`, destroying the diagnostic record ex02 needs.

**Cost.** One reduction over 201 elements per step, ~10 % of the step's own cost.
Worth it.

---

## 14. Pinned dependencies and a src/ layout

**Pinned with `==`, never `>=`.** Floating pins are the commonest cause of "the
figures don't regenerate". `requirements.txt` and `environment.yml` carry
identical pins.

**`src/` layout.** Tests import the *installed* package rather than accidentally
picking up the working directory, so `pip install -e . && pytest` behaves the
same on the examiner's machine as on mine.

**No pandas.** Summary tables are written with the standard library's `csv`
module — fully transparent for a few dozen rows, and one fewer large pinned
dependency on the critical path of "can the examiner regenerate the figures?".

---

## Summary table

| Choice | Alternative rejected | Decisive evidence |
|---|---|---|
| Conservative flux form | `D·V_xx` | Charge conserved to 1e−15 across a 100× `D` step |
| Harmonic mean | Arithmetic | Arithmetic cannot block a 1-node gap at any `ρ` |
| Explicit Euler | `solve_ivp` | Adaptive stepping would hide the stability limit |
| `Δt = 0.02 ms` | Near-limit stepping | Empirical failure at 1.065× the limit |
| `Δx = 0.01 cm` | Coarser / finer | 0.52 % error; 8× cost to halve it |
| Trapezoidal weights | Uniform | Uniform weights break conservation (negative control) |
| `V = 0` activation | max `dV/dt` | Both computed; choice reported |
| Fit window `[0.5, 1.8]` | Whole strand | `R² = 1.00000000` on the retained window |
| Scaled grid in `D` sweep | Fixed grid | Spread 1.92 % → 0.02 % |
| Bisection | Brent | The target is boolean, not continuous |
| Per-step divergence check | Per-snapshot | Blow-up cubes each step; 7 steps to overflow |
