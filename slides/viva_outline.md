# Viva outline — 4 minutes

**Problem 8: Cardiac action-potential propagation and conduction block**

Timings are cumulative. Four minutes is short: it is one story, four figures,
and one honest surprise. Everything else goes in the back pocket for questions.

---

## The four-minute talk

### 0:00–0:30 — The question

> "Fibrosis makes heart tissue conduct badly. I wanted to know: **how weak does
> the coupling have to get, and over how long a patch, before an action
> potential simply fails to cross?**
>
> I solved the monodomain cable equation with FitzHugh–Nagumo kinetics on a 2 cm
> strand, with a patch of reduced coupling in the middle."

*Slide: the equation, the strand cartoon with the gap shaded.*

---

### 0:30–1:15 — It works, and I can prove it

> "First, the tissue is genuinely **excitable, not oscillatory** — the rest
> state at `V* = −1.1994` is a stable spiral, trace negative, determinant
> positive. A 10 % change in stimulus amplitude switches you between nothing and
> a full action potential.
>
> Second, I verified the solver against things I can check independently. With
> the reaction term off it reproduces the exact diffusion solution to 1.8×10⁻⁵,
> and it conserves charge to **1.3×10⁻¹⁵** — machine precision — because I used
> the conservative flux form rather than `D` times the second derivative."

*Slide: `fig_ex01_single_cell` (phase plane panel) + one line of ex04 numbers.*

---

### 1:15–2:00 — The stability limit is real

> "The scheme is explicit Euler, hand-coded. Von Neumann analysis says
> `Δt ≤ 2/(4D/Δx² + |f_V|) = 0.0465 ms`, and the binding mode is the
> **checkerboard** — alternating sign node to node.
>
> That is exactly how it fails. Here it is blowing up, and you can see the sign
> flipping at every single node. The measured amplification matches the
> predicted `|1 − 4DΔt/Δx²|` to 2×10⁻¹⁶.
>
> Note the naive `Δx²/2D` limit ignores the reaction term and is **7.5 % too
> optimistic** — which is the difference between working and not."

*Slide: `fig_ex02_stability`, panels (b) and (d).*

---

### 2:00–2:45 — Velocity scales as √D — and why it's 16 % low

> "Conduction velocity scales as the square root of coupling. Measured exponent
> **0.49996**. And a completely independent route — the sensitivity elasticity
> of velocity to `D₀` — gives **0.5044** against exactly 0.5.
>
> But my measured prefactor is 0.807 against the analytic 0.963: **16 % low**.
> I checked it wasn't the grid — refining to `Δx = 0.00125 cm` leaves only
> 0.52 % discretisation error, so the gap survives refinement.
>
> It's physical. The analytic result freezes the recovery variable at rest. But
> by the time the front arrives, `w` has already risen from −0.624 to −0.591.
> Put the measured value in and the gap closes to 7 %."

*Slide: `fig_ex05_cv_vs_D`, panels (b) and (d).*

---

### 2:45–3:30 — The answer, and the surprise

> "So — the block threshold. **It is not a single number.** It's a curve: a
> one-cell gap blocks only below `ρ = 0.067`, but by 0.1 cm it has risen to
> 0.157 and then saturates. Once the gap is longer than the distance over which
> failure develops, only the local coupling matters.
>
> And here's what I got wrong. I expected the conduction delay to **diverge** at
> the threshold, like a saddle-node — `(ρ − ρ_crit)^(−1/2)`. It doesn't. I
> pushed to nine decimal places of the threshold, six decades of approach, and
> it **saturates at 16 milliseconds**.
>
> The reason is that a stalled wavefront can't outlast its own upstream source,
> which is repolarising on the recovery timescale. I tested that by varying `ε`
> and the ceiling scales inversely with it. So the transition to block is
> **discontinuous in delay**: you cross within about a recovery time, or you
> don't cross at all."

*Slide: `fig_ex06_block_threshold` panel (a) + `fig_ex07_conduction_delay`
panel (b).*

---

### 3:30–4:00 — Close

> "Two closing points.
>
> One: a numerical choice changed a physical conclusion. The interface
> conductance has to be the **harmonic** mean, because `D` goes as one over
> resistance and resistances in series add. Use the arithmetic mean and a
> single-cell gap becomes **impossible to block at any coupling ratio** —
> because its interface conductance tends to `D₀/2` instead of zero.
>
> Two: everything here regenerates from a clean clone in under ten minutes, with
> 93 tests passing. Thank you."

---

# One-page notes sheet

*Print this. Numbers you must not fumble.*

## Core numbers

| | |
|---|---|
| Rest state | `V* = −1.199408`, `w* = −0.624260` |
| Jacobian | `tr = −0.5026`, `det = +0.1081` → **stable spiral** |
| Eigenvalues | `−0.2513 ± 0.2119 i` |
| Bistable roots | `−1.1994`, `−0.7863`, `+1.9857` (sum to zero) |
| Analytic prefactor | `0.9630` → `θ = 30.45 cm/s` at `D₀` |
| **Measured** prefactor | `0.8074` → `θ = 25.5 cm/s` |
| Stimulus threshold | `A* = 0.6063` (we use 1.0 = 1.65×) |
| `Δt` limit | `2/43 = 0.046512 ms`; we use 0.02 (safety 2.33) |
| Pure-diffusion limit | 0.05 ms — **7.5 % too optimistic** |
| `Δx` | 0.01 cm; 0.52 % discretisation error in `θ` |
| `ρ_crit` | 0.067 (1 node) → **0.1575** (saturated) |
| Delay ceiling | **16.0 ms** excess; healthy transit 27.4 ms |

## The one-line derivations

**Rest state.** `w* = (V*+a)/b`, sub into `f = 0` → `V³ + 0.75V + 2.625 = 0`.

**Stability.** `g = 1 + Δt[−(4D/Δx²)sin²(kΔx/2) + f_V]`; worst at `sin² = 1`,
i.e. `k = π/Δx`; `g ≥ −1` gives `Δt ≤ 2/(4D/Δx² + |f_V|)`.
*The 4 is because the second difference of an alternating field is −4×; the 2 is
because the binding side is `g ≥ −1`.*

**Front speed.** Freeze `w` at `w*` → `V³ − 3V + 3w* = 0`;
`θ = √(A/2)(V₁ − 2V₂ + V₃)√D` with `A = 1/3`.
**Roots sum to zero, so `θ/√D = −3V₂/√6`.**

**Harmonic mean.** `D ∝ 1/r`, series resistances add →
`D_{j+½} = 2D_jD_{j+1}/(D_j + D_{j+1})`.

## Answers to have ready

**"Why not `solve_ivp`?"** Part (c) asks me to demonstrate *my* scheme's
stability limit. An adaptive controller would shrink the step and hide it.
SciPy is used only for `brentq` and least squares.

**"Why is the velocity 16 % low?"** Not the grid — 0.52 % discretisation error,
and it survives refinement to `Δx = 0.00125`. The analytic result freezes `w`
at rest; `w` is actually −0.591 at the front. Substituting closes it to 7 %.
The residual is because `w` varies *across* the front.

**"Why does the delay saturate?"** A stalled front can't outlast its upstream
source, which repolarises on `1/ε`. Confirmed by varying `ε`: the ceiling scales
inversely, exponent −1.46 (steeper than −1 because `ε` also moves `ρ_crit`).

**"Why does RK4 measure first order?"** The stimulus switches off
discontinuously at 1 ms. A jump in the forcing kills the Taylor expansion that
higher order needs — caps *any* one-step method at 1. On smooth propagation RK4
gives 3.95.

**"How do you know it's converged?"** Three ways: Richardson extrapolation
(0.52 % error at default `Δx`); the numerical elasticities in ex08 are 257×
smaller than the physical ones; `θ/√D` is flat to 0.02 % on a
resolution-matched grid.

**"Why 0.3 cm margin in the block criterion?"** Charge leaks past a blocked gap
electrotonically and decays over about a space constant. Judging at the gap edge
would mistake passive spread for a propagating front. ex06 panel (d) shows it.

**"How confident are you in `ρ_crit`?"** As a *comparison* — very. As a
*prediction for myocardium* — not at all. It's 1-D and continuum, both of which
make block easier than reality; real tissue conducts saltatorily and can route
around a patch. It's an upper bound on how easily block occurs.

**"What would you do next?"** Sobol sensitivity (the one-at-a-time sweep can't
see interactions); a discrete-cell model to test the continuum assumption; a
premature stimulus to reach the re-entry question — though FHN's action
potential is 4–6× too short for that to be trustworthy.

**"What did the AI do?"** Wrote the code from my specification. I wrote the
spec, set the requirement that nothing analytic be hard-coded — which is what
made the discrepancies findable — reviewed at each gate, and I can explain every
line. Declared fully in `docs/ai_use_declaration.md`.

## If time runs short, cut in this order

1. The verification slide (1:15–2:00) — compress to one sentence.
2. The 16 % explanation — say "and I can explain the 16 % deficit if you'd like".
3. **Never cut** the block-threshold curve or the delay saturation. They are the
   result.
