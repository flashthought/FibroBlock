# Cardiac Action-Potential Propagation and Conduction Block in a 1-D Monodomain Strand

**COE 562 — Engineering Systems Design and Modelling**
**Problem 8**
MPhil, Kwame Nkrumah University of Science and Technology

*Author:* ____________________  *Date:* ____________________

Repository: `https://github.com/REPLACE-ME/FibroBlock`
Commit: ____________________ (from `python scripts/check_environment.py`)

---

> **STATUS: SKELETON.** Section headings follow the marking scheme. Each
> section lists the figures, numbers and claims available to it, so the prose
> can be written straight onto this scaffold. Every number quoted below is
> already computed and lives in `results/`; nothing here needs re-deriving.
>
> Placeholders to fill are marked `[WRITE]`.

---

## Abstract

`[WRITE — 150–200 words]`

*Points to hit:* monodomain cable with FitzHugh–Nagumo kinetics on a 2 cm
strand; conservative finite-volume discretisation with harmonic interface
averaging; hand-coded explicit Euler. Headline results: rest state is a stable
spiral so the medium is excitable; `θ ∝ √D` with measured exponent 0.49996;
the block threshold is a **curve** in `(L_gap, ρ)` rising from 0.067 to 0.157
and saturating; conduction delay **saturates at 16 ms rather than diverging**,
bounded by the recovery timescale. All figures regenerate from a clean clone in
under ten minutes; 93 automated tests pass.

---

## 1. Introduction and problem statement

`[WRITE]`

**Content available:**

- The clinical motivation: fibrosis reduces intercellular coupling; the
  question is how weak coupling must become, over what distance, before an
  action potential fails to cross.
- Why a region that *delays* conduction may matter more than one that blocks
  it: delay creates the timing dispersion underlying re-entry.
- Governing equations (§1.1 of the brief):

$$\frac{\partial V}{\partial t} = \frac{\partial}{\partial x}\!\left(D(x)\frac{\partial V}{\partial x}\right) + f(V,w), \qquad \frac{\partial w}{\partial t} = \varepsilon(V + a - bw)$$

$$f(V,w) = V - \tfrac{1}{3}V^3 - w + I_{\text{stim}}(x,t)$$

- Parameters: `D₀ = 0.001 cm²/ms`, `L = 2.0 cm`, `a = 0.7`, `b = 0.8`,
  `ε = 0.08`.
- **State assumption A1 here**: one dimensionless FHN time unit is *declared*
  equal to 1 ms. This is a calibration choice, not a derivation — say so in the
  introduction, not buried later.

---

## 2. Mathematical model and analytical results

`[WRITE]`

**Figure:** `fig_ex01_single_cell` (four panels: AP and recovery; all-or-none;
phase plane with nullclines; threshold bisection).

**Numbers available** (all in `results/ex01_single_cell_summary.csv`):

| Quantity | Value |
|---|---|
| Rest state | `V* = −1.199408035`, `w* = −0.624260044` |
| Jacobian | `tr J = −0.502580`, `det J = +0.108069`, disc `= −0.179690` |
| Eigenvalues | `−0.251290 ± 0.211949 i` |
| Classification | **stable spiral** → excitable, not oscillatory |
| Bistable roots | `V₁ = −1.199408`, `V₂ = −0.786321`, `V₃ = +1.985729` |
| CV prefactor | `θ/√D = 0.963043` |
| Analytic CV at `D₀` | `0.030454 cm/ms = 30.45 cm/s` |
| Stimulus threshold | `A* = 0.606281` (configured `A = 1.0` is 1.65×) |

**Derivations to present:**

1. Rest state: `w* = (V*+a)/b` and `V*³ + 0.75V* + 2.625 = 0`.
2. Jacobian `J = [[1−V*², −1], [ε, −εb]]`; trace–determinant classification.
3. Frozen-`w` reduction → `V³ − 3V + 3w* = 0`; front speed
   `θ = √(A/2)(V₁ − 2V₂ + V₃)√D` with `A = 1/3`.
4. **The identity worth featuring:** roots sum to zero, so
   `V₁ − 2V₂ + V₃ = −3V₂` and hence `θ/√D = −3V₂/√6`. *The front speed is
   controlled entirely by how far the excitation threshold sits from rest* —
   this is the mechanism of block, stated in one line.

**Discrepancies to report honestly:** the brief quotes `tr J = −0.5034` and
prefactor `0.9634`; recomputing from the exact rest state gives `−0.502580` and
`0.963043` (0.16 % and 0.04 %). Both are rounding in the brief's intermediates,
confirmed by independent recomputation. See `verification_log.md` V3, V9.

---

## 3. Numerical method

`[WRITE]`

**Content available** — the full argument for each choice is in
[`../docs/numerical_choices.md`](../docs/numerical_choices.md).

### 3.1 Spatial discretisation

- Conservative flux form; **never** `D(x)∂²V/∂x²`. Charge is conserved to
  1.33e−15 across a 100-fold `D` step.
- Harmonic interface averaging: `D ∝ 1/r`, series resistances add.
- Sealed ends via reflecting ghost nodes: the end formula
  `2D_{1/2}(V₁−V₀)/Δx²` *is* the interior formula with `V₋₁ = V₁` substituted.
- Trapezoidal quadrature weights (half-weight end cells) — required for exact
  discrete conservation; there is a negative-control test.

### 3.2 Time integration and stability

- Hand-coded explicit Euler; **no adaptive solver**, because an adaptive
  controller would hide the stability limit part (c) asks for.
- Von Neumann analysis → `Δt ≤ 2/(4D/Δx² + |f_V|) = 2/43 = 0.046512 ms`,
  binding at the checkerboard mode `k = π/Δx`.
- The pure-diffusion limit `Δx²/(2D) = 0.05 ms` is **7.50 % too optimistic**.
- Working step `Δt = 0.02 ms`, safety factor **2.33**.

### 3.3 Grid resolution

- `Δx = 0.01 cm`: ≈ one myocyte length; resolves the balance length
  `√(D/|f_V|) = 0.0183 cm`; carries only **0.52 %** discretisation error in the
  reported velocity (Richardson-extrapolated to `0.0256705 cm/ms`).

---

## 4. Verification

`[WRITE]`

**Figures:** `fig_ex02_stability`, `fig_ex03_convergence`,
`fig_ex04_pure_diffusion`.

**Full table:** [`../docs/verification_log.md`](../docs/verification_log.md)
(47 numbered checks, 93 automated tests, no skips).

**Headline verification results:**

| Check | Result |
|---|---|
| Analytic-limit test: pure diffusion vs exact sealed solution | `L∞ = 1.788e−05` |
| Charge conservation, 1000 steps | `1.33e−15` relative drift |
| Spatial order (asymptotic) | **1.9987** (design 2) |
| Temporal order, Euler | **0.9988** (design 1) |
| Temporal order, RK4 (smooth) | **3.9545** (design 4) |
| Measured vs predicted amplification | max error `2.22e−16` |
| Empirical stability boundary | 1.065× the conservative limit |

**Three points that will earn marks if written up properly:**

1. **The instability is the *predicted mode*, not generic noise.** ex02 panel
   (b) shows the sign alternating every single node — `k = π/Δx` exactly as the
   von Neumann analysis says. Amplification matches `|1 − 4DΔt/Δx²|` to machine
   precision.

2. **RK4 measures first order when the stimulus is included** (1.0197, against
   3.9545 on smooth propagation). The stimulus switches off discontinuously at
   1 ms, and a jump in the forcing caps *any* one-step method at order 1. This
   is a property of the problem, not a defect in the implementation — and it is
   why the temporal study is run twice.

3. **The infinite-line Gaussian is the wrong reference.** At 20 ms the boundary
   reflection contributes `2.03e−05`, comparable to the discretisation error
   itself. `analytic_gaussian_sealed` (method of images) is exact for the sealed
   strand and isolates the discretisation error properly.

---

## 5. Results

`[WRITE]`

### 5.1 Conduction velocity and the `√D` law

**Figure:** `fig_ex05_cv_vs_D`.

| Quantity | Value |
|---|---|
| Log-log exponent (resolution-matched grid) | **0.49996** (`R² = 0.99999999`) |
| `θ/√D` spread, resolution-matched grid | **0.02 %** |
| `θ/√D` spread, fixed grid | 1.92 % |
| Measured prefactor | 0.807418 |
| Analytic prefactor (`w` frozen at rest) | 0.963043 |
| Prefactor using measured `w` at the front | 0.869256 |

**The `θ/√D` flatness plot is the sharper test** and should be presented as
such: a log-log fit can return 0.500 even when the data curve, whereas
systematic drift is immediately visible in the ratio. (The project's own test
suite records that `R² > 0.9999` survives a few per cent of curvature.)

**Explain the −16 % discrepancy properly.** It is *physical*: grid refinement
to `Δx = 0.00125 cm` shows only 0.52 % discretisation error at the default
spacing, so the gap survives refinement. The analytic derivation freezes `w` at
`w* = −0.6243`, but `w` has already risen to **−0.5906** by the time the front
passes. Substituting the measured value closes the gap from −16.2 % to −7.1 %;
the residual is because `w` varies *across* the front rather than sitting at any
one value.

**Also report:** the fixed 0.1 cm stimulus fails to launch a wave above
`D ≈ 0.002 cm²/ms`, because the liminal length scales as `√D`.

### 5.2 The conduction-block threshold

**Figure:** `fig_ex06_block_threshold`.

| Gap length | `ρ_crit` (harmonic) |
|---|---|
| 0.01 cm | 0.06703 |
| 0.02 cm | 0.13242 |
| 0.04 cm | 0.14847 |
| 0.06 cm | 0.15438 |
| 0.10 cm | 0.15725 |
| ≥ 0.15 cm | **0.15750 (saturated)** |

**The central claim: this is a curve, not a number.** The threshold rises 135 %
from a single-node gap to its asymptote. Do not let the text imply a single
critical coupling ratio.

**The saturation is itself a result.** Beyond ~0.1 cm the threshold stops
depending on length, because once the gap exceeds the distance over which
failure develops, survival depends on the *local* coupling alone rather than on
how far the wave must travel.

**Numerical-choice sensitivity.** Arithmetic averaging shifts the threshold by
up to 4.2 % — and at a single-node gap **cannot produce block at any coupling
ratio whatsoever**, because its interface conductance tends to `D₀/2` rather
than to zero as `ρ → 0`. This is the strongest available argument for the
harmonic mean.

### 5.3 Conduction delay

**Figure:** `fig_ex07_conduction_delay`.

| Quantity | Value |
|---|---|
| `ρ_crit` (`L_gap = 0.1 cm`) | 0.1572519307 |
| Healthy transit (0.65 → 1.35 cm) | 27.41 ms |
| Maximum transit near threshold | 43.38 ms |
| Saturated excess delay | **15.96 ms** |
| Change over the final 3 decades of approach | 15.5 % |
| Power-law exponent | −0.072 (`R² = 0.59`) |
| Scaling of the ceiling with `ε` | exponent −1.46 (`R² = 0.95`) |

**Report the negative result prominently.** The delay was expected to diverge
as `(ρ − ρ_crit)^(−1/2)`. Measured over **six decades** of approach, it does
not: it saturates. A stalled front cannot outlast its own upstream source,
which repolarises on the recovery timescale `1/ε` — confirmed by varying `ε`.
The transition to block is therefore **discontinuous in delay**: the wave
crosses within roughly the recovery time, or not at all.

### 5.4 Sensitivity

**Figure:** `fig_ex08_sensitivity`.

| Parameter | `S(θ)` | `S(ρ_crit)` |
|---|---|---|
| `a` | −1.2479 | **+7.8549** |
| `b` | −0.8541 | +4.4850 |
| `ε` | −0.2534 | +2.7703 |
| `D₀` | **+0.5044** | −0.0218 |
| `L_gap` | 0.0000 | +0.0216 |
| `Δx` *(numerical)* | −0.0073 | +0.0305 |
| `Δt` *(numerical)* | −0.0017 | −0.0085 |

**Two things to draw out:**

1. `S(θ, D₀) = 0.5044` against **exactly 0.5** — an independent confirmation of
   the `√D` law by a completely different route from §5.1, since a square root
   has elasticity exactly ½.
2. The **numerical** parameters have elasticities ~257× smaller than the leading
   physical one. The conclusions are properties of the model, not of the grid.

---

## 6. Discussion

`[WRITE]`

**Threads worth developing:**

- **Mechanism.** `θ/√D = −3V₂/√6` ties everything together: weakening coupling
  slows the front, `w` accumulates during the slower upstroke, which lifts `V₂`
  towards rest, which slows it further — positive feedback that terminates in
  block. This explains both the velocity deficit (§5.1) and the delay ceiling
  (§5.3) from one idea.
- **Why the delay ceiling matters clinically.** A region conducting 16 ms late
  is arguably more dangerous than one that blocks: block removes a pathway,
  delay creates a late one. Note the caveat that FHN's action potential is 4–6×
  too short, so refractoriness-dependent claims are out of reach here.
- **Where the model is optimistic about block.** 1-D (A2) and continuum (A4)
  both make block *easier* than in real tissue, which can conduct saltatorily
  and can route around a patch. `ρ_crit ≈ 0.157` is an upper bound on how
  easily block occurs.
- **Numerical choices as modelling choices.** The harmonic-mean result shows a
  discretisation decision changing a physical conclusion qualitatively. Worth a
  paragraph.

---

## 7. Limitations

`[WRITE]` — draw from
[`../docs/assumption_register.md`](../docs/assumption_register.md) §E.

Chief among them: 1-D continuum; FHN rather than physiological ion channels;
APD too short by 4–6×; single beat only, so no restitution or re-entry;
one-at-a-time sensitivity cannot detect parameter interactions; the time-unit
calibration (A1) means absolute velocity agreement is weak evidence.

---

## 8. Conclusions

`[WRITE]`

Suggested skeleton:

1. The medium is excitable (stable spiral), not oscillatory.
2. `θ ∝ √D` confirmed to exponent 0.49996 by two independent routes.
3. The measured prefactor is 16 % below the frozen-`w` analytic value, and the
   deficit is explained quantitatively by the rise of `w` at the front.
4. The block threshold is a curve in `(L_gap, ρ)`, saturating at `ρ ≈ 0.157`.
5. Conduction delay saturates at ~16 ms rather than diverging, bounded by the
   recovery timescale.
6. The interface averaging scheme is not a free choice: the arithmetic mean
   makes a thin gap unblockable.

---

## 9. Reproducibility statement

Every figure in this report regenerates from a clean clone:

```bash
git clone <repo> && cd FibroBlock
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/make_all_figures.py
pytest
```

- Total pipeline runtime: **roughly 6–10 min** (6 m 15 s and 9 m 14 s measured
  on two runs of the same machine), producing 8 figures and their results files.
- Test suite: **93 tests, all passing, no skips.**
- Every result file carries a JSON provenance sidecar: UTC timestamp, git
  commit (flagged `-dirty` if the tree is unclean), library versions, platform,
  and the complete `RunConfig` that produced it.
- Dependencies pinned with `==`; `requirements.txt` and `environment.yml` carry
  identical pins.
- Random seed fixed at **20260810** and printed by every script. The model is
  deterministic and consumes no random numbers; the seed is fixed and reported
  because the brief requires it.

**Environment used for the committed figures:** `[PASTE output of
python scripts/check_environment.py]`

---

## Appendix A — Figure index

| Figure | File | Takeaway |
|---|---|---|
| 1 | `fig_ex01_single_cell` | Excitable, not oscillatory; threshold `A* = 0.6063` |
| 2 | `fig_ex02_stability` | The von Neumann limit is real, and fails via `k = π/Δx` |
| 3 | `fig_ex03_convergence` | Design orders achieved; stimulus discontinuity caps RK4 |
| 4 | `fig_ex04_pure_diffusion` | Matches exact solution; charge conserved to 1e−15 |
| 5 | `fig_ex05_cv_vs_D` | `θ ∝ √D`, exponent 0.49996 |
| 6 | `fig_ex06_block_threshold` | Threshold is a curve; arithmetic mean fails qualitatively |
| 7 | `fig_ex07_conduction_delay` | Delay saturates, does not diverge |
| 8 | `fig_ex08_sensitivity` | `S(θ,D₀) = 0.5044`; numerical influence 257× smaller |

Full takeaway captions are stored in `results/fig_*_meta.json`.

## Appendix B — Supporting documents

- [`docs/assumption_register.md`](../docs/assumption_register.md) — 17 numbered
  assumptions with justification and consequences
- [`docs/verification_log.md`](../docs/verification_log.md) — 47 checks,
  expected vs obtained
- [`docs/validation_log.md`](../docs/validation_log.md) — published ranges
  ⚠️ *citations require verification before submission*
- [`docs/numerical_choices.md`](../docs/numerical_choices.md) — 14 decisions
  defended
- [`docs/code_walkthrough.md`](../docs/code_walkthrough.md) — module-by-module
  tour
- [`docs/ai_use_declaration.md`](../docs/ai_use_declaration.md) — AI assistance
  declaration
