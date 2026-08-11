# Code walkthrough

A module-by-module tour of FibroBlock, written to be revised from the night
before the viva. For each file: what it does, why it exists, the parts most
likely to be asked about, and the one-line answer to the obvious question.

Read this alongside [`numerical_choices.md`](numerical_choices.md), which
justifies the decisions; this file explains the *structure*.

---

## The shape of the whole thing

```
config.py ──► grid.py ──► operators.py ──┐
     │           │                       ├──► simulate.py ──► measure.py ──► experiments/
     │           └──► (D_half, weights)  │          ▲
     ├──► fhn.py ──────────────────────┬─┘          │
     │       (kinetics, rest state)    │            │
     └──► stimulus.py ─────────────────┘   solvers.py (euler/rk4 + stability)
```

Everything flows one way. `config.py` depends on nothing; `simulate.py` depends
on almost everything; experiments depend only on the public API. **There is no
global mutable state anywhere** — a run is a pure function of its `RunConfig`,
which is what makes the bitwise-reproducibility tests possible.

---

## `config.py` — the single source of truth

**What.** Every parameter in the project, as frozen dataclasses: `FHNParams`,
`GridParams`, `GapParams`, `StimulusParams`, `SolverParams`,
`MeasurementParams`, all composed into `RunConfig`.

**Why frozen.** A configuration cannot be mutated half-way through a run, so a
result file's recorded configuration is *guaranteed* to be the one that produced
it. Sweeps use `.replace()`, which returns a new object, so one sweep point can
never leak into another.

**The rule it enforces.** No numeric literal anywhere else in `src/` unless it
is an exact mathematical constant with a comment saying so. If you find a
magic number in the codebase, it is a bug.

**Also here.** `BRIEF_*` constants holding the values quoted in the assignment.
These are used **only by the tests**, never by the solver, so that agreement
with the brief is a *check* rather than an input.

> **Likely question: "Why is `time_unit_ms` a parameter when it is always 1?"**
> Because it is an *assumption* (A1), not a fact. Making it explicit means it
> appears in every saved configuration and can be varied to test its effect.
> FHN is dimensionless; attaching milliseconds is a calibration choice.

---

## `fhn.py` — kinetics and everything analytic

**What.** `f(V,w)` and `g(V,w)`; the rest state; the Jacobian and its
classification; the frozen-`w` bistable roots; the analytic front-speed
prefactor; front thickness.

**The key design decision.** Nothing is hard-coded from the brief. `rest_state`
solves `V³ + (3/b − 3)V + 3a/b = 0` at run time — `numpy.roots` for robustness,
then a `brentq` polish for full double precision. Change `a` in the config and
everything downstream follows.

**The chain worth memorising:**

```
rest state (V*, w*)
   └─► freeze w at w*  ─►  V³ − 3V + 3w* = 0  ─►  roots V₁, V₂, V₃
                                                    └─► θ/√D = √(A/2)(V₁ − 2V₂ + V₃)
                                                              = −3V₂/√6   [since ΣVᵢ = 0]
```

**The identity to have ready.** The cubic has no quadratic term, so the roots
sum to zero, so `V₁ − 2V₂ + V₃ = −3V₂`. **The front speed is controlled
entirely by how far the excitation threshold sits from rest.** Push `V₂` towards
`V₁` and the front slows and eventually fails — that is the whole mechanism of
conduction block in one line.

**Later addition.** `bistable_roots_at(w)` and `front_speed_prefactor_at(w)`
generalise the above to any frozen `w`, which is what ex05 uses to explain the
−16 % velocity discrepancy.

> **Likely question: "Why both `numpy.roots` and `brentq`?"**
> `numpy.roots` is robust but only ~10 significant digits; `brentq` on a tight
> bracket converges to full precision. Using both means the answer does not
> depend on either routine's failure modes alone.

---

## `grid.py` — geometry, `D(x)`, and half-node averaging

**What.** Node positions, the piecewise-constant diffusion profile, the
harmonic/arithmetic interface averaging, trapezoidal quadrature weights, and the
`Grid` object that caches them all.

**The layout to be able to draw:**

```
node       0     1     2  ...  N−1    N       (N+1 nodes)
x          0    Δx   2Δx      L−Δx    L
half-node     ½    1½   ...   N−½              (N half-nodes)
```

201 nodes, 200 intervals, 200 half-nodes at `Δx = 0.01 cm` on a 2 cm strand.

**Where the harmonic mean lives.** `half_node_diffusion`, called **once** from
`build_grid`, never inside the time loop. The arrays are then made read-only
with `setflags(write=False)` — a frozen dataclass would still allow its arrays'
*contents* to be mutated, and a stray in-place edit mid-sweep would be very hard
to find.

**The trapezoidal weights are not cosmetic.** `[Δx/2, Δx, …, Δx, Δx/2]`. In the
finite-volume reading the end nodes own half-width cells. With these weights the
no-flux operator conserves charge exactly; with uniform weights it does not, and
there is a negative-control test proving it.

> **Likely question: "Why does `D_max` read `D_half` and not `D_nodes`?"**
> Because the node update uses `D_{j−1/2}` and `D_{j+1/2}`. Those are what enter
> the stability limit.

---

## `operators.py` — the conservative divergence

**The most important file in the project.** Twenty lines of arithmetic that the
whole result rests on.

```python
flux         = D_half * (V[1:] - V[:-1]) / dx     # interfaces
result[1:-1] = (flux[1:] - flux[:-1]) / dx        # interior
result[0]    =  2.0 * flux[0]  / dx               # left, ghost node
result[-1]   = -2.0 * flux[-1] / dx               # right, mirror
```

**Where the factor of 2 comes from.** Substituting the reflecting ghost values
`V₋₁ = V₁` and `D_{−1/2} = D_{1/2}` into the *interior* formula collapses it to
`2 D_{1/2}(V₁ − V₀)/Δx²`. The end formulas are not special cases bolted on
afterwards — they are what the interior formula *becomes*.

**Why fluxes and not `D V_xx`.** Every interior flux appears twice with opposite
signs and cancels exactly, so the only charge that can leave is what crosses the
boundaries — which no-flux sets to zero. The expanded form has no such structure
and creates charge wherever `D` varies, i.e. exactly at the interface being
studied.

**Also here.** `analytic_gaussian` (infinite line) and
`analytic_gaussian_sealed` (method of images, exact for this problem).
Keeping both is what lets ex04 separate discretisation error from boundary
reflection — at 20 ms they are comparable in size.

> **Likely question: "Is `boundary_flux` returning zero a real test?"**
> No, and the docstring says so. The centred derivative at node 0 is
> `(V₁ − V₋₁)/2Δx` with `V₋₁ := V₁`, so it is a float minus itself — zero
> bit-for-bit, by construction. The *substantive* test of the boundary treatment
> is discrete charge conservation, which could fail if the operator were wrong.

---

## `solvers.py` — integrators and the stability analysis

**What.** `euler_step`, `rk4_step`, the stability-limit calculator, and the
checkerboard-mode utilities.

**The derivation, in the module docstring.** Substituting
`V_j^n = g^n e^{ikjΔx}` into the Euler update gives

```
g(k) = 1 + Δt[ −(4D/Δx²)sin²(kΔx/2) + f_V ]
```

`|g| ≤ 1` binds on the `g ≥ −1` side, and the worst case is `sin² = 1`, i.e.
`kΔx/2 = π/2`, i.e. **`k = π/Δx` — the checkerboard mode**, alternating sign
node to node. Hence

```
Δt ≤ 2 / (4D_max/Δx² + |f_V|_max) = 2/43 = 0.046512 ms
```

**Two numbers to keep straight.** The `4` (not 2) is because the second
difference of an *alternating* field is `−4` times the field. The `2` in the
numerator is because the binding constraint is `g ≥ −1`, not `g ≤ 1`.

**RK4 is written out stage by stage**, not table-driven, so it can be checked
against the textbook line by line.

> **Likely question: "Why not just use `dx²/(2D)`?"**
> That ignores the reaction term. It gives 0.05 ms against the true 0.046512 —
> **7.5 % too optimistic.** ex02 reports both for exactly this reason.

---

## `stimulus.py` — the pulse and the bisection helper

**What.** The rectangular stimulus mask and current, plus `bisect_threshold`.

**The half-open time window** `[start, start + duration)` matters: a closed
window would apply one extra step's worth of current whenever `duration/Δt` is
an integer, making the delivered charge depend on the time step.

**`bisect_threshold` is used for four different things** — stimulus amplitude
(ex01), the empirical stability boundary (ex02), the critical coupling ratio
(ex06, ex07, ex08). It bisects on a **boolean**, not a residual, which is why
bisection rather than Brent: there is nothing to interpolate.

**It checks both endpoints first and raises if the bracket does not straddle.**
That is not defensive padding — it is what surfaced the finding that arithmetic
averaging cannot block a single-node gap at any `ρ`.

---

## `simulate.py` — the one time loop

**What.** `run_simulation(config, ...) -> SimulationResult`, plus
`run_single_cell` for the space-clamped case.

**Structure.** Everything is precomputed *before* the loop: the grid, `D_half`,
the stimulus mask, the reusable stimulus buffer. The loop itself does one
`step()` call and some bookkeeping.

**What is stored, and why.** The full `V(x,t)` history is **off by default** —
15 000 × 201 doubles per field, and nothing reported needs it. What is always
recorded is cheap and sufficient:

- **activation times**, accumulated *during* the run (they cannot be recovered
  afterwards from downsampled snapshots without losing sub-step resolution);
- peak `V` and peak `dV/dt` per node;
- downsampled snapshots for the space-time plots;
- total charge at each snapshot.

**Sub-step interpolation of the activation time** is worth understanding: at
`Δt = 0.02 ms` one step is ~1.5 node spacings, so without it the activation map
would be visibly stepped and the velocity fit would inherit the quantisation as
noise.

**The divergence check runs every step.** Above the stability limit the cubic
makes the update `V ← −Δt V³/3`, which *cubes* the magnitude each step: about
seven steps from `V = 10` to overflow. A per-snapshot check missed that window.

> **Likely question: "Why does `_validate_time_step` raise instead of warning?"**
> Because a silently unstable run produces a figure that looks like physics.
> `force=True` exists solely so ex02 can step past the limit deliberately.

---

## `measure.py` — turning a run into reported numbers

**What.** Conduction-velocity fitting, block detection, conduction delay,
recovery-at-front, error norms, observed order, log-log fitting.

**Why it is a separate module.** The *definitions* behind every reported number
live in one readable file rather than being scattered through eight experiment
scripts. If someone asks "what exactly do you mean by conduction velocity?",
there is one function to point at.

**`fit_conduction_velocity`** fits `t_act = x/θ + t₀` over `x ∈ [0.5, 1.8] cm`
and reports `R²` as evidence of steady propagation. **It refuses (returns NaN)
rather than guessing when fewer than three nodes activated in the window** —
with two points `R²` is identically 1 and the velocity is meaningless.

**`detect_block`** deliberately uses `activation_time_crossing`, not the
configured activation rule, because the criterion is *defined* in terms of
`V = 0`. Using the steepest-upstroke definition here would silently change the
criterion.

**`recovery_at_front`** is the measurement behind the ex05 explanation: it reads
`w` at the instant `V` crosses the threshold root, interpolated between
snapshots.

---

## `plotting.py` — house style

**What.** The Okabe-Ito colour-blind-safe palette, `new_figure`, `label_axes`,
`shade_gap`, `annotate_takeaway`, `set_log_ticks`, `save_figure`.

**Two structural enforcements worth knowing:**

- **`label_axes` raises on an empty label.** Every plot routes through it, so an
  unlabelled axis would have to be created deliberately rather than by
  omission.
- **`save_figure` raises on a caption under 40 characters.** Crude, but enough
  to stop a placeholder reaching the report. Captions must state a *takeaway*,
  not a description.

`matplotlib.use("Agg")` is set **before** `pyplot` is imported, so the pipeline
runs head-less.

---

## `utils.py` — reproducibility plumbing

**What.** Path discovery, seeding, timing, provenance, and file output.

**`project_root()`** searches upward for `pyproject.toml` from the package
location *and* from the working directory, so experiments run from anywhere on
any drive.

**`provenance()`** returns timestamp, git commit (with `-dirty` if the tree has
uncommitted changes), library versions, and platform. Every saved array and
every figure gets one.

**`git_commit_hash()` never raises.** A missing git installation or a downloaded
ZIP must degrade the provenance, not break the run.

**Provenance goes in a sidecar JSON, not inside the `.npz`.** NumPy would have
to pickle a dict, and loading pickled data needs `allow_pickle=True` — a
security foot-gun and a portability risk. Plain JSON is readable by anything,
including a text editor.

---

## `experiments/` — one script per figure

Each is standalone (`python experiments/exNN_*.py`), exposes `main() -> dict`,
and puts `src/` on the path so it works without an editable install.

| Script | The question, and the answer |
|---|---|
| **ex01** | Is the medium excitable? Yes — stable spiral, threshold `A* = 0.6063`, all-or-none. |
| **ex02** | Does the stability limit bite? Yes, at 1.065× the conservative bound, via the predicted checkerboard mode. Amplification matches theory to 2e−16. |
| **ex03** | Are the design orders achieved? 2.00 spatial, 1.00 Euler, 3.95 RK4 — and 1.02 for RK4 *with* the discontinuous stimulus. |
| **ex04** | Does pure diffusion match the exact solution? To 1.8e−5, with charge conserved to 1.3e−15. |
| **ex05** | Does `θ ∝ √D`? Exponent 0.49996; `θ/√D` flat to 0.02 % on a resolution-matched grid. |
| **ex06** | Where is the block threshold? A *curve*: 0.067 → 0.157, saturating past 0.1 cm. Arithmetic averaging cannot block a 1-node gap at all. |
| **ex07** | Does the delay diverge? **No** — it saturates at 16 ms, capped by the recovery timescale. |
| **ex08** | What does it depend on? `S(θ, D₀) = 0.5044` (confirming `√D` independently); numerical elasticities 257× smaller than physical ones. |

---

## `tests/` — verification evidence, not hygiene

93 tests, no skips. Each docstring states **which report claim it supports**, so
the suite doubles as the evidence table in `verification_log.md`.

Three tests are deliberately *negative controls*, and they are the ones worth
mentioning in a viva because they show the positive results are not vacuous:

- `test_uniform_weights_would_NOT_conserve_charge` — proves the trapezoidal end
  weights are load-bearing.
- `test_a_different_config_gives_a_different_answer` — proves the determinism
  tests are not passing because everything returns the same thing.
- `test_r_squared_is_a_weak_detector_of_MILD_curvature` — records that
  `R² > 0.9999` is *necessary but not sufficient*, which is why ex05 also plots
  `θ/√D` directly.

---

## `scripts/make_all_figures.py` — the reproducibility entry point

Deletes `figures/`, `results/` and `report/figures/`, runs all eight
experiments, copies figures to the report, writes `results/pipeline_run.json`,
and prints a timing table. **Total runtime ≈ 6–10 min**, dominated by ex06,
ex07 and ex08, which each run hundreds of bisection simulations.

Deleting first is the point: if the directories were merely overwritten, a
figure whose generating code had been removed would survive and the examiner
would see output the current code cannot produce.

Failures are caught per stage rather than aborting, so a break in ex03 still
shows you that 4–8 work — and the exit status is still non-zero.

---

## The five things most worth being able to say cold

1. **Conservative form, harmonic mean.** `D ∝ 1/r`, series resistances add;
   the arithmetic mean cannot even block a single-node gap because its
   interface conductance tends to `D₀/2` rather than zero.
2. **`Δt ≤ 2/(4D/Δx² + |f_V|)` = 0.046512 ms**, binding at the checkerboard mode
   `k = π/Δx`; the familiar `Δx²/2D` is 7.5 % too optimistic.
3. **`θ/√D = −3V₂/√6`.** The front speed is set by how far the threshold root
   sits from rest — which is why weakening coupling eventually blocks.
4. **The −16 % velocity gap is physical, not numerical.** Grid-converged to
   0.52 %; caused by `w` rising to −0.5906 before the front arrives, not
   sitting at `w* = −0.6243`.
5. **The delay saturates, it does not diverge.** A stalled front cannot outlast
   its own upstream source, which repolarises on `1/ε`.
