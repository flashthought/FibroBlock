# FibroBlock

**Cardiac action-potential propagation and conduction block in a 1-D monodomain strand.**

COE 562 — Engineering Systems Design and Modelling (MPhil), Kwame Nkrumah
University of Science and Technology. Problem 8.

---

## What this project does

It solves the **monodomain cable equation** with **FitzHugh–Nagumo** kinetics on
a 2 cm strand of cardiac tissue,

$$\frac{\partial V}{\partial t} = \frac{\partial}{\partial x}\!\left(D(x)\,\frac{\partial V}{\partial x}\right) + f(V, w),
\qquad
\frac{\partial w}{\partial t} = \varepsilon\,(V + a - b\,w),$$

$$f(V, w) = V - \tfrac{1}{3}V^{3} - w + I_{\text{stim}}(x, t),$$

and uses it to answer one question: **how weak does intercellular coupling have
to get, over how long a patch, before an advancing action potential fails to
cross it?** That patch is the model of a fibrotic region — hence *FibroBlock*.

The headline results are:

| Quantity | Result |
|---|---|
| Rest state | `V* = -1.199408`, `w* = -0.624260` |
| Rest-state stability | **stable spiral** (`tr J = -0.5026`, `det J = +0.1081`) — excitable, not oscillatory |
| Explicit-Euler stability limit | `Δt ≤ 0.046512 ms`; the diffusion-only estimate `0.05 ms` is **7.5 % too optimistic** |
| Conduction velocity scaling | `θ ∝ √D`, measured exponent **0.49996**; `θ/√D` flat to **0.02 %** |
| Measured vs analytic velocity | 25.5 vs 30.5 cm/s — a **16 % deficit that is physical, not numerical** (see below) |
| Block threshold | a **curve** in `(L_gap, ρ)`: 0.067 → 0.157, saturating past 0.1 cm |
| Conduction delay | **saturates at 16 ms — it does not diverge** |

Every one of those numbers is computed from the parameters at run time. None is
hard-coded, and each is covered by an automated test.

### Three results worth reading the code for

1. **The velocity deficit is physical.** The measured prefactor is 0.807
   against the analytic 0.963. Grid refinement to `Δx = 0.00125 cm` leaves only
   0.52 % discretisation error, so the gap survives refinement. The analytic
   derivation freezes the recovery variable at rest; `w` has actually risen to
   −0.591 by the time the front passes. Substituting the measured value closes
   the gap to 7 %.

2. **The delay does not diverge.** A `(ρ − ρ_crit)^(−1/2)` divergence was
   expected. Measured over six decades of approach, the excess delay saturates
   at 16 ms: a stalled front cannot outlast its own upstream source, which
   repolarises on the recovery timescale `1/ε`.

3. **The averaging scheme is not a free choice.** With the arithmetic mean, a
   single-node gap **cannot be blocked at any coupling ratio at all**, because
   its interface conductance tends to `D₀/2` rather than to zero as `ρ → 0`.

---

## Reproducing every figure from a clean clone

This block works verbatim on a clean Windows machine with Python 3.11 and git
installed. Copy and paste it as-is.

```
git clone <repo>
cd FibroBlock
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/make_all_figures.py
pytest
```

On Linux or macOS the only change is the activation line:

```bash
source .venv/bin/activate
```

`make_all_figures.py` deletes and rebuilds `figures/`, `results/` and
`report/figures/` from empty, runs all eight experiments in order, and prints a
timing summary. **It takes roughly 6–10 minutes** (measured 6 m 15 s and
9 m 14 s on two runs of the same machine; ex06, ex07 and ex08 dominate, and each
runs hundreds of bisection simulations). `pytest` then re-checks
every analytic result the report claims — **93 tests, no skips.**

Deleting first is the point: if the directories were merely overwritten, a
figure whose generating code had been removed would survive from an earlier run,
and you would be looking at output the current code cannot produce.

If you have GNU make available, `make all` does both steps.

### What gets produced

- `figures/` — one PNG (300 dpi) and one PDF per figure, named `fig_exNN_*`.
- `results/` — `.npz` arrays and `.csv` summary tables, each paired with a
  `.json` provenance stamp recording the exact config, git commit, library
  versions and timestamp that produced it.
- `report/figures/` — copies of the figures, pulled in by the report build.

Both `figures/` and `results/` are committed to the repository so that the
written report builds without running anything. They are nevertheless fully
regenerable, which is the point.

---

## Repository layout

```
src/fibroblock/     the library — all physics and numerics live here
  config.py           every parameter, as frozen dataclasses. Single source of truth.
  fhn.py              FHN kinetics, rest state, Jacobian, bistable roots, analytic CV
  grid.py             spatial grid, D(x) profiles, half-node averaging
  operators.py        conservative divergence operator with Neumann ghost nodes
  solvers.py          hand-coded explicit Euler and RK4; stability-limit calculator
  stimulus.py         stimulus current profiles and threshold bisection
  simulate.py         run_simulation() — the top-level entry point
  measure.py          activation times, CV fit, block detection, conduction delay
  plotting.py         house figure style; every helper labels axes with units
  utils.py            seeding, timing, provenance stamping, npz/csv I/O

experiments/        one script per report figure, each runnable standalone
scripts/            make_all_figures.py (the reproducibility entry point)
tests/              pytest suite — verification evidence for the report
docs/               assumption register, verification log, validation log,
                    numerical choices, AI use declaration, code walkthrough
report/             report skeleton and BibTeX
slides/             viva outline
```

Each experiment can also be run on its own, for example:

```
python experiments/ex05_cv_vs_D.py
```

---

## The eight experiments

| Script | Question it answers |
|---|---|
| `ex01_single_cell.py` | Is the medium excitable? Where is the rest state, and what is the stimulus threshold? |
| `ex02_stability.py` | Does the explicit-Euler stability limit predicted by von Neumann analysis actually bite? |
| `ex03_convergence.py` | Is the scheme second-order in `Δx` and first-order in `Δt`, as designed? |
| `ex04_pure_diffusion.py` | With `f = 0`, does the solver match the analytic Gaussian, and is charge conserved? |
| `ex05_cv_vs_D.py` | Does conduction velocity really scale as `√D`? |
| `ex06_block_threshold.py` | For a gap of length `L_gap`, how weak must coupling `ρ` be to block propagation? |
| `ex07_conduction_delay.py` | How does conduction delay behave as `ρ` approaches the critical value? |
| `ex08_sensitivity.py` | Which parameters does the block threshold actually depend on? |

---

## Key modelling choices (and why)

Three choices matter more than the rest. All three are argued at length in
[`docs/numerical_choices.md`](docs/numerical_choices.md).

1. **Conservative form.** The diffusion term is discretised as
   `∂ₓ(D ∂ₓV)`, never as `D(x)·∂ₓₓV`. The latter is not conservative and
   silently creates and destroys charge at the coupling interface — exactly the
   place this project is trying to measure.

2. **Harmonic mean at half-nodes.** Interface conductances use
   `D_{j+1/2} = 2 D_j D_{j+1} / (D_j + D_{j+1})`, because `D ∝ 1/r` and axial
   resistances in series add. The arithmetic mean over-predicts coupling across
   a sharp interface and shifts the block threshold. Both schemes are
   implemented so the difference can be shown as a numerical-choice sensitivity
   result.

3. **Explicit Euler, not an adaptive solver.** Part (c) of the problem requires
   demonstrating the stability limit of the scheme. An adaptive controller such
   as `scipy.integrate.solve_ivp` would silently shrink the step and hide the
   very phenomenon being studied. SciPy is used only for root finding and
   least-squares fitting.

**Units convention.** One FitzHugh–Nagumo dimensionless time unit is *declared*
equal to 1 ms. This is a stated modelling assumption, not a physical
derivation. It is recorded in `config.py` and as assumption A1 in
[`docs/assumption_register.md`](docs/assumption_register.md).

**Block criterion.** Propagation is declared blocked if no node beyond
`x_gap + L_gap + 0.3 cm` reaches `V = 0` within 200 ms of the stimulus.

---

## Verification and validation

- [`docs/verification_log.md`](docs/verification_log.md) — every check, its
  expected value, the obtained value, and pass/fail.
- [`docs/validation_log.md`](docs/validation_log.md) — model output against
  published physiological ranges, with citations.
- [`docs/assumption_register.md`](docs/assumption_register.md) — numbered
  assumptions with justification and the effect if each is violated.
- [`docs/code_walkthrough.md`](docs/code_walkthrough.md) — module-by-module tour.
- [`docs/ai_use_declaration.md`](docs/ai_use_declaration.md) — honest declaration
  of AI assistance, as required by the brief.

Run `python scripts/check_environment.py` to print the platform, library
versions and git commit for the report appendix.

---

## Licence

MIT — see [LICENSE](LICENSE).
