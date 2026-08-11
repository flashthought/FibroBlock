# Declaration of AI assistance

Required by the assignment brief. Written to be honest rather than minimal.

---

## Summary

**An AI coding assistant (Anthropic Claude, via Claude Code) was used
extensively in producing this project.** It wrote the large majority of the
source code, the experiment scripts, the test suite, and the first drafts of the
documentation in this `docs/` directory, working from a detailed specification I
wrote setting out the problem, the governing equations, the parameters, the
required directory structure, the analytic targets to reproduce, and the
build order.

This declaration should be read as covering the whole repository unless a file
says otherwise.

---

## What the AI did

- **Source code.** All modules in `src/fibroblock/` were AI-written: the
  configuration dataclasses, FitzHugh–Nagumo kinetics, grid and half-node
  averaging, the conservative divergence operator, the hand-coded Euler and RK4
  integrators, the simulation driver, the measurement layer, plotting, and
  utilities.
- **Experiments.** All eight scripts in `experiments/`.
- **Tests.** All 93 tests in `tests/`.
- **Pipeline and scripts.** `scripts/make_all_figures.py`,
  `scripts/check_environment.py`.
- **Documentation.** First drafts of every file in `docs/`, the README, and the
  report and slides skeletons.
- **Diagnosis and redesign.** Several experiments were reworked by the AI after
  their first results contradicted the expected outcome — see "Findings that
  changed the design" below.

## What I did

- Wrote the specification that defined the problem, the physics, the parameters,
  the analytic results to be reproduced, the directory structure, the coding
  standards, and the phased build order with gates.
- Set the requirement that every analytic quantity be **computed from the
  parameters at run time** rather than hard-coded, which is what made the
  discrepancies in the next section detectable at all.
- Reviewed output at each build-phase gate.
- Retain responsibility for the submitted work, including everything below.

## What still needs my own work before submission

Stated plainly because it is not yet done:

- **`report/report.md` is a skeleton.** The prose analysis and discussion are
  mine to write.
- **Citations in `docs/validation_log.md` and `report/references.bib` are
  unverified.** They are flagged `[verify]` in the file. They were written from
  the model's background knowledge, not retrieved and checked. **Every one must
  be confirmed against the actual paper before submission**, including the
  specific numerical ranges attributed to each.
- **`slides/viva_outline.md`** is a skeleton for me to rehearse from.

---

## Findings that changed the design

These are recorded because they show where the work was genuinely
investigative rather than transcription, and because each is likely viva
material. In each case the first implementation produced a result that
contradicted the expectation, and the cause was tracked down rather than
tuned away.

1. **RK4 measured first-order, not fourth.** The stimulus switches off
   discontinuously at 1 ms, and a jump in the forcing caps *any* one-step
   method at order 1. `ex03` now measures the temporal order twice — with the
   stimulus (RK4 → 1.02) and on smooth propagation from a pre-formed wave
   (RK4 → 3.95) — and reports the cap as a result.

2. **Conduction velocity came out 16 % below the analytic prediction.** Grid
   refinement to `Δx = 0.00125 cm` established it was not a discretisation
   error (0.52 % at the default spacing). The cause is that the analytic
   derivation freezes `w` at rest, whereas `w` has already risen to −0.5906 by
   the time the front passes. Substituting the measured value closes the gap to
   7 %.

3. **The conduction delay does not diverge at the block threshold.** `ex07` was
   written expecting a `(ρ − ρ_crit)^(−1/2)` divergence and had to be rewritten
   around the data: over six decades of approach the delay saturates at 16 ms.
   The mechanism — a stalled front cannot outlast its own upstream source,
   which repolarises on the recovery timescale — was then tested by varying `ε`
   and confirmed.

4. **The arithmetic mean cannot block a single-node gap at any coupling
   ratio.** Expected to be a few-per-cent shift; it is qualitative. As `ρ → 0`
   the arithmetic interface conductance tends to `D₀/2` rather than to zero.

5. **A fixed 0.1 cm stimulus fails to excite the strand above
   `D ≈ 0.002 cm²/ms`**, because the liminal length scales as `√D`. The `D`
   sweep scales the stimulus width accordingly.

6. **Blow-up above the stability limit is hyper-exponential.** The cubic makes
   the update cube the magnitude each step — about seven steps from `V = 10` to
   overflow — so the divergence check had to move from the snapshot cadence to
   every step.

---

## My position on this

The AI did the implementation; the specification, the review, the physics
judgement calls, and the responsibility are mine. I can explain every line of
the code and every number in the results — that requirement drove the coding
standards in the specification (type hints, NumPy-style docstrings, the physics
commented rather than the syntax, no unexplained one-liners) and it is why
`docs/code_walkthrough.md` exists.

I would rather declare this fully and be marked on what I actually understand
than under-declare it.

---

*Model used: Anthropic Claude (Claude Code CLI). Assistance spanned the full
build. Last updated 10 August 2026.*
