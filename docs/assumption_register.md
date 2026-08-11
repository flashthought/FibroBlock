# Assumption register

Every modelling assumption made in FibroBlock, why it was made, and what would
change if it were violated. Numbered so the report and the viva can refer to
them directly ("under A1…").

The rightmost column is the honest one: it says what the assumption *costs*,
not just that it was made.

---

## A. Framing assumptions

| # | Assumption | Justification | Effect if violated |
|---|---|---|---|
| **A1** | One dimensionless FitzHugh–Nagumo time unit **is declared equal to 1 ms**. | FHN is a dimensionless caricature; it has no intrinsic timescale. Attaching a millisecond label is what lets the computed conduction velocity be compared with published cardiac values at all. Recorded in `config.py` as `FHNParams.time_unit_ms`. | Every velocity scales as `1/time_unit`. Declaring 1 unit = 2 ms would halve the reported cm/s and move the model out of the physiological range. **This is a calibration choice, not a derivation, and the agreement with measured cardiac velocities in §D is therefore weaker evidence than it looks.** |
| **A2** | The tissue is **one-dimensional**. | The assignment specifies a 1-D strand. A thin trabecula or a narrow isthmus is reasonably 1-D. | In 2-D and 3-D a wave can travel *around* a poorly coupled patch rather than through it, so block requires a barrier spanning the whole width. The critical coupling ratio reported here is therefore an **upper bound on how easily block occurs**; real tissue is harder to block. |
| **A3** | **Monodomain**, not bidomain: the extracellular space is assumed to be a perfect conductor at ground. | Standard for propagation studies where no external field is applied. Halves the number of unknowns. | Bidomain matters for defibrillation and for propagation through poorly coupled regions where extracellular potentials become significant. Here it would slightly change the effective coupling near the gap. |
| **A4** | Tissue is a **continuum**; discrete cell boundaries are not resolved. | `dx = 0.01 cm` is about one myocyte length, so the grid is at the edge of the continuum limit. | At very low coupling, real propagation becomes *saltatory* — hopping cell to cell — and the continuum model over-smooths it. The reported `rho_crit ≈ 0.157` is likely to be an **underestimate of how much uncoupling real tissue tolerates**, since discrete tissue can conduct in a decremental, jumping fashion the continuum model cannot represent. |

---

## B. Kinetics assumptions

| # | Assumption | Justification | Effect if violated |
|---|---|---|---|
| **A5** | **FitzHugh–Nagumo kinetics** stand in for cardiac ion channels. | Specified by the assignment. FHN captures the two features that matter for propagation and block: a cubic (three-branch) fast nullcline giving excitability, and a slow recovery variable giving refractoriness. | FHN has no separate sodium and calcium currents, so it cannot reproduce the two distinct mechanisms by which real tissue fails (sodium-dependent block at reduced excitability, calcium-mediated conduction at very low coupling). Reported thresholds are qualitative, not quantitative predictions for real myocardium. |
| **A6** | Parameters `a = 0.7`, `b = 0.8`, `eps = 0.08` as given. | Assignment brief. Verified to give a **stable spiral** rest state (`tr J = −0.5026`, `det J = 0.1081`), i.e. an excitable rather than oscillatory medium. | `eps` is the sensitive one: the block threshold has elasticity **+2.77** with respect to it (ex08). A 10 % error in `eps` moves `rho_crit` by 28 %. |
| **A7** | The recovery variable may be **frozen at `w*`** when deriving the analytic front speed. | `eps = 0.08` gives a recovery timescale of ~12.5 ms against an upstroke of ~1 ms. | **Measurably violated.** `w` has already risen to −0.5906 by the time the front passes, against `w* = −0.6243`. This is why the measured prefactor is 0.8074 rather than the analytic 0.9630, a −16 % discrepancy. Substituting the measured front value recovers most of it (0.8693, −7 %). Quantified in ex05. |

---

## C. Geometry, coupling and stimulus

| # | Assumption | Justification | Effect if violated |
|---|---|---|---|
| **A8** | `D(x)` is **piecewise constant** with a sharp step at each gap edge. | Simplest representation of a discrete fibrotic patch; makes the interface treatment (harmonic mean) the explicit object of study. | Real fibrosis has graded, irregular borders. A smooth transition over a few space constants raises the critical coupling (block becomes harder), because the wave is not asked to cross a discontinuity. |
| **A9** | Interface conductances use the **harmonic mean**. | `D ∝ 1/r` and series resistances add, so the resistance-correct interface value is harmonic. | Not a free choice — the arithmetic mean is simply wrong here, and ex06 shows the consequence is qualitative, not cosmetic: with arithmetic averaging a single-node gap **cannot be blocked at any coupling ratio**, because its interface conductance tends to `D0/2` rather than to zero as `rho → 0`. |
| **A10** | **Sealed (no-flux) ends.** | Assignment brief. Represents a strand terminating in non-excitable tissue. | The sealed end reflects the arriving wave and accelerates it, which is why the final 0.2 cm is excluded from every velocity fit. |
| **A11** | Coupling is **isotropic and homogeneous** outside the gap. | Keeps the experiment one-variable. | Real myocardium is anisotropic (roughly 3:1 in velocity) and heterogeneous at every scale. |
| **A12** | A single **rectangular stimulus** of amplitude 1.0 over `x ∈ [0, 0.1] cm` for 1 ms. | Assignment brief. Measured to be 1.65× the excitation threshold `A* = 0.6063` (ex01), so it is reliably supra-threshold without being so strong it distorts the early wave. | The stimulus must exceed the **liminal length**, which scales as `√D`. ex05 shows that a fixed 0.1 cm stimulus fails to launch a wave at all above `D ≈ 0.002 cm²/ms`; the sweep therefore scales the stimulus width with `√D`. |
| **A13** | The stimulus turns off **discontinuously** at 1 ms. | As specified. | A jump discontinuity in the forcing caps the achievable temporal order at 1 for any one-step method. ex03 measures this: RK4 achieves order 3.95 on smooth propagation but only **1.02** when the stimulus is included. |

---

## D. Measurement conventions

| # | Assumption | Justification | Effect if violated |
|---|---|---|---|
| **A14** | **Activation** is defined as the first upward crossing of `V = 0`. | `V = 0` lies between the threshold root (−0.786) and the excited plateau (+1.986), so it is crossed exactly once per upstroke. Selectable in `config.py` as `activation_rule`; the alternative is the time of maximum `dV/dt`. | The two definitions differ by a fraction of a millisecond and give slightly different velocities. **The report states which was used.** Both are computed and stored, so a figure can be redrawn under the other without re-running anything. |
| **A15** | Conduction velocity is fitted only over `x ∈ [0.5, 1.8] cm`. | The wave is still forming over roughly the first half-centimetre and is accelerated by the sealed end over the last 0.2 cm. `R² = 1.00000000` over the retained window is the evidence that what remains is genuinely steady propagation. | Including either region biases the velocity in a direction that still looks plausible — the worst kind of error. |
| **A16** | **Block criterion:** propagation has failed if no node at `x > x_gap + L_gap + 0.3 cm` reaches `V = 0` within 200 ms of the stimulus. | The 0.3 cm margin excludes the electrotonic foot, which raises `V` downstream of a blocked gap by passive spread alone (visible in ex06 panel d). 200 ms is far longer than any successful transit measured (maximum 43.4 ms). | A smaller margin would count passive charge spread as propagation. A shorter window would misclassify slow-but-successful conduction as block — though ex07 shows this is not a live risk, since the delay saturates at 16 ms of excess. |
| **A17** | `|f_V|_max = 3` in the stability estimate. | `|1 − V²|` reaches ≈3 at the excited root `V₃ = 1.9857`. | It is an **upper bound**, and deliberately conservative: the solution actually peaks at `V = 1.691`, giving `|f_V| = 1.86` and a true limit of 0.04778 ms. Measured empirically, failure begins at 1.065× the conservative limit (ex02). |

---

## E. What is not modelled at all

Listed explicitly so the report does not have to pretend these were considered
and dismissed.

- **Restitution and rate dependence.** A single beat is simulated; there is no
  premature stimulus, so the interaction between refractoriness and block —
  the actual mechanism of most re-entrant arrhythmia — is out of scope.
- **Stochastic ion-channel gating.** Deterministic throughout. This is why the
  random seed, though fixed and reported as the brief requires, changes
  nothing.
- **Mechanical coupling, metabolism, temperature.**
- **Fibroblast–myocyte electrical coupling**, which in real fibrosis loads the
  myocytes capacitively as well as reducing coupling.
- **Parameter interactions.** ex08 is a one-at-a-time sweep and by construction
  cannot detect them. A Sobol or full-factorial analysis is the natural
  extension.

---

## How to use this register in the viva

If asked "how confident are you in `rho_crit ≈ 0.157`?", the honest answer
chains three of these: it is a 1-D continuum result (**A2**, **A4**), so real
tissue is harder to block than this suggests; it is FHN rather than a
physiological ion-channel model (**A5**), so it is a mechanism study rather than
a prediction; and it is most sensitive to `eps` and `a` (**A6**), which are
inherited from the brief rather than fitted to data.

What the number *is* good for is comparison: harmonic against arithmetic
averaging, short gaps against long ones, one parameter against another. Those
comparisons are internally consistent and grid-converged (numerical elasticities
below 0.03, ex08), and they are what the report actually claims.
