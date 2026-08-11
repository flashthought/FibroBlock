# Validation log

*Validation asks: are these the right equations?* — a different and harder
question than verification (am I solving them correctly?), which is logged
separately in [`verification_log.md`](verification_log.md).

> **⚠️ Citation check required before submission.**
> The references below are standard works in cardiac electrophysiology and the
> numerical ranges quoted are those commonly reported in that literature.
> **They were written from background knowledge, not read and checked against
> the papers during this project.** Before submitting, retrieve each source and
> confirm the specific number attributed to it. Where a range is broad it is
> quoted broadly on purpose. Entries whose provenance is weakest are flagged
> **[verify]**.

---

## What can and cannot be validated here

FitzHugh–Nagumo is a *caricature*. It has no sodium current, no calcium
current, no physical membrane capacitance, and its "voltage" is dimensionless.
Under assumption **A1** one dimensionless time unit is *declared* to be 1 ms —
a calibration choice, not a derivation.

This has a consequence that must be stated plainly: **agreement between the
model's conduction velocity and measured cardiac velocities is weaker evidence
than it appears**, because the time unit was chosen to make the comparison
possible in the first place. What genuinely validates is the *structure* of the
results — the scaling laws, the ordering of effects, and the qualitative shape
of the block boundary — none of which depends on the calibration.

| Claim | Can it be validated? |
|---|---|
| `θ ∝ √D` | **Yes** — a scaling law, independent of the time-unit calibration. |
| `θ ≈ 30 cm/s` in absolute terms | **Weakly** — depends directly on A1. |
| Block occurs at ~10–20 % of normal coupling | **Yes, qualitatively** — a dimensionless ratio, independent of A1. |
| Delay saturates rather than diverging | **Partially** — a structural prediction, but FHN-specific. |

---

## L1. Conduction velocity

| Quantity | This model | Published range | Assessment |
|---|---|---|---|
| Measured CV, homogeneous strand | **25.5 cm/s** | Ventricular myocardium, **transverse** to fibre: ~20–35 cm/s [1,2] | **Consistent.** |
| Analytic CV (frozen `w`) | 30.5 cm/s | as above | Consistent. |
| — | — | Ventricular, **longitudinal**: ~50–70 cm/s [1,2] | Model is below this; unsurprising for a 1-D isotropic strand with no fibre structure. |
| — | — | Purkinje fibres: ~200–400 cm/s [1] | Not represented; a different tissue. |

**Reading.** The model lands in the *transverse* ventricular range. That is the
honest comparison for an isotropic 1-D cable with no fibre architecture — the
transverse direction is where anisotropy contributes least. It should not be
presented as a match to longitudinal conduction.

**Scaling.** `θ ∝ √D` with exponent **0.49996** (R² = 0.99999999) is the more
meaningful validation. The square-root dependence of conduction velocity on
intercellular coupling is a robust, well-established result of cable theory and
is reproduced here to four decimal places without any parameter tuning [3,4].

---

## L2. Conduction block threshold

| Quantity | This model | Published | Assessment |
|---|---|---|---|
| Critical coupling ratio (saturated) | **ρ_crit ≈ 0.157**, i.e. block below ~16 % of normal coupling | Experimental and computational studies report propagation failure when gap-junction coupling falls to roughly **5–20 %** of control [3,5,6] | **Consistent, and at the upper end.** |
| Threshold at a single-node gap | ρ_crit ≈ 0.067 | — | Shows the length dependence; no direct published counterpart. |
| CV just before block | ~40 % of control | Conduction can slow to ~10–20 % of normal before failing in reduced-coupling preparations [3,5] | **Model fails earlier than real tissue.** See below. |

**Where the model is optimistic about block.** Two stated assumptions push in
the same direction:

- **A4 (continuum).** Real tissue at very low coupling conducts *saltatorily* —
  hopping cell to cell with large delays at each junction. A continuum model
  cannot represent this and therefore fails sooner. Discrete-cell studies find
  conduction persisting at coupling levels where continuum models predict block
  [3,5].
- **A2 (one dimension).** In 2-D or 3-D a wave can travel around a poorly
  coupled patch. Block requires a barrier spanning the full width, so real
  tissue is harder to block than this 1-D result suggests.

**Conclusion:** `ρ_crit ≈ 0.157` should be read as an **upper bound on how
easily block occurs**, not as a prediction for myocardium.

---

## L3. Conduction delay

| Quantity | This model | Published | Assessment |
|---|---|---|---|
| Maximum excess delay across the gap | **16.0 ms** over a 0.1 cm gap | Delays of **tens of milliseconds** across discrete uncoupled junctions and fibrotic zones are widely reported [5,7] | **Consistent in magnitude.** |
| Behaviour approaching threshold | **Saturates**; does not diverge | Both graded slowing and abrupt failure are described; the balance depends on the preparation [3,5] | **Partially supported** — see note. |
| Total transit, healthy 0.7 cm span | 27.4 ms | — | — |

**Note on the saturation result.** This experiment was written expecting a
`(ρ − ρ_crit)^(−1/2)` divergence and had to be rewritten around what the data
showed. Measured over **six decades** of approach, the excess delay saturates
at 16.0 ms and changes by only 15 % over the final three decades; a power-law
fit returns −0.07 (R² = 0.59), not −0.5.

The mechanism is that a stalled front cannot outlast its own upstream source,
which repolarises on the recovery timescale `1/ε`. Varying `ε` confirms it: the
ceiling falls with exponent −1.46 (R² = 0.95), against −1 for a pure `1/ε` cap;
the extra steepness comes from `ε` also shifting `ρ_crit` itself.

**How far this generalises is unclear.** The result follows from FHN's
single-recovery-variable structure. Real myocardium has separate sodium
inactivation and calcium recovery on different timescales, and the analogous
bound would involve both. This is a **model-structural prediction**, offered as
a hypothesis rather than a validated physiological claim.

---

## L4. Fibrosis as reduced coupling

The gap represents a fibrotic patch as a **region of reduced intercellular
coupling**. This captures the dominant electrical consequence of fibrosis —
collagen deposition and gap-junction remodelling reducing effective
conductivity [6,8] — but omits three real effects:

1. **Fibroblast–myocyte coupling.** Fibroblasts couple electrotonically to
   myocytes, loading them capacitively and depolarising resting potential.
2. **Geometric tortuosity.** Fibrotic strands force zig-zag conduction paths,
   which lengthen the route without necessarily reducing local coupling [7].
3. **Ion-channel remodelling.** Fibrotic regions typically also show altered
   channel expression, so excitability changes alongside coupling.

The model varies coupling alone. That isolates one mechanism cleanly, which is
appropriate for a mechanism study, but it means the model **cannot** predict
the behaviour of real fibrotic tissue quantitatively.

---

## L5. Excitability

| Quantity | This model | Expected | Assessment |
|---|---|---|---|
| Rest state | Stable spiral (`tr J = −0.503`, `det J = 0.108`) | Working myocardium is excitable, not spontaneously oscillatory | ✅ **Correct qualitative behaviour** |
| All-or-none response | 10 % amplitude change switches peak from −0.59 to +1.70 | Cardiac AP is all-or-none | ✅ |
| Action potential duration | ~50 ms (FHN units = ms under A1) | Ventricular APD ~200–300 ms [1] | ⚠️ **Short by 4–6×.** A known limitation of FHN with `ε = 0.08`. |

**The APD mismatch matters for what is claimed.** Because the model's action
potential is several times shorter than a real one, **any conclusion involving
refractoriness or re-entry would be unreliable** — which is precisely why this
project restricts itself to single-beat propagation and block, and does not
attempt a re-entry study. The delay-saturation result of L3 depends on the
recovery timescale and inherits this limitation.

---

## Summary

| Aspect | Verdict |
|---|---|
| CV magnitude | Consistent with transverse ventricular conduction (weak evidence; depends on A1) |
| CV scaling `∝ √D` | **Strongly validated** — exponent 0.49996, no tuning |
| Block threshold | Consistent with published 5–20 % range, at its upper end; an upper bound on how easily block occurs |
| Delay magnitude | Consistent (tens of ms) |
| Delay saturation | Model-structural prediction; not independently validated |
| APD | **Too short by 4–6×**; limits claims to single-beat phenomena |

**Overall.** The model is validated as a *mechanism* study: the scaling laws,
the ordering of parameter influences, and the shape of the block boundary are
trustworthy and grid-converged. Absolute values are calibration-dependent and
should be quoted with assumption A1 attached. No claim in the report rests on
absolute agreement alone.

---

## References

**[verify] — confirm each citation and the specific value attributed to it
before submission.** BibTeX entries are in
[`../report/references.bib`](../report/references.bib).

1. **[verify]** Katz, A. M. *Physiology of the Heart*, 5th edn. Lippincott
   Williams & Wilkins, 2010. — conduction velocities by tissue type; action
   potential durations.
2. **[verify]** Kléber, A. G., Rudy, Y. (2004). Basic mechanisms of cardiac
   impulse propagation and associated arrhythmias. *Physiological Reviews*
   **84**(2), 431–488. — velocities, anisotropy, propagation safety factor.
3. **[verify]** Rohr, S. (2004). Role of gap junctions in the propagation of
   the cardiac action potential. *Cardiovascular Research* **62**(2), 309–322.
   — coupling reduction and conduction failure.
4. Keener, J., Sneyd, J. *Mathematical Physiology I: Cellular Physiology*,
   2nd edn. Springer, 2009. — bistable front speed `θ = √(A/2)(V₁ − 2V₂ + V₃)√D`;
   this is the derivation used in `fhn.py` and is the one citation whose *use*
   here is directly verifiable against the code.
5. **[verify]** Shaw, R. M., Rudy, Y. (1997). Ionic mechanisms of propagation
   in cardiac tissue: roles of the sodium and L-type calcium currents during
   reduced excitability and decreased gap junction coupling. *Circulation
   Research* **81**(5), 727–741. — conduction at very low coupling; saltatory
   propagation.
6. **[verify]** de Bakker, J. M. T., et al. (1993). Slow conduction in the
   infarcted human heart: 'zigzag' course of activation. *Circulation*
   **88**(3), 915–926. — fibrosis, tortuosity, slow conduction.
7. **[verify]** Spach, M. S., Boineau, J. P. (1997). Microfibrosis produces
   electrical load variations due to loss of side-to-side cell connections.
   *Pacing and Clinical Electrophysiology* **20**(2), 397–413.
8. FitzHugh, R. (1961). Impulses and physiological states in theoretical models
   of nerve membrane. *Biophysical Journal* **1**(6), 445–466. — the kinetics
   themselves.
9. Nagumo, J., Arimoto, S., Yoshizawa, S. (1962). An active pulse transmission
   line simulating nerve axon. *Proceedings of the IRE* **50**(10), 2061–2070.
