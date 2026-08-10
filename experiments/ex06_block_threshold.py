"""Experiment 6: where is the conduction-block threshold?

Question
--------
A patch of reduced coupling either lets the action potential through or stops
it. Where is the boundary?

The answer is **not a single critical coupling ratio**. Block depends on both
how weak the coupling is (``rho``) and how far the wave has to cross at that
coupling (``L_gap``). The threshold is therefore a *curve* in the
``(L_gap, rho)`` plane -- a surface, if further parameters are allowed to vary
-- and this experiment maps it.

Block criterion
---------------
Propagation has blocked if no node at ``x > x_gap + L_gap + 0.3 cm`` reaches
``V = 0`` within 200 ms of the stimulus.

Why the averaging scheme is swept too
-------------------------------------
The interface conductance is the harmonic mean of the two nodal values,
because ``D`` is inversely proportional to axial resistance and resistances in
series add. The arithmetic mean over-predicts coupling across a sharp
interface -- by a factor of about 25 for a hundred-fold drop -- and therefore
shifts the threshold. Running both quantifies how much of the answer is
physics and how much is a numerical choice.

What this produces
------------------
``fig_ex06_block_threshold`` -- four panels:

(a) the phase diagram in ``(L_gap, rho)``, with individual run outcomes and the
    bisected threshold curve;
(b) space-time maps either side of the threshold;
(c) harmonic against arithmetic averaging;
(d) the electrotonic foot downstream of a blocked gap, which is why the block
    criterion needs a margin.

Run standalone with::

    python experiments/ex06_block_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibroblock import config as cfg  # noqa: E402
from fibroblock import measure, plotting, simulate, stimulus, utils  # noqa: E402

# Gap lengths at which the threshold coupling is located. Starts at a
# single-node gap (0.01 cm), where the two interface half-nodes ARE the whole
# gap and the averaging scheme therefore has its largest possible effect, and
# runs out to 0.40 cm, about twenty front thicknesses.
GAP_LENGTHS_CM: tuple[float, ...] = (
    0.01, 0.02, 0.04, 0.06, 0.10, 0.15, 0.20, 0.30, 0.40
)

# Coarse grid of individual runs, drawn as outcome markers behind the curve so
# the reader can see the raw evidence rather than only the fitted boundary.
# Clustered around 0.10-0.20, where the threshold actually lies: a grid spread
# evenly over (0, 1] puts every sampled point far from the boundary and shows
# nothing.
GRID_RHO_VALUES: tuple[float, ...] = (
    0.02, 0.05, 0.10, 0.12, 0.14, 0.16, 0.18, 0.25, 0.50, 1.0
)

# Bisection bracket on rho. The lower end must block and the upper end must
# propagate; both are verified by bisect_threshold itself.
RHO_BRACKET: tuple[float, float] = (0.0005, 1.0)
# Tight enough to resolve how the threshold varies with gap length. At a
# tolerance of 1e-3 every gap length beyond 0.1 cm returned an identical value,
# which said only that the thresholds agree to 1e-3 -- not that they are equal.
RHO_TOLERANCE: float = 1.0e-4

# The criterion is defined over 200 ms after the stimulus, so there is nothing
# to gain by integrating further.
BLOCK_RUN_MS: float = 205.0
SNAPSHOT_EVERY: int = 50


def run_gap(
    base: cfg.RunConfig,
    rho: float,
    gap_length_cm: float,
    averaging: str = "harmonic",
) -> simulate.SimulationResult:
    """Simulate a strand with one reduced-coupling gap.

    Parameters
    ----------
    base : RunConfig
        Baseline configuration.
    rho : float
        Coupling ratio inside the gap, in ``(0, 1]``.
    gap_length_cm : float
        Gap length. cm.
    averaging : {"harmonic", "arithmetic"}, optional
        Interface averaging scheme.

    Returns
    -------
    SimulationResult
    """
    config = base.replace(
        gap=cfg.GapParams(
            rho=rho,
            gap_length_cm=gap_length_cm,
            gap_centre_cm=base.gap.gap_centre_cm,
            averaging=averaging,  # type: ignore[arg-type]
        ),
        solver=cfg.SolverParams(
            dt_ms=base.solver.dt_ms,
            t_end_ms=BLOCK_RUN_MS,
            f_v_bound=base.solver.f_v_bound,
            record_every=SNAPSHOT_EVERY,
        ),
        label=f"ex06_rho{rho:g}_L{gap_length_cm:g}_{averaging}",
    )
    return simulate.run_simulation(config)


def propagates(
    base: cfg.RunConfig, rho: float, gap_length_cm: float, averaging: str
) -> bool:
    """Whether the wave crosses the gap, under the stated block criterion."""
    result = run_gap(base, rho, gap_length_cm, averaging)
    return not measure.detect_block(result).blocked


def threshold_curve(
    base: cfg.RunConfig, averaging: str
) -> tuple[np.ndarray, np.ndarray]:
    """Locate the critical coupling ratio at each gap length, by bisection.

    Parameters
    ----------
    base : RunConfig
        Baseline configuration.
    averaging : {"harmonic", "arithmetic"}
        Interface averaging scheme.

    Returns
    -------
    gap_lengths : ndarray
        Gap lengths at which a threshold was found. cm.
    critical_rho : ndarray
        The critical coupling ratio at each. Dimensionless.

    Notes
    -----
    A gap length whose bracket does not straddle the threshold gets **NaN**,
    not a silent omission. That outcome is a real physical result and the most
    interesting one in this experiment: under arithmetic averaging a
    single-node gap cannot be blocked at any coupling ratio whatsoever, because
    the interface conductance ``(1 + rho) D0 / 2`` tends to ``D0 / 2`` as
    ``rho -> 0`` instead of to zero. The harmonic mean, ``2 rho D0/(1 + rho)``,
    does tend to zero, as a series resistance must.
    """
    lengths, thresholds = [], []
    for gap_length in GAP_LENGTHS_CM:
        lengths.append(gap_length)
        try:
            critical, iterations = stimulus.bisect_threshold(
                succeeds=lambda rho, L=gap_length: propagates(
                    base, rho, L, averaging
                ),
                lower=RHO_BRACKET[0],
                upper=RHO_BRACKET[1],
                tolerance=RHO_TOLERANCE,
            )
        except ValueError:
            thresholds.append(float("nan"))
            print(
                f"      L_gap = {gap_length:.2f} cm -> NEVER BLOCKS: still "
                f"propagates at rho = {RHO_BRACKET[0]}, the lowest tested"
            )
            continue
        thresholds.append(critical)
        print(
            f"      L_gap = {gap_length:.2f} cm -> rho_crit = {critical:.5f} "
            f"({iterations} bisection steps)"
        )
    return np.array(lengths), np.array(thresholds)


def main() -> dict[str, Any]:
    """Run experiment 6 and write its figure and results.

    Returns
    -------
    dict
        Threshold curves for both averaging schemes.
    """
    print("=" * 70)
    print("Experiment 6: conduction-block threshold in the (L_gap, rho) plane")
    print("=" * 70)

    base = cfg.default_config().replace(label="ex06_block_threshold")
    utils.set_seed(base.seed)

    criterion_example = measure.detect_block(run_gap(base, 1.0, 0.1))
    print(f"  criterion: {criterion_example.criterion}")

    # ---- Coarse grid of individual outcomes --------------------------------
    print("\n  [1] coarse grid of individual runs (harmonic averaging)")
    outcomes = np.zeros((len(GAP_LENGTHS_CM), len(GRID_RHO_VALUES)), dtype=bool)
    for i, gap_length in enumerate(GAP_LENGTHS_CM):
        row = []
        for j, rho in enumerate(GRID_RHO_VALUES):
            outcomes[i, j] = propagates(base, rho, gap_length, "harmonic")
            row.append("P" if outcomes[i, j] else ".")
        print(f"      L_gap = {gap_length:.2f} cm : {' '.join(row)}")
    print(f"      (P = propagates, . = blocked; rho = {GRID_RHO_VALUES})")

    # ---- Threshold curves --------------------------------------------------
    print("\n  [2] threshold curve, harmonic averaging (the physical choice)")
    harmonic_L, harmonic_rho = threshold_curve(base, "harmonic")

    print("\n  [3] threshold curve, arithmetic averaging (for comparison only)")
    arithmetic_L, arithmetic_rho = threshold_curve(base, "arithmetic")

    # Compare the two schemes where BOTH found a finite threshold.
    both_blocked = np.isfinite(harmonic_rho) & np.isfinite(arithmetic_rho)
    shared = harmonic_L[both_blocked]
    harmonic_shared = harmonic_rho[both_blocked]
    arithmetic_shared = arithmetic_rho[both_blocked]
    scheme_ratio = harmonic_shared / arithmetic_shared

    # Gap lengths that the arithmetic scheme cannot block at all, but the
    # harmonic scheme can. This is the qualitative failure, not merely a shift.
    unblockable = harmonic_L[np.isfinite(harmonic_rho) & ~np.isfinite(arithmetic_rho)]
    if unblockable.size > 0:
        print(
            f"\n  arithmetic averaging CANNOT block at L_gap = "
            f"{', '.join(f'{L:.2f}' for L in unblockable)} cm at any rho, "
            f"whereas harmonic averaging can. As rho -> 0 the arithmetic "
            f"interface tends to D0/2, not to zero."
        )
    largest_disagreement = float(np.max(np.abs(scheme_ratio - 1.0)))
    print(
        f"\n  harmonic / arithmetic threshold ratio: "
        f"min {scheme_ratio.min():.3f}, max {scheme_ratio.max():.3f}, "
        f"largest disagreement {100.0 * largest_disagreement:.1f} %"
    )
    for L, ratio in zip(shared, scheme_ratio, strict=True):
        print(f"      L_gap = {L:.2f} cm -> ratio {ratio:.4f}")

    # Saturation: beyond a certain gap length the threshold stops depending on
    # length at all, because failure is then governed by whether the gap tissue
    # can sustain a front locally, not by how far the wave must travel in it.
    finite_harmonic = harmonic_rho[np.isfinite(harmonic_rho)]
    saturated = float(finite_harmonic[-1])
    shortest = float(finite_harmonic[0])
    print(
        f"  threshold rises from {shortest:.5f} at L_gap = {harmonic_L[0]:.2f} cm "
        f"to {saturated:.5f} asymptotically ({100.0 * (saturated / shortest - 1.0):.1f} % rise)"
    )

    # ---- Two runs bracketing the threshold, for the space-time maps --------
    demo_gap = 0.10
    demo_index = int(np.flatnonzero(harmonic_L == demo_gap)[0])
    demo_critical = float(harmonic_rho[demo_index])
    just_propagating = run_gap(base, demo_critical * 1.10, demo_gap, "harmonic")
    just_blocked = run_gap(base, demo_critical * 0.90, demo_gap, "harmonic")
    print(
        f"\n  demonstration at L_gap = {demo_gap} cm, rho_crit = {demo_critical:.5f}:"
    )
    print(
        f"      rho = {demo_critical * 1.10:.5f} -> "
        f"blocked = {measure.detect_block(just_propagating).blocked}"
    )
    print(
        f"      rho = {demo_critical * 0.90:.5f} -> "
        f"blocked = {measure.detect_block(just_blocked).blocked}"
    )

    # ---- Figure ------------------------------------------------------------
    fig, axes = plotting.new_figure(
        figsize=(10.5, 7.8), nrows=2, ncols=2, constrained_layout=True
    )
    ax_phase, ax_spacetime, ax_schemes, ax_foot = axes.flatten()

    # (a) Phase diagram.
    for i, gap_length in enumerate(GAP_LENGTHS_CM):
        for j, rho in enumerate(GRID_RHO_VALUES):
            ax_phase.plot(
                [gap_length],
                [rho],
                marker="o" if outcomes[i, j] else "x",
                markersize=5,
                markerfacecolor="none",
                color=(
                    plotting.COLOUR_PROPAGATED
                    if outcomes[i, j]
                    else plotting.COLOUR_BLOCKED
                ),
                linestyle="none",
            )
    finite = np.isfinite(harmonic_rho)
    ax_phase.plot(
        harmonic_L[finite],
        harmonic_rho[finite],
        color=plotting.COLOUR_REFERENCE,
        linewidth=2.0,
        marker="s",
        markersize=4,
        label="threshold (bisected)",
    )
    ax_phase.fill_between(
        harmonic_L[finite],
        harmonic_rho[finite],
        1.0,
        color=plotting.COLOUR_PROPAGATED,
        alpha=0.12,
    )
    ax_phase.fill_between(
        harmonic_L[finite],
        0.0,
        harmonic_rho[finite],
        color=plotting.COLOUR_BLOCKED,
        alpha=0.12,
    )
    # Linear rather than log: the whole boundary lies between rho = 0.1 and
    # 0.2, and a log axis spanning three decades compresses exactly the range
    # the figure exists to show.
    ax_phase.set_ylim(0.0, 0.55)
    ax_phase.text(
        0.28, 0.42, "propagates", fontsize=9, color=plotting.COLOUR_PROPAGATED
    )
    ax_phase.text(0.28, 0.04, "blocked", fontsize=9, color=plotting.COLOUR_BLOCKED)
    plotting.label_axes(
        ax_phase,
        "gap length $L_{\\mathrm{gap}}$ (cm)",
        r"coupling ratio $\rho$ (dimensionless)",
        "(a) Block threshold is a CURVE, not a single value",
    )
    ax_phase.legend(loc="lower left", fontsize=7.5)

    # (b) Space-time maps either side of the threshold.
    combined = np.vstack(
        [just_propagating.V_snapshots, just_blocked.V_snapshots]
    )
    vmin, vmax = float(combined.min()), float(combined.max())
    mesh = ax_spacetime.pcolormesh(
        just_propagating.x,
        just_propagating.snapshot_times,
        just_propagating.V_snapshots,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax_spacetime.axvspan(
        base.gap.gap_centre_cm - demo_gap / 2,
        base.gap.gap_centre_cm + demo_gap / 2,
        facecolor="none",
        edgecolor="white",
        linewidth=1.2,
        linestyle="--",
    )
    plotting.label_axes(
        ax_spacetime,
        "position $x$ (cm)",
        "time $t$ (ms)",
        f"(b) Just above threshold, $\\rho={demo_critical * 1.10:.4f}$",
    )
    fig.colorbar(mesh, ax=ax_spacetime, label="$V$ (dimensionless)")

    # (c) Harmonic against arithmetic.
    ax_schemes.plot(
        harmonic_L[finite],
        harmonic_rho[finite],
        marker="o",
        color=plotting.COLOUR_MEASURED,
        label="harmonic mean (correct)",
    )
    arithmetic_finite = np.isfinite(arithmetic_rho)
    ax_schemes.plot(
        arithmetic_L[arithmetic_finite],
        arithmetic_rho[arithmetic_finite],
        marker="s",
        linestyle="--",
        color=plotting.COLOUR_ANALYTIC,
        label="arithmetic mean (over-couples)",
    )
    # Mark the gap lengths the arithmetic scheme cannot block at ANY coupling.
    # Drawn just above the axis with an arrow rather than at its true rho, which
    # is "none exists": placing it at the bracket floor would force a log axis
    # spanning three decades and compress the two curves -- the actual subject
    # of the panel -- into an indistinguishable band.
    if unblockable.size > 0:
        ax_schemes.plot(
            unblockable,
            np.full(unblockable.size, 0.012),
            marker="v",
            markersize=10,
            linestyle="none",
            color=plotting.COLOUR_ANALYTIC,
            label=r"arithmetic: never blocks, any $\rho$",
        )
    ax_schemes.set_ylim(0.0, 0.20)
    ax_schemes.set_xscale("log")
    plotting.set_log_ticks(ax_schemes, np.array(GAP_LENGTHS_CM), axis="x")
    ax_schemes.tick_params(axis="x", labelrotation=45, labelsize=7)
    plotting.label_axes(
        ax_schemes,
        "gap length $L_{\\mathrm{gap}}$ (cm)",
        r"critical coupling ratio $\rho_{\mathrm{crit}}$",
        "(c) The averaging scheme moves the threshold",
    )
    ax_schemes.legend(loc="lower right", fontsize=7)
    plotting.annotate_takeaway(
        ax_schemes,
        f"curves disagree by up to {100.0 * largest_disagreement:.1f} %",
        loc="upper left",
    )

    # (d) The electrotonic foot: why the criterion needs a margin.
    peak_blocked = just_blocked.V_peak
    peak_propagating = just_propagating.V_peak
    ax_foot.plot(
        just_blocked.x,
        peak_propagating,
        color=plotting.COLOUR_PROPAGATED,
        label=f"$\\rho={demo_critical * 1.10:.4f}$ (propagates)",
    )
    ax_foot.plot(
        just_blocked.x,
        peak_blocked,
        color=plotting.COLOUR_BLOCKED,
        label=f"$\\rho={demo_critical * 0.90:.4f}$ (blocked)",
    )
    plotting.shade_gap(
        ax_foot,
        base.gap.gap_centre_cm - demo_gap / 2,
        base.gap.gap_centre_cm + demo_gap / 2,
    )
    ax_foot.axhline(
        base.measurement.activation_level,
        color=plotting.COLOUR_REFERENCE,
        linestyle=":",
        linewidth=1.0,
        label="activation level $V=0$",
    )
    ax_foot.axvline(
        criterion_example.detection_x_cm,
        color=plotting.PALETTE["purple"],
        linestyle="-.",
        linewidth=1.2,
        label=f"detection point {criterion_example.detection_x_cm:.2f} cm",
    )
    ax_foot.set_xlim(0.6, 1.8)
    plotting.label_axes(
        ax_foot,
        "position $x$ (cm)",
        "peak potential reached $V_{\\max}$ (dimensionless)",
        "(d) The electrotonic foot beyond a blocked gap",
    )
    ax_foot.legend(loc="upper right", fontsize=7.5)

    caption = (
        f"The conduction-block threshold is a curve in the (L_gap, rho) plane, "
        f"not a single critical coupling ratio: a single-node gap blocks only "
        f"below rho = {shortest:.4f}, rising {100.0 * (saturated / shortest - 1.0):.0f} % "
        f"to rho = {saturated:.4f} and then saturating beyond about 0.1 cm. The "
        f"saturation is itself informative -- once the gap is longer than the "
        f"distance over which failure develops, whether the wave survives "
        f"depends on the local coupling alone, not on how far it must travel. "
        f"Panel (c) shows the numerical-choice sensitivity is not merely "
        f"quantitative: the arithmetic mean shifts the threshold by up to "
        f"{100.0 * largest_disagreement:.1f} %, and for a single-node gap it "
        f"cannot produce block at ANY coupling ratio, because its interface "
        f"conductance tends to D0/2 rather than to zero as rho falls. Panel (d) "
        f"shows why the "
        f"criterion needs a 0.3 cm margin -- charge leaks passively past a "
        f"blocked gap and decays over about one space constant without ever "
        f"becoming a propagating front."
    )

    measurements = {
        "gap_lengths_cm": harmonic_L.tolist(),
        "critical_rho_harmonic": harmonic_rho.tolist(),
        "gap_lengths_arithmetic_cm": arithmetic_L.tolist(),
        "critical_rho_arithmetic": arithmetic_rho.tolist(),
        "harmonic_over_arithmetic_ratio": scheme_ratio.tolist(),
        "largest_scheme_disagreement_relative": largest_disagreement,
        "arithmetic_unblockable_gap_lengths_cm": unblockable.tolist(),
        "critical_rho_shortest_gap": shortest,
        "critical_rho_saturated": saturated,
        "threshold_rise_relative": float(saturated / shortest - 1.0),
        "demonstration_gap_length_cm": demo_gap,
        "demonstration_critical_rho": demo_critical,
        "block_criterion": criterion_example.criterion,
    }

    plotting.save_figure(
        fig,
        "fig_ex06_block_threshold",
        caption,
        config_dict=base.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_arrays(
        "ex06_block_threshold",
        {
            "gap_lengths_cm": harmonic_L,
            "critical_rho_harmonic": harmonic_rho,
            "gap_lengths_arithmetic_cm": arithmetic_L,
            "critical_rho_arithmetic": arithmetic_rho,
            "grid_gap_lengths_cm": np.array(GAP_LENGTHS_CM),
            "grid_rho_values": np.array(GRID_RHO_VALUES),
            "grid_propagates": outcomes,
            "x_cm": just_blocked.x,
            "peak_V_blocked": peak_blocked,
            "peak_V_propagating": peak_propagating,
            "spacetime_times_ms": just_propagating.snapshot_times,
            "spacetime_V_propagating": just_propagating.V_snapshots,
            "spacetime_V_blocked": just_blocked.V_snapshots,
        },
        config_dict=base.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_table(
        "ex06_block_threshold",
        header=[
            "gap_length_cm",
            "critical_rho_harmonic",
            "critical_rho_arithmetic",
            "harmonic_over_arithmetic",
        ],
        rows=[
            (
                f"{L:.4f}",
                f"{h:.6f}",
                f"{a:.6f}",
                f"{h / a:.4f}",
            )
            for L, h, a in zip(shared, harmonic_shared, arithmetic_shared, strict=True)
        ],
    )

    print("\n  figure -> figures/fig_ex06_block_threshold.png / .pdf")
    return measurements


if __name__ == "__main__":
    main()
