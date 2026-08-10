"""Experiment 4: pure diffusion against the analytic Gaussian.

Question
--------
Switch the reaction term off. The model then reduces to the heat equation,
``V_t = D V_xx``, whose solution from a Gaussian initial condition is known in
closed form. Does the solver reproduce it, and does it conserve charge?

This is the strongest available check on the spatial operator and the sealed
ends, because it compares against an exact answer rather than against another
numerical solution.

What this produces
------------------
``fig_ex04_pure_diffusion`` -- four panels:

(a) computed profiles against the analytic Gaussian at several times;
(b) the pointwise error at the final time, and where it lives;
(c) total charge over time, showing conservation to machine precision;
(d) peak height decay against the analytic ``sigma_0 / sigma(t)`` law.

Run standalone with::

    python experiments/ex04_pure_diffusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibroblock import config as cfg  # noqa: E402
from fibroblock import measure, operators, plotting, simulate, utils  # noqa: E402

# --- Initial Gaussian -------------------------------------------------------
# Centred on the strand so both sealed ends are equally far away.
GAUSSIAN_CENTRE_CM: float = 1.0
# Ten node spacings wide at dx = 0.01 cm, so the initial profile is well
# resolved. Narrower would make the discretisation error dominate; wider would
# reach the boundaries sooner.
GAUSSIAN_SIGMA0_CM: float = 0.1
GAUSSIAN_AMPLITUDE: float = 1.0

# Run length. The analytic solution is the INFINITE-LINE one, so it is only
# valid while the Gaussian has not felt the sealed ends. At t = 20 ms,
# sigma = sqrt(0.01 + 2*0.001*20) = 0.224 cm, so each boundary is 4.5 standard
# deviations away and the neglected image contribution is about 5e-5 of the
# peak -- two orders of magnitude below the discretisation error being
# measured. The experiment reports this contamination rather than assuming it.
DIFFUSION_DURATION_MS: float = 20.0

# With the reaction term switched off, the only stability constraint is the
# pure-diffusion one, so |f_V|_max is genuinely zero here -- not an
# approximation. dt = 0.02 ms then sits a factor 2.5 below the 0.05 ms limit.
PURE_DIFFUSION_F_V_BOUND: float = 0.0

# Times at which profiles are drawn in panel (a).
PROFILE_TIMES_MS: tuple[float, ...] = (0.0, 2.0, 6.0, 12.0, 20.0)


def main() -> dict[str, Any]:
    """Run experiment 4 and write its figure and results.

    Returns
    -------
    dict
        Summary of the measured errors and conservation quality.
    """
    print("=" * 70)
    print("Experiment 4: pure diffusion vs the analytic Gaussian")
    print("=" * 70)

    base = cfg.default_config()
    utils.set_seed(base.seed)

    config = base.replace(
        label="ex04_pure_diffusion",
        # Homogeneous strand: this experiment tests the operator, not the gap.
        gap=cfg.GapParams(rho=1.0, gap_length_cm=0.0),
        solver=cfg.SolverParams(
            dt_ms=base.solver.dt_ms,
            t_end_ms=DIFFUSION_DURATION_MS,
            method="euler",
            f_v_bound=PURE_DIFFUSION_F_V_BOUND,
            record_every=base.solver.record_every,
        ),
    )

    D = config.grid.baseline_D

    # ---- Initial condition -------------------------------------------------
    x = np.linspace(0.0, config.grid.length_cm, config.grid.n_nodes)
    initial_V = operators.analytic_gaussian_sealed(
        x,
        t=0.0,
        D=D,
        x0=GAUSSIAN_CENTRE_CM,
        sigma0=GAUSSIAN_SIGMA0_CM,
        length=config.grid.length_cm,
        amplitude=GAUSSIAN_AMPLITUDE,
    )

    print(f"  D            = {D} cm^2/ms")
    print(f"  sigma_0      = {GAUSSIAN_SIGMA0_CM} cm ({GAUSSIAN_SIGMA0_CM / config.grid.dx_cm:.0f} nodes)")
    print(f"  dt           = {config.solver.dt_ms} ms over {DIFFUSION_DURATION_MS} ms")

    # ---- Integrate ---------------------------------------------------------
    result = simulate.run_simulation(
        config,
        initial_V=initial_V,
        initial_w=np.zeros_like(initial_V),
        include_reaction=False,
        include_stimulus=False,
    )

    print(f"  stability    dt limit = {result.stability.pure_diffusion_dt_ms:.5f} ms")
    print(f"               safety factor = {result.stability.safety_factor:.2f}")
    print(f"  integration  {result.n_steps_taken} steps in {result.wall_seconds:.2f} s")

    # ---- Compare with the closed-form solutions ----------------------------
    # TWO references are used, and the distinction matters:
    #   * the SEALED solution (method of images) is exact for this problem, so
    #     the difference from it is purely discretisation error;
    #   * the INFINITE-LINE solution is the textbook Gaussian, which stops
    #     describing a sealed strand once the profile reaches the ends.
    # Reporting only the second would blame the scheme for a modelling
    # difference that has nothing to do with it.
    l2_errors = np.empty(result.snapshot_times.size)
    linf_errors = np.empty(result.snapshot_times.size)
    linf_errors_infinite = np.empty(result.snapshot_times.size)
    sealed_snapshots = np.empty_like(result.V_snapshots)
    infinite_snapshots = np.empty_like(result.V_snapshots)

    for index, t in enumerate(result.snapshot_times):
        exact_sealed = operators.analytic_gaussian_sealed(
            x,
            t=float(t),
            D=D,
            x0=GAUSSIAN_CENTRE_CM,
            sigma0=GAUSSIAN_SIGMA0_CM,
            length=config.grid.length_cm,
            amplitude=GAUSSIAN_AMPLITUDE,
        )
        exact_infinite = operators.analytic_gaussian(
            x,
            t=float(t),
            D=D,
            x0=GAUSSIAN_CENTRE_CM,
            sigma0=GAUSSIAN_SIGMA0_CM,
            amplitude=GAUSSIAN_AMPLITUDE,
        )
        sealed_snapshots[index] = exact_sealed
        infinite_snapshots[index] = exact_infinite

        l2_errors[index] = measure.l2_error(
            result.V_snapshots[index], exact_sealed, result.grid.quadrature_weights
        )
        linf_errors[index] = measure.linf_error(result.V_snapshots[index], exact_sealed)
        linf_errors_infinite[index] = measure.linf_error(
            result.V_snapshots[index], exact_infinite
        )

    final_l2 = float(l2_errors[-1])
    final_linf = float(linf_errors[-1])
    final_linf_infinite = float(linf_errors_infinite[-1])
    print(f"  vs sealed-exact solution (discretisation error only):")
    print(f"               L2 = {final_l2:.3e}, Linf = {final_linf:.3e}")
    print(f"  vs infinite-line Gaussian (includes the boundary difference):")
    print(f"               Linf = {final_linf_infinite:.3e}")

    # ---- How much of the discrepancy is the boundary? ----------------------
    final_sealed = sealed_snapshots[-1]
    final_infinite = infinite_snapshots[-1]
    boundary_contamination = float(
        np.max(np.abs(final_sealed - final_infinite))
    )
    sigma_final = float(np.sqrt(GAUSSIAN_SIGMA0_CM**2 + 2.0 * D * DIFFUSION_DURATION_MS))
    print(
        f"  boundary     sigma(T) = {sigma_final:.4f} cm "
        f"({(config.grid.length_cm / 2.0) / sigma_final:.1f} sigma to each end)"
    )
    print(
        f"               sealed vs infinite-line difference = "
        f"{boundary_contamination:.2e}"
    )

    # ---- Charge conservation ----------------------------------------------
    charge = result.charge_history
    charge_drift = np.abs(charge - charge[0]) / abs(charge[0])
    max_drift = float(np.max(charge_drift))

    analytic_charge = (
        GAUSSIAN_AMPLITUDE * GAUSSIAN_SIGMA0_CM * float(np.sqrt(2.0 * np.pi))
    )
    print(f"  charge       Q(0) = {charge[0]:.12f}")
    print(f"               Q(T) = {charge[-1]:.12f}")
    print(f"               analytic integral = {analytic_charge:.12f}")
    print(f"               max relative drift = {max_drift:.3e}")

    # ---- Peak decay --------------------------------------------------------
    computed_peaks = np.max(result.V_snapshots, axis=1)
    sealed_peaks = np.max(sealed_snapshots, axis=1)
    # Infinite-line law: the peak falls as sigma_0 / sigma(t) exactly, because
    # the Gaussian widens and shortens in step to conserve its integral.
    sigma_of_t = np.sqrt(GAUSSIAN_SIGMA0_CM**2 + 2.0 * D * result.snapshot_times)
    analytic_peaks = GAUSSIAN_AMPLITUDE * GAUSSIAN_SIGMA0_CM / sigma_of_t
    peak_relative_error = float(
        np.max(np.abs(computed_peaks - sealed_peaks) / sealed_peaks)
    )
    print(f"  peak decay   max relative error vs sealed = {peak_relative_error:.3e}")

    # ---- Figure ------------------------------------------------------------
    fig, axes = plotting.new_figure(
        figsize=(10.0, 7.5), nrows=2, ncols=2, constrained_layout=True
    )
    ax_profiles, ax_error, ax_charge, ax_peak = axes.flatten()

    # (a) Profiles.
    for target_time in PROFILE_TIMES_MS:
        index = int(np.argmin(np.abs(result.snapshot_times - target_time)))
        actual_time = float(result.snapshot_times[index])
        line, = ax_profiles.plot(
            x,
            sealed_snapshots[index],
            linewidth=1.6,
            label=f"exact, $t={actual_time:.0f}$ ms",
        )
        # Markers on every eighth node so the computed points are visible
        # without hiding the analytic curve underneath them.
        ax_profiles.plot(
            x[::8],
            result.V_snapshots[index][::8],
            linestyle="none",
            marker="o",
            markersize=3.5,
            markerfacecolor="none",
            color=line.get_color(),
        )
    plotting.label_axes(
        ax_profiles,
        "position $x$ (cm)",
        "membrane potential $V$ (dimensionless)",
        "(a) Computed (markers) vs exact sealed solution (lines)",
    )
    ax_profiles.legend(loc="upper right", fontsize=8)

    # (b) Pointwise error at the final time, against both references.
    ax_error.plot(
        x,
        result.V_snapshots[-1] - final_sealed,
        color=plotting.COLOUR_MEASURED,
        label="vs exact sealed solution\n(discretisation error)",
    )
    ax_error.plot(
        x,
        result.V_snapshots[-1] - final_infinite,
        color=plotting.COLOUR_ANALYTIC,
        linestyle="--",
        label="vs infinite-line Gaussian\n(includes boundary difference)",
    )
    plotting.label_axes(
        ax_error,
        "position $x$ (cm)",
        "computed $-$ reference (dimensionless)",
        f"(b) Pointwise error at $t={DIFFUSION_DURATION_MS:.0f}$ ms",
    )
    ax_error.legend(loc="lower right", fontsize=7.5)
    plotting.annotate_takeaway(
        ax_error,
        f"vs sealed: $\\|e\\|_\\infty={final_linf:.2e}$",
        loc="upper right",
    )

    # (c) Charge conservation.
    ax_charge.plot(
        result.snapshot_times,
        charge_drift,
        color=plotting.COLOUR_MEASURED,
        label="relative drift $|Q(t)-Q(0)|/Q(0)$",
    )
    ax_charge.axhline(
        np.finfo(float).eps,
        color=plotting.COLOUR_ANALYTIC,
        linestyle="--",
        label=f"machine epsilon = {np.finfo(float).eps:.1e}",
    )
    ax_charge.set_yscale("log")
    plotting.label_axes(
        ax_charge,
        "time $t$ (ms)",
        "relative charge drift (dimensionless)",
        "(c) Charge conservation with sealed ends",
    )
    ax_charge.legend(loc="lower right", fontsize=8)

    # (d) Peak decay.
    ax_peak.plot(
        result.snapshot_times,
        sealed_peaks,
        color=plotting.COLOUR_ANALYTIC,
        label="exact sealed solution",
    )
    ax_peak.plot(
        result.snapshot_times,
        analytic_peaks,
        color=plotting.COLOUR_REFERENCE,
        linestyle=":",
        linewidth=1.2,
        label=r"infinite line, $A\sigma_0/\sigma(t)$",
    )
    ax_peak.plot(
        result.snapshot_times[::4],
        computed_peaks[::4],
        linestyle="none",
        marker="s",
        markersize=4,
        markerfacecolor="none",
        color=plotting.COLOUR_MEASURED,
        label="computed peak",
    )
    plotting.label_axes(
        ax_peak,
        "time $t$ (ms)",
        "peak potential $V_{\\max}$ (dimensionless)",
        "(d) Peak decay follows the spreading law",
    )
    ax_peak.legend(loc="upper right", fontsize=8)
    plotting.annotate_takeaway(
        ax_peak,
        f"max relative error vs sealed: {peak_relative_error:.1e}",
        loc="lower left",
    )

    caption = (
        f"With the reaction term switched off the solver reproduces the exact "
        f"sealed-strand solution of the heat equation to {final_linf:.1e} in the "
        f"maximum norm after {DIFFUSION_DURATION_MS:.0f} ms, and conserves total "
        f"charge to a relative drift of {max_drift:.1e} -- machine precision. "
        f"Panel (b) separates the two error sources: the conservative operator "
        f"itself is accurate to {final_linf:.1e}, whereas comparing against the "
        f"textbook infinite-line Gaussian would wrongly attribute a further "
        f"{boundary_contamination:.1e} of boundary reflection to the scheme."
    )

    measurements = {
        "final_l2_error_vs_sealed": final_l2,
        "final_linf_error_vs_sealed": final_linf,
        "final_linf_error_vs_infinite_line": final_linf_infinite,
        "max_charge_drift_relative": max_drift,
        "initial_charge": float(charge[0]),
        "final_charge": float(charge[-1]),
        "analytic_charge": analytic_charge,
        "peak_max_relative_error": peak_relative_error,
        "sigma_final_cm": sigma_final,
        "sealed_vs_infinite_difference": boundary_contamination,
        "wall_seconds": result.wall_seconds,
    }

    plotting.save_figure(
        fig,
        "fig_ex04_pure_diffusion",
        caption,
        config_dict=config.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_arrays(
        "ex04_pure_diffusion",
        {
            "x_cm": x,
            "snapshot_times_ms": result.snapshot_times,
            "V_computed": result.V_snapshots,
            "V_exact_sealed": sealed_snapshots,
            "V_exact_infinite_line": infinite_snapshots,
            "l2_error_vs_sealed": l2_errors,
            "linf_error_vs_sealed": linf_errors,
            "linf_error_vs_infinite_line": linf_errors_infinite,
            "charge": charge,
            "charge_drift_relative": charge_drift,
            "computed_peaks": computed_peaks,
            "sealed_peaks": sealed_peaks,
            "infinite_line_peaks": analytic_peaks,
        },
        config_dict=config.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_table(
        "ex04_pure_diffusion_summary",
        header=["quantity", "value", "units"],
        rows=[
            ("final_L2_error_vs_sealed", f"{final_l2:.6e}", "dimensionless x sqrt(cm)"),
            ("final_Linf_error_vs_sealed", f"{final_linf:.6e}", "dimensionless"),
            (
                "final_Linf_error_vs_infinite_line",
                f"{final_linf_infinite:.6e}",
                "dimensionless",
            ),
            ("initial_charge", f"{charge[0]:.15e}", "dimensionless x cm"),
            ("final_charge", f"{charge[-1]:.15e}", "dimensionless x cm"),
            ("analytic_charge", f"{analytic_charge:.15e}", "dimensionless x cm"),
            ("max_relative_charge_drift", f"{max_drift:.6e}", "dimensionless"),
            ("peak_max_relative_error", f"{peak_relative_error:.6e}", "dimensionless"),
            ("sigma_at_final_time", f"{sigma_final:.6f}", "cm"),
            (
                "sealed_vs_infinite_line_difference",
                f"{boundary_contamination:.6e}",
                "dimensionless",
            ),
        ],
    )

    print("  figure -> figures/fig_ex04_pure_diffusion.png / .pdf")
    return measurements


if __name__ == "__main__":
    main()
