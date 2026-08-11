"""Experiment 2: the explicit-Euler stability limit, and the mode that breaks it.

Question
--------
Von Neumann analysis predicts that explicit Euler is stable only for

    dt <= 2 / (4 D_max/dx^2 + |f_V|_max)

with the binding mode being the checkerboard ``k = pi/dx``. Does the code
actually fail there, and does it fail in that specific way? And how misleading
is the pure-diffusion limit ``dx^2/(2D)`` that ignores the reaction term?

What this produces
------------------
``fig_ex02_stability`` -- four panels:

(a) the solution just below the limit -- a clean propagating front;
(b) the solution just above it, with the node-to-node alternation of the
    checkerboard mode visible in a zoomed inset region;
(c) growth of the checkerboard component over time for several time steps,
    on a log scale;
(d) measured per-step amplification against the von Neumann prediction
    ``|g| = |1 - 4 D dt / dx^2|`` for the linear problem, with both candidate
    limits marked.

Run standalone with::

    python experiments/ex02_stability.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibroblock import config as cfg  # noqa: E402
from fibroblock import grid as gridmod  # noqa: E402
from fibroblock import (  # noqa: E402
    operators,
    plotting,
    simulate,
    solvers,
    stimulus,
    utils,
)

# Short run: an unstable step blows up within a few milliseconds, and a stable
# one only needs long enough to show a clean front. Neither needs 300 ms.
STABILITY_RUN_MS: float = 40.0

# Steps used for the two illustrative panels, as fractions of the computed
# reaction-diffusion limit.
STABLE_FRACTION: float = 0.90
UNSTABLE_FRACTION: float = 1.08

# Fractions of the limit sampled for the empirical stability boundary in (c).
SWEEP_FRACTIONS: tuple[float, ...] = (0.80, 0.95, 1.02, 1.05, 1.10)

# Bracket for bisecting the empirical stability boundary, as fractions of the
# computed limit. The lower end must be safely stable and the upper end safely
# unstable, which the sweep above confirms.
BOUNDARY_BRACKET: tuple[float, float] = (0.90, 1.30)
BOUNDARY_TOLERANCE: float = 0.002
# Long enough that slow growth just above the limit is still caught: at
# 1.03x the limit the checkerboard mode grows by only a few per cent per step.
BOUNDARY_RUN_MS: float = 150.0

# Magnitude at which the unstable solution is displayed in panel (b): large
# enough that the instability clearly dominates the physiological signal
# (|V| <~ 2), small enough that the alternating structure is still legible on a
# linear axis rather than being one enormous spike.
DISPLAY_MAGNITUDE: float = 8.0

# Steps used for the clean linear-theory comparison in (d), as fractions of
# the PURE-DIFFUSION limit (the linear problem has no reaction term).
LINEAR_SWEEP_FRACTIONS: np.ndarray = np.linspace(0.5, 1.3, 25)
LINEAR_SWEEP_STEPS: int = 40


def linear_checkerboard_growth(dt: float, dx: float, D: float, n_steps: int) -> float:
    """Per-step amplification of the checkerboard mode under pure diffusion.

    Parameters
    ----------
    dt : float
        Time step. ms.
    dx : float
        Node spacing. cm.
    D : float
        Diffusion coefficient. cm^2/ms.
    n_steps : int
        Number of steps to take.

    Returns
    -------
    float
        Geometric mean amplification per step, ``(A_final / A_initial)^(1/n)``.

    Notes
    -----
    Deliberately linear: no reaction term, and the initial condition is the
    checkerboard mode itself. This isolates the diffusive part of the
    amplification factor so it can be compared directly with the closed-form
    prediction, without the nonlinear cubic muddying the comparison.
    """
    grid_params = cfg.GridParams(length_cm=1.0, dx_cm=dx, baseline_D=D)
    gap_params = cfg.GapParams(rho=1.0, gap_length_cm=0.0)
    g = gridmod.build_grid(grid_params, gap_params)

    V = solvers.checkerboard_mode(g.n_nodes)
    initial = solvers.checkerboard_amplitude(V)

    for _ in range(n_steps):
        V = V + dt * operators.divergence(V, g.D_half, g.dx)

    final = solvers.checkerboard_amplitude(V)
    return float((final / initial) ** (1.0 / n_steps))


def main() -> dict[str, Any]:
    """Run experiment 2 and write its figure and results.

    Returns
    -------
    dict
        Summary of the stability limits and the empirical boundary.
    """
    print("=" * 70)
    print("Experiment 2: explicit-Euler stability limit and the checkerboard mode")
    print("=" * 70)

    base = cfg.default_config().replace(label="ex02_stability")
    utils.set_seed(base.seed)

    D = base.grid.baseline_D
    dx = base.grid.dx_cm
    f_v = base.solver.f_v_bound

    limits = solvers.stability_limits(D, dx, f_v, base.solver.dt_ms)

    print(f"  4 D / dx^2                     = {limits.diffusion_number:.4f} 1/ms")
    print(f"  |f_V|_max                      = {limits.reaction_bound:.4f} 1/ms")
    print(
        f"  reaction-diffusion limit       = "
        f"{limits.reaction_diffusion_dt_ms:.6f} ms   (2/43)"
    )
    print(
        f"  pure-diffusion limit           = "
        f"{limits.pure_diffusion_dt_ms:.6f} ms   (dx^2/2D)"
    )
    print(
        f"  pure-diffusion overestimate    = "
        f"{100.0 * limits.relative_overestimate:.2f} %"
    )
    print(
        f"  working step dt = {base.solver.dt_ms} ms -> safety factor "
        f"{limits.safety_factor:.2f}"
    )

    dt_limit = limits.reaction_diffusion_dt_ms

    # ---- Two illustrative full-model runs ----------------------------------
    def run_at(
        fraction: float, record_every: int | None = None
    ) -> simulate.SimulationResult:
        """Run the full model at a given fraction of the stability limit.

        ``record_every`` defaults to roughly one snapshot per 0.5 ms; pass 1 to
        capture every step, which is needed to see an unstable run before it
        saturates.
        """
        dt = fraction * dt_limit
        cadence = (
            record_every
            if record_every is not None
            else max(1, int(round(0.5 / dt)))
        )
        config = base.replace(
            solver=cfg.SolverParams(
                dt_ms=dt,
                t_end_ms=STABILITY_RUN_MS,
                method="euler",
                f_v_bound=f_v,
                record_every=cadence,
            )
        )
        # force=True is required above the limit and harmless below it: this
        # experiment exists precisely to step past the guard deliberately.
        return simulate.run_simulation(config, force=True)

    stable = run_at(STABLE_FRACTION)
    # Every step, because blow-up is hyper-exponential once the cubic takes
    # over: between two snapshots 10 steps apart the solution jumps from
    # order 1 to order 1e15, and any plot of the later state is a single spike
    # with no visible structure. Per-step recording catches it mid-growth.
    unstable = run_at(UNSTABLE_FRACTION, record_every=1)

    print(
        f"  dt = {STABLE_FRACTION:.2f} x limit = {STABLE_FRACTION * dt_limit:.6f} ms"
        f"  -> diverged = {stable.diverged}"
    )
    print(
        f"  dt = {UNSTABLE_FRACTION:.2f} x limit = "
        f"{UNSTABLE_FRACTION * dt_limit:.6f} ms"
        f"  -> diverged = {unstable.diverged}"
        + (
            f" at t = {unstable.divergence_time_ms:.3f} ms"
            if unstable.diverged
            else ""
        )
    )

    # ---- Empirical boundary: where does the full model actually fail? ------
    sweep_rows = []
    checkerboard_traces = {}
    for fraction in SWEEP_FRACTIONS:
        result = run_at(fraction)
        amplitudes = np.array(
            [solvers.checkerboard_amplitude(V) for V in result.V_snapshots]
        )
        checkerboard_traces[fraction] = (result.snapshot_times, amplitudes)
        sweep_rows.append(
            (
                fraction,
                fraction * dt_limit,
                result.diverged,
                result.divergence_time_ms,
                float(amplitudes[-1] / max(amplitudes[0], np.finfo(float).tiny)),
            )
        )
        print(
            f"    dt/limit = {fraction:.2f}  diverged = {str(result.diverged):<5} "
            f"checkerboard growth = {sweep_rows[-1][4]:.3e}"
        )

    # ---- Locate the empirical boundary precisely, by bisection -------------
    def blows_up(fraction: float) -> bool:
        """Whether the full model diverges within BOUNDARY_RUN_MS at this step."""
        config = base.replace(
            solver=cfg.SolverParams(
                dt_ms=fraction * dt_limit,
                t_end_ms=BOUNDARY_RUN_MS,
                method="euler",
                f_v_bound=f_v,
                # Snapshots are irrelevant here; keep them sparse for speed.
                record_every=1000,
            )
        )
        return simulate.run_simulation(config, force=True).diverged

    empirical_boundary, boundary_iterations = stimulus.bisect_threshold(
        succeeds=blows_up,
        lower=BOUNDARY_BRACKET[0],
        upper=BOUNDARY_BRACKET[1],
        tolerance=BOUNDARY_TOLERANCE,
    )
    print(
        f"  empirical boundary  dt/limit = {empirical_boundary:.4f} "
        f"({boundary_iterations} bisection steps)"
    )

    # ---- Why the empirical boundary sits slightly ABOVE 1 ------------------
    # |f_V|_max = 3 is an UPPER BOUND taken from |1 - V^2| at the excited root
    # V3 = 1.986. The solution never actually reaches V3 -- w rises during the
    # upstroke and caps the peak lower -- so the true reaction contribution is
    # smaller and the true limit is correspondingly larger. Recomputing the
    # limit from the peak the solution ACTUALLY attains explains the gap.
    actual_peak_V = float(np.max(stable.V_peak))
    actual_f_v = abs(1.0 - actual_peak_V**2)
    refined_limit = solvers.explicit_euler_dt_limit(D, dx, actual_f_v)
    refined_fraction = refined_limit / dt_limit
    print(f"  actual peak V                  = {actual_peak_V:.4f}")
    print(f"  actual |f_V| = |1 - V_peak^2|  = {actual_f_v:.4f}  (bound used: {f_v})")
    print(
        f"  limit from the actual peak     = {refined_limit:.6f} ms "
        f"= {refined_fraction:.4f} x the conservative limit"
    )

    # ---- Clean linear comparison with the closed-form amplification --------
    pure_limit = limits.pure_diffusion_dt_ms
    linear_dt = LINEAR_SWEEP_FRACTIONS * pure_limit
    measured_growth = np.array(
        [
            linear_checkerboard_growth(float(dt), dx, D, LINEAR_SWEEP_STEPS)
            for dt in linear_dt
        ]
    )
    # von Neumann: for the checkerboard mode sin^2(k dx/2) = 1, so
    # g = 1 - 4 D dt / dx^2.
    predicted_growth = np.abs(1.0 - 4.0 * D * linear_dt / dx**2)

    max_theory_error = float(
        np.max(np.abs(measured_growth - predicted_growth))
    )
    print(
        f"  linear test: max |measured - predicted| amplification = "
        f"{max_theory_error:.3e}"
    )

    # ---- Figure ------------------------------------------------------------
    fig, axes = plotting.new_figure(
        figsize=(10.0, 7.5), nrows=2, ncols=2, constrained_layout=True
    )
    ax_stable, ax_unstable, ax_growth, ax_theory = axes.flatten()

    # (a) Stable run.
    for target in (5.0, 15.0, 25.0, 35.0):
        index = int(np.argmin(np.abs(stable.snapshot_times - target)))
        ax_stable.plot(
            stable.x,
            stable.V_snapshots[index],
            label=f"$t={stable.snapshot_times[index]:.0f}$ ms",
        )
    plotting.label_axes(
        ax_stable,
        "position $x$ (cm)",
        "membrane potential $V$ (dimensionless)",
        f"(a) Stable: $\\Delta t = {STABLE_FRACTION:.2f}\\,\\Delta t_{{\\max}}"
        f" = {STABLE_FRACTION * dt_limit:.4f}$ ms",
    )
    ax_stable.legend(loc="upper right", fontsize=8)

    # (b) Unstable run, shown while the instability is still legible.
    # The FINAL snapshot is useless for seeing structure: by then the solution
    # has reached 1e15 and any plot of it is one spike. Instead, take the first
    # snapshot at which the instability clearly dominates the physiological
    # signal but has not yet saturated the axis.
    snapshot_magnitudes = np.max(np.abs(unstable.V_snapshots), axis=1)
    beyond_display = np.flatnonzero(snapshot_magnitudes > DISPLAY_MAGNITUDE)
    display_index = (
        int(beyond_display[0])
        if beyond_display.size > 0
        else len(snapshot_magnitudes) - 1
    )
    display_V = unstable.V_snapshots[display_index]
    display_time = float(unstable.snapshot_times[display_index])

    ax_unstable.plot(
        unstable.x,
        display_V,
        color=plotting.COLOUR_BLOCKED,
        linewidth=1.0,
        label=f"$t={display_time:.2f}$ ms",
    )
    plotting.label_axes(
        ax_unstable,
        "position $x$ (cm)",
        "membrane potential $V$ (dimensionless)",
        f"(b) Unstable: $\\Delta t = {UNSTABLE_FRACTION:.2f}\\,\\Delta t_{{\\max}}"
        f" = {UNSTABLE_FRACTION * dt_limit:.4f}$ ms",
    )
    ax_unstable.legend(loc="upper right", fontsize=8)

    # Inset showing the node-to-node alternation that identifies the mode.
    # Sixteen nodes is enough to see eight full periods of the k = pi/dx mode.
    peak_node = int(np.argmax(np.abs(display_V)))
    lo = max(0, peak_node - 8)
    hi = min(display_V.size, peak_node + 9)
    inset = ax_unstable.inset_axes((0.42, 0.10, 0.55, 0.38))
    inset.plot(
        unstable.x[lo:hi],
        display_V[lo:hi],
        marker="o",
        markersize=3.5,
        color=plotting.COLOUR_BLOCKED,
        linewidth=1.0,
    )
    inset.set_title(
        "sign alternates every node:\n$k=\\pi/\\Delta x$", fontsize=7
    )
    inset.set_xlabel("$x$ (cm)", fontsize=6)
    inset.tick_params(labelsize=6)
    inset.grid(alpha=0.3)

    # (c) Checkerboard growth.
    for fraction, (times, amplitudes) in checkerboard_traces.items():
        style = "-" if fraction < 1.0 else "--"
        ax_growth.plot(
            times,
            np.maximum(amplitudes, np.finfo(float).tiny),
            style,
            label=f"$\\Delta t/\\Delta t_{{\\max}} = {fraction:.2f}$",
        )
    ax_growth.set_yscale("log")
    plotting.label_axes(
        ax_growth,
        "time $t$ (ms)",
        "checkerboard amplitude (dimensionless)",
        "(c) Growth of the $k=\\pi/\\Delta x$ component",
    )
    ax_growth.legend(loc="lower right", fontsize=7.5)

    # (d) Theory comparison.
    ax_theory.plot(
        linear_dt,
        predicted_growth,
        color=plotting.COLOUR_ANALYTIC,
        label=r"von Neumann $|g|=|1-4D\Delta t/\Delta x^2|$",
    )
    ax_theory.plot(
        linear_dt[::2],
        measured_growth[::2],
        linestyle="none",
        marker="o",
        markersize=4,
        markerfacecolor="none",
        color=plotting.COLOUR_MEASURED,
        label="measured amplification",
    )
    ax_theory.axhline(
        1.0,
        color=plotting.COLOUR_REFERENCE,
        linewidth=1.0,
        linestyle=":",
        label="$|g|=1$ (stability boundary)",
    )
    ax_theory.axvline(
        limits.reaction_diffusion_dt_ms,
        color=plotting.PALETTE["green"],
        linestyle="-.",
        linewidth=1.2,
        label=f"reaction-diffusion limit {limits.reaction_diffusion_dt_ms:.5f} ms",
    )
    ax_theory.axvline(
        limits.pure_diffusion_dt_ms,
        color=plotting.PALETTE["purple"],
        linestyle="-.",
        linewidth=1.2,
        label=f"pure-diffusion limit {limits.pure_diffusion_dt_ms:.5f} ms",
    )
    ax_theory.axvline(
        empirical_boundary * dt_limit,
        color=plotting.PALETTE["orange"],
        linestyle="-",
        linewidth=1.2,
        label=f"empirical failure {empirical_boundary * dt_limit:.5f} ms",
    )
    plotting.label_axes(
        ax_theory,
        "time step $\\Delta t$ (ms)",
        "amplification per step $|g|$ (dimensionless)",
        "(d) Measured vs predicted amplification (linear problem)",
    )
    ax_theory.legend(loc="upper left", fontsize=7)
    plotting.annotate_takeaway(
        ax_theory,
        f"pure-diffusion limit is\n"
        f"{100.0 * limits.relative_overestimate:.1f} % too optimistic",
        loc="lower right",
    )

    caption = (
        f"The von Neumann stability limit dt = {dt_limit:.5f} ms is the limit the "
        f"code obeys, and it fails in the predicted way: the growing component is "
        f"the checkerboard mode k = pi/dx, not generic noise. Measured "
        f"amplification matches the closed-form |1 - 4D dt/dx^2| to "
        f"{max_theory_error:.1e}. Failure is first seen at "
        f"{empirical_boundary:.3f} times the limit rather than exactly at it, "
        f"because |f_V| = 3 is an upper bound taken at the excited root V3 = 1.986 "
        f"whereas the solution peaks at only {actual_peak_V:.3f}; using that peak "
        f"predicts {refined_fraction:.3f}. Ignoring the reaction term entirely "
        f"would give {limits.pure_diffusion_dt_ms:.4f} ms, "
        f"{100.0 * limits.relative_overestimate:.1f} % too optimistic."
    )

    measurements = {
        "diffusion_number_per_ms": limits.diffusion_number,
        "f_v_bound": limits.reaction_bound,
        "reaction_diffusion_dt_limit_ms": limits.reaction_diffusion_dt_ms,
        "pure_diffusion_dt_limit_ms": limits.pure_diffusion_dt_ms,
        "pure_diffusion_relative_overestimate": limits.relative_overestimate,
        "working_dt_ms": base.solver.dt_ms,
        "working_safety_factor": limits.safety_factor,
        "stable_run_diverged": stable.diverged,
        "unstable_run_diverged": unstable.diverged,
        "unstable_divergence_time_ms": unstable.divergence_time_ms,
        "empirical_boundary_fraction_of_limit": empirical_boundary,
        "empirical_boundary_dt_ms": empirical_boundary * dt_limit,
        "actual_peak_V": actual_peak_V,
        "actual_f_v_from_peak": actual_f_v,
        "refined_limit_ms": refined_limit,
        "refined_limit_fraction": refined_fraction,
        "linear_theory_max_amplification_error": max_theory_error,
    }

    plotting.save_figure(
        fig,
        "fig_ex02_stability",
        caption,
        config_dict=base.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_arrays(
        "ex02_stability",
        {
            "x_cm": stable.x,
            "stable_snapshot_times_ms": stable.snapshot_times,
            "stable_V": stable.V_snapshots,
            "unstable_snapshot_times_ms": unstable.snapshot_times,
            "unstable_V": unstable.V_snapshots,
            "linear_dt_ms": linear_dt,
            "linear_measured_amplification": measured_growth,
            "linear_predicted_amplification": predicted_growth,
        },
        config_dict=base.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_table(
        "ex02_stability_sweep",
        header=[
            "dt_over_limit",
            "dt_ms",
            "diverged",
            "divergence_time_ms",
            "checkerboard_growth_factor",
        ],
        rows=[
            (
                f"{fraction:.3f}",
                f"{dt_value:.6f}",
                str(diverged),
                f"{time_value:.6f}",
                f"{growth:.6e}",
            )
            for fraction, dt_value, diverged, time_value, growth in sweep_rows
        ],
    )

    print("  figure -> figures/fig_ex02_stability.png / .pdf")
    return measurements


if __name__ == "__main__":
    main()
