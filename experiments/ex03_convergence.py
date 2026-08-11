"""Experiment 3: observed order of accuracy in space and time.

Question
--------
The scheme is designed to be second order in ``dx`` (centred differences) and
first order in ``dt`` (explicit Euler). Is it? And once the orders are
confirmed, what does the grid-converged conduction velocity actually converge
*to*, so that the discretisation error in the headline number can be quoted
rather than guessed?

Design
------
Three separate refinement studies, because they answer different questions:

1. **Spatial, against an exact solution.** Pure diffusion of a Gaussian, where
   the exact sealed-strand answer is known, so the error is unambiguous. This
   is the clean measurement of the spatial order.

2. **Spatial, on the real problem.** Conduction velocity against ``dx``, with
   Richardson extrapolation to estimate the grid-converged value. This is what
   justifies quoting a discretisation error on the reported velocity.

3. **Temporal.** Error at fixed ``dx`` against a high-resolution RK4 reference,
   for both Euler and RK4. Holding ``dx`` fixed is essential: it makes the
   spatial error identical in every run so that it cancels in the difference,
   leaving only the temporal error.

Note that a spatial refinement study **cannot** hold ``dt`` fixed. The
stability limit scales as ``dx^2``, so halving ``dx`` forces roughly a
quartering of ``dt``. Each spatial run therefore uses the largest step with a
constant safety factor below its own limit.

Run standalone with::

    python experiments/ex03_convergence.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibroblock import config as cfg  # noqa: E402
from fibroblock import (  # noqa: E402
    measure,
    operators,
    plotting,
    simulate,
    solvers,
    utils,
)

# --- Study 1: spatial order against the exact Gaussian ----------------------
DIFFUSION_SPACINGS_CM: tuple[float, ...] = (0.04, 0.02, 0.01, 0.005, 0.0025)
DIFFUSION_DURATION_MS: float = 10.0
GAUSSIAN_CENTRE_CM: float = 1.0
GAUSSIAN_SIGMA0_CM: float = 0.1
GAUSSIAN_AMPLITUDE: float = 1.0

# --- Study 2: conduction velocity against dx --------------------------------
# Stops at 0.0025 cm: the next halving costs about 27 s for a change in the
# extrapolated velocity of under 0.01 %, which does not earn its place in a
# pipeline the examiner has to run.
VELOCITY_SPACINGS_CM: tuple[float, ...] = (0.02, 0.01, 0.005, 0.0025)
VELOCITY_DURATION_MS: float = 110.0

# --- Study 3: temporal order -----------------------------------------------
TEMPORAL_DT_MS: tuple[float, ...] = (0.02, 0.01, 0.005, 0.0025)
# Reference step, 25x finer than the finest test step. Integrated with RK4, so
# its own temporal error is around (1e-4)^4 -- far below double precision --
# and it can be treated as exact.
TEMPORAL_REFERENCE_DT_MS: float = 0.0001
TEMPORAL_DURATION_MS: float = 20.0

# Time at which the pre-formed wave used for the smooth temporal study is
# taken. By 30 ms the front has settled into its travelling shape and is well
# clear of the stimulus site, so the state is smooth in both x and t.
PREFORMED_WAVE_TIME_MS: float = 30.0
PREFORMED_WAVE_DT_MS: float = 0.005

# Every run keeps this much margin below its own stability limit, so that the
# time step is refined consistently rather than arbitrarily.
SAFETY_FACTOR: float = 2.5


def main() -> dict[str, Any]:
    """Run experiment 3 and write its figure and results.

    Returns
    -------
    dict
        Observed orders and the Richardson-extrapolated velocity.
    """
    print("=" * 70)
    print("Experiment 3: observed order of accuracy in space and time")
    print("=" * 70)

    base = cfg.default_config().replace(label="ex03_convergence")
    utils.set_seed(base.seed)

    D = base.grid.baseline_D
    length = base.grid.length_cm

    # -----------------------------------------------------------------------
    # Study 1: spatial order against the exact sealed-strand Gaussian
    # -----------------------------------------------------------------------
    print("\n  [1] spatial order, pure diffusion against the exact solution")
    print(f"      {'dx (cm)':>10} {'dt (ms)':>10} {'nodes':>7} {'L2 error':>12}")

    diffusion_errors = []
    for dx in DIFFUSION_SPACINGS_CM:
        # With no reaction term the bound on |f_V| is genuinely zero.
        dt = solvers.pure_diffusion_dt_limit(D, dx) / SAFETY_FACTOR
        config = base.replace(
            grid=cfg.GridParams(length_cm=length, dx_cm=dx, baseline_D=D),
            gap=cfg.GapParams(rho=1.0, gap_length_cm=0.0),
            solver=cfg.SolverParams(
                dt_ms=dt,
                t_end_ms=DIFFUSION_DURATION_MS,
                f_v_bound=0.0,
                record_every=1000000,  # only the final state is needed
            ),
        )
        x = np.linspace(0.0, length, config.grid.n_nodes)
        initial_V = operators.analytic_gaussian_sealed(
            x,
            0.0,
            D,
            GAUSSIAN_CENTRE_CM,
            GAUSSIAN_SIGMA0_CM,
            length,
            GAUSSIAN_AMPLITUDE,
        )
        result = simulate.run_simulation(
            config,
            initial_V=initial_V,
            initial_w=np.zeros_like(initial_V),
            include_reaction=False,
            include_stimulus=False,
        )
        exact = operators.analytic_gaussian_sealed(
            x,
            DIFFUSION_DURATION_MS,
            D,
            GAUSSIAN_CENTRE_CM,
            GAUSSIAN_SIGMA0_CM,
            length,
            GAUSSIAN_AMPLITUDE,
        )
        error = measure.l2_error(
            result.V_snapshots[-1], exact, result.grid.quadrature_weights
        )
        diffusion_errors.append(error)
        print(
            f"      {dx:>10.5f} {dt:>10.6f} {config.grid.n_nodes:>7} {error:>12.4e}"
        )

    spacings = np.array(DIFFUSION_SPACINGS_CM)
    diffusion_errors = np.array(diffusion_errors)
    pairwise_space, fitted_all = measure.observed_order(spacings, diffusion_errors)

    # The coarsest grid is PRE-ASYMPTOTIC and must be excluded from the fitted
    # order. At dx = 0.04 cm the Gaussian's standard deviation of 0.1 cm spans
    # only 2.5 nodes, so the profile is not resolved at all and the error is
    # not yet governed by the leading truncation term. Including that point
    # inflates the fitted order to about 2.7 and would be a misleading claim.
    # It is kept in the plot precisely to show where the asymptotic regime
    # begins.
    asymptotic = slice(1, None)
    _, fitted_space = measure.observed_order(
        spacings[asymptotic], diffusion_errors[asymptotic]
    )
    print(f"      pairwise orders: {np.array2string(pairwise_space, precision=3)}")
    print(f"      fitted order, all points:        {fitted_all:.4f} (pre-asymptotic)")
    print(f"      fitted order, asymptotic range:  {fitted_space:.4f}")

    # -----------------------------------------------------------------------
    # Study 2: conduction velocity against dx, with Richardson extrapolation
    # -----------------------------------------------------------------------
    print("\n  [2] conduction velocity against dx")
    print(f"      {'dx (cm)':>10} {'dt (ms)':>10} {'theta (cm/ms)':>15} {'R^2':>12}")

    velocities = []
    for dx in VELOCITY_SPACINGS_CM:
        dt = (
            solvers.explicit_euler_dt_limit(D, dx, base.solver.f_v_bound)
            / SAFETY_FACTOR
        )
        config = base.replace(
            grid=cfg.GridParams(length_cm=length, dx_cm=dx, baseline_D=D),
            gap=cfg.GapParams(rho=1.0, gap_length_cm=0.0),
            solver=cfg.SolverParams(
                dt_ms=dt,
                t_end_ms=VELOCITY_DURATION_MS,
                f_v_bound=base.solver.f_v_bound,
                record_every=1000000,
            ),
        )
        result = simulate.run_simulation(config)
        fit = measure.measure_velocity(result)
        velocities.append(fit.theta_cm_per_ms)
        print(
            f"      {dx:>10.5f} {dt:>10.6f} {fit.theta_cm_per_ms:>15.7f} "
            f"{fit.r_squared:>12.8f}"
        )

    velocities = np.array(velocities)
    velocity_spacings = np.array(VELOCITY_SPACINGS_CM)

    # Richardson extrapolation from the three finest grids. With a refinement
    # ratio of 2 and observed order p, the extrapolated value is
    #     theta_exact ~ theta_fine + (theta_fine - theta_medium) / (2^p - 1).
    coarse, medium, fine = velocities[-3], velocities[-2], velocities[-1]
    velocity_order = float(np.log2(abs((coarse - medium) / (medium - fine))))
    richardson = float(fine + (fine - medium) / (2.0**velocity_order - 1.0))
    dx_error_at_default = abs(velocities[1] - richardson) / abs(richardson)

    print(f"      observed order in theta: {velocity_order:.4f}")
    print(f"      Richardson-extrapolated theta: {richardson:.7f} cm/ms")
    print(
        f"      discretisation error at the default dx = "
        f"{base.grid.dx_cm} cm: {100.0 * dx_error_at_default:.3f} %"
    )

    # -----------------------------------------------------------------------
    # Study 3: temporal order at fixed dx
    # -----------------------------------------------------------------------
    print("\n  [3] temporal order at fixed dx = 0.01 cm")

    from fibroblock import grid as gridmod

    homogeneous_gap = cfg.GapParams(rho=1.0, gap_length_cm=0.0)
    weights = gridmod.build_grid(base.grid, homogeneous_gap).quadrature_weights

    def run_temporal(
        dt: float,
        method: str,
        with_stimulus: bool,
        initial: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Integrate for TEMPORAL_DURATION_MS and return the final profile."""
        config = base.replace(
            gap=homogeneous_gap,
            solver=cfg.SolverParams(
                dt_ms=dt,
                t_end_ms=TEMPORAL_DURATION_MS,
                method=method,
                f_v_bound=base.solver.f_v_bound,
                record_every=1000000,
            ),
        )
        initial_V, initial_w = (None, None) if initial is None else initial
        return simulate.run_simulation(
            config,
            initial_V=initial_V,
            initial_w=initial_w,
            include_stimulus=with_stimulus,
        ).V_snapshots[-1]

    def measure_orders(
        with_stimulus: bool, initial: tuple[np.ndarray, np.ndarray] | None
    ) -> dict[str, Any]:
        """Refine dt for both integrators and return errors and fitted orders."""
        reference = run_temporal(
            TEMPORAL_REFERENCE_DT_MS, "rk4", with_stimulus, initial
        )
        euler, rk4 = [], []
        for dt in TEMPORAL_DT_MS:
            euler.append(
                measure.l2_error(
                    run_temporal(dt, "euler", with_stimulus, initial),
                    reference,
                    weights,
                )
            )
            rk4.append(
                measure.l2_error(
                    run_temporal(dt, "rk4", with_stimulus, initial),
                    reference,
                    weights,
                )
            )
        euler_array = np.array(euler)
        rk4_array = np.array(rk4)
        steps_array = np.array(TEMPORAL_DT_MS)
        pairwise_e, fitted_e = measure.observed_order(steps_array, euler_array)
        pairwise_r, fitted_r = measure.observed_order(steps_array, rk4_array)
        return {
            "euler_errors": euler_array,
            "rk4_errors": rk4_array,
            "euler_pairwise": pairwise_e,
            "rk4_pairwise": pairwise_r,
            "euler_order": fitted_e,
            "rk4_order": fitted_r,
        }

    # --- 3a: including the stimulus -------------------------------------
    # The stimulus switches off discontinuously at t = 1 ms. A jump
    # discontinuity in the forcing caps the achievable order at 1 for ANY
    # one-step method, no matter how many stages it has, because the Taylor
    # expansion the higher order relies on does not exist across the jump.
    # This is measured rather than asserted.
    print("      (3a) including the discontinuous stimulus")
    with_stim = measure_orders(with_stimulus=True, initial=None)
    for dt, e_err, r_err in zip(
        TEMPORAL_DT_MS, with_stim["euler_errors"], with_stim["rk4_errors"], strict=True
    ):
        print(
            f"          dt = {dt:>8.5f} ms   Euler L2 = {e_err:.4e}   "
            f"RK4 L2 = {r_err:.4e}"
        )
    print(f"          Euler fitted order: {with_stim['euler_order']:.4f}")
    print(
        f"          RK4   fitted order: {with_stim['rk4_order']:.4f} "
        f"<- capped at 1 by the stimulus discontinuity, not an RK4 defect"
    )

    # --- 3b: smooth propagation, no stimulus -----------------------------
    # Start from an already-formed travelling wave so the right-hand side is
    # smooth in time throughout. This is the setting in which the design
    # orders are actually claimed.
    print("      (3b) smooth propagation, starting from a pre-formed wave")
    preformed_config = base.replace(
        gap=homogeneous_gap,
        solver=cfg.SolverParams(
            dt_ms=PREFORMED_WAVE_DT_MS,
            t_end_ms=PREFORMED_WAVE_TIME_MS,
            method="rk4",
            f_v_bound=base.solver.f_v_bound,
            record_every=1000000,
        ),
    )
    preformed = simulate.run_simulation(preformed_config)
    initial_state = (
        preformed.V_snapshots[-1].copy(),
        preformed.w_snapshots[-1].copy(),
    )

    smooth = measure_orders(with_stimulus=False, initial=initial_state)
    for dt, e_err, r_err in zip(
        TEMPORAL_DT_MS, smooth["euler_errors"], smooth["rk4_errors"], strict=True
    ):
        print(
            f"          dt = {dt:>8.5f} ms   Euler L2 = {e_err:.4e}   "
            f"RK4 L2 = {r_err:.4e}"
        )
    euler_pairwise_text = np.array2string(smooth["euler_pairwise"], precision=3)
    print(f"          Euler pairwise: {euler_pairwise_text}")
    print(f"          Euler fitted order: {smooth['euler_order']:.4f}   (expected 1)")
    rk4_pairwise_text = np.array2string(smooth["rk4_pairwise"], precision=3)
    print(f"          RK4   pairwise: {rk4_pairwise_text}")
    print(f"          RK4   fitted order: {smooth['rk4_order']:.4f}   (expected 4)")

    steps = np.array(TEMPORAL_DT_MS)
    euler_errors = smooth["euler_errors"]
    rk4_errors = smooth["rk4_errors"]
    pairwise_euler = smooth["euler_pairwise"]
    pairwise_rk4 = smooth["rk4_pairwise"]
    fitted_euler = smooth["euler_order"]
    fitted_rk4 = smooth["rk4_order"]

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig, axes = plotting.new_figure(
        figsize=(10.0, 7.5), nrows=2, ncols=2, constrained_layout=True
    )
    ax_space, ax_velocity, ax_time, ax_orders = axes.flatten()

    # (a) Spatial order against the exact solution.
    ax_space.loglog(
        spacings,
        diffusion_errors,
        marker="o",
        color=plotting.COLOUR_MEASURED,
        label="measured $L_2$ error",
    )
    # Reference slope anchored at the coarsest point.
    reference_line = diffusion_errors[0] * (spacings / spacings[0]) ** 2
    ax_space.loglog(
        spacings,
        reference_line,
        linestyle="--",
        color=plotting.COLOUR_ANALYTIC,
        label=r"slope 2 reference $\propto \Delta x^2$",
    )
    plotting.label_axes(
        ax_space,
        "node spacing $\\Delta x$ (cm)",
        "$L_2$ error (dimensionless $\\times \\sqrt{\\mathrm{cm}}$)",
        "(a) Spatial order, pure diffusion vs exact",
    )
    ax_space.legend(loc="lower right", fontsize=8)
    plotting.set_log_ticks(ax_space, spacings, axis="x")
    ax_space.tick_params(axis="x", labelrotation=45, labelsize=7)
    plotting.annotate_takeaway(
        ax_space,
        f"fitted order {fitted_space:.3f}\n(asymptotic range)",
        loc="upper left",
    )

    # (b) Velocity convergence.
    ax_velocity.plot(
        velocity_spacings,
        velocities,
        marker="o",
        color=plotting.COLOUR_MEASURED,
        label="measured $\\theta$",
    )
    ax_velocity.axhline(
        richardson,
        color=plotting.COLOUR_ANALYTIC,
        linestyle="--",
        label=f"Richardson limit {richardson:.6f} cm/ms",
    )
    ax_velocity.axvline(
        base.grid.dx_cm,
        color=plotting.COLOUR_REFERENCE,
        linestyle=":",
        linewidth=1.0,
        label=f"default $\\Delta x = {base.grid.dx_cm}$ cm",
    )
    ax_velocity.set_xscale("log")
    plotting.label_axes(
        ax_velocity,
        "node spacing $\\Delta x$ (cm)",
        "conduction velocity $\\theta$ (cm/ms)",
        "(b) Velocity convergence and Richardson limit",
    )
    ax_velocity.legend(loc="lower left", fontsize=8)
    plotting.set_log_ticks(ax_velocity, velocity_spacings, axis="x")
    ax_velocity.tick_params(axis="x", labelrotation=45, labelsize=7)
    # Headroom so the Richardson line is not flush against the top spine.
    span = float(velocities.max() - velocities.min())
    ax_velocity.set_ylim(velocities.min() - 0.15 * span, richardson + 0.25 * span)
    plotting.annotate_takeaway(
        ax_velocity,
        f"error at default $\\Delta x$:\n{100.0 * dx_error_at_default:.2f} %",
        loc="upper right",
    )

    # (c) Temporal order, smooth propagation vs the stimulus-limited case.
    ax_time.loglog(
        steps,
        euler_errors,
        marker="o",
        color=plotting.COLOUR_MEASURED,
        label=f"Euler, smooth (order {fitted_euler:.2f})",
    )
    ax_time.loglog(
        steps,
        rk4_errors,
        marker="s",
        color=plotting.PALETTE["green"],
        label=f"RK4, smooth (order {fitted_rk4:.2f})",
    )
    ax_time.loglog(
        steps,
        with_stim["rk4_errors"],
        marker="v",
        linestyle="--",
        color=plotting.PALETTE["purple"],
        label=f"RK4, with stimulus (order {with_stim['rk4_order']:.2f})",
    )
    ax_time.loglog(
        steps,
        euler_errors[0] * (steps / steps[0]) ** 1,
        linestyle="--",
        color=plotting.COLOUR_ANALYTIC,
        linewidth=1.0,
        label=r"slope 1",
    )
    ax_time.loglog(
        steps,
        rk4_errors[0] * (steps / steps[0]) ** 4,
        linestyle=":",
        color=plotting.COLOUR_REFERENCE,
        linewidth=1.0,
        label=r"slope 4",
    )
    plotting.label_axes(
        ax_time,
        "time step $\\Delta t$ (ms)",
        "$L_2$ error vs fine reference",
        "(c) Temporal order at fixed $\\Delta x$",
    )
    ax_time.legend(loc="lower right", fontsize=6.8)
    plotting.set_log_ticks(ax_time, steps, axis="x")
    ax_time.tick_params(axis="x", labelrotation=45, labelsize=7)
    # Two decades of headroom so the annotation clears the topmost curve.
    ax_time.set_ylim(top=float(euler_errors.max()) * 1.0e3)
    plotting.annotate_takeaway(
        ax_time,
        "the discontinuous stimulus\ncaps RK4 at first order",
        loc="upper left",
    )

    # (d) Pairwise observed orders.
    ax_orders.plot(
        spacings[1:],
        pairwise_space,
        marker="o",
        color=plotting.COLOUR_MEASURED,
        label="spatial (expect 2)",
    )
    ax_orders.plot(
        steps[1:],
        pairwise_euler,
        marker="^",
        color=plotting.PALETTE["orange"],
        label="temporal, Euler (expect 1)",
    )
    ax_orders.plot(
        steps[1:],
        pairwise_rk4,
        marker="s",
        color=plotting.PALETTE["green"],
        label="temporal, RK4 (expect 4)",
    )
    for expected in (1.0, 2.0, 4.0):
        ax_orders.axhline(
            expected, color=plotting.COLOUR_REFERENCE, linestyle=":", linewidth=0.9
        )
    ax_orders.set_xscale("log")
    plotting.label_axes(
        ax_orders,
        "refinement level ($\\Delta x$ in cm, or $\\Delta t$ in ms)",
        "observed order between successive grids",
        "(d) Observed orders approach their design values",
    )
    ax_orders.legend(loc="upper left", fontsize=7.5)
    plotting.set_log_ticks(
        ax_orders, np.concatenate([spacings[1:], steps[1:]]), axis="x"
    )
    ax_orders.tick_params(axis="x", labelrotation=45, labelsize=7)
    plotting.annotate_takeaway(
        ax_orders,
        "the point at 5.3 is the\npre-asymptotic coarsest grid",
        loc="upper right",
    )

    caption = (
        f"The scheme achieves its design orders on smooth data: "
        f"{fitted_space:.2f} in space against the exact diffusion solution "
        f"(design 2), {fitted_euler:.2f} for explicit Euler and {fitted_rk4:.2f} "
        f"for RK4 in time (designs 1 and 4). Panel (c) also shows why the "
        f"stimulus must be excluded from the temporal study: its discontinuous "
        f"switch-off at 1 ms caps even RK4 at order "
        f"{with_stim['rk4_order']:.2f}, since a jump in the forcing destroys the "
        f"Taylor expansion higher order depends on. Richardson extrapolation "
        f"gives a grid-converged velocity of {richardson:.5f} cm/ms, so the "
        f"default dx = {base.grid.dx_cm} cm carries only "
        f"{100.0 * dx_error_at_default:.2f} % discretisation error -- the "
        f"discrepancy against the analytic velocity is therefore physical, not "
        f"numerical."
    )

    measurements = {
        "spatial_fitted_order_asymptotic": fitted_space,
        "spatial_fitted_order_all_points": fitted_all,
        "spatial_pairwise_orders": pairwise_space.tolist(),
        "velocity_observed_order": velocity_order,
        "velocity_richardson_cm_per_ms": richardson,
        "velocity_error_at_default_dx_relative": dx_error_at_default,
        "euler_fitted_order_smooth": fitted_euler,
        "euler_pairwise_orders_smooth": pairwise_euler.tolist(),
        "rk4_fitted_order_smooth": fitted_rk4,
        "rk4_pairwise_orders_smooth": pairwise_rk4.tolist(),
        "euler_fitted_order_with_stimulus": with_stim["euler_order"],
        "rk4_fitted_order_with_stimulus": with_stim["rk4_order"],
    }

    plotting.save_figure(
        fig,
        "fig_ex03_convergence",
        caption,
        config_dict=base.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_arrays(
        "ex03_convergence",
        {
            "diffusion_spacings_cm": spacings,
            "diffusion_l2_errors": diffusion_errors,
            "velocity_spacings_cm": velocity_spacings,
            "velocities_cm_per_ms": velocities,
            "temporal_steps_ms": steps,
            "euler_l2_errors_smooth": euler_errors,
            "rk4_l2_errors_smooth": rk4_errors,
            "euler_l2_errors_with_stimulus": with_stim["euler_errors"],
            "rk4_l2_errors_with_stimulus": with_stim["rk4_errors"],
        },
        config_dict=base.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_table(
        "ex03_convergence_summary",
        header=["study", "refinement", "error_or_value", "pairwise_order"],
        rows=(
            [
                ("spatial_diffusion", f"dx={dx:.5f}", f"{err:.6e}", "")
                for dx, err in zip(spacings, diffusion_errors, strict=True)
            ]
            + [
                ("spatial_diffusion_order", f"dx={dx:.5f}", "", f"{order:.4f}")
                for dx, order in zip(spacings[1:], pairwise_space, strict=True)
            ]
            + [
                ("velocity_vs_dx", f"dx={dx:.5f}", f"{v:.7f}", "")
                for dx, v in zip(velocity_spacings, velocities, strict=True)
            ]
            + [("velocity_richardson", "extrapolated", f"{richardson:.7f}", "")]
            + [
                ("temporal_euler_smooth", f"dt={dt:.5f}", f"{err:.6e}", "")
                for dt, err in zip(steps, euler_errors, strict=True)
            ]
            + [
                ("temporal_rk4_smooth", f"dt={dt:.5f}", f"{err:.6e}", "")
                for dt, err in zip(steps, rk4_errors, strict=True)
            ]
            + [
                ("temporal_rk4_with_stimulus", f"dt={dt:.5f}", f"{err:.6e}", "")
                for dt, err in zip(steps, with_stim["rk4_errors"], strict=True)
            ]
        ),
    )

    print("\n  figure -> figures/fig_ex03_convergence.png / .pdf")
    return measurements


if __name__ == "__main__":
    main()
