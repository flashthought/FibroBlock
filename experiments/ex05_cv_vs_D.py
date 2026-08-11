"""Experiment 5: testing whether conduction velocity scales as the square root of D.

Question
--------
Bistable front theory predicts ``theta = C sqrt(D)``. Two things need testing:
the **exponent** (is it really 1/2?) and the **prefactor** (is it really
0.9630, and if not, why not?).

Two tests of the exponent, because they are not equally sharp
-------------------------------------------------------------
A log-log fit of ``theta`` against ``D`` can return a slope of 0.500 even when
the data curve systematically, because taking logarithms compresses exactly the
deviations that matter. Plotting ``theta / sqrt(D)`` directly is far more
demanding: the prediction is a horizontal line, and any curvature is
immediately visible. Both are produced.

Two grid families, because a fixed grid biases the test
-------------------------------------------------------
The wavefront thickness scales as ``sqrt(D / |f_V|)``. On a fixed grid, a small
``D`` therefore means a thinner front spread over fewer nodes -- so the
low-``D`` end of the sweep is systematically less well resolved than the
high-``D`` end, and the resulting drift in ``theta / sqrt(D)`` is a numerical
artefact, not physics. A second family with ``dx`` scaled as ``sqrt(D)`` holds
the nodes-per-front-thickness constant and removes that bias. It also, neatly,
holds ``4 D / dx^2`` constant, so every run in that family shares one time step.

A stimulus subtlety
-------------------
The liminal length -- the smallest patch of tissue that can be excited into a
propagating wave -- scales with the space constant, hence as ``sqrt(D)``. A
stimulus of fixed 0.1 cm width therefore becomes **sub-threshold** at large
``D``, and the wave fails to launch for reasons that have nothing to do with
the scaling law being tested. The sweep scales the stimulus width as
``sqrt(D)`` to hold it at a fixed number of space constants, and panel (c)
documents the failure that occurs otherwise.

Run standalone with::

    python experiments/ex05_cv_vs_D.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibroblock import config as cfg  # noqa: E402
from fibroblock import fhn, measure, plotting, simulate, solvers, utils  # noqa: E402

# Twenty-fold range in D, geometrically spaced so the log-log fit is evenly
# weighted. Centred on the assignment's D0 = 0.001 cm^2/ms.
D_VALUES: tuple[float, ...] = (0.0002, 0.0004, 0.0007, 0.001, 0.0015, 0.0025, 0.004)

# Reference point against which the scaled grid and stimulus are defined.
D_REFERENCE: float = 0.001
DX_REFERENCE_CM: float = 0.01
STIMULUS_WIDTH_REFERENCE_CM: float = 0.1

SAFETY_FACTOR: float = 2.3
MAX_DT_MS: float = 0.02

# Node at which the recovery variable is sampled to explain the prefactor.
# Inside the velocity fit window, and far from both the stimulus and the end.
FRONT_PROBE_X_CM: float = 1.2

# Snapshots every few tenths of a millisecond, so the upstroke is straddled by
# enough snapshots for the w-at-front interpolation to be meaningful.
SNAPSHOTS_PER_MS: float = 10.0


def duration_for(D: float, length_cm: float) -> float:
    """Compute the simulated time needed to clear the velocity fit window.

    Parameters
    ----------
    D : float
        Diffusion coefficient. cm^2/ms.
    length_cm : float
        Strand length. cm.

    Returns
    -------
    float
        Run duration in ms.

    Notes
    -----
    Slower waves need longer runs. The estimate uses the *measured* prefactor
    of about 0.81 rather than the analytic 0.963, with a 60 % margin and a
    fixed additive allowance for the time the wave takes to form out of the
    stimulus. Using the analytic prefactor would under-estimate the time needed
    and truncate the slowest runs.
    """
    approximate_prefactor = 0.81  # measured, not analytic; see module docstring
    crossing_time = length_cm / (approximate_prefactor * np.sqrt(D))
    return float(1.6 * crossing_time + 40.0)


def run_for_D(
    base: cfg.RunConfig,
    D: float,
    scale_grid: bool,
    scale_stimulus: bool,
) -> tuple[simulate.SimulationResult, cfg.RunConfig]:
    """Run one homogeneous strand at a given diffusion coefficient.

    Parameters
    ----------
    base : RunConfig
        Baseline configuration.
    D : float
        Diffusion coefficient. cm^2/ms.
    scale_grid : bool
        Scale ``dx`` as ``sqrt(D)`` to hold nodes-per-front-thickness constant.
    scale_stimulus : bool
        Scale the stimulus width as ``sqrt(D)`` to hold it at a fixed number of
        space constants.

    Returns
    -------
    result : SimulationResult
    config : RunConfig
        The configuration actually used.
    """
    scale = np.sqrt(D / D_REFERENCE)

    if scale_grid:
        # Round to a spacing that still divides the strand exactly, since
        # GridParams rejects a non-integer cell count.
        target_dx = DX_REFERENCE_CM * scale
        n_intervals = max(2, int(round(base.grid.length_cm / target_dx)))
        dx = base.grid.length_cm / n_intervals
    else:
        dx = base.grid.dx_cm

    width = (
        STIMULUS_WIDTH_REFERENCE_CM * scale
        if scale_stimulus
        else STIMULUS_WIDTH_REFERENCE_CM
    )

    dt = min(
        MAX_DT_MS,
        solvers.explicit_euler_dt_limit(D, dx, base.solver.f_v_bound) / SAFETY_FACTOR,
    )
    t_end = duration_for(D, base.grid.length_cm)

    config = base.replace(
        grid=cfg.GridParams(length_cm=base.grid.length_cm, dx_cm=dx, baseline_D=D),
        gap=cfg.GapParams(rho=1.0, gap_length_cm=0.0),
        stimulus=cfg.StimulusParams(
            amplitude=base.stimulus.amplitude,
            width_cm=width,
            duration_ms=base.stimulus.duration_ms,
        ),
        solver=cfg.SolverParams(
            dt_ms=dt,
            t_end_ms=t_end,
            f_v_bound=base.solver.f_v_bound,
            record_every=max(1, int(round(1.0 / (SNAPSHOTS_PER_MS * dt)))),
        ),
        label=f"ex05_D{D:g}",
    )
    return simulate.run_simulation(config), config


def main() -> dict[str, Any]:
    """Run experiment 5 and write its figure and results.

    Returns
    -------
    dict
        Fitted exponent, prefactors and the recovery-corrected prediction.
    """
    print("=" * 70)
    print("Experiment 5: conduction velocity against diffusion coefficient")
    print("=" * 70)

    base = cfg.default_config().replace(label="ex05_cv_vs_D")
    utils.set_seed(base.seed)

    params = base.fhn
    _, w_rest = fhn.rest_state(params)
    _, V2_rest, _ = fhn.bistable_roots(params)
    analytic_prefactor = fhn.analytic_cv_prefactor(params)
    print(f"  analytic prefactor (w frozen at rest) = {analytic_prefactor:.6f}")
    print(f"  threshold root V2 at rest             = {V2_rest:.6f}")

    # ---- Family 1: fixed grid ---------------------------------------------
    print("\n  [1] fixed grid, dx = 0.01 cm, stimulus width scaled as sqrt(D)")
    header = (
        f"      {'D':>8} {'dx':>8} {'nodes':>6} {'theta':>10} "
        f"{'theta/sqrtD':>12} {'R^2':>11} {'w at front':>11}"
    )
    print(header)

    fixed_velocities = []
    fixed_w_front = []
    for D in D_VALUES:
        result, config = run_for_D(base, D, scale_grid=False, scale_stimulus=True)
        fit = measure.measure_velocity(result)
        w_front = measure.recovery_at_front(result, FRONT_PROBE_X_CM, V2_rest)
        fixed_velocities.append(fit.theta_cm_per_ms)
        fixed_w_front.append(w_front)
        print(
            f"      {D:>8.4f} {config.grid.dx_cm:>8.5f} {config.grid.n_nodes:>6} "
            f"{fit.theta_cm_per_ms:>10.6f} {fit.theta_cm_per_ms / np.sqrt(D):>12.5f} "
            f"{fit.r_squared:>11.8f} {w_front:>11.6f}"
        )

    # ---- Family 2: grid scaled with sqrt(D) --------------------------------
    print("\n  [2] scaled grid, dx proportional to sqrt(D) (constant resolution)")
    print(header)

    scaled_velocities = []
    for D in D_VALUES:
        result, config = run_for_D(base, D, scale_grid=True, scale_stimulus=True)
        fit = measure.measure_velocity(result)
        w_front = measure.recovery_at_front(result, FRONT_PROBE_X_CM, V2_rest)
        scaled_velocities.append(fit.theta_cm_per_ms)
        print(
            f"      {D:>8.4f} {config.grid.dx_cm:>8.5f} {config.grid.n_nodes:>6} "
            f"{fit.theta_cm_per_ms:>10.6f} {fit.theta_cm_per_ms / np.sqrt(D):>12.5f} "
            f"{fit.r_squared:>11.8f} {w_front:>11.6f}"
        )

    D_array = np.array(D_VALUES)
    fixed_velocities = np.array(fixed_velocities)
    scaled_velocities = np.array(scaled_velocities)
    fixed_w_front = np.array(fixed_w_front)

    # ---- Test 1: log-log exponent -----------------------------------------
    fixed_exponent, fixed_C, fixed_r2 = measure.log_log_slope(D_array, fixed_velocities)
    scaled_exponent, scaled_C, scaled_r2 = measure.log_log_slope(
        D_array, scaled_velocities
    )
    print(
        f"\n  log-log fit, fixed grid : theta = {fixed_C:.5f} D^{fixed_exponent:.5f}"
        f"  (R^2 = {fixed_r2:.8f})"
    )
    print(
        f"  log-log fit, scaled grid: theta = {scaled_C:.5f} D^{scaled_exponent:.5f}"
        f"  (R^2 = {scaled_r2:.8f})"
    )

    # ---- Test 2: flatness of theta / sqrt(D) ------------------------------
    fixed_ratio = fixed_velocities / np.sqrt(D_array)
    scaled_ratio = scaled_velocities / np.sqrt(D_array)
    fixed_spread = float((fixed_ratio.max() - fixed_ratio.min()) / fixed_ratio.mean())
    scaled_spread = float(
        (scaled_ratio.max() - scaled_ratio.min()) / scaled_ratio.mean()
    )
    print(
        f"  theta/sqrt(D) spread: fixed grid {100.0 * fixed_spread:.2f} %, "
        f"scaled grid {100.0 * scaled_spread:.2f} %"
    )
    print(f"  theta/sqrt(D) mean, scaled grid = {scaled_ratio.mean():.5f}")

    # ---- Explaining the prefactor -----------------------------------------
    # The analytic prefactor freezes w at rest. Substituting the recovery level
    # actually measured at the front recovers most of the gap.
    mean_w_front = float(np.nanmean(fixed_w_front))
    corrected_prefactor = fhn.front_speed_prefactor_at(mean_w_front)
    measured_prefactor = float(scaled_ratio.mean())

    gap_uncorrected = (measured_prefactor - analytic_prefactor) / analytic_prefactor
    gap_corrected = (measured_prefactor - corrected_prefactor) / corrected_prefactor
    print(f"\n  mean w at the front        = {mean_w_front:.6f} (rest: {w_rest:.6f})")
    print(f"  prefactor, w frozen at rest  = {analytic_prefactor:.6f}")
    print(f"  prefactor, w at the front    = {corrected_prefactor:.6f}")
    print(f"  prefactor, measured          = {measured_prefactor:.6f}")
    print(f"  discrepancy vs resting-w     = {100.0 * gap_uncorrected:+.2f} %")
    print(f"  discrepancy vs front-w       = {100.0 * gap_corrected:+.2f} %")

    # ---- The liminal-length effect ----------------------------------------
    print("\n  [3] fixed 0.1 cm stimulus: where does it stop exciting the strand?")
    fixed_width_peaks = []
    for D in D_VALUES:
        result, _ = run_for_D(base, D, scale_grid=False, scale_stimulus=False)
        peak = float(np.max(result.V_peak))
        fixed_width_peaks.append(peak)
        launched = peak >= base.measurement.activation_level
        print(
            f"      D = {D:>8.4f}  peak V = {peak:>8.4f}  "
            f"{'propagates' if launched else 'FAILS TO LAUNCH'}"
        )
    fixed_width_peaks = np.array(fixed_width_peaks)
    scaled_width_peaks = np.full_like(D_array, np.nan)
    for index, D in enumerate(D_VALUES):
        result, _ = run_for_D(base, D, scale_grid=False, scale_stimulus=True)
        scaled_width_peaks[index] = float(np.max(result.V_peak))

    # ---- Figure ------------------------------------------------------------
    fig, axes = plotting.new_figure(
        figsize=(10.0, 7.5), nrows=2, ncols=2, constrained_layout=True
    )
    ax_loglog, ax_flat, ax_liminal, ax_prefactor = axes.flatten()

    # (a) log-log.
    ax_loglog.loglog(
        D_array,
        fixed_velocities,
        marker="o",
        color=plotting.COLOUR_MEASURED,
        label=f"fixed grid (slope {fixed_exponent:.4f})",
    )
    ax_loglog.loglog(
        D_array,
        scaled_velocities,
        marker="s",
        color=plotting.PALETTE["green"],
        label=f"scaled grid (slope {scaled_exponent:.4f})",
    )
    ax_loglog.loglog(
        D_array,
        analytic_prefactor * np.sqrt(D_array),
        linestyle="--",
        color=plotting.COLOUR_ANALYTIC,
        label=r"analytic $0.9630\sqrt{D}$ ($w$ at rest)",
    )
    plotting.label_axes(
        ax_loglog,
        "diffusion coefficient $D$ (cm$^2$/ms)",
        "conduction velocity $\\theta$ (cm/ms)",
        "(a) Log-log: the exponent is 1/2",
    )
    ax_loglog.legend(loc="upper left", fontsize=7.5)
    plotting.set_log_ticks(ax_loglog, D_array, axis="x")
    ax_loglog.tick_params(axis="x", labelrotation=45, labelsize=7)

    # (b) Flatness -- the sharper test.
    ax_flat.plot(
        D_array,
        fixed_ratio,
        marker="o",
        color=plotting.COLOUR_MEASURED,
        label=f"fixed grid (spread {100.0 * fixed_spread:.2f} %)",
    )
    ax_flat.plot(
        D_array,
        scaled_ratio,
        marker="s",
        color=plotting.PALETTE["green"],
        label=f"scaled grid (spread {100.0 * scaled_spread:.2f} %)",
    )
    ax_flat.axhline(
        analytic_prefactor,
        linestyle="--",
        color=plotting.COLOUR_ANALYTIC,
        label=f"analytic {analytic_prefactor:.4f}",
    )
    ax_flat.axhline(
        corrected_prefactor,
        linestyle="-.",
        color=plotting.PALETTE["purple"],
        label=f"corrected for $w$ at front {corrected_prefactor:.4f}",
    )
    ax_flat.set_xscale("log")
    plotting.label_axes(
        ax_flat,
        "diffusion coefficient $D$ (cm$^2$/ms)",
        # theta is cm/ms and sqrt(D) is cm/sqrt(ms), so the ratio is 1/sqrt(ms).
        r"$\theta/\sqrt{D}$ (ms$^{-1/2}$)",
        r"(b) $\theta/\sqrt{D}$ flatness: the sharper test",
    )
    ax_flat.legend(loc="center right", fontsize=7)
    plotting.set_log_ticks(ax_flat, D_array, axis="x")
    ax_flat.tick_params(axis="x", labelrotation=45, labelsize=7)
    # Headroom so the legend does not sit on the data.
    ax_flat.set_ylim(0.78, 1.02)

    # (c) The liminal-length effect.
    ax_liminal.plot(
        D_array,
        fixed_width_peaks,
        marker="o",
        color=plotting.COLOUR_BLOCKED,
        label="fixed 0.1 cm stimulus",
    )
    ax_liminal.plot(
        D_array,
        scaled_width_peaks,
        marker="s",
        color=plotting.COLOUR_PROPAGATED,
        label=r"stimulus width $\propto \sqrt{D}$",
    )
    ax_liminal.axhline(
        base.measurement.activation_level,
        color=plotting.COLOUR_REFERENCE,
        linestyle=":",
        label="activation level $V=0$",
    )
    ax_liminal.set_xscale("log")
    plotting.label_axes(
        ax_liminal,
        "diffusion coefficient $D$ (cm$^2$/ms)",
        "peak potential reached $V_{\\max}$ (dimensionless)",
        "(c) Liminal length: a fixed stimulus fails at large $D$",
    )
    ax_liminal.legend(loc="center left", fontsize=7.5)
    plotting.set_log_ticks(ax_liminal, D_array, axis="x")
    ax_liminal.tick_params(axis="x", labelrotation=45, labelsize=7)

    # (d) Where the prefactor comes from.
    w_grid = np.linspace(w_rest, mean_w_front + 0.10, 200)
    prefactor_curve = np.array([fhn.front_speed_prefactor_at(w) for w in w_grid])
    ax_prefactor.plot(
        w_grid,
        prefactor_curve,
        color=plotting.COLOUR_REFERENCE,
        label=r"$\theta/\sqrt{D}$ as a function of frozen $w$",
    )
    ax_prefactor.plot(
        [w_rest],
        [analytic_prefactor],
        marker="o",
        markersize=8,
        linestyle="none",
        color=plotting.COLOUR_ANALYTIC,
        label=f"$w^*$ (analytic): {analytic_prefactor:.4f}",
    )
    ax_prefactor.plot(
        [mean_w_front],
        [corrected_prefactor],
        marker="D",
        markersize=8,
        linestyle="none",
        color=plotting.PALETTE["purple"],
        label=f"$w$ at front: {corrected_prefactor:.4f}",
    )
    ax_prefactor.axhline(
        measured_prefactor,
        linestyle="--",
        color=plotting.COLOUR_MEASURED,
        label=f"measured: {measured_prefactor:.4f}",
    )
    plotting.label_axes(
        ax_prefactor,
        "frozen recovery variable $w$ (dimensionless)",
        r"prefactor $\theta/\sqrt{D}$ (ms$^{-1/2}$)",
        "(d) Why the measured prefactor is below the analytic one",
    )
    ax_prefactor.legend(loc="lower left", fontsize=7)

    caption = (
        f"Conduction velocity scales as the square root of the diffusion "
        f"coefficient over a twenty-fold range: the log-log slope is "
        f"{scaled_exponent:.4f} against a predicted 0.5, and the sharper "
        f"theta/sqrt(D) test is flat to {100.0 * scaled_spread:.2f} % once the "
        f"grid is scaled to hold front resolution constant (versus "
        f"{100.0 * fixed_spread:.2f} % on a fixed grid). The measured prefactor "
        f"{measured_prefactor:.4f} sits {100.0 * abs(gap_uncorrected):.0f} % below "
        f"the analytic {analytic_prefactor:.4f} because that derivation freezes w "
        f"at rest; substituting the recovery level actually measured at the "
        f"front, w = {mean_w_front:.4f}, closes most of the gap to "
        f"{100.0 * abs(gap_corrected):.0f} %."
    )

    measurements = {
        "D_values": D_array.tolist(),
        "theta_fixed_grid": fixed_velocities.tolist(),
        "theta_scaled_grid": scaled_velocities.tolist(),
        "log_log_exponent_fixed": fixed_exponent,
        "log_log_exponent_scaled": scaled_exponent,
        "log_log_r_squared_scaled": scaled_r2,
        "theta_over_sqrtD_spread_fixed": fixed_spread,
        "theta_over_sqrtD_spread_scaled": scaled_spread,
        "measured_prefactor": measured_prefactor,
        "analytic_prefactor_w_rest": analytic_prefactor,
        "corrected_prefactor_w_front": corrected_prefactor,
        "mean_w_at_front": mean_w_front,
        "w_rest": w_rest,
        "relative_gap_vs_resting_w": gap_uncorrected,
        "relative_gap_vs_front_w": gap_corrected,
        "fixed_width_peak_V": fixed_width_peaks.tolist(),
        "scaled_width_peak_V": scaled_width_peaks.tolist(),
    }

    plotting.save_figure(
        fig,
        "fig_ex05_cv_vs_D",
        caption,
        config_dict=base.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_arrays(
        "ex05_cv_vs_D",
        {
            "D_values": D_array,
            "theta_fixed_grid": fixed_velocities,
            "theta_scaled_grid": scaled_velocities,
            "theta_over_sqrtD_fixed": fixed_ratio,
            "theta_over_sqrtD_scaled": scaled_ratio,
            "w_at_front": fixed_w_front,
            "fixed_width_peak_V": fixed_width_peaks,
            "scaled_width_peak_V": scaled_width_peaks,
        },
        config_dict=base.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_table(
        "ex05_cv_vs_D",
        header=[
            "D_cm2_per_ms",
            "theta_fixed_grid_cm_per_ms",
            "theta_scaled_grid_cm_per_ms",
            "theta_over_sqrtD_fixed",
            "theta_over_sqrtD_scaled",
            "w_at_front",
            "peak_V_fixed_width_stimulus",
        ],
        rows=[
            (
                f"{D:.6f}",
                f"{tf:.7f}",
                f"{ts:.7f}",
                f"{rf:.6f}",
                f"{rs:.6f}",
                f"{wf:.6f}",
                f"{pk:.6f}",
            )
            for D, tf, ts, rf, rs, wf, pk in zip(
                D_array,
                fixed_velocities,
                scaled_velocities,
                fixed_ratio,
                scaled_ratio,
                fixed_w_front,
                fixed_width_peaks,
                strict=True,
            )
        ],
    )

    print("\n  figure -> figures/fig_ex05_cv_vs_D.png / .pdf")
    return measurements


if __name__ == "__main__":
    main()
