"""Experiment 1: the single cell -- rest state, excitability, and threshold.

Question
--------
Before any wave can propagate, the tissue has to be *excitable*: it must sit
quietly at rest until pushed hard enough, then produce a large stereotyped
excursion and return. Is that what the FitzHugh-Nagumo parameters in the brief
actually give, and where exactly is the threshold?

What this produces
------------------
``fig_ex01_single_cell`` -- four panels:

(a) the action potential and recovery variable following a supra-threshold
    stimulus;
(b) sub- and supra-threshold responses on the same axes, showing the
    all-or-none behaviour that defines excitability;
(c) the phase plane with both nullclines and the trajectory, which is where the
    cubic nullcline's three branches become visible;
(d) peak potential against stimulus amplitude, with the bisected threshold
    marked.

Run standalone with::

    python experiments/ex01_single_cell.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow "python experiments/ex01_single_cell.py" to work without the package
# being installed, by putting src/ on the path. With the editable install the
# README prescribes this is a no-op, but it makes each script genuinely
# standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibroblock import config as cfg  # noqa: E402
from fibroblock import fhn, plotting, simulate, stimulus, utils  # noqa: E402

# Stimulus amplitudes used for the illustrative sub/supra-threshold pair, as
# multiples of the bisected threshold. Chosen to sit clearly either side of it
# without being so far away that the comparison stops being informative.
SUBTHRESHOLD_FACTOR: float = 0.95
SUPRATHRESHOLD_FACTOR: float = 1.05

# Duration long enough to contain the whole excursion and the return to rest.
SINGLE_CELL_DURATION_MS: float = 200.0

# Shorter window used inside the bisection loop: a cell that is going to fire
# does so within a few milliseconds, so integrating for the full 200 ms on
# every one of ~40 bisection steps would be wasted work.
THRESHOLD_PROBE_DURATION_MS: float = 60.0

# Bracket for the stimulus-amplitude bisection, in the units of f(V, w).
THRESHOLD_SEARCH_LOWER: float = 0.0
THRESHOLD_SEARCH_UPPER: float = 3.0
THRESHOLD_TOLERANCE: float = 1.0e-6

# Amplitudes sampled for panel (d).
N_AMPLITUDE_SAMPLES: int = 60


def fires(config: cfg.RunConfig, amplitude: float) -> bool:
    """Whether a single cell reaches the activation level at this amplitude.

    Parameters
    ----------
    config : RunConfig
        Base configuration; only its kinetics, solver and stimulus are used.
    amplitude : float
        Stimulus amplitude to test, in the units of ``f(V, w)``.

    Returns
    -------
    bool
        True if the peak potential crosses the activation level.

    Notes
    -----
    "Fired" is defined by the same activation level used everywhere else
    (``V = 0``), which sits between the threshold root ``V2 = -0.786`` and the
    excited plateau ``V3 = +1.986``. A merely passive charging response cannot
    reach it, so the test genuinely separates regenerative from passive.
    """
    _, V, _ = simulate.run_single_cell(config, stimulus_amplitude=amplitude)
    return bool(np.max(V) >= config.measurement.activation_level)


def main() -> dict[str, Any]:
    """Run experiment 1 and write its figure and results.

    Returns
    -------
    dict
        Summary of the measured quantities, for the pipeline log.
    """
    print("=" * 70)
    print("Experiment 1: single cell -- rest state, excitability, threshold")
    print("=" * 70)

    base = cfg.default_config().replace(label="ex01_single_cell")
    utils.set_seed(base.seed)

    params = base.fhn

    # ---- Analytic characterisation ----------------------------------------
    summary = fhn.excitability(params)
    V1, V2, V3 = fhn.bistable_roots(params)
    prefactor = fhn.analytic_cv_prefactor(params)

    print(f"  rest state           V* = {summary.V_rest:.9f}")
    print(f"                       w* = {summary.w_rest:.9f}")
    print(f"  Jacobian trace       tr(J)  = {summary.trace:+.6f}")
    print(f"  Jacobian determinant det(J) = {summary.determinant:+.6f}")
    print(f"  discriminant                = {summary.discriminant:+.6f}")
    print(f"  eigenvalues          {summary.eigenvalues[0]:.6f}")
    print(f"                       {summary.eigenvalues[1]:.6f}")
    print(f"  classification       {summary.classification}")
    print(f"  excitable            {summary.is_excitable}")
    print(f"  bistable roots       V1 = {V1:.6f} (rest)")
    print(f"                       V2 = {V2:.6f} (threshold)")
    print(f"                       V3 = {V3:.6f} (excited)")
    print(f"  CV prefactor         theta/sqrt(D) = {prefactor:.6f}")

    # ---- Stimulus threshold by bisection -----------------------------------
    probe_config = base.replace(
        solver=cfg.SolverParams(
            dt_ms=base.solver.dt_ms,
            t_end_ms=THRESHOLD_PROBE_DURATION_MS,
            method=base.solver.method,
            f_v_bound=base.solver.f_v_bound,
            record_every=base.solver.record_every,
        )
    )

    threshold, iterations = stimulus.bisect_threshold(
        succeeds=lambda amplitude: fires(probe_config, amplitude),
        lower=THRESHOLD_SEARCH_LOWER,
        upper=THRESHOLD_SEARCH_UPPER,
        tolerance=THRESHOLD_TOLERANCE,
    )
    print(
        f"  stimulus threshold   A* = {threshold:.6f} "
        f"(bisection, {iterations} iterations)"
    )
    print(
        f"  configured amplitude A  = {base.stimulus.amplitude:.6f} "
        f"({base.stimulus.amplitude / threshold:.2f}x threshold)"
    )

    # ---- Long runs for the figure ------------------------------------------
    display_config = base.replace(
        solver=cfg.SolverParams(
            dt_ms=base.solver.dt_ms,
            t_end_ms=SINGLE_CELL_DURATION_MS,
            method=base.solver.method,
            f_v_bound=base.solver.f_v_bound,
            record_every=base.solver.record_every,
        )
    )

    t_supra, V_supra, w_supra = simulate.run_single_cell(
        display_config, stimulus_amplitude=SUPRATHRESHOLD_FACTOR * threshold
    )
    t_sub, V_sub, w_sub = simulate.run_single_cell(
        display_config, stimulus_amplitude=SUBTHRESHOLD_FACTOR * threshold
    )

    peak_supra = float(np.max(V_supra))
    peak_sub = float(np.max(V_sub))
    print(f"  peak V, {SUPRATHRESHOLD_FACTOR:.2f}x threshold: {peak_supra:+.4f}")
    print(f"  peak V, {SUBTHRESHOLD_FACTOR:.2f}x threshold: {peak_sub:+.4f}")
    print(f"  all-or-none ratio    {peak_supra - peak_sub:.4f} V units apart")

    # ---- Amplitude sweep for panel (d) -------------------------------------
    amplitudes = np.linspace(
        0.5 * threshold, 1.5 * threshold, N_AMPLITUDE_SAMPLES
    )
    peaks = np.empty_like(amplitudes)
    for index, amplitude in enumerate(amplitudes):
        _, V_probe, _ = simulate.run_single_cell(
            probe_config, stimulus_amplitude=float(amplitude)
        )
        peaks[index] = np.max(V_probe)

    # ---- Figure ------------------------------------------------------------
    fig, axes = plotting.new_figure(
        figsize=(10.0, 7.5), nrows=2, ncols=2, constrained_layout=True
    )
    ax_ap, ax_allornone, ax_phase, ax_threshold = axes.flatten()

    # (a) Action potential and recovery variable.
    ax_ap.plot(t_supra, V_supra, color=plotting.COLOUR_MEASURED, label="$V$ (fast)")
    ax_ap.plot(
        t_supra,
        w_supra,
        color=plotting.COLOUR_ANALYTIC,
        linestyle="--",
        label="$w$ (recovery)",
    )
    ax_ap.axhline(
        summary.V_rest,
        color=plotting.COLOUR_REFERENCE,
        linewidth=0.9,
        linestyle=":",
        label=f"rest $V^*={summary.V_rest:.3f}$",
    )
    plotting.label_axes(
        ax_ap,
        "time $t$ (ms)",
        "state variable (dimensionless)",
        "(a) Action potential and recovery",
    )
    ax_ap.legend(loc="upper right")
    plotting.annotate_takeaway(
        ax_ap,
        f"$\\varepsilon={params.eps}$ makes $w$ about\n"
        f"{1.0 / params.eps:.0f}$\\times$ slower than $V$",
        loc="lower right",
    )

    # (b) All-or-none response.
    ax_allornone.plot(
        t_sub,
        V_sub,
        color=plotting.COLOUR_ANALYTIC,
        linestyle="--",
        label=f"${SUBTHRESHOLD_FACTOR:.2f}A^*$ (sub-threshold)",
    )
    ax_allornone.plot(
        t_supra,
        V_supra,
        color=plotting.COLOUR_MEASURED,
        label=f"${SUPRATHRESHOLD_FACTOR:.2f}A^*$ (supra-threshold)",
    )
    ax_allornone.axhline(
        base.measurement.activation_level,
        color=plotting.COLOUR_REFERENCE,
        linewidth=0.9,
        linestyle=":",
        label="activation level $V=0$",
    )
    ax_allornone.set_xlim(0.0, 60.0)
    plotting.label_axes(
        ax_allornone,
        "time $t$ (ms)",
        "membrane potential $V$ (dimensionless)",
        "(b) All-or-none: a 10 % change in amplitude",
    )
    ax_allornone.legend(loc="upper right")

    # (c) Phase plane with nullclines.
    V_grid = np.linspace(-2.5, 2.5, 400)
    # V-nullcline: f = 0 with no stimulus gives w = V - V^3/3.
    v_nullcline = V_grid - fhn.CUBIC_COEFFICIENT * V_grid**3
    # w-nullcline: g = 0 gives w = (V + a) / b.
    w_nullcline = (V_grid + params.a) / params.b

    ax_phase.plot(
        V_grid,
        v_nullcline,
        color=plotting.COLOUR_ANALYTIC,
        linestyle="--",
        label="$V$-nullcline  $w=V-V^3/3$",
    )
    ax_phase.plot(
        V_grid,
        w_nullcline,
        color=plotting.COLOUR_REFERENCE,
        linestyle="-.",
        label="$w$-nullcline  $w=(V+a)/b$",
    )
    ax_phase.plot(
        V_supra,
        w_supra,
        color=plotting.COLOUR_MEASURED,
        linewidth=1.4,
        label="trajectory",
    )
    ax_phase.plot(
        [summary.V_rest],
        [summary.w_rest],
        marker="o",
        color=plotting.COLOUR_REFERENCE,
        markersize=7,
        linestyle="none",
        label="rest state",
    )
    for root, name in ((V1, "$V_1$"), (V2, "$V_2$"), (V3, "$V_3$")):
        ax_phase.axvline(
            root, color=plotting.PALETTE["green"], linewidth=0.8, alpha=0.6
        )
        ax_phase.text(
            root,
            2.6,
            name,
            ha="center",
            fontsize=8,
            color=plotting.PALETTE["green"],
        )
    ax_phase.set_ylim(-1.5, 2.8)
    plotting.label_axes(
        ax_phase,
        "membrane potential $V$ (dimensionless)",
        "recovery variable $w$ (dimensionless)",
        "(c) Phase plane",
    )
    ax_phase.legend(loc="lower right", fontsize=8)

    # (d) Threshold.
    ax_threshold.plot(
        amplitudes,
        peaks,
        marker="o",
        markersize=3,
        color=plotting.COLOUR_MEASURED,
        label="peak $V$ reached",
    )
    ax_threshold.axvline(
        threshold,
        color=plotting.COLOUR_ANALYTIC,
        linestyle="--",
        label=f"threshold $A^*={threshold:.4f}$",
    )
    ax_threshold.axhline(
        base.measurement.activation_level,
        color=plotting.COLOUR_REFERENCE,
        linewidth=0.9,
        linestyle=":",
        label="activation level $V=0$",
    )
    plotting.label_axes(
        ax_threshold,
        "stimulus amplitude $A$ (units of $f$)",
        "peak membrane potential $V$ (dimensionless)",
        "(d) Excitation threshold by bisection",
    )
    ax_threshold.legend(loc="upper left", fontsize=8)

    caption = (
        f"The FitzHugh-Nagumo medium is excitable, not oscillatory: the rest "
        f"state at V* = {summary.V_rest:.4f} is a stable spiral "
        f"(tr J = {summary.trace:.4f} < 0, det J = {summary.determinant:.4f} > 0), "
        f"so it stays put until a stimulus exceeds A* = {threshold:.4f}, after "
        f"which a 10 % change in amplitude switches the response between a "
        f"passive decay and a full {peak_supra:.2f}-amplitude action potential."
    )

    measurements = {
        "V_rest": summary.V_rest,
        "w_rest": summary.w_rest,
        "jacobian_trace": summary.trace,
        "jacobian_determinant": summary.determinant,
        "jacobian_discriminant": summary.discriminant,
        "eigenvalue_real": summary.eigenvalues[0].real,
        "eigenvalue_imag": abs(summary.eigenvalues[0].imag),
        "classification": summary.classification,
        "is_excitable": summary.is_excitable,
        "bistable_root_V1": V1,
        "bistable_root_V2": V2,
        "bistable_root_V3": V3,
        "cv_prefactor": prefactor,
        "stimulus_threshold": threshold,
        "bisection_iterations": iterations,
        "peak_V_suprathreshold": peak_supra,
        "peak_V_subthreshold": peak_sub,
    }

    plotting.save_figure(
        fig,
        "fig_ex01_single_cell",
        caption,
        config_dict=base.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_arrays(
        "ex01_single_cell",
        {
            "t_suprathreshold_ms": t_supra,
            "V_suprathreshold": V_supra,
            "w_suprathreshold": w_supra,
            "t_subthreshold_ms": t_sub,
            "V_subthreshold": V_sub,
            "w_subthreshold": w_sub,
            "sweep_amplitudes": amplitudes,
            "sweep_peak_V": peaks,
        },
        config_dict=base.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_table(
        "ex01_single_cell_summary",
        header=["quantity", "value", "units"],
        rows=[
            ("V_rest", f"{summary.V_rest:.9f}", "dimensionless"),
            ("w_rest", f"{summary.w_rest:.9f}", "dimensionless"),
            ("jacobian_trace", f"{summary.trace:.9f}", "1/ms"),
            ("jacobian_determinant", f"{summary.determinant:.9f}", "1/ms^2"),
            ("jacobian_discriminant", f"{summary.discriminant:.9f}", "1/ms^2"),
            ("eigenvalue_real_part", f"{summary.eigenvalues[0].real:.9f}", "1/ms"),
            (
                "eigenvalue_imag_part",
                f"{abs(summary.eigenvalues[0].imag):.9f}",
                "1/ms",
            ),
            ("classification", summary.classification, "-"),
            ("bistable_root_V1_rest", f"{V1:.9f}", "dimensionless"),
            ("bistable_root_V2_threshold", f"{V2:.9f}", "dimensionless"),
            ("bistable_root_V3_excited", f"{V3:.9f}", "dimensionless"),
            ("cv_prefactor", f"{prefactor:.9f}", "1/sqrt(ms)"),
            ("stimulus_threshold", f"{threshold:.9f}", "units of f"),
        ],
    )

    print("  figure -> figures/fig_ex01_single_cell.png / .pdf")
    return measurements


if __name__ == "__main__":
    main()
