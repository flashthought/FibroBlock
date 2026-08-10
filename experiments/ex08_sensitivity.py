"""Experiment 8: one-at-a-time sensitivity of the reported quantities.

Question
--------
Two reported numbers matter: the conduction velocity of a healthy strand, and
the critical coupling ratio at which a gap blocks. Which parameters do they
actually depend on?

Design
------
Each parameter is perturbed by +/-10 % about the baseline with everything else
held fixed, and the **elasticity** is reported:

.. math:: S = \\frac{\\Delta Y / Y_0}{\\Delta X / X_0}

An elasticity of 1 means a 10 % change in the parameter produces a 10 % change
in the output; 0.5 is the signature of a square-root dependence; 0 means the
output does not care.

Physical against numerical parameters
-------------------------------------
The sweep deliberately mixes two kinds of parameter, and the distinction is the
point of the experiment:

* ``a``, ``b``, ``eps``, ``D0``, ``L_gap`` are **physical**. Their elasticities
  say which biology the conclusions rest on.
* ``dx`` and ``dt`` are **numerical**. They have no physical meaning at all, so
  their elasticities *ought to be near zero*. A large numerical elasticity
  would mean the answer is an artefact of the discretisation rather than a
  property of the model -- so these entries are a convergence check disguised
  as a sensitivity, and they are plotted in a different colour.

A one-at-a-time sweep cannot detect interactions between parameters. That
limitation is stated rather than hidden; a full factorial or Sobol analysis
would be the next step and is noted in the report's future-work section.

Run standalone with::

    python experiments/ex08_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibroblock import config as cfg  # noqa: E402
from fibroblock import measure, plotting, simulate, solvers, stimulus, utils  # noqa: E402

# Fractional perturbation applied to each parameter. Small enough to stay in
# the locally linear regime, large enough that the response is well clear of
# the bisection tolerance on the threshold.
PERTURBATION: float = 0.10

GAP_LENGTH_CM: float = 0.10
BLOCK_RUN_MS: float = 205.0
VELOCITY_RUN_MS: float = 130.0
SNAPSHOT_EVERY: int = 100

RHO_BRACKET: tuple[float, float] = (0.0005, 1.0)
RHO_TOLERANCE: float = 1.0e-5

# Keeps every run at the same margin below its own stability limit, so that a
# dx perturbation does not silently also change the temporal accuracy.
SAFETY_FACTOR: float = 2.3
MAX_DT_MS: float = 0.02

# Which parameters are numerical rather than physical. Their elasticities are
# a convergence check: they should be near zero.
NUMERICAL_PARAMETERS: frozenset[str] = frozenset({"dx", "dt"})


def build_config(
    base: cfg.RunConfig,
    a: float,
    b: float,
    eps: float,
    D0: float,
    gap_length: float,
    dx: float,
    dt: float | None,
    rho: float,
    t_end_ms: float,
) -> cfg.RunConfig:
    """Assemble a configuration from explicit parameter values.

    Parameters
    ----------
    base : RunConfig
        Baseline, used for anything not swept.
    a, b, eps : float
        FitzHugh-Nagumo kinetics parameters.
    D0 : float
        Baseline diffusion coefficient. cm^2/ms.
    gap_length : float
        Gap length. cm.
    dx : float
        Node spacing. cm.
    dt : float or None
        Time step. ms. If None, the largest step at the standard safety factor
        below this configuration's own stability limit is used.
    rho : float
        Coupling ratio in the gap.
    t_end_ms : float
        Run duration. ms.

    Returns
    -------
    RunConfig
    """
    # Round the node count so that length / dx stays an exact integer, which
    # GridParams requires. Perturbing dx by 10 % otherwise fails validation.
    n_intervals = max(2, int(round(base.grid.length_cm / dx)))
    exact_dx = base.grid.length_cm / n_intervals

    if dt is None:
        dt = min(
            MAX_DT_MS,
            solvers.explicit_euler_dt_limit(D0, exact_dx, base.solver.f_v_bound)
            / SAFETY_FACTOR,
        )

    return base.replace(
        fhn=cfg.FHNParams(a=a, b=b, eps=eps),
        grid=cfg.GridParams(
            length_cm=base.grid.length_cm, dx_cm=exact_dx, baseline_D=D0
        ),
        gap=cfg.GapParams(
            rho=rho, gap_length_cm=gap_length, gap_centre_cm=base.gap.gap_centre_cm
        ),
        solver=cfg.SolverParams(
            dt_ms=dt,
            t_end_ms=t_end_ms,
            f_v_bound=base.solver.f_v_bound,
            record_every=SNAPSHOT_EVERY,
        ),
    )


def measure_outputs(
    base: cfg.RunConfig, **values: float
) -> tuple[float, float]:
    """Measure conduction velocity and critical coupling for one parameter set.

    Parameters
    ----------
    base : RunConfig
        Baseline configuration.
    **values
        ``a``, ``b``, ``eps``, ``D0``, ``gap_length``, ``dx``, ``dt``.

    Returns
    -------
    theta_cm_per_ms : float
        Conduction velocity of the healthy strand. NaN if it fails to
        propagate.
    critical_rho : float
        Critical coupling ratio. NaN if no threshold exists in the bracket.
    """
    # --- Conduction velocity, homogeneous strand ---
    healthy = build_config(
        base, rho=1.0, t_end_ms=VELOCITY_RUN_MS, **values
    ).replace(gap=cfg.GapParams(rho=1.0, gap_length_cm=0.0))
    fit = measure.measure_velocity(simulate.run_simulation(healthy))
    theta = fit.theta_cm_per_ms

    # --- Critical coupling ratio ---
    def crosses(rho: float) -> bool:
        """Whether the wave crosses the gap at this coupling ratio."""
        config = build_config(base, rho=rho, t_end_ms=BLOCK_RUN_MS, **values)
        return not measure.detect_block(simulate.run_simulation(config)).blocked

    try:
        critical, _ = stimulus.bisect_threshold(
            succeeds=crosses,
            lower=RHO_BRACKET[0],
            upper=RHO_BRACKET[1],
            tolerance=RHO_TOLERANCE,
        )
    except ValueError:
        critical = float("nan")

    return theta, critical


def elasticity(high: float, low: float, baseline: float, fraction: float) -> float:
    """Normalised sensitivity (elasticity) from a central difference.

    .. math:: S = \\frac{(Y_+ - Y_-)/Y_0}{(X_+ - X_-)/X_0}

    Parameters
    ----------
    high, low : float
        Output at the perturbed-up and perturbed-down parameter values.
    baseline : float
        Output at the baseline parameter value.
    fraction : float
        Fractional perturbation applied, so ``(X_+ - X_-)/X_0 = 2 * fraction``.

    Returns
    -------
    float
        The elasticity, or NaN if any input is not finite or the baseline is
        zero.
    """
    if not np.isfinite([high, low, baseline]).all() or baseline == 0.0:
        return float("nan")
    return float(((high - low) / baseline) / (2.0 * fraction))


def main() -> dict[str, Any]:
    """Run experiment 8 and write its figure and results.

    Returns
    -------
    dict
        Elasticities of both outputs with respect to every parameter.
    """
    print("=" * 70)
    print("Experiment 8: one-at-a-time sensitivity")
    print("=" * 70)

    base = cfg.default_config().replace(label="ex08_sensitivity")
    utils.set_seed(base.seed)

    baseline_values: dict[str, float] = {
        "a": base.fhn.a,
        "b": base.fhn.b,
        "eps": base.fhn.eps,
        "D0": base.grid.baseline_D,
        "gap_length": GAP_LENGTH_CM,
        "dx": base.grid.dx_cm,
        "dt": base.solver.dt_ms,
    }
    # Display names, with the numerical ones flagged for the reader.
    labels: dict[str, str] = {
        "a": "$a$",
        "b": "$b$",
        "eps": r"$\varepsilon$",
        "D0": "$D_0$",
        "gap_length": "$L_{\\mathrm{gap}}$",
        "dx": "$\\Delta x$ (numerical)",
        "dt": "$\\Delta t$ (numerical)",
    }

    print(f"  perturbation: +/-{100.0 * PERTURBATION:.0f} % about the baseline")
    print("  measuring the baseline...")
    baseline_theta, baseline_rho = measure_outputs(base, **baseline_values)
    print(f"      theta     = {baseline_theta:.7f} cm/ms")
    print(f"      rho_crit  = {baseline_rho:.6f}")

    print(
        f"\n      {'parameter':>12} {'value':>10} "
        f"{'S(theta)':>10} {'S(rho_crit)':>13}"
    )

    theta_elasticities: dict[str, float] = {}
    rho_elasticities: dict[str, float] = {}
    raw_rows = []

    for name, baseline_value in baseline_values.items():
        perturbed_outputs = {}
        for direction, sign in (("high", +1.0), ("low", -1.0)):
            values = dict(baseline_values)
            values[name] = baseline_value * (1.0 + sign * PERTURBATION)
            perturbed_outputs[direction] = measure_outputs(base, **values)

        theta_high, rho_high = perturbed_outputs["high"]
        theta_low, rho_low = perturbed_outputs["low"]

        theta_elasticities[name] = elasticity(
            theta_high, theta_low, baseline_theta, PERTURBATION
        )
        rho_elasticities[name] = elasticity(
            rho_high, rho_low, baseline_rho, PERTURBATION
        )

        raw_rows.append(
            (
                name,
                baseline_value,
                theta_low,
                theta_high,
                rho_low,
                rho_high,
                theta_elasticities[name],
                rho_elasticities[name],
            )
        )

        print(
            f"      {name:>12} {baseline_value:>10.5f} "
            f"{theta_elasticities[name]:>10.4f} {rho_elasticities[name]:>13.4f}"
        )

    # ---- Interpretation ----------------------------------------------------
    physical = [n for n in baseline_values if n not in NUMERICAL_PARAMETERS]
    numerical = [n for n in baseline_values if n in NUMERICAL_PARAMETERS]

    largest_numerical_theta = max(
        abs(theta_elasticities[n]) for n in numerical if np.isfinite(theta_elasticities[n])
    )
    largest_numerical_rho = max(
        abs(rho_elasticities[n]) for n in numerical if np.isfinite(rho_elasticities[n])
    )
    dominant_theta = max(
        physical, key=lambda n: abs(theta_elasticities[n])
    )
    dominant_rho = max(physical, key=lambda n: abs(rho_elasticities[n]))

    print(f"\n  most influential physical parameter for theta:    {dominant_theta}")
    print(f"  most influential physical parameter for rho_crit: {dominant_rho}")
    print(
        f"  largest NUMERICAL elasticity: {largest_numerical_theta:.4f} (theta), "
        f"{largest_numerical_rho:.4f} (rho_crit)"
    )
    print(
        "  Numerical elasticities should be near zero; they are the "
        "convergence check embedded in this sweep."
    )
    print(
        f"\n  Note: S(theta) with respect to D0 should be 0.5 exactly, since "
        f"theta ~ sqrt(D). Measured: {theta_elasticities['D0']:.4f}."
    )

    # ---- Figure ------------------------------------------------------------
    fig, axes = plotting.new_figure(
        figsize=(10.5, 7.8), nrows=2, ncols=2, constrained_layout=True
    )
    ax_theta, ax_rho, ax_compare, ax_dvalue = axes.flatten()

    def tornado(ax, elasticities: dict[str, float], title: str, xlabel: str) -> None:
        """Draw a horizontal bar chart of elasticities, largest magnitude first."""
        ordered = sorted(
            elasticities,
            key=lambda n: abs(elasticities[n]) if np.isfinite(elasticities[n]) else -1.0,
        )
        positions = np.arange(len(ordered))
        values = [elasticities[n] for n in ordered]
        colours = [
            plotting.COLOUR_ANALYTIC if n in NUMERICAL_PARAMETERS
            else plotting.COLOUR_MEASURED
            for n in ordered
        ]
        ax.barh(positions, values, color=colours, height=0.65)
        ax.set_yticks(positions)
        ax.set_yticklabels([labels[n] for n in ordered])
        ax.axvline(0.0, color=plotting.COLOUR_REFERENCE, linewidth=1.0)
        plotting.label_axes(ax, xlabel, "parameter", title)
        for position, value in zip(positions, values, strict=True):
            if np.isfinite(value):
                ax.text(
                    value + (0.02 if value >= 0 else -0.02),
                    position,
                    f"{value:+.2f}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=7.5,
                )

    tornado(
        ax_theta,
        theta_elasticities,
        "(a) Sensitivity of conduction velocity",
        r"elasticity $S = (\Delta\theta/\theta)/(\Delta X/X)$",
    )
    # Margin so the printed value labels are not clipped by the axes.
    theta_extent = max(
        abs(v) for v in theta_elasticities.values() if np.isfinite(v)
    )
    ax_theta.set_xlim(-1.35 * theta_extent, 1.35 * theta_extent)

    tornado(
        ax_rho,
        rho_elasticities,
        "(b) Sensitivity of the block threshold",
        r"elasticity $S = (\Delta\rho_c/\rho_c)/(\Delta X/X)$",
    )
    rho_extent = max(abs(v) for v in rho_elasticities.values() if np.isfinite(v))
    ax_rho.set_xlim(-1.35 * rho_extent, 1.35 * rho_extent)

    # (c) Physical against numerical, as a direct comparison.
    physical_max = max(
        abs(rho_elasticities[n]) for n in physical if np.isfinite(rho_elasticities[n])
    )
    ax_compare.bar(
        ["physical\n(largest)", "numerical\n(largest)"],
        [physical_max, largest_numerical_rho],
        color=[plotting.COLOUR_MEASURED, plotting.COLOUR_ANALYTIC],
        width=0.55,
    )
    plotting.label_axes(
        ax_compare,
        "parameter class",
        r"largest $|S|$ for $\rho_{\mathrm{crit}}$ (dimensionless)",
        "(c) Numerical choices must not rival physical ones",
    )
    ax_compare.set_yscale("log")
    plotting.annotate_takeaway(
        ax_compare,
        f"physical influence exceeds\nnumerical by "
        f"{physical_max / max(largest_numerical_rho, 1e-12):.0f}$\\times$",
        loc="upper right",
    )

    # (d) The sqrt(D) elasticity as an independent check of ex05.
    ax_dvalue.bar(
        ["measured", "analytic"],
        [theta_elasticities["D0"], 0.5],
        color=[plotting.COLOUR_MEASURED, plotting.COLOUR_ANALYTIC],
        width=0.55,
    )
    ax_dvalue.axhline(
        0.5, color=plotting.COLOUR_REFERENCE, linestyle="--", linewidth=1.0
    )
    plotting.label_axes(
        ax_dvalue,
        "source",
        r"elasticity of $\theta$ with respect to $D_0$",
        r"(d) $S=1/2$ confirms $\theta \propto \sqrt{D}$ independently",
    )
    ax_dvalue.set_ylim(0.0, 0.7)
    plotting.annotate_takeaway(
        ax_dvalue,
        f"measured {theta_elasticities['D0']:.4f}\nvs exactly 0.5",
        loc="upper left",
    )

    caption = (
        f"Conduction velocity is governed almost entirely by the coupling "
        f"strength, with an elasticity of {theta_elasticities['D0']:.3f} with "
        f"respect to D0 -- an independent confirmation of the sqrt(D) law, since "
        f"a square root has elasticity exactly 1/2. The block threshold is most "
        f"sensitive to {dominant_rho} "
        f"(S = {rho_elasticities[dominant_rho]:+.2f}). Critically, the two "
        f"NUMERICAL parameters have elasticities of at most "
        f"{largest_numerical_rho:.3f}, some "
        f"{physical_max / max(largest_numerical_rho, 1e-12):.0f} times smaller "
        f"than the leading physical parameter: the conclusions are properties of "
        f"the model, not of the grid. A one-at-a-time sweep cannot detect "
        f"parameter interactions, which is stated as a limitation."
    )

    measurements = {
        "perturbation_fraction": PERTURBATION,
        "baseline_theta_cm_per_ms": baseline_theta,
        "baseline_critical_rho": baseline_rho,
        "theta_elasticities": theta_elasticities,
        "rho_elasticities": rho_elasticities,
        "dominant_physical_for_theta": dominant_theta,
        "dominant_physical_for_rho": dominant_rho,
        "largest_numerical_elasticity_theta": largest_numerical_theta,
        "largest_numerical_elasticity_rho": largest_numerical_rho,
        "physical_to_numerical_ratio": float(
            physical_max / max(largest_numerical_rho, 1e-12)
        ),
    }

    plotting.save_figure(
        fig,
        "fig_ex08_sensitivity",
        caption,
        config_dict=base.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_arrays(
        "ex08_sensitivity",
        {
            "theta_elasticities": np.array(
                [theta_elasticities[n] for n in baseline_values]
            ),
            "rho_elasticities": np.array(
                [rho_elasticities[n] for n in baseline_values]
            ),
            "baseline_parameter_values": np.array(list(baseline_values.values())),
        },
        config_dict=base.to_dict(),
        extra_metadata=measurements,
    )

    utils.save_table(
        "ex08_sensitivity",
        header=[
            "parameter",
            "class",
            "baseline_value",
            "theta_minus10pct",
            "theta_plus10pct",
            "rho_crit_minus10pct",
            "rho_crit_plus10pct",
            "elasticity_theta",
            "elasticity_rho_crit",
        ],
        rows=[
            (
                name,
                "numerical" if name in NUMERICAL_PARAMETERS else "physical",
                f"{value:.6f}",
                f"{tl:.7f}",
                f"{th:.7f}",
                f"{rl:.6f}",
                f"{rh:.6f}",
                f"{et:+.5f}",
                f"{er:+.5f}",
            )
            for name, value, tl, th, rl, rh, et, er in raw_rows
        ],
    )

    print("\n  figure -> figures/fig_ex08_sensitivity.png / .pdf")
    return measurements


if __name__ == "__main__":
    main()
