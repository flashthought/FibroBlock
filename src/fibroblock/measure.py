"""Extracting reported quantities from a completed simulation.

Conduction velocity, block detection, conduction delay and observed order of
accuracy all live here, so that the *definitions* behind the report's numbers
are in one readable file rather than scattered through the experiment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fibroblock.config import MeasurementParams, RunConfig
from fibroblock.simulate import SimulationResult


# ---------------------------------------------------------------------------
# Conduction velocity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VelocityFit:
    """Least-squares fit of activation time against position.

    Attributes
    ----------
    theta_cm_per_ms : float
        Conduction velocity. cm/ms. NaN if the fit could not be made.
    theta_cm_per_s : float
        The same velocity in cm/s, the unit used in the physiology literature.
    slope_ms_per_cm : float
        Fitted slope ``dt/dx``. Its reciprocal is the velocity.
    intercept_ms : float
        Fitted intercept. ms.
    r_squared : float
        Coefficient of determination. Evidence that the wave really is
        travelling at constant speed over the fitting window; anything below
        about 0.999 means the window still contains acceleration.
    residual_rms_ms : float
        Root-mean-square fit residual. ms.
    n_points : int
        Number of nodes used.
    window_start_cm, window_end_cm : float
        Bounds of the fitting window. cm.
    rule : str
        Which activation definition was used.
    """

    theta_cm_per_ms: float
    theta_cm_per_s: float
    slope_ms_per_cm: float
    intercept_ms: float
    r_squared: float
    residual_rms_ms: float
    n_points: int
    window_start_cm: float
    window_end_cm: float
    rule: str


def fit_conduction_velocity(
    x: np.ndarray,
    activation_times: np.ndarray,
    measurement: MeasurementParams,
    length_cm: float,
    rule: str = "v_zero_crossing",
) -> VelocityFit:
    """Fit conduction velocity over the steady-propagation window.

    Activation time should be linear in distance once the wave has settled:

    .. math:: t_{\\text{act}}(x) = \\frac{x}{\\theta} + t_0

    so a straight-line fit of ``t`` against ``x`` gives ``theta = 1 / slope``.

    Parameters
    ----------
    x : ndarray
        Node positions. cm.
    activation_times : ndarray
        Activation time at each node, NaN where the node never activated. ms.
    measurement : MeasurementParams
        Supplies the distances to discard at each end.
    length_cm : float
        Strand length, for locating the far end. cm.
    rule : str, optional
        Name of the activation definition used, recorded in the result.

    Returns
    -------
    VelocityFit
        The fitted velocity and the evidence for it.

    Notes
    -----
    **The window matters.** The first ``cv_fit_skip_start_cm`` (default 0.5 cm)
    is discarded because the wave is still forming out of the stimulus and is
    accelerating; the last ``cv_fit_skip_end_cm`` (default 0.2 cm) is discarded
    because the sealed end reflects charge back into the approaching front and
    speeds it up. Fitting through either region gives a velocity that is wrong
    in a direction that looks plausible, which is the worst kind of wrong.

    ``R^2`` is reported precisely so this can be checked rather than trusted:
    if the window still contained acceleration, the fit would curve and ``R^2``
    would fall away from 1.

    A fit is refused (all fields NaN) rather than fudged if fewer than three
    nodes activated in the window -- two points would give an ``R^2`` of
    exactly 1 and a meaningless velocity.
    """
    window_start = measurement.cv_fit_skip_start_cm
    window_end = length_cm - measurement.cv_fit_skip_end_cm

    in_window = (x >= window_start) & (x <= window_end)
    activated = np.isfinite(activation_times)
    usable = in_window & activated

    n_points = int(np.count_nonzero(usable))

    # Three points is the minimum at which R^2 carries any information about
    # linearity; with two it is identically 1 whatever the data.
    if n_points < 3:
        return VelocityFit(
            theta_cm_per_ms=float("nan"),
            theta_cm_per_s=float("nan"),
            slope_ms_per_cm=float("nan"),
            intercept_ms=float("nan"),
            r_squared=float("nan"),
            residual_rms_ms=float("nan"),
            n_points=n_points,
            window_start_cm=window_start,
            window_end_cm=window_end,
            rule=rule,
        )

    x_fit = x[usable]
    t_fit = activation_times[usable]

    slope, intercept = np.polyfit(x_fit, t_fit, deg=1)

    predicted = slope * x_fit + intercept
    residuals = t_fit - predicted

    sum_squared_residuals = float(np.sum(residuals**2))
    sum_squared_total = float(np.sum((t_fit - np.mean(t_fit)) ** 2))

    if sum_squared_total > 0.0:
        r_squared = 1.0 - sum_squared_residuals / sum_squared_total
    else:
        # Every node activated at the same instant: no propagation to speak of.
        r_squared = float("nan")

    theta = 1.0 / slope if slope != 0.0 else float("inf")

    return VelocityFit(
        theta_cm_per_ms=float(theta),
        # 1 ms = 1e-3 s, so cm/ms -> cm/s is a factor of 1000. Exact unit
        # conversion, not a fitted quantity.
        theta_cm_per_s=float(theta * 1000.0),
        slope_ms_per_cm=float(slope),
        intercept_ms=float(intercept),
        r_squared=float(r_squared),
        residual_rms_ms=float(np.sqrt(sum_squared_residuals / n_points)),
        n_points=n_points,
        window_start_cm=window_start,
        window_end_cm=window_end,
        rule=rule,
    )


def measure_velocity(result: SimulationResult) -> VelocityFit:
    """Fit the conduction velocity of a completed run.

    Parameters
    ----------
    result : SimulationResult
        A finished simulation.

    Returns
    -------
    VelocityFit
        Velocity, ``R^2`` and the window used.
    """
    return fit_conduction_velocity(
        x=result.x,
        activation_times=result.activation_times,
        measurement=result.config.measurement,
        length_cm=result.config.grid.length_cm,
        rule=result.config.measurement.activation_rule,
    )


# ---------------------------------------------------------------------------
# Conduction block
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockResult:
    """Outcome of applying the block criterion to a run.

    Attributes
    ----------
    blocked : bool
        True if propagation failed.
    detection_x_cm : float
        The position beyond which activation had to occur. cm.
    deadline_ms : float
        The time by which it had to occur. ms.
    n_activated_beyond : int
        How many nodes past the detection point activated in time.
    first_arrival_ms : float
        Earliest qualifying activation time beyond the detection point. ms.
        NaN if none.
    furthest_activation_cm : float
        Furthest node that activated at all, at any time. cm. Useful for
        distinguishing "blocked at the gap" from "blocked just after it".
    criterion : str
        The criterion in words, stored so every result file carries its own
        definition.
    """

    blocked: bool
    detection_x_cm: float
    deadline_ms: float
    n_activated_beyond: int
    first_arrival_ms: float
    furthest_activation_cm: float
    criterion: str


def detect_block(result: SimulationResult) -> BlockResult:
    """Apply the block criterion to a completed run.

    **Criterion.** Propagation has blocked if no node at
    ``x > x_gap + L_gap + block_margin`` reaches ``V = activation_level``
    within ``block_window_ms`` of the stimulus.

    Parameters
    ----------
    result : SimulationResult
        A finished simulation.

    Returns
    -------
    BlockResult
        Whether it blocked, and the evidence.

    Notes
    -----
    The margin exists because ``V`` rises *somewhat* just downstream of the gap
    even when propagation fails: charge leaks across the interface
    electrotonically and decays over roughly one space constant. Judging block
    at the gap edge itself would therefore mistake passive spread for a
    regenerative upstroke. At 0.3 cm the passive contribution is negligible and
    any activation is a genuine propagating front.

    The level crossing is used regardless of the configured
    ``activation_rule``, because the criterion is *defined* in terms of
    ``V = 0``. Using the steepest-upstroke definition here would silently
    change the criterion.
    """
    config: RunConfig = result.config
    measurement = config.measurement

    detection_x = config.gap.gap_end_cm + measurement.block_margin_cm
    deadline = config.stimulus.start_ms + measurement.block_window_ms

    # Deliberately the crossing array, not result.activation_times: the
    # criterion is stated in terms of V reaching the activation level.
    crossings = result.activation_time_crossing

    beyond = result.x > detection_x
    qualifying = beyond & np.isfinite(crossings) & (crossings <= deadline)

    n_qualifying = int(np.count_nonzero(qualifying))
    first_arrival = (
        float(np.min(crossings[qualifying])) if n_qualifying > 0 else float("nan")
    )

    any_activation = np.isfinite(crossings)
    furthest = (
        float(np.max(result.x[any_activation]))
        if np.any(any_activation)
        else float("nan")
    )

    criterion = (
        f"blocked if no node at x > {detection_x:.4f} cm reaches "
        f"V = {measurement.activation_level} within "
        f"{measurement.block_window_ms} ms of the stimulus"
    )

    return BlockResult(
        blocked=(n_qualifying == 0),
        detection_x_cm=detection_x,
        deadline_ms=deadline,
        n_activated_beyond=n_qualifying,
        first_arrival_ms=first_arrival,
        furthest_activation_cm=furthest,
        criterion=criterion,
    )


# ---------------------------------------------------------------------------
# Conduction delay across the gap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DelayResult:
    """Time taken to cross the gap, and how much of that is excess.

    Attributes
    ----------
    upstream_x_cm, downstream_x_cm : float
        Probe positions, placed symmetrically about the gap at
        ``block_margin_cm`` outside each edge. cm.
    upstream_time_ms, downstream_time_ms : float
        Activation times at the probes. ms. NaN if the probe never activated.
    transit_ms : float
        ``downstream_time - upstream_time``. ms. NaN if propagation failed.
    reference_transit_ms : float
        Transit time between the same two probes in a healthy strand. ms.
    excess_delay_ms : float
        ``transit - reference``: the delay attributable to the gap alone. ms.
        This is the quantity that diverges as coupling approaches its critical
        value.
    propagated : bool
        Whether the wave reached the downstream probe at all.
    """

    upstream_x_cm: float
    downstream_x_cm: float
    upstream_time_ms: float
    downstream_time_ms: float
    transit_ms: float
    reference_transit_ms: float
    excess_delay_ms: float
    propagated: bool


def _activation_at(
    x: np.ndarray, activation_times: np.ndarray, target_x: float
) -> tuple[float, float]:
    """Activation time at the node nearest a target position.

    Parameters
    ----------
    x : ndarray
        Node positions. cm.
    activation_times : ndarray
        Activation times. ms.
    target_x : float
        Position of interest. cm.

    Returns
    -------
    actual_x : float
        Position of the node actually used. cm.
    activation_time : float
        Its activation time, or NaN. ms.

    Notes
    -----
    The nearest node is used rather than interpolating between neighbours,
    because activation time is not smooth across a node when the wave is close
    to failing -- one node activates and the next does not. The actual position
    used is returned so the reported delay is honest about where it was
    measured.
    """
    index = int(np.argmin(np.abs(x - target_x)))
    return float(x[index]), float(activation_times[index])


def measure_delay(
    result: SimulationResult,
    reference_transit_ms: float = float("nan"),
) -> DelayResult:
    """Measure the time the wave takes to cross the gap region.

    Parameters
    ----------
    result : SimulationResult
        A finished simulation.
    reference_transit_ms : float, optional
        Transit time between the same probes in a healthy (``rho = 1``) strand,
        so the excess delay can be reported. Pass NaN to omit.

    Returns
    -------
    DelayResult
        Probe positions, transit time, and excess delay.

    Notes
    -----
    The probes sit ``block_margin_cm`` outside each gap edge -- the same offset
    at which block is judged -- so "the wave crossed" and "the wave took this
    long to cross" refer to exactly the same span. Measuring the delay at the
    gap edges themselves would contaminate it with the electrotonic foot, which
    grows as coupling weakens and would inflate the apparent delay for the
    wrong reason.
    """
    config = result.config
    margin = config.measurement.block_margin_cm

    upstream_target = config.gap.gap_start_cm - margin
    downstream_target = config.gap.gap_end_cm + margin

    crossings = result.activation_time_crossing

    upstream_x, upstream_t = _activation_at(result.x, crossings, upstream_target)
    downstream_x, downstream_t = _activation_at(result.x, crossings, downstream_target)

    propagated = bool(np.isfinite(upstream_t) and np.isfinite(downstream_t))
    transit = downstream_t - upstream_t if propagated else float("nan")
    excess = transit - reference_transit_ms

    return DelayResult(
        upstream_x_cm=upstream_x,
        downstream_x_cm=downstream_x,
        upstream_time_ms=upstream_t,
        downstream_time_ms=downstream_t,
        transit_ms=float(transit),
        reference_transit_ms=float(reference_transit_ms),
        excess_delay_ms=float(excess),
        propagated=propagated,
    )


# ---------------------------------------------------------------------------
# Error norms and observed order of accuracy
# ---------------------------------------------------------------------------


def l2_error(
    computed: np.ndarray, exact: np.ndarray, quadrature_weights: np.ndarray
) -> float:
    """Grid-independent L2 norm of the error.

    .. math:: \\|e\\|_2 = \\sqrt{\\sum_j w_j (u_j - u^{\\text{exact}}_j)^{2}}

    Parameters
    ----------
    computed, exact : ndarray
        Numerical and exact solutions on the same grid.
    quadrature_weights : ndarray
        Trapezoidal weights. cm.

    Returns
    -------
    float
        The error norm.

    Notes
    -----
    The quadrature weights are essential for a convergence study. A plain
    ``sqrt(sum(e^2))`` grows as the grid is refined simply because there are
    more terms in the sum, which would corrupt the measured order of accuracy.
    Weighting by cell width makes the norm approximate a fixed integral, so
    successive grids can be compared.
    """
    if computed.shape != exact.shape:
        raise ValueError(
            f"Shapes must match, got {computed.shape} and {exact.shape}"
        )
    error = computed - exact
    return float(np.sqrt(np.sum(quadrature_weights * error**2)))


def linf_error(computed: np.ndarray, exact: np.ndarray) -> float:
    """Maximum absolute error.

    Parameters
    ----------
    computed, exact : ndarray
        Numerical and exact solutions on the same grid.

    Returns
    -------
    float
        ``max |computed - exact|``.
    """
    return float(np.max(np.abs(computed - exact)))


def observed_order(
    resolutions: np.ndarray, errors: np.ndarray
) -> tuple[np.ndarray, float]:
    """Observed order of accuracy from a refinement sequence.

    Between consecutive grids,

    .. math:: p = \\frac{\\log(e_1 / e_2)}{\\log(h_1 / h_2)}

    and the overall order is the slope of a straight line fitted to
    ``log e`` against ``log h``.

    Parameters
    ----------
    resolutions : ndarray
        Grid spacings or time steps, in decreasing order.
    errors : ndarray
        Corresponding error norms.

    Returns
    -------
    pairwise : ndarray, shape (n - 1,)
        Order between each consecutive pair.
    fitted : float
        Slope of the log-log least-squares fit over all points.

    Raises
    ------
    ValueError
        If fewer than two resolutions are supplied, or if any error is
        non-positive (which makes the logarithm undefined and usually means the
        "error" has hit round-off rather than truncation).

    Notes
    -----
    Both are reported because they answer different questions. The pairwise
    values expose non-asymptotic behaviour -- an order that drifts from 1.6 to
    1.9 to 2.0 as the grid refines is a scheme entering its asymptotic regime,
    which is normal and worth showing. The single fitted slope is the summary
    number for the report.
    """
    resolutions = np.asarray(resolutions, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if resolutions.size < 2:
        raise ValueError(
            f"Need at least two resolutions, got {resolutions.size}"
        )
    if resolutions.shape != errors.shape:
        raise ValueError(
            f"Shapes must match, got {resolutions.shape} and {errors.shape}"
        )
    if np.any(errors <= 0.0):
        raise ValueError(
            f"All errors must be positive to take logarithms, got {errors}. "
            f"A zero error usually means round-off has been reached."
        )

    pairwise = np.log(errors[:-1] / errors[1:]) / np.log(
        resolutions[:-1] / resolutions[1:]
    )

    fitted_slope, _ = np.polyfit(np.log(resolutions), np.log(errors), deg=1)

    return pairwise, float(fitted_slope)


def log_log_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit ``y = C x^p`` by least squares in log-log space.

    Parameters
    ----------
    x, y : ndarray
        Strictly positive data.

    Returns
    -------
    exponent : float
        The fitted power ``p``.
    prefactor : float
        The fitted constant ``C``.
    r_squared : float
        Coefficient of determination of the log-log fit.

    Raises
    ------
    ValueError
        If any value is non-positive.

    Notes
    -----
    Used for the conduction-velocity scaling test, where the prediction is
    ``theta = C sqrt(D)``, i.e. ``p = 0.5``. The report also plots
    ``theta / sqrt(D)`` directly, because a flat line there is a far sharper
    test than a fitted slope: a log-log fit can return 0.50 even when the data
    curve, whereas systematic curvature is immediately visible in the ratio.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("Log-log fitting requires strictly positive data.")

    log_x = np.log(x)
    log_y = np.log(y)

    exponent, log_prefactor = np.polyfit(log_x, log_y, deg=1)

    predicted = exponent * log_x + log_prefactor
    residual_sum = float(np.sum((log_y - predicted) ** 2))
    total_sum = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else float("nan")

    return float(exponent), float(np.exp(log_prefactor)), float(r_squared)
