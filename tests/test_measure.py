"""Verification tests for the measurement layer.

These support every number the report quotes that is *derived* from a
simulation rather than computed directly by it: conduction velocity, the block
verdict, conduction delay, and observed orders of accuracy. A measurement
routine that is wrong would corrupt the results while every solver test still
passed, so these are checked against synthetic data with known answers.
"""

from __future__ import annotations

import numpy as np
import pytest

from fibroblock import config as cfg
from fibroblock import measure, simulate

# ---------------------------------------------------------------------------
# Conduction velocity from a synthetic activation map
# ---------------------------------------------------------------------------


def synthetic_activation_map(
    theta: float, offset_ms: float = 3.0, length_cm: float = 2.0, dx: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Build an exactly linear activation map for a known velocity.

    ``t_act(x) = x / theta + offset``, so the fit must recover ``theta``
    exactly.
    """
    x = np.linspace(0.0, length_cm, int(round(length_cm / dx)) + 1)
    return x, x / theta + offset_ms


def test_velocity_is_recovered_exactly_from_a_linear_map() -> None:
    """Supports every reported conduction velocity.

    If the fit cannot recover a velocity from data that is exactly linear, no
    velocity it reports elsewhere can be trusted.
    """
    true_theta = 0.0305  # cm/ms, close to the analytic value
    x, activation = synthetic_activation_map(true_theta)

    fit = measure.fit_conduction_velocity(
        x, activation, cfg.MeasurementParams(), length_cm=2.0
    )

    assert fit.theta_cm_per_ms == pytest.approx(true_theta, rel=1.0e-12)
    assert fit.theta_cm_per_s == pytest.approx(true_theta * 1000.0, rel=1.0e-12)
    assert fit.r_squared == pytest.approx(1.0, abs=1.0e-12)
    assert fit.residual_rms_ms == pytest.approx(0.0, abs=1.0e-12)


def test_velocity_fit_ignores_the_discarded_windows() -> None:
    """Supports the choice to discard the first 0.5 cm and last 0.2 cm.

    Nodes outside the window are corrupted deliberately. The fit must be
    unaffected, proving the window is genuinely applied.
    """
    true_theta = 0.0305
    x, activation = synthetic_activation_map(true_theta)
    measurement = cfg.MeasurementParams()

    corrupted = activation.copy()
    corrupted[x < measurement.cv_fit_skip_start_cm] += 5.0
    corrupted[x > 2.0 - measurement.cv_fit_skip_end_cm] -= 5.0

    fit = measure.fit_conduction_velocity(x, corrupted, measurement, length_cm=2.0)

    assert fit.theta_cm_per_ms == pytest.approx(true_theta, rel=1.0e-12)
    assert fit.r_squared == pytest.approx(1.0, abs=1.0e-12)


def test_r_squared_falls_when_the_wave_is_clearly_accelerating() -> None:
    """Supports the use of R^2 as evidence of steady propagation.

    A quadratic activation map means the wave is accelerating rather than
    travelling at constant speed. With a curvature strong enough to change the
    local velocity by tens of per cent across the fitting window, R^2 must
    detect it; otherwise it would be worthless as evidence.
    """
    x, linear = synthetic_activation_map(0.0305)
    # Curvature chosen so the local velocity 1/(1/theta - 2 c x) rises by about
    # 30 % between the two ends of the fitting window.
    curvature = 4.0
    accelerating = linear - curvature * x**2

    fit = measure.fit_conduction_velocity(
        x, accelerating, cfg.MeasurementParams(), length_cm=2.0
    )

    assert fit.r_squared < 0.999
    assert fit.residual_rms_ms > 0.1


def test_r_squared_is_a_weak_detector_of_MILD_curvature() -> None:
    """Supports the report's argument for the theta/sqrt(D) flatness plot.

    This is a negative result, and it is deliberately recorded. A curvature
    that changes the local velocity by only a few per cent still leaves R^2
    above 0.9999. R^2 near 1 is therefore NECESSARY but not SUFFICIENT evidence
    of steady propagation -- which is exactly why ex05 does not rely on a
    fitted log-log slope alone, and plots theta/sqrt(D) directly instead, where
    a few per cent of drift is plainly visible.
    """
    x, linear = synthetic_activation_map(0.0305)
    mildly_curved = linear - 0.5 * x**2

    fit = measure.fit_conduction_velocity(
        x, mildly_curved, cfg.MeasurementParams(), length_cm=2.0
    )

    # Real curvature is present...
    assert fit.residual_rms_ms > 0.0
    # ...yet R^2 barely moves off 1.
    assert fit.r_squared > 0.9999


def test_velocity_fit_tolerates_nodes_that_never_activated() -> None:
    """Supports the blocked-propagation cases, where downstream nodes are NaN."""
    true_theta = 0.0305
    x, activation = synthetic_activation_map(true_theta)
    activation[x > 1.5] = np.nan  # wave died at 1.5 cm

    fit = measure.fit_conduction_velocity(
        x, activation, cfg.MeasurementParams(), length_cm=2.0
    )

    assert fit.theta_cm_per_ms == pytest.approx(true_theta, rel=1.0e-12)
    assert fit.n_points == int(np.count_nonzero((x >= 0.5) & (x <= 1.5)))


def test_velocity_fit_refuses_rather_than_guesses_with_too_few_points() -> None:
    """Supports the 'fail loudly' requirement in the measurement layer.

    With two points R^2 is identically 1 and the velocity is meaningless, so a
    fit must be refused rather than reported with false confidence.
    """
    x, activation = synthetic_activation_map(0.0305)
    activation[:] = np.nan
    # Leave exactly two activated nodes inside the fitting window.
    inside = np.flatnonzero((x >= 0.5) & (x <= 1.8))
    activation[inside[0]] = 20.0
    activation[inside[1]] = 20.5

    fit = measure.fit_conduction_velocity(
        x, activation, cfg.MeasurementParams(), length_cm=2.0
    )

    assert fit.n_points == 2
    assert np.isnan(fit.theta_cm_per_ms)
    assert np.isnan(fit.r_squared)


# ---------------------------------------------------------------------------
# Observed order and log-log fitting
# ---------------------------------------------------------------------------


def test_observed_order_recovers_a_known_power_law() -> None:
    """Supports the convergence study in ex03.

    Errors constructed as exactly C h^2 must give an observed order of exactly
    2, both pairwise and fitted.
    """
    spacings = np.array([0.04, 0.02, 0.01, 0.005])
    errors = 3.7 * spacings**2

    pairwise, fitted = measure.observed_order(spacings, errors)

    np.testing.assert_allclose(pairwise, 2.0, rtol=1.0e-12)
    assert fitted == pytest.approx(2.0, rel=1.0e-12)


def test_observed_order_rejects_non_positive_errors() -> None:
    """Supports the 'fail loudly' requirement: log of zero is not a number."""
    with pytest.raises(ValueError, match="must be positive"):
        measure.observed_order(np.array([0.02, 0.01]), np.array([1.0e-3, 0.0]))


def test_log_log_slope_recovers_a_square_root_law() -> None:
    """Supports the sqrt(D) scaling test in ex05."""
    D = np.array([0.0002, 0.0005, 0.001, 0.002, 0.004])
    theta = 0.8074 * np.sqrt(D)

    exponent, prefactor, r_squared = measure.log_log_slope(D, theta)

    assert exponent == pytest.approx(0.5, rel=1.0e-12)
    assert prefactor == pytest.approx(0.8074, rel=1.0e-10)
    assert r_squared == pytest.approx(1.0, abs=1.0e-12)


def test_log_log_slope_rejects_non_positive_data() -> None:
    """Supports the 'fail loudly' requirement."""
    with pytest.raises(ValueError, match="strictly positive"):
        measure.log_log_slope(np.array([1.0, 2.0]), np.array([1.0, -1.0]))


# ---------------------------------------------------------------------------
# Error norms
# ---------------------------------------------------------------------------


def test_l2_error_is_grid_independent() -> None:
    """Supports the convergence study's use of a weighted norm.

    The same continuous error function sampled on grids of different density
    must give nearly the same weighted L2 norm. An unweighted sqrt(sum(e^2))
    would grow as sqrt(n) and destroy the measured order of accuracy.
    """
    from fibroblock import grid as gridmod

    norms = []
    unweighted = []
    for n_intervals in (100, 200, 400):
        dx = 1.0 / n_intervals
        g = gridmod.build_grid(
            cfg.GridParams(length_cm=1.0, dx_cm=dx),
            cfg.GapParams(rho=1.0, gap_length_cm=0.0),
        )
        error_field = np.sin(np.pi * g.x)
        norms.append(
            measure.l2_error(
                error_field, np.zeros_like(error_field), g.quadrature_weights
            )
        )
        unweighted.append(float(np.sqrt(np.sum(error_field**2))))

    # Weighted norm converges to sqrt(integral sin^2) = sqrt(1/2).
    for value in norms:
        assert value == pytest.approx(np.sqrt(0.5), rel=1.0e-3)
    # The unweighted sum, by contrast, doubles with each refinement.
    assert unweighted[-1] / unweighted[0] > 1.9


def test_linf_error_is_the_maximum_absolute_difference() -> None:
    """Supports the maximum-norm figures quoted in ex04."""
    computed = np.array([1.0, 2.0, 3.0])
    exact = np.array([1.1, 2.0, 2.5])
    assert measure.linf_error(computed, exact) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Block detection and delay, on real (short) simulations
# ---------------------------------------------------------------------------


def short_run(rho: float, gap_length_cm: float = 0.1) -> simulate.SimulationResult:
    """Run a strand with a given coupling gap, long enough to resolve the verdict."""
    config = cfg.default_config().replace(
        gap=cfg.GapParams(rho=rho, gap_length_cm=gap_length_cm, gap_centre_cm=1.0),
        solver=cfg.SolverParams(dt_ms=0.02, t_end_ms=250.0, record_every=50),
    )
    return simulate.run_simulation(config)


def test_healthy_strand_is_not_blocked() -> None:
    """Supports the control case: with rho = 1 the wave must cross."""
    result = short_run(rho=1.0)
    verdict = measure.detect_block(result)

    assert verdict.blocked is False
    assert verdict.n_activated_beyond > 0
    assert np.isfinite(verdict.first_arrival_ms)
    assert verdict.first_arrival_ms <= verdict.deadline_ms


def test_severely_uncoupled_gap_blocks_propagation() -> None:
    """Supports the central result: weak enough coupling stops the wave."""
    result = short_run(rho=0.02)
    verdict = measure.detect_block(result)

    assert verdict.blocked is True
    assert verdict.n_activated_beyond == 0
    # The wave must still have reached the gap -- otherwise the run failed for
    # some unrelated reason and the test would pass for the wrong cause.
    assert verdict.furthest_activation_cm >= result.config.gap.gap_start_cm


def test_block_detection_point_is_where_the_criterion_says() -> None:
    """Supports the stated criterion, x > x_gap + L_gap + margin."""
    result = short_run(rho=0.5, gap_length_cm=0.2)
    verdict = measure.detect_block(result)

    expected = (
        result.config.gap.gap_end_cm + result.config.measurement.block_margin_cm
    )
    assert verdict.detection_x_cm == pytest.approx(expected)
    assert verdict.deadline_ms == pytest.approx(
        result.config.measurement.block_window_ms
    )
    assert "V = 0" in verdict.criterion or "V = 0.0" in verdict.criterion


def test_weakening_coupling_increases_conduction_delay() -> None:
    """Supports the delay result in ex07: a weaker gap is slower to cross."""
    healthy = measure.measure_delay(short_run(rho=1.0))
    impaired = measure.measure_delay(short_run(rho=0.2))

    assert healthy.propagated is True
    assert impaired.propagated is True
    assert impaired.transit_ms > healthy.transit_ms


def test_delay_probes_bracket_the_gap_symmetrically() -> None:
    """Supports the stated probe placement for the delay measurement."""
    result = short_run(rho=0.5, gap_length_cm=0.2)
    delay = measure.measure_delay(result)

    margin = result.config.measurement.block_margin_cm
    # Nearest node, so allow half a node spacing of slack.
    tolerance = 0.5 * result.grid.dx + 1.0e-12
    assert delay.upstream_x_cm == pytest.approx(
        result.config.gap.gap_start_cm - margin, abs=tolerance
    )
    assert delay.downstream_x_cm == pytest.approx(
        result.config.gap.gap_end_cm + margin, abs=tolerance
    )


def test_blocked_run_reports_no_transit_time() -> None:
    """Supports the handling of failed propagation in the delay sweep."""
    delay = measure.measure_delay(short_run(rho=0.02))

    assert delay.propagated is False
    assert np.isnan(delay.transit_ms)


# ---------------------------------------------------------------------------
# Recovery at the wavefront
# ---------------------------------------------------------------------------


def test_recovery_at_front_lies_between_rest_and_the_plateau() -> None:
    """Supports the ex05 explanation of the conduction-velocity prefactor.

    The recovery variable at the moment the front crosses the threshold root
    must be above its resting value (w rises during the upstroke) but well
    below the value it reaches at the plateau.
    """
    from fibroblock import fhn

    params = cfg.FHNParams()
    _, w_rest = fhn.rest_state(params)
    _, V2, _ = fhn.bistable_roots(params)

    result = short_run(rho=1.0)
    w_front = measure.recovery_at_front(result, probe_x_cm=1.2, threshold_V=V2)

    assert np.isfinite(w_front)
    assert w_front > w_rest
    assert w_front < w_rest + 0.3


def test_recovery_at_front_is_nan_where_the_wave_never_arrived() -> None:
    """Supports the blocked cases, where there is no front to sample."""
    from fibroblock import fhn

    _, V2, _ = fhn.bistable_roots(cfg.FHNParams())
    result = short_run(rho=0.02)

    w_front = measure.recovery_at_front(result, probe_x_cm=1.8, threshold_V=V2)
    assert np.isnan(w_front)
