"""Verification tests for the time integrators and the stability analysis.

These support the report's part (c): that the explicit-Euler stability limit
derived by von Neumann analysis is the limit the code actually obeys, and that
the integrators have the orders of accuracy claimed for them.
"""

from __future__ import annotations

import numpy as np
import pytest

from fibroblock import config as cfg
from fibroblock import grid as gridmod
from fibroblock import operators, simulate, solvers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def decay_rhs(
    t: float, V: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Right-hand side of dV/dt = -V, with w held fixed.

    The exact solution is V(t) = V(0) exp(-t), which is the standard test
    problem for measuring an integrator's order of accuracy.
    """
    del t
    return -V, np.zeros_like(w)


def integrate_decay(stepper, dt: float, t_end: float) -> float:
    """Integrate dV/dt = -V from V(0) = 1 and return V(t_end)."""
    V = np.array([1.0])
    w = np.array([0.0])
    n_steps = int(round(t_end / dt))
    for step_index in range(n_steps):
        V, w = stepper(step_index * dt, V, w, dt, decay_rhs)
    return float(V[0])


# ---------------------------------------------------------------------------
# Order of accuracy on a problem with a known solution
# ---------------------------------------------------------------------------


def test_euler_is_first_order_on_exponential_decay() -> None:
    """Supports the report's claim that the primary scheme is first order in dt.

    Halving dt must halve the error.
    """
    t_end = 1.0
    exact = float(np.exp(-t_end))

    steps = [0.1, 0.05, 0.025, 0.0125]
    errors = [
        abs(integrate_decay(solvers.euler_step, dt, t_end) - exact) for dt in steps
    ]

    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        observed_order = np.log2(coarse / fine)
        assert observed_order == pytest.approx(1.0, abs=0.05)


def test_rk4_is_fourth_order_on_exponential_decay() -> None:
    """Supports the use of RK4 as the accuracy reference in ex03.

    Halving dt must reduce the error by a factor of about 16.
    """
    t_end = 1.0
    exact = float(np.exp(-t_end))

    # Steps kept well above the point where round-off would swamp truncation:
    # at dt = 0.0125 the RK4 error is about 4e-9, still 7 orders of magnitude
    # above double-precision round-off.
    steps = [0.1, 0.05, 0.025, 0.0125]
    errors = [
        abs(integrate_decay(solvers.rk4_step, dt, t_end) - exact) for dt in steps
    ]

    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        observed_order = np.log2(coarse / fine)
        assert observed_order == pytest.approx(4.0, abs=0.1)


def test_rk4_is_far_more_accurate_than_euler_at_the_same_step() -> None:
    """Supports the report's cost-accuracy discussion."""
    t_end = 1.0
    dt = 0.05
    exact = float(np.exp(-t_end))

    euler_error = abs(integrate_decay(solvers.euler_step, dt, t_end) - exact)
    rk4_error = abs(integrate_decay(solvers.rk4_step, dt, t_end) - exact)

    assert rk4_error < euler_error / 1000.0


# ---------------------------------------------------------------------------
# The stability limit itself
# ---------------------------------------------------------------------------


def test_stability_limit_matches_the_brief() -> None:
    """Supports the reported limit dt <= 2/43 = 0.04651 ms.

    With D = 0.001 cm^2/ms, dx = 0.01 cm and |f_V|_max = 3, the diffusive term
    4D/dx^2 is exactly 40 per ms, so the limit is 2/(40 + 3).
    """
    D = cfg.GridParams().baseline_D
    dx = cfg.GridParams().dx_cm
    f_v = cfg.SolverParams().f_v_bound

    limit = solvers.explicit_euler_dt_limit(D, dx, f_v)

    assert 4.0 * D / dx**2 == pytest.approx(40.0, rel=1.0e-12)
    assert limit == pytest.approx(2.0 / 43.0, rel=1.0e-12)
    assert limit == pytest.approx(cfg.BRIEF_DT_LIMIT_MS, abs=1.0e-5)


def test_pure_diffusion_limit_is_seven_percent_too_optimistic() -> None:
    """Supports the report's warning about the naive diffusion-only limit.

    The brief notes that dx^2/(2D) = 0.05 ms is 7 % too generous. This test
    computes the exact overestimate from the two formulas.
    """
    D = cfg.GridParams().baseline_D
    dx = cfg.GridParams().dx_cm
    f_v = cfg.SolverParams().f_v_bound
    dt = cfg.SolverParams().dt_ms

    limits = solvers.stability_limits(D, dx, f_v, dt)

    assert limits.pure_diffusion_dt_ms == pytest.approx(0.05, rel=1.0e-12)
    assert limits.pure_diffusion_dt_ms > limits.reaction_diffusion_dt_ms
    # 0.05 / (2/43) - 1 = 0.075 exactly.
    assert limits.relative_overestimate == pytest.approx(0.075, rel=1.0e-9)


def test_default_step_has_the_stated_safety_factor() -> None:
    """Supports the reported safety factor of about 2.3 at dt = 0.02 ms."""
    D = cfg.GridParams().baseline_D
    dx = cfg.GridParams().dx_cm
    solver = cfg.SolverParams()

    limits = solvers.stability_limits(D, dx, solver.f_v_bound, solver.dt_ms)

    assert limits.is_stable is True
    assert limits.safety_factor == pytest.approx(2.33, abs=0.02)


def test_stability_limit_scales_as_dx_squared_when_diffusion_dominates() -> None:
    """Supports the numerical-choices discussion of refinement cost.

    Halving dx must roughly quarter the permitted step, which is why a
    convergence study in dx cannot hold dt fixed.
    """
    D = 0.001
    f_v = 3.0

    coarse = solvers.explicit_euler_dt_limit(D, 0.01, f_v)
    fine = solvers.explicit_euler_dt_limit(D, 0.005, f_v)

    # Not exactly 4 because the reaction term does not scale with dx; the ratio
    # approaches 4 from below as diffusion dominates further.
    assert 3.5 < coarse / fine < 4.0


def test_stability_limit_rejects_invalid_inputs() -> None:
    """Supports the 'fail loudly' requirement."""
    with pytest.raises(ValueError, match="dx must be positive"):
        solvers.explicit_euler_dt_limit(0.001, 0.0, 3.0)
    with pytest.raises(ValueError, match="D_max must be non-negative"):
        solvers.explicit_euler_dt_limit(-1.0, 0.01, 3.0)
    with pytest.raises(ValueError, match="f_v_bound must be non-negative"):
        solvers.explicit_euler_dt_limit(0.001, 0.01, -1.0)


# ---------------------------------------------------------------------------
# The limit actually bites
# ---------------------------------------------------------------------------


def _diffusion_only_growth(dt: float, n_steps: int, dx: float, D: float) -> float:
    """Amplification of a checkerboard initial condition under explicit Euler.

    Returns the ratio of final to initial checkerboard amplitude for the pure
    diffusion problem, which is what the von Neumann analysis predicts.
    """
    n_nodes = int(round(1.0 / dx)) + 1
    grid_params = cfg.GridParams(length_cm=1.0, dx_cm=dx, baseline_D=D)
    gap_params = cfg.GapParams(rho=1.0, gap_length_cm=0.0)
    g = gridmod.build_grid(grid_params, gap_params)
    assert g.n_nodes == n_nodes

    V = solvers.checkerboard_mode(n_nodes)
    initial = solvers.checkerboard_amplitude(V)

    for _ in range(n_steps):
        V = V + dt * operators.divergence(V, g.D_half, g.dx)

    return solvers.checkerboard_amplitude(V) / initial


def test_checkerboard_mode_decays_below_the_limit_and_grows_above_it() -> None:
    """Supports the central claim of part (c): the derived limit is the real one.

    For pure diffusion the limit is dx^2/(2D). Stepping just below it must make
    the checkerboard mode decay; stepping just above it must make it grow.
    """
    D = 0.001
    dx = 0.01
    limit = solvers.pure_diffusion_dt_limit(D, dx)

    below = _diffusion_only_growth(0.98 * limit, n_steps=200, dx=dx, D=D)
    above = _diffusion_only_growth(1.02 * limit, n_steps=200, dx=dx, D=D)

    assert below < 1.0
    assert above > 1.0


def test_checkerboard_amplification_matches_the_von_neumann_prediction() -> None:
    """Supports the derivation itself, not merely its conclusion.

    For the checkerboard mode the predicted per-step amplification factor is
    g = 1 - 4 D dt / dx^2. The measured growth over n steps must equal g^n.
    """
    D = 0.001
    dx = 0.01
    dt = 0.03
    n_steps = 50

    predicted_g = 1.0 - 4.0 * D * dt / dx**2
    measured = _diffusion_only_growth(dt, n_steps=n_steps, dx=dx, D=D)

    assert measured == pytest.approx(abs(predicted_g) ** n_steps, rel=1.0e-9)


def test_simulation_refuses_an_unstable_step_unless_forced() -> None:
    """Supports the 'fail loudly' requirement for the headline pipeline.

    A step past the limit must raise, so no figure can be produced from an
    unstable run by accident. ex02 passes force=True deliberately.
    """
    base = cfg.default_config()
    unstable = base.replace(
        solver=cfg.SolverParams(dt_ms=0.06, t_end_ms=1.0, record_every=1)
    )

    with pytest.raises(ValueError, match="exceeds the explicit-Euler stability limit"):
        simulate.run_simulation(unstable)

    # With force=True it runs, and reports that it was not stable.
    result = simulate.run_simulation(unstable, force=True)
    assert result.stability.is_stable is False


def test_forced_unstable_run_actually_diverges() -> None:
    """Supports the empirical demonstration in ex02.

    A step well past the limit must not merely be flagged as unstable, it must
    blow up.
    """
    base = cfg.default_config()
    unstable = base.replace(
        solver=cfg.SolverParams(dt_ms=0.055, t_end_ms=50.0, record_every=10)
    )

    result = simulate.run_simulation(unstable, force=True)

    assert result.diverged is True
    assert np.isfinite(result.divergence_time_ms)


def test_stable_run_does_not_diverge() -> None:
    """Supports the choice of dt = 0.02 ms as the working step."""
    config = cfg.default_config().replace(
        solver=cfg.SolverParams(dt_ms=0.02, t_end_ms=50.0, record_every=50)
    )
    result = simulate.run_simulation(config)

    assert result.diverged is False
    assert result.stability.is_stable is True
    # The solution must stay within the physically reachable range.
    assert float(np.max(result.V_peak)) < 3.0


# ---------------------------------------------------------------------------
# Checkerboard mode utilities
# ---------------------------------------------------------------------------


def test_checkerboard_mode_alternates() -> None:
    """Supports the identification of k = pi/dx as the worst-case mode."""
    mode = solvers.checkerboard_mode(5)
    np.testing.assert_allclose(mode, [1.0, -1.0, 1.0, -1.0, 1.0])


def test_checkerboard_amplitude_isolates_the_alternating_component() -> None:
    """Supports the diagnostic used in ex02 to identify the growing mode."""
    n_nodes = 101
    x = np.linspace(0.0, 1.0, n_nodes)

    # A smooth field has essentially no checkerboard content.
    smooth = np.sin(2.0 * np.pi * x)
    assert solvers.checkerboard_amplitude(smooth) < 1.0e-2

    # A pure checkerboard of amplitude 0.3 must be recovered as 0.3.
    checkerboard = 0.3 * solvers.checkerboard_mode(n_nodes)
    assert solvers.checkerboard_amplitude(checkerboard) == pytest.approx(0.3, rel=1e-12)


def test_get_stepper_rejects_an_unknown_method() -> None:
    """Supports the 'fail loudly' requirement for configuration typos."""
    with pytest.raises(ValueError, match="Unknown integrator"):
        solvers.get_stepper("midpoint")


# ---------------------------------------------------------------------------
# Euler against RK4 on the real problem
# ---------------------------------------------------------------------------


def test_euler_and_rk4_agree_on_the_strand_at_the_working_step() -> None:
    """Supports the claim that spatial error dominates at dt = 0.02 ms.

    If the two integrators, which differ by three orders in temporal accuracy,
    produce conduction velocities that agree to well under a percent, then the
    temporal error is not what limits the answer -- the spatial discretisation
    is. That is the justification for using the cheaper scheme.
    """
    from fibroblock import measure

    base = cfg.default_config().replace(
        solver=cfg.SolverParams(dt_ms=0.02, t_end_ms=120.0, record_every=100)
    )

    euler_result = simulate.run_simulation(base)
    rk4_result = simulate.run_simulation(
        base.replace(
            solver=cfg.SolverParams(
                dt_ms=0.02, t_end_ms=120.0, method="rk4", record_every=100
            )
        )
    )

    euler_theta = measure.measure_velocity(euler_result).theta_cm_per_ms
    rk4_theta = measure.measure_velocity(rk4_result).theta_cm_per_ms

    assert np.isfinite(euler_theta)
    assert np.isfinite(rk4_theta)
    relative_difference = abs(euler_theta - rk4_theta) / rk4_theta
    assert relative_difference < 0.01
