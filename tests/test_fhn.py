"""Verification tests for the FitzHugh-Nagumo kinetics module.

Each test states which claim in the report it supports. Together these tests
are the evidence behind report section "Verification", and their expected-vs-
obtained values are transcribed into ``docs/verification_log.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from fibroblock import config as cfg
from fibroblock import fhn

# ---------------------------------------------------------------------------
# Rest state
# ---------------------------------------------------------------------------


def test_rest_state_matches_brief_values() -> None:
    """Supports the report claim: rest state is V* = -1.199408, w* = -0.624260.

    The values are computed from the cubic at run time, never hard-coded, and
    must agree with the brief to 1e-6.
    """
    params = cfg.FHNParams()
    V_rest, w_rest = fhn.rest_state(params)

    assert V_rest == pytest.approx(cfg.BRIEF_REST_V, abs=1.0e-6)
    assert w_rest == pytest.approx(cfg.BRIEF_REST_W, abs=1.0e-6)


def test_rest_state_actually_annihilates_both_right_hand_sides() -> None:
    """Supports the claim that (V*, w*) is an equilibrium, not merely a root.

    A root of the reduced cubic could in principle fail to zero the recovery
    equation if the algebra were wrong, so both are checked directly.
    """
    params = cfg.FHNParams()
    V_rest, w_rest = fhn.rest_state(params)

    assert fhn.reaction_f(V_rest, w_rest) == pytest.approx(0.0, abs=1.0e-12)
    assert fhn.recovery_g(V_rest, w_rest, params) == pytest.approx(0.0, abs=1.0e-12)


def test_rest_cubic_reduces_to_the_brief_polynomial() -> None:
    """Supports the derivation printed in the report: V^3 + 0.75 V + 2.625 = 0.

    Confirms the symbolic reduction in :func:`fhn.rest_cubic_coefficients` for
    the assignment values a = 0.7, b = 0.8.
    """
    coefficients = fhn.rest_cubic_coefficients(cfg.FHNParams())

    # [V^3, V^2, V^1, V^0] = [1, 0, 0.75, 2.625]
    expected = np.array([1.0, 0.0, 0.75, 2.625])
    np.testing.assert_allclose(coefficients, expected, rtol=0.0, atol=1.0e-12)


def test_rest_state_is_found_for_a_perturbed_parameter_set() -> None:
    """Supports the sensitivity study: the rest state is computed, not assumed.

    If the rest state had been hard-coded, changing ``a`` would leave it
    unchanged and the equilibrium condition below would fail.
    """
    perturbed = cfg.FHNParams(a=0.6, b=0.9)
    V_rest, w_rest = fhn.rest_state(perturbed)

    assert fhn.reaction_f(V_rest, w_rest) == pytest.approx(0.0, abs=1.0e-12)
    assert fhn.recovery_g(V_rest, w_rest, perturbed) == pytest.approx(0.0, abs=1.0e-12)
    # It must genuinely differ from the baseline rest state.
    baseline_V, _ = fhn.rest_state(cfg.FHNParams())
    assert abs(V_rest - baseline_V) > 1.0e-3


# ---------------------------------------------------------------------------
# Excitability (linear stability of the rest state)
# ---------------------------------------------------------------------------


def test_rest_state_is_a_stable_spiral() -> None:
    """Supports the report claim: the medium is excitable, not oscillatory.

    Requires tr(J) < 0 (perturbations decay), det(J) > 0 (not a saddle) and a
    negative discriminant (complex eigenvalues, so the return to rest spirals).
    """
    summary = fhn.excitability(cfg.FHNParams())

    assert summary.trace < 0.0
    assert summary.determinant > 0.0
    assert summary.discriminant < 0.0
    assert summary.classification == "stable spiral"
    assert summary.is_excitable is True


def test_jacobian_determinant_matches_brief() -> None:
    """Supports the reported det(J) = 0.1081.

    The determinant is checked to 1e-4 absolute, which is the precision to
    which the brief quotes it.
    """
    summary = fhn.excitability(cfg.FHNParams())
    assert summary.determinant == pytest.approx(cfg.BRIEF_JACOBIAN_DET, abs=1.0e-4)


def test_jacobian_trace_is_close_to_brief_value() -> None:
    """Supports the reported tr(J), and records a known small discrepancy.

    Computing tr(J) = (1 - V*^2) - eps*b from the exact rest state gives
    -0.502584, whereas the brief quotes -0.5034. The 0.16 % difference comes
    from rounding in the brief's intermediate arithmetic, not from an error
    here: this test independently confirms the trace from the definition.
    See docs/verification_log.md, check V3.
    """
    params = cfg.FHNParams()
    V_rest, _ = fhn.rest_state(params)
    summary = fhn.excitability(params)

    # Independent recomputation straight from the definition tr = J11 + J22.
    trace_from_definition = (1.0 - V_rest**2) - params.eps * params.b
    assert summary.trace == pytest.approx(trace_from_definition, abs=1.0e-12)

    # Agreement with the brief, to the tolerance its own rounding permits.
    assert summary.trace == pytest.approx(cfg.BRIEF_JACOBIAN_TRACE, abs=1.0e-3)


def test_jacobian_entries_have_the_derived_form() -> None:
    """Supports the Jacobian printed in the report: [[1 - V^2, -1], [eps, -eps*b]]."""
    params = cfg.FHNParams()
    V = -0.5  # arbitrary evaluation point; the form must hold everywhere
    J = fhn.jacobian(V, params)

    assert J[0, 0] == pytest.approx(1.0 - V**2)
    assert J[0, 1] == pytest.approx(-1.0)
    assert J[1, 0] == pytest.approx(params.eps)
    assert J[1, 1] == pytest.approx(-params.eps * params.b)


def test_eigenvalues_are_a_conjugate_pair_with_negative_real_part() -> None:
    """Supports the spiral classification with the eigenvalues themselves."""
    summary = fhn.excitability(cfg.FHNParams())
    lambda_1, lambda_2 = summary.eigenvalues

    assert lambda_1.real < 0.0
    assert lambda_2.real < 0.0
    assert lambda_1.real == pytest.approx(lambda_2.real, abs=1.0e-12)
    assert lambda_1.imag == pytest.approx(-lambda_2.imag, abs=1.0e-12)
    assert abs(lambda_1.imag) > 0.0

    # Trace and determinant must equal the sum and product of the eigenvalues.
    assert (lambda_1 + lambda_2).real == pytest.approx(summary.trace, abs=1.0e-12)
    assert (lambda_1 * lambda_2).real == pytest.approx(summary.determinant, abs=1.0e-12)


# ---------------------------------------------------------------------------
# Frozen-w bistable reduction and the analytic front speed
# ---------------------------------------------------------------------------


def test_bistable_roots_match_brief() -> None:
    """Supports the reported upstroke roots V1, V2, V3 = -1.1994, -0.7864, 1.9858."""
    roots = fhn.bistable_roots(cfg.FHNParams())

    for obtained, expected in zip(roots, cfg.BRIEF_BISTABLE_ROOTS, strict=True):
        assert obtained == pytest.approx(expected, abs=1.0e-4)


def test_lowest_bistable_root_is_exactly_the_rest_state() -> None:
    """Supports the internal consistency of the frozen-w reduction.

    Freezing w at w* must leave V* a zero of f, by the definition of w*. If
    this failed, the reduction used for the front-speed formula would be wrong.
    """
    params = cfg.FHNParams()
    V_rest, _ = fhn.rest_state(params)
    V1, _, _ = fhn.bistable_roots(params)

    assert V1 == pytest.approx(V_rest, abs=1.0e-10)


def test_bistable_roots_sum_to_zero() -> None:
    """Supports the algebraic claim that the frozen-w cubic has no V^2 term."""
    V1, V2, V3 = fhn.bistable_roots(cfg.FHNParams())
    assert V1 + V2 + V3 == pytest.approx(0.0, abs=1.0e-10)


def test_bistable_roots_are_ordered_rest_threshold_excited() -> None:
    """Supports the physical reading of the roots used throughout the report."""
    V1, V2, V3 = fhn.bistable_roots(cfg.FHNParams())
    assert V1 < V2 < V3
    # The threshold must sit strictly between rest and the excited plateau,
    # otherwise there is no excitation gap to overcome.
    assert V1 < 0.0
    assert V3 > 0.0


def test_analytic_cv_prefactor_matches_brief() -> None:
    """Supports the reported prefactor theta / sqrt(D) = 0.9634.

    Computed from the roots at run time. The brief's 0.9634 comes from its
    rounded roots; recomputing from the exact roots gives 0.96304, a relative
    difference of 4e-4. Tolerance is set accordingly.
    """
    prefactor = fhn.analytic_cv_prefactor(cfg.FHNParams())
    assert prefactor == pytest.approx(cfg.BRIEF_CV_PREFACTOR, rel=1.0e-3)


def test_prefactor_equals_minus_three_v2_over_sqrt_six() -> None:
    """Supports the simplification theta/sqrt(D) = -3 V2 / sqrt(6).

    Because the roots sum to zero, V1 - 2 V2 + V3 = -3 V2. This identity is the
    basis of the report's argument that the front speed is controlled by how
    far the excitation threshold sits from rest.
    """
    params = cfg.FHNParams()
    _, V2, _ = fhn.bistable_roots(params)

    prefactor = fhn.analytic_cv_prefactor(params)
    # sqrt(A/2) * (-3 V2) with A = 1/3 gives -3 V2 / sqrt(6).
    expected = -3.0 * V2 / np.sqrt(6.0)

    assert prefactor == pytest.approx(expected, rel=1.0e-12)


def test_analytic_cv_matches_brief_value_at_baseline_D() -> None:
    """Supports the reported theta = 0.03047 cm/ms (about 30.5 cm/s) at D = 0.001."""
    params = cfg.FHNParams()
    D = cfg.GridParams().baseline_D

    theta = fhn.analytic_cv(D, params)

    assert theta == pytest.approx(0.03047, rel=1.0e-3)
    # Restated in cm/s, the unit used when comparing with the literature.
    assert theta * 1000.0 == pytest.approx(30.5, rel=1.0e-2)


def test_analytic_cv_scales_as_sqrt_D() -> None:
    """Supports the report's central scaling claim, theta proportional to sqrt(D).

    Quadrupling D must exactly double the analytic velocity.
    """
    params = cfg.FHNParams()
    D = cfg.GridParams().baseline_D

    assert fhn.analytic_cv(4.0 * D, params) == pytest.approx(
        2.0 * fhn.analytic_cv(D, params), rel=1.0e-12
    )


def test_analytic_cv_rejects_negative_diffusion() -> None:
    """Supports the 'fail loudly' requirement: invalid input raises, not returns NaN."""
    with pytest.raises(ValueError, match="non-negative"):
        fhn.analytic_cv(-1.0, cfg.FHNParams())


# ---------------------------------------------------------------------------
# Front thickness (the grid-resolution justification)
# ---------------------------------------------------------------------------


def test_front_thickness_justifies_the_default_grid_spacing() -> None:
    """Supports the numerical-choices argument for dx = 0.01 cm.

    The reaction-diffusion balance length sqrt(D / |f_V|) must be resolved. The
    default spacing has to be no coarser than that length, or the front is
    under-resolved and the measured velocity becomes grid-dependent.
    """
    grid = cfg.GridParams()
    solver = cfg.SolverParams()

    delta = fhn.front_thickness(grid.baseline_D, cfg.FHNParams(), solver.f_v_bound)

    # sqrt(0.001 / 3) = 0.01826 cm.
    assert delta == pytest.approx(np.sqrt(grid.baseline_D / solver.f_v_bound))
    assert grid.dx_cm <= delta


# ---------------------------------------------------------------------------
# Configuration validation ("fail loudly")
# ---------------------------------------------------------------------------


def test_config_rejects_non_integer_cell_count() -> None:
    """Supports the claim that grid setup cannot silently change the strand length."""
    with pytest.raises(ValueError, match="not an integer"):
        cfg.GridParams(length_cm=2.0, dx_cm=0.03)


def test_config_rejects_out_of_range_coupling_ratio() -> None:
    """Supports the stated domain of rho, (0, 1]."""
    with pytest.raises(ValueError, match=r"rho must lie in \(0, 1\]"):
        cfg.GapParams(rho=0.0)
    with pytest.raises(ValueError, match=r"rho must lie in \(0, 1\]"):
        cfg.GapParams(rho=1.5)


def test_config_rejects_unknown_averaging_scheme() -> None:
    """Supports the claim that only the two documented schemes can be selected."""
    with pytest.raises(ValueError, match="harmonic"):
        cfg.GapParams(averaging="geometric")  # type: ignore[arg-type]


def test_run_config_round_trips_to_a_plain_dictionary() -> None:
    """Supports the provenance requirement: every result carries its own config."""
    config = cfg.default_config()
    as_dict = config.to_dict()

    assert as_dict["fhn"]["a"] == config.fhn.a
    assert as_dict["grid"]["dx_cm"] == config.grid.dx_cm
    assert as_dict["gap"]["averaging"] == config.gap.averaging
    assert as_dict["seed"] == config.seed
    # Must be plain built-in types so that json.dump succeeds without a helper.
    import json

    json.dumps(as_dict)


def test_run_config_replace_leaves_the_original_untouched() -> None:
    """Supports the sweep design: frozen configs cannot leak between runs."""
    base = cfg.default_config()
    modified = base.replace(gap=cfg.GapParams(rho=0.2), label="blocked")

    assert base.gap.rho == 1.0
    assert modified.gap.rho == 0.2
    assert base.label == "default"
    assert modified.label == "blocked"


def test_grid_node_count_is_intervals_plus_one() -> None:
    """Supports the stated discretisation: 2 cm at dx = 0.01 cm gives 201 nodes."""
    grid = cfg.GridParams()
    assert grid.n_intervals == 200
    assert grid.n_nodes == 201
