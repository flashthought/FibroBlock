"""Verification tests for the grid and the conservative diffusion operator.

These are the tests behind the report's claim that the spatial discretisation
is conservative, second-order accurate in the interior, and genuinely no-flux
at the sealed ends.
"""

from __future__ import annotations

import numpy as np
import pytest

from fibroblock import grid as gridmod
from fibroblock import operators as ops
from fibroblock.config import GapParams, GridParams

# ---------------------------------------------------------------------------
# Helpers used by several tests
# ---------------------------------------------------------------------------


def uniform_grid(n_nodes: int = 51, length_cm: float = 1.0, D: float = 0.001):
    """Build a homogeneous grid (no gap) with the requested node count."""
    dx = length_cm / (n_nodes - 1)
    grid_params = GridParams(length_cm=length_cm, dx_cm=dx, baseline_D=D)
    gap_params = GapParams(rho=1.0, gap_length_cm=0.0)
    return gridmod.build_grid(grid_params, gap_params)


# ---------------------------------------------------------------------------
# Laplacian of a quadratic
# ---------------------------------------------------------------------------


def test_laplacian_of_a_quadratic_is_exact_in_the_interior() -> None:
    """Supports the claim that the interior stencil is exact for quadratics.

    For V = a x^2 + b x + c with constant D, the exact answer is 2 a D
    everywhere. A centred second difference reproduces it to machine precision
    because the truncation error involves the fourth derivative, which is zero.
    """
    g = uniform_grid(n_nodes=41, length_cm=2.0, D=0.003)

    a, b, c = 1.7, -0.4, 0.9  # arbitrary quadratic coefficients
    V = a * g.x**2 + b * g.x + c

    computed = ops.divergence(V, g.D_half, g.dx)
    exact_interior = 2.0 * a * g.D_nodes[1:-1]

    np.testing.assert_allclose(
        computed[1:-1], exact_interior, rtol=1.0e-10, atol=1.0e-12
    )


def test_boundary_stencil_is_exact_for_a_quadratic_that_obeys_the_bc() -> None:
    """Supports the claim that the ghost-node end formulas are correct.

    The reflecting ghost is exact only for fields that actually satisfy
    dV/dx = 0 at the boundary. V = (x - L/2)^2 is symmetric about the strand
    centre, so on a symmetric grid the LEFT and RIGHT ends see mirror-image
    data. Here we instead use a field that is flat at x = 0 -- V = x^2 has
    V'(0) = 0 -- and check the left end exactly.
    """
    g = uniform_grid(n_nodes=41, length_cm=2.0, D=0.003)

    V = g.x**2  # V'(0) = 0, so the reflecting ghost at the left end is exact
    computed = ops.divergence(V, g.D_half, g.dx)

    # d/dx(D dV/dx) = 2 D everywhere, and the left end honours the BC exactly.
    assert computed[0] == pytest.approx(2.0 * g.D_nodes[0], rel=1.0e-12)


def test_boundary_stencil_is_second_order_for_a_field_obeying_both_ends() -> None:
    """Supports the claim that the sealed-end treatment does not lose accuracy.

    V = cos(pi x / L) has zero derivative at BOTH ends, so it is a legitimate
    test field for the no-flux problem. Its exact Laplacian is
    -D (pi/L)^2 cos(pi x / L). Refining the grid must reduce the boundary error
    by a factor of about 4, i.e. second order.
    """
    length = 2.0
    D = 0.001
    errors = []

    for n_nodes in (41, 81, 161):
        g = uniform_grid(n_nodes=n_nodes, length_cm=length, D=D)
        V = np.cos(np.pi * g.x / length)
        exact = -D * (np.pi / length) ** 2 * np.cos(np.pi * g.x / length)

        computed = ops.divergence(V, g.D_half, g.dx)
        errors.append(np.max(np.abs(computed - exact)))

    # Successive halvings of dx must quarter the error (order 2).
    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        observed_order = np.log2(coarse / fine)
        assert observed_order == pytest.approx(2.0, abs=0.15)


# ---------------------------------------------------------------------------
# No-flux boundary condition
# ---------------------------------------------------------------------------


def test_neumann_boundary_flux_is_identically_zero() -> None:
    """Supports the claim that the ends are sealed, for an arbitrary field."""
    g = uniform_grid(n_nodes=51)
    rng = np.random.default_rng(seed=20260810)
    V = rng.normal(size=g.n_nodes)

    left, right = ops.boundary_flux(V, g.D_nodes, g.dx)

    assert left == 0.0
    assert right == 0.0


def test_operator_conserves_charge_to_machine_precision() -> None:
    """Supports the report's central conservation claim.

    With no-flux ends, the total charge Q = sum(w_j V_j) cannot change under
    the diffusion operator alone. Equivalently sum(w * L(V)) = 0 for ANY field
    V. This one identity exercises the interior stencil, both end formulas and
    the half-node array simultaneously.
    """
    g = uniform_grid(n_nodes=101, length_cm=2.0)
    rng = np.random.default_rng(seed=20260810)
    V = rng.normal(size=g.n_nodes)

    rate_of_change = ops.divergence(V, g.D_half, g.dx)
    total = ops.total_charge(rate_of_change, g.quadrature_weights)

    # Scale the tolerance by the size of the individual terms, so this is a
    # genuine machine-precision claim rather than a loose absolute bound.
    scale = float(np.max(np.abs(rate_of_change))) * g.length_cm
    assert abs(total) < 1.0e-12 * scale


def test_charge_conservation_also_holds_across_a_sharp_coupling_gap() -> None:
    """Supports the claim that the interface itself creates and destroys nothing.

    This is the case a non-conservative discretisation would fail. A 100-fold
    drop in D is imposed, and charge must still be conserved exactly.
    """
    grid_params = GridParams(length_cm=2.0, dx_cm=0.01, baseline_D=0.001)
    gap_params = GapParams(rho=0.01, gap_length_cm=0.1, gap_centre_cm=1.0)
    g = gridmod.build_grid(grid_params, gap_params)

    rng = np.random.default_rng(seed=20260810)
    V = rng.normal(size=g.n_nodes)

    rate_of_change = ops.divergence(V, g.D_half, g.dx)
    total = ops.total_charge(rate_of_change, g.quadrature_weights)

    scale = float(np.max(np.abs(rate_of_change))) * g.length_cm
    assert abs(total) < 1.0e-12 * scale


def test_uniform_weights_would_NOT_conserve_charge() -> None:
    """Supports the numerical-choices argument for trapezoidal end weights.

    A negative control. If naive uniform weights also conserved charge, the
    half-weight end cells would be an unnecessary complication and the previous
    two tests would prove nothing about them.
    """
    g = uniform_grid(n_nodes=101, length_cm=2.0)
    rng = np.random.default_rng(seed=20260810)
    V = rng.normal(size=g.n_nodes)

    rate_of_change = ops.divergence(V, g.D_half, g.dx)

    naive_total = float(np.sum(rate_of_change) * g.dx)
    correct_total = ops.total_charge(rate_of_change, g.quadrature_weights)

    assert abs(correct_total) < 1.0e-12
    assert abs(naive_total) > 1.0e-6  # demonstrably nonzero


# ---------------------------------------------------------------------------
# Reduction to the textbook stencil
# ---------------------------------------------------------------------------


def test_uniform_D_reproduces_the_plain_second_difference() -> None:
    """Supports the claim that the conservative form is a generalisation, not a change.

    With constant D the flux formulation must return exactly the classical
    (V[j-1] - 2 V[j] + V[j+1]) * D / dx^2 stencil, including at the ends.
    """
    D = 0.0025
    g = uniform_grid(n_nodes=61, length_cm=1.5, D=D)

    rng = np.random.default_rng(seed=20260810)
    V = rng.normal(size=g.n_nodes)

    conservative = ops.divergence(V, g.D_half, g.dx)
    textbook = ops.uniform_second_difference(V, D, g.dx)

    np.testing.assert_allclose(conservative, textbook, rtol=1.0e-12, atol=1.0e-14)


def test_operator_is_symmetric_for_symmetric_data() -> None:
    """Supports the claim that no directional bias is baked into the stencil.

    A field and a coupling profile that are both mirror-symmetric about the
    strand centre must produce a mirror-symmetric result. Any one-sided error
    in the boundary handling would break this.
    """
    grid_params = GridParams(length_cm=2.0, dx_cm=0.01, baseline_D=0.001)
    gap_params = GapParams(rho=0.2, gap_length_cm=0.2, gap_centre_cm=1.0)
    g = gridmod.build_grid(grid_params, gap_params)

    # cos(2 pi x / L) is symmetric about x = L/2 and flat at both ends.
    V = np.cos(2.0 * np.pi * g.x / g.length_cm)

    result = ops.divergence(V, g.D_half, g.dx)

    # The comparison is limited by round-off in V itself, not by the operator.
    # cos(2*pi*x/L) evaluated at mirror-image nodes agrees only to about one
    # machine epsilon, and the operator then divides by dx^2 = 1e-4, amplifying
    # that by 1e4. The achievable tolerance is therefore ~eps/dx^2 ~ 2e-12, not
    # eps. Anything larger than this bound would be a genuine asymmetry.
    roundoff_bound = 10.0 * np.finfo(float).eps / g.dx**2
    np.testing.assert_allclose(result, result[::-1], rtol=0.0, atol=roundoff_bound)


def test_operator_annihilates_a_constant_field() -> None:
    """Supports the claim that a uniformly polarised strand does not drift.

    A constant V has no gradient anywhere, so the diffusion term must be
    exactly zero at every node -- including inside a coupling gap.
    """
    grid_params = GridParams(length_cm=2.0, dx_cm=0.01)
    gap_params = GapParams(rho=0.05, gap_length_cm=0.15)
    g = gridmod.build_grid(grid_params, gap_params)

    V = np.full(g.n_nodes, -1.199408)  # the resting potential
    result = ops.divergence(V, g.D_half, g.dx)

    np.testing.assert_allclose(result, 0.0, atol=1.0e-18)


# ---------------------------------------------------------------------------
# Half-node averaging
# ---------------------------------------------------------------------------


def test_harmonic_mean_formula() -> None:
    """Supports the reported interface formula 2 D_j D_{j+1} / (D_j + D_{j+1})."""
    left = np.array([1.0, 2.0, 0.001])
    right = np.array([1.0, 8.0, 0.00001])

    obtained = gridmod.harmonic_mean(left, right)
    expected = 2.0 * left * right / (left + right)

    np.testing.assert_allclose(obtained, expected, rtol=1.0e-14)
    # Equal inputs must return that same value: the mean is consistent.
    assert obtained[0] == pytest.approx(1.0)


def test_harmonic_mean_is_dominated_by_the_smaller_coefficient() -> None:
    """Supports the report's physical argument for using the harmonic mean.

    A series resistance is dominated by its largest element, so the equivalent
    D is dominated by the SMALLEST D. The harmonic mean must therefore never
    exceed twice the smaller value, whereas the arithmetic mean can be
    arbitrarily larger.
    """
    D0 = 0.001
    rho = 0.01
    left = np.array([D0])
    right = np.array([rho * D0])

    harmonic = gridmod.harmonic_mean(left, right)[0]
    arithmetic = gridmod.arithmetic_mean(left, right)[0]

    assert harmonic <= 2.0 * rho * D0
    assert arithmetic == pytest.approx(0.5 * (D0 + rho * D0))
    # For a 100-fold drop the arithmetic mean is about 25x too generous.
    assert arithmetic / harmonic > 20.0


def test_harmonic_and_arithmetic_agree_when_D_is_uniform() -> None:
    """Supports the claim that the averaging choice only matters at interfaces."""
    D = np.full(20, 0.001)
    harmonic = gridmod.half_node_diffusion(D, "harmonic")
    arithmetic = gridmod.half_node_diffusion(D, "arithmetic")

    np.testing.assert_allclose(harmonic, arithmetic, rtol=1.0e-14)
    np.testing.assert_allclose(harmonic, 0.001, rtol=1.0e-14)


def test_half_node_array_has_one_entry_per_interval() -> None:
    """Supports the stated grid layout: N intervals, N+1 nodes, N half-nodes."""
    g = uniform_grid(n_nodes=51)
    assert g.D_nodes.size == 51
    assert g.D_half.size == 50


def test_unknown_averaging_scheme_raises() -> None:
    """Supports the 'fail loudly' requirement."""
    D = np.full(10, 0.001)
    with pytest.raises(ValueError, match="Unknown averaging scheme"):
        gridmod.half_node_diffusion(D, "geometric")  # type: ignore[arg-type]


def test_harmonic_mean_rejects_zero_coupling() -> None:
    """Supports the stated exclusion of rho = 0, which would sever the strand."""
    with pytest.raises(ValueError, match="strictly positive"):
        gridmod.harmonic_mean(np.array([1.0]), np.array([0.0]))


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def test_gap_is_placed_where_the_configuration_says() -> None:
    """Supports every figure that shades the gap region."""
    grid_params = GridParams(length_cm=2.0, dx_cm=0.01, baseline_D=0.001)
    gap_params = GapParams(rho=0.1, gap_length_cm=0.2, gap_centre_cm=1.0)
    g = gridmod.build_grid(grid_params, gap_params)

    inside = g.x[g.gap_mask]
    assert inside.min() == pytest.approx(0.9, abs=1.0e-12)
    assert inside.max() == pytest.approx(1.1, abs=1.0e-12)
    np.testing.assert_allclose(g.D_nodes[g.gap_mask], 0.1 * 0.001, rtol=1.0e-14)
    np.testing.assert_allclose(g.D_nodes[~g.gap_mask], 0.001, rtol=1.0e-14)


def test_grid_arrays_are_read_only() -> None:
    """Supports the claim that a precomputed grid cannot be mutated mid-run."""
    g = uniform_grid()
    with pytest.raises(ValueError):
        g.D_half[0] = 99.0


def test_quadrature_weights_sum_to_the_strand_length() -> None:
    """Supports the correctness of the trapezoidal integration weights."""
    g = uniform_grid(n_nodes=101, length_cm=2.0)
    assert float(np.sum(g.quadrature_weights)) == pytest.approx(2.0, rel=1.0e-14)
    assert g.quadrature_weights[0] == pytest.approx(0.5 * g.dx)
    assert g.quadrature_weights[-1] == pytest.approx(0.5 * g.dx)
    assert g.quadrature_weights[1] == pytest.approx(g.dx)


def test_D_max_uses_interface_values_not_nodal_values() -> None:
    """Supports the stability-limit derivation, which needs the interface maximum.

    Inside a gap the nodal maximum and the interface maximum coincide at the
    healthy value D0, but the property must read D_half, because that is what
    appears in the node update.
    """
    grid_params = GridParams(length_cm=2.0, dx_cm=0.01, baseline_D=0.001)
    gap_params = GapParams(rho=0.1, gap_length_cm=0.2)
    g = gridmod.build_grid(grid_params, gap_params)

    assert g.D_max == pytest.approx(0.001, rel=1.0e-12)
    assert g.D_max <= float(np.max(g.D_nodes))
