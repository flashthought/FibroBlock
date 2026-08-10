"""The conservative diffusion operator, with sealed (no-flux) ends.

This module contains the single most important piece of numerics in the
project: the discretisation of

.. math:: \\nabla \\cdot (D(x) \\nabla V)
          \\quad\\longrightarrow\\quad
          \\frac{\\partial}{\\partial x}\\!\\left(D(x) \\frac{\\partial V}{\\partial x}\\right)

in **conservative (flux) form**, never as ``D(x) * d2V/dx2``.

Why conservative form
---------------------
Expanding the derivative gives ``D V_xx + D_x V_x``. Discretising that
non-conservatively means the numerical scheme has no exact discrete
divergence-theorem analogue, so charge is created and destroyed at any place
where ``D`` varies. In this project ``D`` varies precisely at the coupling
interface whose effect we are trying to measure, so a non-conservative scheme
would corrupt the answer at the one location that matters.

The conservative form computes fluxes on interfaces and differences them. Every
interior flux appears twice with opposite signs, so it cancels exactly in the
total, and the only charge that can leave the strand is what crosses the two
boundaries -- which the no-flux condition sets to zero.

Discretisation
--------------
Flux at the interface between nodes ``j`` and ``j+1``:

.. math:: F_{j+1/2} = D_{j+1/2}\\,\\frac{V_{j+1} - V_j}{\\Delta x}

Interior node update:

.. math:: \\mathcal{L}(V)_j = \\frac{F_{j+1/2} - F_{j-1/2}}{\\Delta x}
    = \\frac{D_{j+1/2}(V_{j+1}-V_j) - D_{j-1/2}(V_j - V_{j-1})}{\\Delta x^{2}}

Sealed ends by ghost nodes: reflecting ``V_{-1} = V_1`` and
``D_{-1/2} = D_{1/2}`` into the interior formula collapses the left-end update
to

.. math:: \\mathcal{L}(V)_0 = \\frac{2 D_{1/2}(V_1 - V_0)}{\\Delta x^{2}}

and the mirror image holds at the right end.
"""

from __future__ import annotations

import numpy as np


def interface_flux(V: np.ndarray, D_half: np.ndarray, dx: float) -> np.ndarray:
    """Diffusive flux at every interior interface.

    .. math:: F_{j+1/2} = D_{j+1/2}\\,\\frac{V_{j+1} - V_j}{\\Delta x}

    Parameters
    ----------
    V : ndarray, shape (n_nodes,)
        Nodal potential. Dimensionless.
    D_half : ndarray, shape (n_nodes - 1,)
        Interface diffusion coefficients. cm^2/ms.
    dx : float
        Node spacing. cm.

    Returns
    -------
    ndarray, shape (n_nodes - 1,)
        Flux at each interface, positive when charge flows towards larger
        ``x``. Units of cm/ms.

    Raises
    ------
    ValueError
        If the array shapes are inconsistent or ``dx`` is non-positive.

    Notes
    -----
    Fully vectorised: ``V[1:] - V[:-1]`` is a single NumPy operation over all
    interfaces. There is no Python-level loop over nodes anywhere in this
    module -- with 201 nodes and 15 000 time steps, a per-node loop would cost
    three million interpreter round trips per run.
    """
    if V.ndim != 1:
        raise ValueError(f"V must be one-dimensional, got shape {V.shape}")
    if D_half.shape != (V.size - 1,):
        raise ValueError(
            f"D_half must have shape ({V.size - 1},) for {V.size} nodes, "
            f"got {D_half.shape}"
        )
    if dx <= 0.0:
        raise ValueError(f"Node spacing must be positive, got {dx}")

    return D_half * (V[1:] - V[:-1]) / dx


def divergence(V: np.ndarray, D_half: np.ndarray, dx: float) -> np.ndarray:
    """Conservative divergence operator with sealed (no-flux) ends.

    Computes ``d/dx (D(x) dV/dx)`` at every node.

    Parameters
    ----------
    V : ndarray, shape (n_nodes,)
        Nodal potential. Dimensionless.
    D_half : ndarray, shape (n_nodes - 1,)
        Interface diffusion coefficients, from
        :func:`fibroblock.grid.half_node_diffusion`. cm^2/ms.
    dx : float
        Node spacing. cm.

    Returns
    -------
    ndarray, shape (n_nodes,)
        The diffusion term, per millisecond.

    Raises
    ------
    ValueError
        If the array shapes are inconsistent or ``dx`` is non-positive.

    Notes
    -----
    The two end formulas are not a special case bolted on afterwards: they are
    what the interior formula becomes once the reflecting ghost values
    ``V_{-1} = V_1`` and ``D_{-1/2} = D_{1/2}`` are substituted. The factor of
    2 is the ghost node's contribution, which doubles the single real
    neighbour's influence.

    Discrete conservation: with the trapezoidal weights from
    :func:`fibroblock.grid.trapezoidal_weights`, ``sum(w * divergence(V)) = 0``
    to machine precision for any ``V``. That identity is tested directly in
    ``tests/test_operators.py`` and demonstrated numerically in ``ex04``.
    """
    flux = interface_flux(V, D_half, dx)

    result = np.empty_like(V)

    # Interior nodes j = 1 .. N-1: difference the two bracketing fluxes.
    result[1:-1] = (flux[1:] - flux[:-1]) / dx

    # Left end, j = 0. Ghost node V_{-1} = V_1 makes the incoming flux the
    # negative of the outgoing one, so the two contributions add rather than
    # cancel: L_0 = 2 D_{1/2} (V_1 - V_0) / dx^2.
    result[0] = 2.0 * flux[0] / dx

    # Right end, j = N. Mirror image: L_N = 2 D_{N-1/2} (V_{N-1} - V_N) / dx^2,
    # and flux[-1] already carries (V_N - V_{N-1}), hence the minus sign.
    result[-1] = -2.0 * flux[-1] / dx

    return result


def boundary_flux(V: np.ndarray, D_nodes: np.ndarray, dx: float) -> tuple[float, float]:
    """Physical flux through each sealed end, evaluated with ghost nodes.

    Parameters
    ----------
    V : ndarray, shape (n_nodes,)
        Nodal potential. Dimensionless.
    D_nodes : ndarray, shape (n_nodes,)
        Nodal diffusion coefficients. cm^2/ms.
    dx : float
        Node spacing. cm.

    Returns
    -------
    left_flux, right_flux : float
        Flux through ``x = 0`` and ``x = L``. Both are identically zero.

    Notes
    -----
    This function exists to make the boundary condition *checkable* rather than
    merely asserted. The centred derivative at node 0 is
    ``(V_1 - V_{-1}) / (2 dx)``; the reflecting ghost sets ``V_{-1} = V_1``, so
    the numerator is exactly zero -- not zero to within truncation error, but
    zero bit-for-bit, because it is the subtraction of a float from itself.

    That exactness is the point. It is also why the substantive test of the
    boundary treatment is discrete charge conservation (which *could* fail if
    the operator were wrong) rather than this function's return value (which
    could not).
    """
    if D_nodes.shape != V.shape:
        raise ValueError(
            f"D_nodes and V must have the same shape, got {D_nodes.shape} and {V.shape}"
        )
    if dx <= 0.0:
        raise ValueError(f"Node spacing must be positive, got {dx}")

    # Reflecting ghosts: V[-1] := V[1] at the left, V[N+1] := V[N-1] at the right.
    ghost_left = V[1]
    ghost_right = V[-2]

    # Centred first derivative at the boundary node, times the local D.
    left_flux = float(D_nodes[0] * (V[1] - ghost_left) / (2.0 * dx))
    right_flux = float(D_nodes[-1] * (ghost_right - V[-2]) / (2.0 * dx))

    return left_flux, right_flux


def total_charge(V: np.ndarray, quadrature_weights: np.ndarray) -> float:
    """Total charge on the strand, ``Q = integral of V dx``.

    Parameters
    ----------
    V : ndarray, shape (n_nodes,)
        Nodal potential. Dimensionless.
    quadrature_weights : ndarray, shape (n_nodes,)
        Trapezoidal weights from
        :func:`fibroblock.grid.trapezoidal_weights`. cm.

    Returns
    -------
    float
        The integral in units of (dimensionless voltage) x cm.

    Raises
    ------
    ValueError
        If the shapes do not match.

    Notes
    -----
    With ``f = 0`` and sealed ends this quantity is conserved exactly, and that
    is the strongest single check on the spatial operator: it tests the
    interior stencil, both boundary formulas and the half-node array all at
    once, and it fails loudly if any of them is wrong.
    """
    if quadrature_weights.shape != V.shape:
        raise ValueError(
            f"Weights and V must have the same shape, got "
            f"{quadrature_weights.shape} and {V.shape}"
        )
    return float(np.dot(quadrature_weights, V))


def analytic_gaussian(
    x: np.ndarray,
    t: float,
    D: float,
    x0: float,
    sigma0: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Closed-form solution of pure diffusion from a Gaussian initial condition.

    For ``V_t = D V_xx`` on the infinite line with
    ``V(x, 0) = A exp(-(x - x0)^2 / (2 sigma0^2))``, the solution stays Gaussian
    and simply spreads:

    .. math::
        V(x, t) = A \\frac{\\sigma_0}{\\sigma(t)}
                  \\exp\\!\\left(-\\frac{(x - x_0)^{2}}{2\\sigma(t)^{2}}\\right),
        \\qquad \\sigma(t)^{2} = \\sigma_0^{2} + 2 D t

    Parameters
    ----------
    x : ndarray
        Positions. cm.
    t : float
        Time. ms.
    D : float
        Diffusion coefficient. cm^2/ms.
    x0 : float
        Centre of the initial Gaussian. cm.
    sigma0 : float
        Initial standard deviation. cm.
    amplitude : float, optional
        Initial peak height. Default 1.

    Returns
    -------
    ndarray
        The exact solution at time ``t``.

    Raises
    ------
    ValueError
        If ``sigma0`` is non-positive, or ``D`` or ``t`` is negative.

    Notes
    -----
    This is the reference for the ``ex04`` verification. It is the
    **infinite-line** solution, so it is only valid while the Gaussian has not
    yet felt the sealed ends. The experiment keeps ``sigma(t)`` small compared
    with the distance to either boundary and reports the residual boundary
    contamination, rather than assuming it away.

    The prefactor ``sigma0 / sigma(t)`` is what conserves the integral: the
    Gaussian gets wider and shorter in exact proportion, so
    ``integral V dx = A sigma0 sqrt(2 pi)`` for all time. That is the analytic
    counterpart of the discrete charge conservation tested in
    :func:`total_charge`.
    """
    if sigma0 <= 0.0:
        raise ValueError(f"sigma0 must be positive, got {sigma0}")
    if D < 0.0:
        raise ValueError(f"D must be non-negative, got {D}")
    if t < 0.0:
        raise ValueError(f"t must be non-negative, got {t}")

    # Variance grows linearly in time: sigma^2 = sigma0^2 + 2 D t. The 2 is
    # exact, from the fundamental solution of the heat equation.
    variance = sigma0**2 + 2.0 * D * t
    sigma = np.sqrt(variance)

    return amplitude * (sigma0 / sigma) * np.exp(-((x - x0) ** 2) / (2.0 * variance))


def analytic_gaussian_sealed(
    x: np.ndarray,
    t: float,
    D: float,
    x0: float,
    sigma0: float,
    length: float,
    amplitude: float = 1.0,
    n_images: int = 4,
) -> np.ndarray:
    """Exact pure-diffusion solution on a **sealed** strand, by method of images.

    :func:`analytic_gaussian` solves the problem on the infinite line, so it
    slowly stops matching a sealed strand as the profile reaches the ends. The
    no-flux problem on ``[0, L]`` is solved exactly by summing image Gaussians
    reflected in both boundaries:

    .. math::
        V(x, t) = \\sum_{n=-\\infty}^{\\infty}
            \\Big[ G\\big(x - (x_0 + 2nL)\\big)
                 + G\\big(x - (-x_0 + 2nL)\\big) \\Big]

    with ``G(u) = A (sigma_0/sigma) exp(-u^2 / 2 sigma^2)`` and
    ``sigma^2 = sigma_0^2 + 2 D t``.

    Parameters
    ----------
    x : ndarray
        Positions. cm.
    t : float
        Time. ms.
    D : float
        Diffusion coefficient. cm^2/ms.
    x0 : float
        Centre of the initial Gaussian. cm.
    sigma0 : float
        Initial standard deviation. cm.
    length : float
        Strand length ``L``. cm.
    amplitude : float, optional
        Initial peak height. Default 1.
    n_images : int, optional
        Number of image pairs on each side. Default 4, which places the
        nearest neglected image at least ``8L`` away; for the parameters used
        here that is more than 30 standard deviations, so its contribution is
        below double precision.

    Returns
    -------
    ndarray
        The exact solution for the sealed strand at time ``t``.

    Raises
    ------
    ValueError
        If ``sigma0`` or ``length`` is non-positive, or ``D`` or ``t`` is
        negative.

    Notes
    -----
    Why both solutions are kept: comparing the computed result against the
    *infinite-line* Gaussian conflates two different errors -- the
    discretisation error, and the physical difference between an open line and
    a sealed strand. Comparing against this images solution isolates the
    discretisation error, which is the thing a convergence study is supposed to
    measure. ``ex04`` reports both and shows that at 20 ms they are comparable,
    which is exactly why the distinction is worth making.

    Each reflection flips the sign of the derivative at the wall, so the summed
    field has zero slope at both ends by construction -- the same condition the
    ghost nodes impose numerically.
    """
    if sigma0 <= 0.0:
        raise ValueError(f"sigma0 must be positive, got {sigma0}")
    if length <= 0.0:
        raise ValueError(f"length must be positive, got {length}")
    if D < 0.0:
        raise ValueError(f"D must be non-negative, got {D}")
    if t < 0.0:
        raise ValueError(f"t must be non-negative, got {t}")
    if n_images < 0:
        raise ValueError(f"n_images must be non-negative, got {n_images}")

    variance = sigma0**2 + 2.0 * D * t
    sigma = np.sqrt(variance)
    peak = amplitude * (sigma0 / sigma)

    total = np.zeros_like(x, dtype=float)
    for n in range(-n_images, n_images + 1):
        shift = 2.0 * n * length
        # Direct image, reflected an even number of times.
        total += peak * np.exp(-((x - (x0 + shift)) ** 2) / (2.0 * variance))
        # Mirror image about x = 0, reflected an odd number of times.
        total += peak * np.exp(-((x - (-x0 + shift)) ** 2) / (2.0 * variance))

    return total


def uniform_second_difference(V: np.ndarray, D: float, dx: float) -> np.ndarray:
    """Plain constant-coefficient second difference, for cross-checking only.

    .. math:: D \\frac{V_{j-1} - 2 V_j + V_{j+1}}{\\Delta x^{2}}

    with the same reflecting-ghost treatment at the ends.

    Parameters
    ----------
    V : ndarray, shape (n_nodes,)
        Nodal potential. Dimensionless.
    D : float
        Constant diffusion coefficient. cm^2/ms.
    dx : float
        Node spacing. cm.

    Returns
    -------
    ndarray, shape (n_nodes,)
        The Laplacian term, per millisecond.

    Notes
    -----
    Not used by the solver. It exists so that ``tests/test_operators.py`` can
    confirm that :func:`divergence` reduces to the textbook stencil when ``D``
    is uniform -- an independent check written from the standard formula rather
    than from the flux formulation, so an error in the flux code cannot hide
    behind an identical error in the reference.
    """
    if dx <= 0.0:
        raise ValueError(f"Node spacing must be positive, got {dx}")

    result = np.empty_like(V)
    # Exact constants: the 2 and the -2 are the centred second-difference
    # stencil [1, -2, 1], not tunable numbers.
    result[1:-1] = D * (V[:-2] - 2.0 * V[1:-1] + V[2:]) / dx**2
    result[0] = D * 2.0 * (V[1] - V[0]) / dx**2
    result[-1] = D * 2.0 * (V[-2] - V[-1]) / dx**2
    return result
