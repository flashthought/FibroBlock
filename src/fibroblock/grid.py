"""Spatial grid, the diffusion profile ``D(x)``, and half-node averaging.

Everything here is precomputed **once per run, outside the time loop**. The
half-node diffusion array in particular is fixed for the whole simulation;
recomputing it every step would be both slow and pointless.

Grid layout
-----------
Nodes sit at ``x_j = j * dx`` for ``j = 0 .. N``, so a 2.0 cm strand at
``dx = 0.01 cm`` has ``N = 200`` intervals and 201 nodes. Half-nodes sit at
``x_{j+1/2} = (j + 1/2) * dx`` for ``j = 0 .. N-1``, so there are ``N``
half-nodes -- one per interval, one fewer than the number of nodes.

::

    node      0     1     2   ...   N-1    N
    x         0    dx    2dx        L-dx   L
    half-node    0+1/2 1+1/2  ...      N-1+1/2
    index          0     1              N-1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fibroblock.config import AveragingScheme, GapParams, GridParams


def build_x(grid: GridParams) -> np.ndarray:
    """Build the node coordinate array.

    Parameters
    ----------
    grid : GridParams
        Grid description supplying ``length_cm`` and ``dx_cm``.

    Returns
    -------
    ndarray, shape (n_nodes,)
        Node positions in cm, from 0 to ``L`` inclusive.

    Notes
    -----
    ``numpy.linspace`` is used rather than ``numpy.arange`` because linspace
    guarantees the final value is exactly ``L``. With arange, accumulated
    floating-point error can leave the last node a few ULPs short of ``L``,
    which then quietly corrupts any measurement that keys off distance.
    """
    return np.linspace(0.0, grid.length_cm, grid.n_nodes)


def diffusion_profile(
    x: np.ndarray,
    grid: GridParams,
    gap: GapParams,
) -> np.ndarray:
    r"""Evaluate the piecewise-constant diffusion coefficient at each node.

    .. math::
        D(x) = \begin{cases}
            \rho D_0 & x_{gap} \le x \le x_{gap} + L_{gap} \\
            D_0       & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    x : ndarray
        Node positions. cm.
    grid : GridParams
        Supplies the baseline coefficient ``D0``. cm^2/ms.
    gap : GapParams
        Supplies ``rho``, the gap length and the gap centre.

    Returns
    -------
    ndarray, shape (x.size,)
        Nodal diffusion coefficients in cm^2/ms.

    Notes
    -----
    The interval is treated as closed on both sides, with a tolerance of a
    millionth of a node spacing. Without the tolerance, a gap edge that lands
    exactly on a node (which is the common case, since the gap is centred at
    1.0 cm on a grid whose nodes are multiples of 0.01 cm) would be included or
    excluded depending on the last bit of a floating-point subtraction.

    Because ``D`` is defined at nodes, a gap edge falling on a node makes that
    node's whole cell part of the gap. The discrete gap width is therefore
    ``L_gap`` to within one ``dx``. Experiment ``ex08`` includes ``dx`` in the
    sensitivity sweep precisely so this discretisation effect is quantified
    rather than assumed negligible.
    """
    if x.size < 2:
        raise ValueError(f"Need at least two nodes to build a profile, got {x.size}")

    # Tolerance is a tiny fraction of the node spacing: large enough to absorb
    # round-off in the gap edge arithmetic, far too small to move a real edge.
    spacing = float(x[1] - x[0])
    tolerance = 1.0e-6 * spacing

    inside_gap = (x >= gap.gap_start_cm - tolerance) & (
        x <= gap.gap_end_cm + tolerance
    )

    D = np.full_like(x, grid.baseline_D)
    D[inside_gap] = gap.rho * grid.baseline_D
    return D


def harmonic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    r"""Harmonic mean of adjacent diffusion coefficients.

    .. math:: D_{j+1/2} = \frac{2 D_j D_{j+1}}{D_j + D_{j+1}}

    Parameters
    ----------
    left, right : ndarray
        Diffusion coefficients on either side of an interface. cm^2/ms. Both
        must be strictly positive.

    Returns
    -------
    ndarray
        Interface coefficients. cm^2/ms.

    Raises
    ------
    ValueError
        If any input is non-positive. A zero coefficient makes the harmonic
        mean zero, which severs the strand; that is a different problem, and it
        should be raised rather than silently produced.

    Notes
    -----
    This is the physically correct interface value. The diffusion coefficient
    is inversely proportional to axial resistance, ``D ~ 1/r``, and resistances
    in series add. Averaging the *resistances* (arithmetic in ``1/D``) and
    inverting gives the harmonic mean of ``D``.

    The distinction only matters where ``D`` jumps sharply -- which is exactly
    the interface this project is built to study. The arithmetic mean
    over-predicts coupling there and shifts the measured block threshold. For a
    100-fold drop, harmonic gives roughly ``2 rho D0`` while arithmetic gives
    roughly ``D0 / 2``: a factor of about 25 too generous.
    """
    if np.any(left <= 0.0) or np.any(right <= 0.0):
        raise ValueError(
            "Harmonic mean requires strictly positive diffusion coefficients; "
            "a zero coefficient would sever the strand."
        )
    # The 2 is exact algebra from 1/D_half = (1/2)(1/D_left + 1/D_right).
    return 2.0 * left * right / (left + right)


def arithmetic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    r"""Arithmetic mean of adjacent diffusion coefficients.

    .. math:: D_{j+1/2} = \tfrac{1}{2}(D_j + D_{j+1})

    Parameters
    ----------
    left, right : ndarray
        Diffusion coefficients on either side of an interface. cm^2/ms.

    Returns
    -------
    ndarray
        Interface coefficients. cm^2/ms.

    Notes
    -----
    Provided **only** so that the effect of the wrong choice can be measured
    and reported as a numerical-choice sensitivity result. It is not the
    default and should not be used for the headline numbers. See
    :func:`harmonic_mean` for why.
    """
    # The 1/2 is exact algebra, not a tunable weight.
    return 0.5 * (left + right)


def half_node_diffusion(
    D_nodes: np.ndarray,
    scheme: AveragingScheme = "harmonic",
) -> np.ndarray:
    """Build interface (half-node) diffusion coefficients from nodal values.

    Parameters
    ----------
    D_nodes : ndarray, shape (n_nodes,)
        Nodal diffusion coefficients. cm^2/ms.
    scheme : {"harmonic", "arithmetic"}, optional
        Averaging scheme. Harmonic by default, for the reason given in
        :func:`harmonic_mean`.

    Returns
    -------
    ndarray, shape (n_nodes - 1,)
        Coefficient at each half-node ``x_{j+1/2}``, ``j = 0 .. N-1``. One
        value per interval.

    Raises
    ------
    ValueError
        If fewer than two nodes are supplied, or the scheme is unknown.

    Notes
    -----
    Called once per run and cached on the :class:`Grid` object. It is never
    called inside the time loop.
    """
    if D_nodes.size < 2:
        raise ValueError(
            f"Need at least two nodes to form interfaces, got {D_nodes.size}"
        )

    left = D_nodes[:-1]
    right = D_nodes[1:]

    if scheme == "harmonic":
        return harmonic_mean(left, right)
    if scheme == "arithmetic":
        return arithmetic_mean(left, right)
    raise ValueError(
        f"Unknown averaging scheme {scheme!r}; expected 'harmonic' or 'arithmetic'."
    )


@dataclass(frozen=True)
class Grid:
    """Everything geometric about a run, precomputed once.

    Attributes
    ----------
    x : ndarray, shape (n_nodes,)
        Node positions. cm.
    dx : float
        Node spacing. cm.
    D_nodes : ndarray, shape (n_nodes,)
        Nodal diffusion coefficients. cm^2/ms.
    D_half : ndarray, shape (n_nodes - 1,)
        Interface diffusion coefficients. cm^2/ms.
    quadrature_weights : ndarray, shape (n_nodes,)
        Trapezoidal weights used to integrate a nodal field over the strand.
        The two end nodes carry half weight. See :attr:`quadrature_weights`
        notes below -- this is not cosmetic, it is what makes discrete charge
        conservation exact.
    gap_mask : ndarray of bool, shape (n_nodes,)
        True at nodes inside the reduced-coupling gap. Used only for plotting
        and reporting.

    Notes
    -----
    The arrays are made read-only (``setflags(write=False)``) after
    construction. A frozen dataclass would otherwise still allow the *contents*
    of its arrays to be mutated, and a stray in-place edit of ``D_half`` in the
    middle of a sweep would be very hard to find.
    """

    x: np.ndarray
    dx: float
    D_nodes: np.ndarray
    D_half: np.ndarray
    quadrature_weights: np.ndarray
    gap_mask: np.ndarray

    @property
    def n_nodes(self) -> int:
        """Number of grid nodes."""
        return int(self.x.size)

    @property
    def length_cm(self) -> float:
        """Strand length in cm."""
        return float(self.x[-1] - self.x[0])

    @property
    def D_max(self) -> float:
        """Largest interface diffusion coefficient. cm^2/ms.

        This, not the nodal maximum, is what enters the explicit stability
        limit: the update at node ``j`` uses ``D_{j-1/2}`` and ``D_{j+1/2}``.
        """
        return float(np.max(self.D_half))


def trapezoidal_weights(n_nodes: int, dx: float) -> np.ndarray:
    r"""Quadrature weights for integrating a nodal field over the strand.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    dx : float
        Node spacing. cm.

    Returns
    -------
    ndarray, shape (n_nodes,)
        Weights ``[dx/2, dx, dx, ..., dx, dx/2]``, summing to ``L``.

    Raises
    ------
    ValueError
        If fewer than two nodes are requested or ``dx`` is non-positive.

    Notes
    -----
    The half weights at the ends are essential, not decorative. In the
    finite-volume reading of this discretisation, the end nodes own cells of
    half the width, because half of their control volume lies outside the
    domain. With these weights the total charge

    .. math:: Q = \sum_j w_j V_j

    is conserved to machine precision by the no-flux operator. With uniform
    weights it is not: the residual is exactly the spurious flux implied by
    giving the end nodes full-width cells. ``ex04`` demonstrates this.
    """
    if n_nodes < 2:
        raise ValueError(f"Need at least two nodes, got {n_nodes}")
    if dx <= 0.0:
        raise ValueError(f"Node spacing must be positive, got {dx}")

    weights = np.full(n_nodes, dx)
    # Half weight at each end: the end cells extend only dx/2 into the domain.
    weights[0] = 0.5 * dx
    weights[-1] = 0.5 * dx
    return weights


def build_grid(grid_params: GridParams, gap_params: GapParams) -> Grid:
    """Assemble the complete precomputed grid for a run.

    Parameters
    ----------
    grid_params : GridParams
        Strand length, node spacing and baseline coupling.
    gap_params : GapParams
        Reduced-coupling patch and the averaging scheme to use at interfaces.

    Returns
    -------
    Grid
        Node positions, nodal and interface diffusion coefficients, quadrature
        weights and the gap mask -- all read-only.

    Notes
    -----
    This is the only place the half-node array is built. Everything downstream
    reads ``grid.D_half``, so the "precompute outside the time loop"
    requirement is satisfied structurally rather than by discipline.
    """
    x = build_x(grid_params)
    dx = grid_params.dx_cm

    D_nodes = diffusion_profile(x, grid_params, gap_params)
    D_half = half_node_diffusion(D_nodes, gap_params.averaging)
    weights = trapezoidal_weights(x.size, dx)

    # Recompute the mask from the nodal profile rather than from the geometry a
    # second time, so the mask can never disagree with the coefficients.
    gap_mask = D_nodes < grid_params.baseline_D

    for array in (x, D_nodes, D_half, weights, gap_mask):
        array.setflags(write=False)

    return Grid(
        x=x,
        dx=dx,
        D_nodes=D_nodes,
        D_half=D_half,
        quadrature_weights=weights,
        gap_mask=gap_mask,
    )
