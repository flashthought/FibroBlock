"""Hand-coded explicit time integrators and the analytic stability limit.

Why the integrators are hand-coded
----------------------------------
Part (c) of the assignment requires demonstrating the stability limit of *this*
scheme. ``scipy.integrate.solve_ivp`` and every other adaptive solver would
detect the instability and silently shrink the step until it went away, which
is precisely the behaviour that would hide the result being asked for. Explicit
Euler is therefore the primary scheme, written out in full, with RK4 available
only as an accuracy comparison.

Von Neumann stability analysis
------------------------------
Linearise the reaction-diffusion system about a state where
``df/dV = f_V`` and substitute a Fourier mode ``V_j^n = g^n e^{i k j \\Delta x}``
into the explicit-Euler update

.. math::
    V_j^{n+1} = V_j^n + \\Delta t \\left[
        \\frac{D (V_{j+1}^n - 2V_j^n + V_{j-1}^n)}{\\Delta x^{2}} + f_V V_j^n
    \\right]

The second difference of the Fourier mode gives a factor
``2(cos(k dx) - 1) = -4 sin^2(k dx / 2)``, so the amplification factor is

.. math::
    g(k) = 1 + \\Delta t\\left[-\\frac{4D}{\\Delta x^{2}}
           \\sin^{2}\\!\\left(\\frac{k \\Delta x}{2}\\right) + f_V\\right]

Stability requires ``|g| <= 1`` for every mode. The binding constraint is
``g >= -1``, and the worst case is the mode that maximises the bracket's
magnitude: ``sin^2 = 1``, i.e. ``k dx / 2 = pi / 2``, which is

.. math:: k = \\frac{\\pi}{\\Delta x}

the **checkerboard mode**, alternating sign from node to node -- the shortest
wavelength the grid can represent. Imposing ``g(pi/dx) >= -1`` with the worst
reaction contribution ``|f_V|_{\\max}`` gives

.. math::
    \\boxed{\\Delta t \\le \\frac{2}{\\dfrac{4 D_{\\max}}{\\Delta x^{2}}
            + |f_V|_{\\max}}}

Ignoring the reaction term entirely gives the familiar pure-diffusion limit
``dt <= dx^2 / (2D)``, which is **too optimistic**: at the assignment's values
it says 0.05 ms where the true limit is 0.04651 ms, an overestimate of 7.5 %.
Experiment ``ex02`` reports both and shows the instability appearing as the
checkerboard mode, exactly as the analysis predicts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Type of the right-hand-side function the integrators drive.
# Signature: rhs(t_ms, V, w) -> (dV/dt, dw/dt).
RHSFunction = Callable[
    [float, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]


# ---------------------------------------------------------------------------
# Stability limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityLimits:
    """The two candidate explicit-Euler step limits, and the step in use.

    Attributes
    ----------
    reaction_diffusion_dt_ms : float
        The correct limit, ``2 / (4 D_max / dx^2 + |f_V|_max)``. ms.
    pure_diffusion_dt_ms : float
        The limit if the reaction term is ignored, ``dx^2 / (2 D_max)``. ms.
        Always the larger of the two, hence always over-optimistic.
    relative_overestimate : float
        How much too large the pure-diffusion limit is, as a fraction. About
        0.075 (7.5 %) at the assignment's values.
    diffusion_number : float
        ``4 D_max / dx^2``, in 1/ms. The diffusive contribution to the limit.
    reaction_bound : float
        ``|f_V|_max``, in 1/ms. The reaction contribution.
    dt_ms : float
        The step actually configured. ms.
    safety_factor : float
        ``reaction_diffusion_dt_ms / dt_ms``. Greater than 1 means stable;
        the default configuration gives about 2.3.
    is_stable : bool
        True when ``dt_ms`` does not exceed the reaction-diffusion limit.
    """

    reaction_diffusion_dt_ms: float
    pure_diffusion_dt_ms: float
    relative_overestimate: float
    diffusion_number: float
    reaction_bound: float
    dt_ms: float
    safety_factor: float
    is_stable: bool


def explicit_euler_dt_limit(D_max: float, dx: float, f_v_bound: float) -> float:
    """Explicit-Euler stability limit for the reaction-diffusion system.

    .. math:: \\Delta t \\le \\frac{2}{4 D_{\\max}/\\Delta x^{2} + |f_V|_{\\max}}

    Parameters
    ----------
    D_max : float
        Largest interface diffusion coefficient on the grid. cm^2/ms.
    dx : float
        Node spacing. cm.
    f_v_bound : float
        Bound on ``|df/dV| = |1 - V^2|`` over an action potential. 1/ms.

    Returns
    -------
    float
        Maximum stable time step, in ms. About 0.04651 ms at
        ``D = 0.001 cm^2/ms``, ``dx = 0.01 cm``, ``|f_V| = 3``.

    Raises
    ------
    ValueError
        If ``D_max`` or ``f_v_bound`` is negative, or ``dx`` is non-positive.

    Notes
    -----
    The ``4`` comes from the checkerboard mode: the second difference of an
    alternating field is ``-4`` times the field, not ``-2``. Forgetting this and
    writing ``2 D/dx^2`` is the single most common slip in this analysis, and it
    is why the pure-diffusion limit quoted alongside it is 7.5 % too generous.

    The ``2`` in the numerator comes from the binding constraint being
    ``g >= -1`` rather than ``g <= 1``: the amplification factor is driven
    negative, and it is the excursion below ``-1`` that blows up.

    The two contributions add because both push ``g`` in the same direction on
    the checkerboard mode. Treating them separately and taking the smaller of
    the two limits would be less restrictive, and wrong.
    """
    if D_max < 0.0:
        raise ValueError(f"D_max must be non-negative, got {D_max}")
    if dx <= 0.0:
        raise ValueError(f"dx must be positive, got {dx}")
    if f_v_bound < 0.0:
        raise ValueError(f"f_v_bound must be non-negative, got {f_v_bound}")

    # 4 D / dx^2 : diffusive eigenvalue magnitude at the checkerboard mode.
    diffusion_number = 4.0 * D_max / dx**2
    denominator = diffusion_number + f_v_bound

    if denominator == 0.0:
        # No diffusion and no reaction: nothing can go unstable.
        return float("inf")

    # The 2 is from |g| <= 1 with the binding side being g >= -1.
    return 2.0 / denominator


def pure_diffusion_dt_limit(D_max: float, dx: float) -> float:
    """Stability limit if the reaction term is (incorrectly) ignored.

    .. math:: \\Delta t \\le \\frac{\\Delta x^{2}}{2 D_{\\max}}

    Parameters
    ----------
    D_max : float
        Largest interface diffusion coefficient. cm^2/ms.
    dx : float
        Node spacing. cm.

    Returns
    -------
    float
        The over-optimistic limit, in ms. 0.05 ms at the assignment's values.

    Raises
    ------
    ValueError
        If ``D_max`` is negative or ``dx`` is non-positive.

    Notes
    -----
    Reported alongside the correct limit specifically so the report can quantify
    how misleading it is. It is the same expression with ``f_V`` set to zero:
    ``2 / (4D/dx^2) = dx^2 / (2D)``.
    """
    if D_max < 0.0:
        raise ValueError(f"D_max must be non-negative, got {D_max}")
    if dx <= 0.0:
        raise ValueError(f"dx must be positive, got {dx}")
    if D_max == 0.0:
        return float("inf")
    return dx**2 / (2.0 * D_max)


def stability_limits(
    D_max: float, dx: float, f_v_bound: float, dt: float
) -> StabilityLimits:
    """Compute both stability limits and compare them with the configured step.

    Parameters
    ----------
    D_max : float
        Largest interface diffusion coefficient. cm^2/ms.
    dx : float
        Node spacing. cm.
    f_v_bound : float
        Bound on ``|df/dV|``. 1/ms.
    dt : float
        The time step in use. ms.

    Returns
    -------
    StabilityLimits
        Both limits, their ratio, and whether ``dt`` is safe.
    """
    reaction_diffusion = explicit_euler_dt_limit(D_max, dx, f_v_bound)
    pure_diffusion = pure_diffusion_dt_limit(D_max, dx)

    if np.isfinite(reaction_diffusion) and reaction_diffusion > 0.0:
        overestimate = pure_diffusion / reaction_diffusion - 1.0
        safety = reaction_diffusion / dt
    else:
        overestimate = 0.0
        safety = float("inf")

    return StabilityLimits(
        reaction_diffusion_dt_ms=reaction_diffusion,
        pure_diffusion_dt_ms=pure_diffusion,
        relative_overestimate=float(overestimate),
        diffusion_number=4.0 * D_max / dx**2,
        reaction_bound=f_v_bound,
        dt_ms=dt,
        safety_factor=float(safety),
        is_stable=bool(dt <= reaction_diffusion),
    )


def checkerboard_mode(n_nodes: int, amplitude: float = 1.0) -> np.ndarray:
    """The most unstable Fourier mode, ``k = pi / dx``.

    Parameters
    ----------
    n_nodes : int
        Number of grid nodes.
    amplitude : float, optional
        Peak magnitude. Default 1.

    Returns
    -------
    ndarray, shape (n_nodes,)
        ``[+A, -A, +A, -A, ...]``.

    Notes
    -----
    This is the shortest wavelength representable on the grid (two nodes per
    period) and the mode the von Neumann analysis identifies as binding. When
    ``dt`` exceeds the limit, the blow-up is not generic noise: it has this
    specific node-to-node alternating structure, which is what makes it
    identifiable. ``ex02`` projects the numerical solution onto this vector to
    show that the growing component really is the predicted mode.
    """
    if n_nodes < 2:
        raise ValueError(f"Need at least two nodes, got {n_nodes}")
    # (-1)^j alternation: exp(i * pi * j) evaluated on the grid.
    indices = np.arange(n_nodes)
    return amplitude * (-1.0) ** indices


def checkerboard_amplitude(V: np.ndarray) -> float:
    """Amplitude of the checkerboard component of a field.

    Parameters
    ----------
    V : ndarray
        Nodal field.

    Returns
    -------
    float
        The projection of ``V`` onto the normalised checkerboard mode.

    Notes
    -----
    Computed as ``|<V, s>| / n`` with ``s = (-1)^j``. If the instability really
    is the predicted mode, this quantity grows geometrically while the smooth
    part of the solution does not.
    """
    mode = checkerboard_mode(V.size)
    return float(abs(np.dot(V, mode)) / V.size)


# ---------------------------------------------------------------------------
# Time integrators
# ---------------------------------------------------------------------------


def euler_step(
    t: float,
    V: np.ndarray,
    w: np.ndarray,
    dt: float,
    rhs: RHSFunction,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one step with explicit (forward) Euler.

    .. math:: y^{n+1} = y^{n} + \\Delta t \\, F(t^{n}, y^{n})

    Parameters
    ----------
    t : float
        Current time. ms.
    V, w : ndarray
        Current state. Dimensionless.
    dt : float
        Time step. ms.
    rhs : callable
        ``rhs(t, V, w) -> (dV/dt, dw/dt)``.

    Returns
    -------
    V_new, w_new : ndarray
        State at ``t + dt``.

    Notes
    -----
    First-order accurate and conditionally stable. One right-hand-side
    evaluation per step, which is why it is four times cheaper than RK4 for the
    same step -- but RK4 tolerates a step roughly 2.8 times larger, so the two
    are closer in real cost than the stage count suggests. ``ex03`` measures
    both the order and the cost.
    """
    dV, dw = rhs(t, V, w)
    return V + dt * dV, w + dt * dw


def rk4_step(
    t: float,
    V: np.ndarray,
    w: np.ndarray,
    dt: float,
    rhs: RHSFunction,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one step with the classical fourth-order Runge-Kutta method.

    .. math::
        k_1 &= F(t, y) \\\\
        k_2 &= F(t + \\tfrac{\\Delta t}{2}, y + \\tfrac{\\Delta t}{2} k_1) \\\\
        k_3 &= F(t + \\tfrac{\\Delta t}{2}, y + \\tfrac{\\Delta t}{2} k_2) \\\\
        k_4 &= F(t + \\Delta t, y + \\Delta t\\, k_3) \\\\
        y^{n+1} &= y^{n} + \\tfrac{\\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)

    Parameters
    ----------
    t : float
        Current time. ms.
    V, w : ndarray
        Current state. Dimensionless.
    dt : float
        Time step. ms.
    rhs : callable
        ``rhs(t, V, w) -> (dV/dt, dw/dt)``.

    Returns
    -------
    V_new, w_new : ndarray
        State at ``t + dt``.

    Notes
    -----
    Written out stage by stage rather than looped over a Butcher tableau,
    because the four lines are easier to check against the textbook than a
    table-driven implementation would be.

    RK4 is **not** unconditionally stable. Its stability region reaches to
    about ``-2.785`` on the negative real axis rather than Euler's ``-2``, so
    it tolerates a step about 1.39 times larger for a purely diffusive problem
    -- useful, but not a licence to ignore the limit. It is used here only to
    show that the spatial error, not the temporal one, dominates at the
    working step.

    The weights ``1, 2, 2, 1`` and the divisor ``6`` are the exact classical
    RK4 coefficients, not tunable parameters.
    """
    half_dt = 0.5 * dt

    k1_V, k1_w = rhs(t, V, w)
    k2_V, k2_w = rhs(t + half_dt, V + half_dt * k1_V, w + half_dt * k1_w)
    k3_V, k3_w = rhs(t + half_dt, V + half_dt * k2_V, w + half_dt * k2_w)
    k4_V, k4_w = rhs(t + dt, V + dt * k3_V, w + dt * k3_w)

    V_new = V + (dt / 6.0) * (k1_V + 2.0 * k2_V + 2.0 * k3_V + k4_V)
    w_new = w + (dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)
    return V_new, w_new


# Lookup from the configuration string to the stepper. Kept next to the
# functions so that adding an integrator means touching exactly one place.
STEPPERS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "euler": euler_step,
    "rk4": rk4_step,
}


def get_stepper(method: str) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Return the stepping function named by a configuration string.

    Parameters
    ----------
    method : {"euler", "rk4"}
        Integrator name.

    Returns
    -------
    callable
        The corresponding step function.

    Raises
    ------
    ValueError
        If the name is not recognised.
    """
    try:
        return STEPPERS[method]
    except KeyError:
        raise ValueError(
            f"Unknown integrator {method!r}; available: {sorted(STEPPERS)}"
        ) from None
