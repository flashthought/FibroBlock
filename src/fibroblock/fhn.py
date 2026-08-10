"""FitzHugh-Nagumo kinetics and every analytic result derived from them.

Nothing in this module is hard-coded from the assignment brief. The rest state,
the Jacobian, the bistable roots and the analytic conduction-velocity prefactor
are all computed from :class:`~fibroblock.config.FHNParams` at run time. The
brief's quoted values live in ``config.py`` as ``BRIEF_*`` constants and are
used *only* by the tests, to check agreement.

The kinetics are

.. math::
    f(V, w) &= V - \\tfrac{1}{3} V^{3} - w + I_{\\text{stim}} \\\\
    g(V, w) &= \\varepsilon \\, (V + a - b w)

References
----------
FitzHugh, R. (1961). Impulses and physiological states in theoretical models of
nerve membrane. *Biophysical Journal* 1(6), 445-466.

Nagumo, J., Arimoto, S., Yoshizawa, S. (1962). An active pulse transmission line
simulating nerve axon. *Proceedings of the IRE* 50(10), 2061-2070.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from fibroblock.config import FHNParams

# Exact mathematical constant: the coefficient of the cubic term in the
# FitzHugh-Nagumo nonlinearity, f = V - V^3/3 - w. It is written here once and
# referred to by name so that the same number appears in the kinetics, in the
# bistable-root polynomial and in the front-speed prefactor.
CUBIC_COEFFICIENT: float = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Right-hand sides
# ---------------------------------------------------------------------------


def reaction_f(
    V: np.ndarray | float,
    w: np.ndarray | float,
    i_stim: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """Fast (voltage) kinetics of the FitzHugh-Nagumo system.

    .. math:: f(V, w) = V - \\frac{V^{3}}{3} - w + I_{\\text{stim}}

    Parameters
    ----------
    V : ndarray or float
        Membrane potential. Dimensionless.
    w : ndarray or float
        Recovery variable. Dimensionless.
    i_stim : ndarray or float, optional
        Applied stimulus current, same units as ``f``. Default 0.

    Returns
    -------
    ndarray or float
        ``dV/dt`` contribution from the reaction term, per millisecond.

    Notes
    -----
    The cubic gives the system its three-branch ``V``-nullcline: two stable
    outer branches (rest and excited) separated by an unstable middle branch.
    That is the entire source of excitability.
    """
    return V - CUBIC_COEFFICIENT * V**3 - w + i_stim


def recovery_g(
    V: np.ndarray | float,
    w: np.ndarray | float,
    params: FHNParams,
) -> np.ndarray | float:
    """Slow (recovery) kinetics of the FitzHugh-Nagumo system.

    .. math:: g(V, w) = \\varepsilon \\, (V + a - b w)

    Parameters
    ----------
    V : ndarray or float
        Membrane potential. Dimensionless.
    w : ndarray or float
        Recovery variable. Dimensionless.
    params : FHNParams
        Kinetics parameters supplying ``a``, ``b`` and ``eps``.

    Returns
    -------
    ndarray or float
        ``dw/dt``, per millisecond.

    Notes
    -----
    ``eps`` is small (0.08), so ``w`` changes roughly twelve times more slowly
    than ``V``. This separation is why the upstroke can be analysed with ``w``
    frozen -- the assumption behind :func:`bistable_roots` and
    :func:`analytic_cv_prefactor`.
    """
    return params.eps * (V + params.a - params.b * w)


# ---------------------------------------------------------------------------
# Rest state
# ---------------------------------------------------------------------------


def rest_cubic_coefficients(params: FHNParams) -> np.ndarray:
    """Coefficients of the polynomial whose real root is the rest potential.

    Setting both derivatives to zero gives ``w* = (V* + a) / b``. Substituting
    that into ``f(V*, w*) = 0`` and multiplying through by ``-3`` gives

    .. math:: V^{3} + \\left(\\frac{3}{b} - 3\\right) V + \\frac{3a}{b} = 0

    which for the assignment values ``a = 0.7``, ``b = 0.8`` is
    ``V^3 + 0.75 V + 2.625 = 0``.

    Parameters
    ----------
    params : FHNParams
        Kinetics parameters.

    Returns
    -------
    ndarray, shape (4,)
        Coefficients in NumPy's descending-power order,
        ``[1, 0, 3/b - 3, 3a/b]``.

    Notes
    -----
    There is no quadratic term, so the three roots sum to zero. That identity
    is used as an internal consistency check in :func:`bistable_roots`.
    """
    # The 3s below come from multiplying f = V - V^3/3 - w by -3 to clear the
    # cubic's denominator; they are exact algebra, not tunable parameters.
    linear_coefficient = 3.0 / params.b - 3.0
    constant_coefficient = 3.0 * params.a / params.b
    return np.array([1.0, 0.0, linear_coefficient, constant_coefficient])


def rest_state(params: FHNParams) -> tuple[float, float]:
    """Compute the resting equilibrium ``(V*, w*)`` from the parameters.

    Parameters
    ----------
    params : FHNParams
        Kinetics parameters.

    Returns
    -------
    V_rest : float
        Resting membrane potential. Dimensionless. About ``-1.199408`` for the
        assignment values.
    w_rest : float
        Resting recovery variable, ``(V* + a) / b``. About ``-0.624260``.

    Raises
    ------
    RuntimeError
        If the cubic has no real root, which cannot happen for physical
        parameter values but is checked rather than assumed.

    Notes
    -----
    The root is found in two stages. ``numpy.roots`` gives all three roots of
    the companion matrix, which is robust but only about 10 significant digits;
    the real one is then polished with Brent's method on a tight bracket, which
    converges to full double precision. Doing both means the answer does not
    depend on either routine's failure modes alone.
    """
    coefficients = rest_cubic_coefficients(params)
    all_roots = np.roots(coefficients)

    # For b = 0.8 the linear coefficient 3/b - 3 = 0.75 is positive, so the
    # cubic is strictly increasing and has exactly one real root. Rather than
    # rely on that, filter numerically: a genuinely real root has an imaginary
    # part at the level of round-off.
    real_roots = all_roots[np.abs(all_roots.imag) < 1e-9].real
    if real_roots.size == 0:
        raise RuntimeError(
            f"No real rest state for FHN parameters {params!r}; "
            f"cubic roots were {all_roots}."
        )

    # If several real roots exist, the rest state is the most negative one:
    # the lower branch of the cubic nullcline is the polarised state.
    approximate_root = float(np.min(real_roots))

    def cubic(V: float) -> float:
        """Evaluate the rest-state cubic at ``V``."""
        return float(np.polyval(coefficients, V))

    # Polish with Brent's method on a bracket straddling the approximate root.
    # A half-width of 1e-3 is far larger than numpy.roots' error yet small
    # enough that no other root can sneak into the bracket.
    bracket_half_width = 1.0e-3
    lower = approximate_root - bracket_half_width
    upper = approximate_root + bracket_half_width
    if cubic(lower) * cubic(upper) < 0.0:
        V_rest = float(brentq(cubic, lower, upper, xtol=1.0e-15, rtol=1.0e-15))
    else:
        # Bracket failed to straddle (possible if numpy.roots was unusually
        # inaccurate). Fall back to the unpolished root rather than crash.
        V_rest = approximate_root

    # w-nullcline: g = 0 implies V + a - b w = 0, hence w = (V + a) / b.
    w_rest = (V_rest + params.a) / params.b
    return V_rest, w_rest


# ---------------------------------------------------------------------------
# Linear stability of the rest state
# ---------------------------------------------------------------------------


def jacobian(V: float, params: FHNParams) -> np.ndarray:
    """Jacobian of the space-clamped FitzHugh-Nagumo system at potential ``V``.

    .. math::
        J = \\begin{bmatrix} 1 - V^{2} & -1 \\\\
                             \\varepsilon & -\\varepsilon b \\end{bmatrix}

    Parameters
    ----------
    V : float
        Potential at which to linearise. Dimensionless.
    params : FHNParams
        Kinetics parameters.

    Returns
    -------
    ndarray, shape (2, 2)
        The Jacobian matrix, in units of inverse milliseconds.

    Notes
    -----
    The entries are the partial derivatives of ``(f, g)`` with respect to
    ``(V, w)``: ``df/dV = 1 - V^2``, ``df/dw = -1``, ``dg/dV = eps``,
    ``dg/dw = -eps*b``. The Jacobian does not depend on ``w`` because both
    right-hand sides are linear in ``w``.
    """
    return np.array(
        [
            [1.0 - V**2, -1.0],
            [params.eps, -params.eps * params.b],
        ]
    )


@dataclass(frozen=True)
class ExcitabilitySummary:
    """Linear-stability classification of the rest state.

    Attributes
    ----------
    V_rest, w_rest : float
        The equilibrium being classified. Dimensionless.
    trace : float
        ``tr(J)``. Negative means perturbations decay. 1/ms.
    determinant : float
        ``det(J)``. Positive (with negative trace) means the equilibrium is
        stable rather than a saddle. 1/ms^2.
    discriminant : float
        ``tr(J)^2 - 4 det(J)``. Negative means the eigenvalues are a complex
        conjugate pair, so the return to rest spirals rather than creeping.
    eigenvalues : tuple of complex
        The two eigenvalues of ``J``, in 1/ms.
    classification : str
        Plain-English name of the equilibrium type.
    is_excitable : bool
        True when the rest state is linearly stable, which is the definition of
        an *excitable* (as opposed to *oscillatory*) medium: it stays put until
        pushed hard enough, then makes a large excursion and returns.
    """

    V_rest: float
    w_rest: float
    trace: float
    determinant: float
    discriminant: float
    eigenvalues: tuple[complex, complex]
    classification: str
    is_excitable: bool


def excitability(params: FHNParams) -> ExcitabilitySummary:
    """Classify the rest state by linear stability analysis.

    Parameters
    ----------
    params : FHNParams
        Kinetics parameters.

    Returns
    -------
    ExcitabilitySummary
        Trace, determinant, discriminant, eigenvalues and a classification.

    Notes
    -----
    For the assignment values the result is a **stable spiral**:
    ``tr(J) = -0.5026 < 0`` and ``det(J) = 0.1081 > 0`` with a negative
    discriminant. This is the key qualitative fact about the model -- the
    tissue is excitable, not spontaneously oscillatory, so every action
    potential in the simulations is a *response to the stimulus* and not
    self-generated pacemaker activity.

    The classification follows the standard trace-determinant plane:
    ``det < 0`` is a saddle; ``det > 0`` with ``tr < 0`` is stable, spiral if
    the discriminant is negative and a node if it is positive; ``tr > 0``
    mirrors that as unstable.
    """
    V_rest, w_rest = rest_state(params)
    J = jacobian(V_rest, params)

    trace = float(np.trace(J))
    determinant = float(np.linalg.det(J))
    discriminant = trace**2 - 4.0 * determinant

    eigenvalues_array = np.linalg.eigvals(J)
    eigenvalues = (complex(eigenvalues_array[0]), complex(eigenvalues_array[1]))

    if determinant < 0.0:
        classification = "saddle (unstable)"
    elif trace < 0.0:
        classification = "stable spiral" if discriminant < 0.0 else "stable node"
    elif trace > 0.0:
        classification = "unstable spiral" if discriminant < 0.0 else "unstable node"
    else:
        classification = "centre (marginal)"

    # Excitable means the rest state is linearly stable: negative trace AND
    # positive determinant. Either condition alone is not enough.
    is_excitable = (trace < 0.0) and (determinant > 0.0)

    return ExcitabilitySummary(
        V_rest=V_rest,
        w_rest=w_rest,
        trace=trace,
        determinant=determinant,
        discriminant=discriminant,
        eigenvalues=eigenvalues,
        classification=classification,
        is_excitable=is_excitable,
    )


# ---------------------------------------------------------------------------
# Frozen-w bistable reduction and the analytic front speed
# ---------------------------------------------------------------------------


def bistable_roots(params: FHNParams) -> tuple[float, float, float]:
    """Roots of the frozen-``w`` cubic that governs the fast upstroke.

    During the upstroke ``V`` moves in about 1 ms while ``w`` moves on the
    ``1/eps ~ 12 ms`` scale, so ``w`` may be held at its resting value ``w*``.
    Setting ``f(V, w*) = 0`` and multiplying by ``-3`` gives the bistable cubic

    .. math:: V^{3} - 3V + 3w^{*} = 0

    Parameters
    ----------
    params : FHNParams
        Kinetics parameters.

    Returns
    -------
    V1, V2, V3 : float
        The three real roots in ascending order: the rest state ``V1``, the
        excitation threshold ``V2``, and the excited plateau ``V3``. For the
        assignment values these are about ``-1.1994``, ``-0.7863`` and
        ``+1.9857``.

    Raises
    ------
    RuntimeError
        If the cubic does not have three distinct real roots, meaning the
        parameters do not give a bistable upstroke and the front-speed formula
        below does not apply.

    Notes
    -----
    ``V1`` must equal the rest potential exactly, because ``w*`` was defined as
    the value making ``V*`` a zero of ``f``. That identity is a free correctness
    check and is asserted in ``tests/test_fhn.py``.

    The cubic has no quadratic term, so ``V1 + V2 + V3 = 0``.
    """
    _, w_rest = rest_state(params)

    # Coefficients of V^3 + 0*V^2 - 3*V + 3*w_rest, in descending powers.
    # The -3 and +3 come from multiplying f = V - V^3/3 - w* by -3.
    coefficients = np.array([1.0, 0.0, -3.0, 3.0 * w_rest])
    all_roots = np.roots(coefficients)

    real_roots = np.sort(all_roots[np.abs(all_roots.imag) < 1e-9].real)
    if real_roots.size != 3:
        raise RuntimeError(
            f"Frozen-w cubic is not bistable for parameters {params!r}: "
            f"expected 3 real roots, got {real_roots.size} "
            f"(all roots were {all_roots})."
        )

    return float(real_roots[0]), float(real_roots[1]), float(real_roots[2])


def analytic_cv_prefactor(params: FHNParams) -> float:
    """Prefactor in the analytic front speed ``theta = prefactor * sqrt(D)``.

    Writing the frozen-``w`` reaction term in factored form

    .. math:: f(V, w^{*}) = -A (V - V_1)(V - V_2)(V - V_3),
              \\qquad A = \\tfrac{1}{3}

    the classical bistable travelling-front solution of
    ``V_t = D V_xx + f`` has speed

    .. math:: \\theta = \\sqrt{\\tfrac{A}{2}}\\,(V_1 - 2 V_2 + V_3)\\,\\sqrt{D}

    Parameters
    ----------
    params : FHNParams
        Kinetics parameters.

    Returns
    -------
    float
        The prefactor ``theta / sqrt(D)``, in units of 1/sqrt(ms). About
        ``0.9630`` for the assignment values.

    Notes
    -----
    Because the cubic has no quadratic term the roots sum to zero, so
    ``V1 - 2 V2 + V3 = -3 V2``. The speed is therefore controlled entirely by
    how far the threshold root sits from rest: push ``V2`` towards ``V1`` and
    the front slows and eventually fails, which is exactly the mechanism behind
    conduction block.

    The value ``A = 1/3`` is the coefficient of the cubic term in ``f``, not a
    free parameter: expanding ``-A(V-V1)(V-V2)(V-V3)`` reproduces
    ``-V^3/3 + ...`` only for ``A = 1/3``.

    References
    ----------
    Keener, J., Sneyd, J. (2009). *Mathematical Physiology I: Cellular
    Physiology*, 2nd edn, Springer. Section 6.2 (bistable front speed).
    """
    V1, V2, V3 = bistable_roots(params)

    # A is the cubic coefficient of f; the factored and expanded forms agree
    # only for this value, so it is derived, not chosen.
    A = CUBIC_COEFFICIENT

    # The exact 2 below is from the standard bistable-front result
    # theta = sqrt(A/2) * (V1 - 2 V2 + V3) * sqrt(D).
    return float(np.sqrt(A / 2.0) * (V1 - 2.0 * V2 + V3))


def analytic_cv(D: float, params: FHNParams) -> float:
    """Analytic conduction velocity at a given diffusion coefficient.

    .. math:: \\theta = \\left[\\sqrt{A/2}\\,(V_1 - 2V_2 + V_3)\\right] \\sqrt{D}

    Parameters
    ----------
    D : float
        Diffusion coefficient. cm^2/ms.
    params : FHNParams
        Kinetics parameters.

    Returns
    -------
    float
        Front speed in cm/ms. About ``0.03046 cm/ms`` (30.5 cm/s) at
        ``D = 0.001 cm^2/ms``.

    Raises
    ------
    ValueError
        If ``D`` is negative.

    Notes
    -----
    The measured velocity is expected to come out slightly *below* this. The
    derivation freezes ``w`` at ``w*``, but in the real solution ``w`` does rise
    a little during the upstroke, which lifts the effective threshold root
    ``V2`` towards rest and therefore slows the front. Experiment ``ex05``
    reports measured, analytic, and the relative discrepancy between them.
    """
    if D < 0.0:
        raise ValueError(f"Diffusion coefficient must be non-negative, got {D}")
    return analytic_cv_prefactor(params) * float(np.sqrt(D))


def front_thickness(D: float, params: FHNParams, f_v_bound: float) -> float:
    """Characteristic thickness of the propagating wavefront.

    .. math:: \\delta \\sim \\sqrt{D / |f_V|_{\\max}}

    Parameters
    ----------
    D : float
        Diffusion coefficient. cm^2/ms.
    params : FHNParams
        Kinetics parameters. Present for interface consistency; the estimate
        depends on them only through ``f_v_bound``.
    f_v_bound : float
        Bound on ``|df/dV| = |1 - V^2|`` over the action potential, 1/ms.

    Returns
    -------
    float
        Front thickness in cm. About ``0.018 cm`` at ``D = 0.001`` and
        ``|f_V| = 3``.

    Raises
    ------
    ValueError
        If ``f_v_bound`` is not positive.

    Notes
    -----
    This is the length over which diffusion and reaction balance, and it is the
    quantity the grid must resolve. It is why ``dx = 0.01 cm`` is defensible:
    it puts roughly two nodes inside the balance length and about ten across
    the full visible upstroke. Under-resolving it makes the measured
    conduction velocity grid-dependent.
    """
    if f_v_bound <= 0.0:
        raise ValueError(f"f_v_bound must be positive, got {f_v_bound}")
    del params  # kept in the signature for a uniform call style
    return float(np.sqrt(D / f_v_bound))
