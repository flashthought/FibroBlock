"""Stimulus current profiles and stimulus-threshold bisection.

The stimulus is a rectangular pulse in both space and time: amplitude ``A`` for
``x`` in ``[0, width]`` and ``t`` in ``[start, start + duration)``, zero
elsewhere. It enters the model inside the fast kinetics,
``f(V, w) = V - V^3/3 - w + I_stim``, so it has the units of ``dV/dt``.
"""

from __future__ import annotations

import numpy as np

from fibroblock.config import StimulusParams


def stimulus_mask(x: np.ndarray, stimulus: StimulusParams) -> np.ndarray:
    """Boolean mask of the nodes the stimulus is applied to.

    Parameters
    ----------
    x : ndarray
        Node positions. cm.
    stimulus : StimulusParams
        Stimulus description.

    Returns
    -------
    ndarray of bool, shape (x.size,)
        True at nodes with ``0 <= x <= width_cm``.

    Notes
    -----
    Computed once per run and reused every step. The tolerance handles a width
    that lands exactly on a node, which is the usual case (0.1 cm on a 0.01 cm
    grid): without it, whether node 10 is stimulated would depend on the last
    bit of a floating-point comparison.
    """
    if x.size < 2:
        raise ValueError(f"Need at least two nodes, got {x.size}")

    spacing = float(x[1] - x[0])
    tolerance = 1.0e-6 * spacing
    return x <= stimulus.width_cm + tolerance


def is_stimulus_active(t: float, stimulus: StimulusParams) -> bool:
    """Whether the stimulus is on at time ``t``.

    Parameters
    ----------
    t : float
        Time. ms.
    stimulus : StimulusParams
        Stimulus description.

    Returns
    -------
    bool
        True for ``start_ms <= t < start_ms + duration_ms``.

    Notes
    -----
    The window is half-open. A closed window would apply one extra step's worth
    of current whenever ``duration / dt`` happens to be an integer, making the
    delivered charge depend on the time step -- a subtle way for a convergence
    study to look worse than it is.
    """
    return stimulus.start_ms <= t < stimulus.start_ms + stimulus.duration_ms


def stimulus_current(
    t: float,
    mask: np.ndarray,
    stimulus: StimulusParams,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate ``I_stim(x, t)`` on the grid.

    Parameters
    ----------
    t : float
        Time. ms.
    mask : ndarray of bool
        Output of :func:`stimulus_mask`.
    stimulus : StimulusParams
        Stimulus description.
    out : ndarray, optional
        Buffer to write into, to avoid allocating a new array every step. Must
        have the same shape as ``mask``.

    Returns
    -------
    ndarray, shape (mask.size,)
        Stimulus current at each node, in the same units as ``f(V, w)``.

    Notes
    -----
    The ``out`` buffer matters: this is called once per time step (four times
    per step under RK4), so 15 000 steps would otherwise allocate 60 000
    throwaway arrays. Reusing one buffer is the only performance concession in
    the whole codebase, and it is here because it costs nothing in clarity.
    """
    if out is None:
        current = np.zeros(mask.shape)
    else:
        if out.shape != mask.shape:
            raise ValueError(
                f"Output buffer shape {out.shape} does not match mask {mask.shape}"
            )
        current = out
        current[:] = 0.0

    if is_stimulus_active(t, stimulus):
        current[mask] = stimulus.amplitude

    return current


def delivered_charge(stimulus: StimulusParams) -> float:
    """Total stimulus charge, ``A * width * duration``.

    Parameters
    ----------
    stimulus : StimulusParams
        Stimulus description.

    Returns
    -------
    float
        Charge in (dimensionless current) x cm x ms.

    Notes
    -----
    Useful when comparing stimuli of different shapes: two pulses with the same
    charge but different durations do *not* generally have the same threshold,
    because the tissue leaks charge to its neighbours while the pulse is on.
    Reporting the charge makes that comparison explicit rather than implied.
    """
    return stimulus.amplitude * stimulus.width_cm * stimulus.duration_ms


def bisect_threshold(
    succeeds: callable[[float], bool],
    lower: float,
    upper: float,
    tolerance: float,
    max_iterations: int = 60,
) -> tuple[float, int]:
    """Find the smallest parameter value at which a binary outcome flips.

    Parameters
    ----------
    succeeds : callable
        ``succeeds(value) -> bool``. Must be monotone: False below the
        threshold, True above it.
    lower : float
        A value at which ``succeeds`` is known (or expected) to be False.
    upper : float
        A value at which ``succeeds`` is expected to be True.
    tolerance : float
        Stop once the bracket is narrower than this.
    max_iterations : int, optional
        Hard cap on iterations. Default 60, which is far more than the ~40
        needed to reduce any sensible bracket to double precision.

    Returns
    -------
    threshold : float
        Midpoint of the final bracket.
    iterations : int
        Number of evaluations of ``succeeds`` performed inside the loop.

    Raises
    ------
    ValueError
        If the initial bracket does not straddle the threshold, or if
        ``lower >= upper``.

    Notes
    -----
    Plain bisection rather than a root finder, because the quantity being
    bracketed is a **boolean** -- "did the wave propagate?" -- not a continuous
    function with a sign change. There is no residual to interpolate, so
    Brent's method has nothing to work with, and bisection's guaranteed halving
    is exactly the right tool.

    The monotonicity assumption is checked at the endpoints, which catches the
    common failure of starting with a bracket that is entirely on one side.
    Monotonicity in between is a physical assumption: stronger coupling never
    makes propagation less likely.
    """
    if lower >= upper:
        raise ValueError(f"Need lower < upper, got lower={lower}, upper={upper}")
    if tolerance <= 0.0:
        raise ValueError(f"Tolerance must be positive, got {tolerance}")

    if succeeds(lower):
        raise ValueError(
            f"Bracket does not straddle the threshold: the outcome is already "
            f"True at the lower end ({lower}). Widen the bracket downwards."
        )
    if not succeeds(upper):
        raise ValueError(
            f"Bracket does not straddle the threshold: the outcome is still "
            f"False at the upper end ({upper}). Widen the bracket upwards."
        )

    iterations = 0
    while (upper - lower) > tolerance and iterations < max_iterations:
        midpoint = 0.5 * (lower + upper)
        if succeeds(midpoint):
            upper = midpoint
        else:
            lower = midpoint
        iterations += 1

    return 0.5 * (lower + upper), iterations
