"""Top-level simulation driver: assemble, integrate, and return a typed result.

This is the only place the time loop lives. Everything it needs -- the grid,
the half-node diffusion coefficients, the stimulus mask -- is built once before
the loop starts and never rebuilt inside it.

What is stored, and why
-----------------------
The full ``V(x, t)`` history is **off by default**. A 300 ms run at
``dt = 0.02 ms`` on 201 nodes is 15 000 x 201 doubles per field, and none of the
reported measurements need it. What is always recorded is cheap and sufficient:

* activation times at every node, accumulated during the run by watching for
  the crossing as it happens (there is no way to recover them afterwards from
  downsampled snapshots without losing sub-step resolution);
* the peak potential and peak upstroke rate at every node;
* downsampled spatial snapshots for the space-time plots;
* the total charge at each snapshot, for the conservation check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from fibroblock import fhn, operators, solvers, stimulus
from fibroblock import grid as gridmod
from fibroblock.config import RunConfig
from fibroblock.grid import Grid
from fibroblock.solvers import StabilityLimits

# A solution this large is not physics, it is a blow-up in progress. The
# excited plateau sits near V = 2, so 1e6 is six orders of magnitude past
# anything meaningful, while still being far below the overflow that would
# turn the arrays into inf and NaN and destroy the diagnostic record.
DIVERGENCE_THRESHOLD: float = 1.0e6


@dataclass(frozen=True)
class SimulationResult:
    """Everything one run produced.

    Attributes
    ----------
    config : RunConfig
        The configuration that produced this result. Saved alongside every
        output file.
    grid : Grid
        Node positions, diffusion profile, quadrature weights, gap mask.
    snapshot_times : ndarray, shape (n_snapshots,)
        Times at which spatial snapshots were stored. ms.
    V_snapshots : ndarray, shape (n_snapshots, n_nodes)
        Potential at those times. Dimensionless.
    w_snapshots : ndarray, shape (n_snapshots, n_nodes)
        Recovery variable at those times. Dimensionless.
    charge_history : ndarray, shape (n_snapshots,)
        Total charge at those times, from the trapezoidal quadrature.
    activation_time_crossing : ndarray, shape (n_nodes,)
        Time at which ``V`` first crossed the activation level, linearly
        interpolated within the step. ms. NaN where it never did.
    activation_time_max_dvdt : ndarray, shape (n_nodes,)
        Time of the steepest upstroke at each node. ms. NaN where the node
        never activated.
    max_dvdt : ndarray, shape (n_nodes,)
        Peak upstroke rate at each node. 1/ms.
    V_peak : ndarray, shape (n_nodes,)
        Highest potential reached at each node. Dimensionless.
    V_history : ndarray or None
        Full history, shape ``(n_steps + 1, n_nodes)``, only when
        ``store_full_history`` is set.
    stability : StabilityLimits
        The stability analysis for this configuration.
    wall_seconds : float
        Wall-clock integration time.
    diverged : bool
        True if the solution exceeded :data:`DIVERGENCE_THRESHOLD` or became
        non-finite. Expected (and deliberately provoked) in ``ex02``.
    divergence_time_ms : float
        When that happened. NaN if it did not.
    n_steps_taken : int
        Steps actually completed, which is fewer than requested if the run
        diverged and was stopped early.
    """

    config: RunConfig
    grid: Grid
    snapshot_times: np.ndarray
    V_snapshots: np.ndarray
    w_snapshots: np.ndarray
    charge_history: np.ndarray
    activation_time_crossing: np.ndarray
    activation_time_max_dvdt: np.ndarray
    max_dvdt: np.ndarray
    V_peak: np.ndarray
    V_history: np.ndarray | None
    stability: StabilityLimits
    wall_seconds: float
    diverged: bool
    divergence_time_ms: float
    n_steps_taken: int

    @property
    def x(self) -> np.ndarray:
        """Node positions. cm. Convenience alias for ``result.grid.x``."""
        return self.grid.x

    @property
    def activation_times(self) -> np.ndarray:
        """Activation times under the rule named in the configuration.

        Returns
        -------
        ndarray, shape (n_nodes,)
            ms, NaN where the node never activated.

        Notes
        -----
        Both definitions are always computed and stored, because they differ
        slightly and the report has to state which was used. Selecting here
        rather than at recording time means a figure can be redrawn under the
        other definition without re-running anything.
        """
        if self.config.measurement.activation_rule == "v_zero_crossing":
            return self.activation_time_crossing
        return self.activation_time_max_dvdt

    @property
    def final_V(self) -> np.ndarray:
        """Potential at the last recorded snapshot. Dimensionless."""
        return self.V_snapshots[-1]

    @property
    def charge_drift(self) -> float:
        """Relative change in total charge over the run.

        Returns
        -------
        float
            ``|Q_end - Q_start| / |Q_start|``. For a pure-diffusion run with
            sealed ends this should be at machine-precision level; with the
            reaction term active it is not expected to be small, because the
            reaction genuinely creates and destroys charge.
        """
        start = self.charge_history[0]
        end = self.charge_history[-1]
        if start == 0.0:
            return float(abs(end))
        return float(abs(end - start) / abs(start))


def _validate_time_step(
    config: RunConfig, grid_obj: Grid, force: bool
) -> StabilityLimits:
    """Check the configured step against the von Neumann limit.

    Parameters
    ----------
    config : RunConfig
        Run configuration.
    grid_obj : Grid
        Built grid, needed for ``D_max`` at the interfaces.
    force : bool
        Skip the check. Used only by the stability experiment, which needs to
        step past the limit deliberately.

    Returns
    -------
    StabilityLimits
        The analysis, whether or not it passed.

    Raises
    ------
    ValueError
        If ``dt`` exceeds the limit and ``force`` is False.
    """
    limits = solvers.stability_limits(
        D_max=grid_obj.D_max,
        dx=grid_obj.dx,
        f_v_bound=config.solver.f_v_bound,
        dt=config.solver.dt_ms,
    )

    if not limits.is_stable and not force:
        raise ValueError(
            f"Time step dt = {config.solver.dt_ms} ms exceeds the explicit-Euler "
            f"stability limit of {limits.reaction_diffusion_dt_ms:.6f} ms "
            f"(D_max = {grid_obj.D_max} cm^2/ms, dx = {grid_obj.dx} cm, "
            f"|f_V|_max = {config.solver.f_v_bound}). "
            f"Reduce dt, or pass force=True if the instability is the point."
        )

    return limits


def run_simulation(
    config: RunConfig,
    force: bool = False,
    initial_V: np.ndarray | None = None,
    initial_w: np.ndarray | None = None,
    include_reaction: bool = True,
    include_stimulus: bool = True,
) -> SimulationResult:
    r"""Integrate the monodomain FitzHugh-Nagumo system on the strand.

    Solves

    .. math::
        \frac{\partial V}{\partial t}
            &= \frac{\partial}{\partial x}\!\left(D(x)
               \frac{\partial V}{\partial x}\right) + f(V, w) \\
        \frac{\partial w}{\partial t} &= \varepsilon (V + a - b w)

    with sealed ends and a rectangular stimulus near ``x = 0``.

    Parameters
    ----------
    config : RunConfig
        Complete run description.
    force : bool, optional
        Integrate even if ``dt`` exceeds the stability limit. Default False, so
        an unstable step is an error rather than a silent garbage result. Only
        ``ex02`` sets this.
    initial_V, initial_w : ndarray, optional
        Initial condition. Defaults to the whole strand at the resting state.
        Supplied explicitly by the pure-diffusion verification, which starts
        from a Gaussian.
    include_reaction : bool, optional
        Include ``f(V, w)`` and the recovery equation. Default True. Set False
        for the pure-diffusion analytic check, where the equation must reduce
        to ``V_t = D V_xx`` exactly.
    include_stimulus : bool, optional
        Apply the stimulus current. Default True.

    Returns
    -------
    SimulationResult
        Snapshots, activation times, peak values, charge history, the stability
        analysis, and timing.

    Raises
    ------
    ValueError
        If ``dt`` exceeds the stability limit and ``force`` is False, or if a
        supplied initial condition has the wrong shape.

    Notes
    -----
    The spatial operator is applied with NumPy slicing over the whole strand at
    once; there is no Python loop over nodes anywhere. The only Python-level
    loop is over time steps, which is unavoidable for an explicit scheme.

    ``D_half`` is built by :func:`fibroblock.grid.build_grid` before the loop
    starts and is read-only thereafter, so the "precompute the half-node array
    once per run" requirement holds structurally.
    """
    fhn_params = config.fhn
    solver_params = config.solver
    stim_params = config.stimulus

    # ---- Precompute everything the loop will need --------------------------
    grid_obj = gridmod.build_grid(config.grid, config.gap)
    dx = grid_obj.dx
    dt = solver_params.dt_ms
    n_nodes = grid_obj.n_nodes
    D_half = grid_obj.D_half

    limits = _validate_time_step(config, grid_obj, force)

    stim_mask = stimulus.stimulus_mask(grid_obj.x, stim_params)
    # Reused every right-hand-side evaluation instead of allocating afresh.
    stim_buffer = np.zeros(n_nodes)

    # ---- Initial condition -------------------------------------------------
    V_rest, w_rest = fhn.rest_state(fhn_params)

    if initial_V is None:
        V = np.full(n_nodes, V_rest)
    else:
        if initial_V.shape != (n_nodes,):
            raise ValueError(
                f"initial_V must have shape ({n_nodes},), got {initial_V.shape}"
            )
        V = initial_V.astype(float).copy()

    if initial_w is None:
        w = np.full(n_nodes, w_rest)
    else:
        if initial_w.shape != (n_nodes,):
            raise ValueError(
                f"initial_w must have shape ({n_nodes},), got {initial_w.shape}"
            )
        w = initial_w.astype(float).copy()

    # ---- The right-hand side ----------------------------------------------
    def rhs(
        t: float, V_in: np.ndarray, w_in: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate (dV/dt, dw/dt) for the semi-discrete system."""
        # Conservative divergence: d/dx (D dV/dx), with sealed ends.
        dV = operators.divergence(V_in, D_half, dx)

        if include_reaction:
            if include_stimulus:
                i_stim = stimulus.stimulus_current(
                    t, stim_mask, stim_params, out=stim_buffer
                )
            else:
                i_stim = 0.0
            dV = dV + fhn.reaction_f(V_in, w_in, i_stim)
            dw = fhn.recovery_g(V_in, w_in, fhn_params)
        else:
            # Pure diffusion: the recovery variable is frozen so that the
            # system reduces exactly to V_t = D V_xx and can be compared with
            # the closed-form Gaussian solution.
            dw = np.zeros_like(w_in)

        return dV, dw

    step = solvers.get_stepper(solver_params.method)

    # ---- Diagnostics accumulated during the run ----------------------------
    activation_crossing = np.full(n_nodes, np.nan)
    activation_max_dvdt = np.full(n_nodes, np.nan)
    max_dvdt = np.full(n_nodes, -np.inf)
    V_peak = V.copy()

    activation_level = config.measurement.activation_level

    snapshot_times: list[float] = []
    V_snapshots: list[np.ndarray] = []
    w_snapshots: list[np.ndarray] = []
    charge_history: list[float] = []

    full_history: list[np.ndarray] | None = (
        [V.copy()] if solver_params.store_full_history else None
    )

    def record_snapshot(t: float) -> None:
        """Store the current state for later plotting."""
        snapshot_times.append(t)
        V_snapshots.append(V.copy())
        w_snapshots.append(w.copy())
        charge_history.append(
            operators.total_charge(V, grid_obj.quadrature_weights)
        )

    record_snapshot(0.0)

    # ---- Time loop ---------------------------------------------------------
    n_steps = solver_params.n_steps
    diverged = False
    divergence_time = float("nan")
    steps_taken = 0

    start_clock = time.perf_counter()

    for step_index in range(n_steps):
        t = step_index * dt
        V_previous = V

        V, w = step(t, V, w, dt, rhs)
        steps_taken = step_index + 1
        t_new = t + dt

        # --- Activation by level crossing, interpolated within the step -----
        # A node activates when V rises through the activation level for the
        # FIRST time. Linear interpolation inside the step recovers a time
        # resolution finer than dt, which matters: at dt = 0.02 ms and a
        # conduction velocity of 0.03 cm/ms, one step is 1.5 node spacings, so
        # without interpolation the activation map would be visibly stepped and
        # the velocity fit would inherit that quantisation as noise.
        newly_crossed = (
            (V_previous < activation_level)
            & (V >= activation_level)
            & np.isnan(activation_crossing)
        )
        if np.any(newly_crossed):
            rise = V[newly_crossed] - V_previous[newly_crossed]
            fraction = (activation_level - V_previous[newly_crossed]) / rise
            activation_crossing[newly_crossed] = t + fraction * dt

        # --- Activation by steepest upstroke -------------------------------
        rate = (V - V_previous) / dt
        steeper = rate > max_dvdt
        max_dvdt[steeper] = rate[steeper]
        # The rate is a difference over [t, t+dt], so it is centred at t+dt/2.
        activation_max_dvdt[steeper] = t + 0.5 * dt

        np.maximum(V_peak, V, out=V_peak)

        if full_history is not None:
            full_history.append(V.copy())

        # --- Divergence check, every step ----------------------------------
        # This is checked every step, not at the snapshot cadence, because
        # blow-up here is hyper-exponential rather than merely exponential.
        # Once |V| passes sqrt(3/dt) the cubic in f dominates and the update
        # becomes V <- -dt V^3/3, so the magnitude is CUBED each step: about
        # seven steps carry it from V = 10 to double-precision overflow. A
        # check every 25 steps would routinely miss that window entirely and
        # leave the arrays full of inf and NaN, destroying the diagnostic
        # record that ex02 needs. The cost is one reduction over 201 elements
        # per step, around a tenth of the step's own cost.
        largest = np.max(np.abs(V))
        is_diverging = (not np.isfinite(largest)) or (largest > DIVERGENCE_THRESHOLD)

        # --- Snapshots ------------------------------------------------------
        is_snapshot = (step_index + 1) % solver_params.record_every == 0
        is_final = step_index == n_steps - 1

        if is_snapshot or is_final or is_diverging:
            record_snapshot(t_new)

        if is_diverging:
            diverged = True
            divergence_time = t_new
            break

    wall_seconds = time.perf_counter() - start_clock

    # A node that never activated keeps NaN, and its "peak rate" of -inf is
    # meaningless; replace it with NaN so downstream code sees one consistent
    # missing-data marker rather than two.
    max_dvdt[np.isinf(max_dvdt)] = np.nan
    activation_max_dvdt[np.isnan(max_dvdt)] = np.nan
    # A node whose steepest rise never reached the activation level did not
    # activate at all, regardless of what the rate tracker recorded.
    activation_max_dvdt[np.isnan(activation_crossing)] = np.nan

    return SimulationResult(
        config=config,
        grid=grid_obj,
        snapshot_times=np.asarray(snapshot_times),
        V_snapshots=np.asarray(V_snapshots),
        w_snapshots=np.asarray(w_snapshots),
        charge_history=np.asarray(charge_history),
        activation_time_crossing=activation_crossing,
        activation_time_max_dvdt=activation_max_dvdt,
        max_dvdt=max_dvdt,
        V_peak=V_peak,
        V_history=(np.asarray(full_history) if full_history is not None else None),
        stability=limits,
        wall_seconds=wall_seconds,
        diverged=diverged,
        divergence_time_ms=divergence_time,
        n_steps_taken=steps_taken,
    )


def run_single_cell(
    config: RunConfig,
    stimulus_amplitude: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the space-clamped (single-cell) system, with no diffusion.

    Parameters
    ----------
    config : RunConfig
        Run configuration. Only the kinetics, solver and stimulus sections are
        used.
    stimulus_amplitude : float, optional
        Override the configured amplitude, for threshold bisection.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
        Time. ms.
    V : ndarray, shape (n_steps + 1,)
        Potential. Dimensionless.
    w : ndarray, shape (n_steps + 1,)
        Recovery variable. Dimensionless.

    Notes
    -----
    Setting ``D = 0`` in the full solver would work, but this dedicated routine
    is clearer for the single-cell figure and removes any doubt about whether
    a residual diffusive coupling is affecting the threshold measurement. It
    uses the same integrator as the full solver, so it is not a separate
    implementation of the kinetics.

    A single cell has no diffusion, so the only stability constraint is
    ``dt <= 2 / |f_V|_max``, roughly 0.67 ms -- far looser than the strand's
    limit. The configured step is used regardless, for consistency.
    """
    fhn_params = config.fhn
    solver_params = config.solver
    stim_params = config.stimulus

    amplitude = (
        stim_params.amplitude if stimulus_amplitude is None else stimulus_amplitude
    )

    dt = solver_params.dt_ms
    n_steps = solver_params.n_steps

    V_rest, w_rest = fhn.rest_state(fhn_params)

    t_values = np.empty(n_steps + 1)
    V_values = np.empty(n_steps + 1)
    w_values = np.empty(n_steps + 1)

    # The state is held as one-element arrays so the same stepper functions
    # used for the strand can drive it unchanged.
    V = np.array([V_rest])
    w = np.array([w_rest])

    t_values[0] = 0.0
    V_values[0] = V[0]
    w_values[0] = w[0]

    def rhs(
        t: float, V_in: np.ndarray, w_in: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Space-clamped right-hand side: kinetics only, no diffusion."""
        i_stim = amplitude if stimulus.is_stimulus_active(t, stim_params) else 0.0
        return (
            fhn.reaction_f(V_in, w_in, i_stim),
            fhn.recovery_g(V_in, w_in, fhn_params),
        )

    step = solvers.get_stepper(solver_params.method)

    for step_index in range(n_steps):
        t = step_index * dt
        V, w = step(t, V, w, dt, rhs)
        t_values[step_index + 1] = t + dt
        V_values[step_index + 1] = V[0]
        w_values[step_index + 1] = w[0]

    return t_values, V_values, w_values
