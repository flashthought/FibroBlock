"""Every parameter used anywhere in FibroBlock, in one place.

This module is the **single source of truth** for numbers. No other module in
``src/`` may contain a numeric literal unless it is an exact mathematical
constant (such as the ``2`` in a centred second difference, or the ``1/3`` in
the FitzHugh-Nagumo cubic) accompanied by a comment saying so.

All parameter containers are ``@dataclass(frozen=True)``. Freezing them means a
configuration cannot be mutated half-way through a run, so a result file's
recorded configuration is guaranteed to be the configuration that produced it.

Units convention
----------------
**One FitzHugh-Nagumo dimensionless time unit is DECLARED equal to 1 ms.**

This is a stated modelling assumption, not a physical derivation. The
FitzHugh-Nagumo system is dimensionless; attaching millisecond and centimetre
labels to it is what lets us compare the computed conduction velocity against
published cardiac values. The consequence is that ``D`` carries units of
cm^2/ms and ``eps`` is a pure number. See assumption A1 in
``docs/assumption_register.md``.

Notes
-----
Every field below carries its units in the trailing comment or in the class
docstring. Fields without units are dimensionless by construction.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Literal

# --- Type aliases for the selectable numerical options ----------------------
# Written as Literal types so that a typo such as "harmonics" is caught by a
# type checker rather than silently falling through to a default.
AveragingScheme = Literal["harmonic", "arithmetic"]
IntegratorName = Literal["euler", "rk4"]
ActivationRule = Literal["v_zero_crossing", "max_dvdt"]


@dataclass(frozen=True)
class FHNParams:
    """FitzHugh-Nagumo reaction kinetics parameters.

    The kinetics are

    .. math::
        f(V, w) = V - \\tfrac{1}{3} V^{3} - w + I_{\\text{stim}}
        \\qquad
        g(V, w) = \\varepsilon\\,(V + a - b w)

    Attributes
    ----------
    a : float
        Shifts the ``w``-nullcline. Dimensionless. Assignment value 0.7.
    b : float
        Slope parameter of the ``w``-nullcline. Dimensionless. Value 0.8.
    eps : float
        Time-scale separation between the fast voltage variable and the slow
        recovery variable. Dimensionless. Value 0.08. Small ``eps`` is what
        makes the medium *excitable*: recovery is slow compared with the
        upstroke, so a supra-threshold stimulus produces a long excursion
        before returning to rest.
    time_unit_ms : float
        Number of milliseconds represented by one dimensionless FHN time unit.
        Declared to be 1.0 (assumption A1). Kept as an explicit parameter
        rather than an implicit 1 so that the assumption is visible in every
        saved configuration and can be varied to test its effect.
    """

    a: float = 0.7  # dimensionless
    b: float = 0.8  # dimensionless
    eps: float = 0.08  # dimensionless
    time_unit_ms: float = 1.0  # ms per dimensionless FHN time unit (assumption A1)

    def __post_init__(self) -> None:
        """Validate the kinetics parameters.

        Raises
        ------
        ValueError
            If ``b`` or ``eps`` is non-positive, or ``time_unit_ms`` is
            non-positive. ``b <= 0`` would invert the recovery nullcline and
            ``eps <= 0`` would remove or reverse recovery entirely; neither is
            a cardiac model.
        """
        if self.b <= 0.0:
            raise ValueError(f"FHNParams.b must be positive, got {self.b}")
        if self.eps <= 0.0:
            raise ValueError(f"FHNParams.eps must be positive, got {self.eps}")
        if self.time_unit_ms <= 0.0:
            raise ValueError(
                f"FHNParams.time_unit_ms must be positive, got {self.time_unit_ms}"
            )


@dataclass(frozen=True)
class GridParams:
    """Uniform 1-D spatial grid over the strand.

    Nodes are placed at ``x_j = j * dx_cm`` for ``j = 0 .. n_intervals``, so a
    2.0 cm strand at ``dx = 0.01 cm`` has 200 intervals and **201 nodes**.

    Attributes
    ----------
    length_cm : float
        Strand length ``L``. cm. Assignment value 2.0.
    dx_cm : float
        Node spacing. cm. Default 0.01, which is roughly one myocyte length and
        places about 10 nodes across the front thickness
        ``sqrt(D / |f_V|) ~ 0.032 cm``. Justified in
        ``docs/numerical_choices.md``; it is not an arbitrary round number.
    baseline_D : float
        Baseline diffusion coefficient ``D0``, representing healthy
        intercellular coupling. cm^2/ms. Assignment value 0.001.
    """

    length_cm: float = 2.0  # cm
    dx_cm: float = 0.01  # cm
    baseline_D: float = 0.001  # cm^2/ms

    def __post_init__(self) -> None:
        """Validate the grid and check that the length is a whole number of cells.

        Raises
        ------
        ValueError
            If any quantity is non-positive, or if ``length_cm / dx_cm`` is not
            (within floating-point tolerance) an integer. A non-integer ratio
            would silently shorten or lengthen the strand, which would corrupt
            every distance-based measurement including conduction velocity.
        """
        if self.length_cm <= 0.0:
            raise ValueError(
                f"GridParams.length_cm must be positive, got {self.length_cm}"
            )
        if self.dx_cm <= 0.0:
            raise ValueError(f"GridParams.dx_cm must be positive, got {self.dx_cm}")
        if self.baseline_D <= 0.0:
            raise ValueError(
                f"GridParams.baseline_D must be positive, got {self.baseline_D}"
            )

        ratio = self.length_cm / self.dx_cm
        nearest_integer = round(ratio)
        # Tolerance is relative: 1e-9 of the ratio itself, so it scales sensibly
        # whether the strand holds 20 cells or 20 000.
        if abs(ratio - nearest_integer) > 1e-9 * max(1.0, ratio):
            raise ValueError(
                f"length_cm / dx_cm = {ratio} is not an integer; "
                f"choose dx_cm that divides length_cm exactly."
            )

    @property
    def n_intervals(self) -> int:
        """Number of grid intervals (cells). ``L / dx``."""
        return round(self.length_cm / self.dx_cm)

    @property
    def n_nodes(self) -> int:
        """Number of grid nodes. One more than the number of intervals."""
        return self.n_intervals + 1


@dataclass(frozen=True)
class GapParams:
    """The reduced-coupling gap that models a fibrotic patch.

    The diffusion profile is piecewise constant:

    .. math::
        D(x) = \\rho D_0 \\ \\text{for}\\ x \\in [x_{gap},\\, x_{gap}+L_{gap}],
        \\qquad D(x) = D_0 \\ \\text{otherwise}

    Attributes
    ----------
    rho : float
        Coupling ratio inside the gap, in ``(0, 1]``. ``rho = 1`` is a healthy
        homogeneous strand; ``rho -> 0`` is complete electrical uncoupling.
        Dimensionless.
    gap_length_cm : float
        Length ``L_gap`` of the reduced-coupling patch. cm.
    gap_centre_cm : float
        Centre of the patch. cm. Assignment value 1.0, i.e. the middle of the
        2 cm strand, far enough from the stimulus that the wave has reached
        steady propagation before it arrives.
    averaging : {"harmonic", "arithmetic"}
        Scheme used to build interface (half-node) diffusion coefficients.
        **Harmonic is the physically correct default**: ``D`` is inversely
        proportional to axial resistance, and resistances in series add, so the
        resistance-correct interface value is the harmonic mean. Arithmetic is
        provided only so the difference can be shown as a numerical-choice
        sensitivity result -- it over-predicts coupling across a sharp
        interface and therefore shifts the block threshold.
    """

    rho: float = 1.0  # dimensionless coupling ratio in (0, 1]
    gap_length_cm: float = 0.1  # cm
    gap_centre_cm: float = 1.0  # cm
    averaging: AveragingScheme = "harmonic"

    def __post_init__(self) -> None:
        """Validate the gap description.

        Raises
        ------
        ValueError
            If ``rho`` is outside ``(0, 1]``, if the gap length is negative, or
            if the averaging scheme is not recognised. ``rho = 0`` is excluded
            because it makes the harmonic mean identically zero and severs the
            strand into two independent halves -- a different problem, not a
            limiting case of this one.
        """
        if not (0.0 < self.rho <= 1.0):
            raise ValueError(
                f"GapParams.rho must lie in (0, 1], got {self.rho}. "
                f"rho = 0 severs the strand and is a different problem."
            )
        if self.gap_length_cm < 0.0:
            raise ValueError(
                f"GapParams.gap_length_cm must be non-negative, "
                f"got {self.gap_length_cm}"
            )
        if self.averaging not in ("harmonic", "arithmetic"):
            raise ValueError(
                f"GapParams.averaging must be 'harmonic' or 'arithmetic', "
                f"got {self.averaging!r}"
            )

    @property
    def gap_start_cm(self) -> float:
        """Left edge of the gap, ``x_gap``. cm."""
        return self.gap_centre_cm - 0.5 * self.gap_length_cm

    @property
    def gap_end_cm(self) -> float:
        """Right edge of the gap, ``x_gap + L_gap``. cm."""
        return self.gap_centre_cm + 0.5 * self.gap_length_cm


@dataclass(frozen=True)
class StimulusParams:
    """Rectangular stimulus current applied near the left end of the strand.

    ``I_stim = amplitude`` for ``x`` in ``[0, width_cm]`` and ``t`` in
    ``[start_ms, start_ms + duration_ms)``; zero elsewhere.

    Attributes
    ----------
    amplitude : float
        Stimulus current in the same dimensionless units as ``f(V, w)``. The
        default 1.0 is comfortably supra-threshold: it raises ``V`` by about
        1.0 over the 1 ms pulse, taking the tissue from rest at ``V* = -1.199``
        to roughly ``-0.2``, well past the excitation threshold at
        ``V2 = -0.786``. ``experiments/ex01_single_cell.py`` measures the true
        threshold by bisection rather than assuming it.
    width_cm : float
        Spatial extent of the stimulated region, measured from ``x = 0``. cm.
        Assignment value 0.1.
    duration_ms : float
        Pulse duration. ms. Assignment value 1.0.
    start_ms : float
        Time at which the pulse begins. ms. Zero by default, so the stimulus is
        applied "for the first 1 ms" as the brief specifies.
    """

    amplitude: float = 1.0  # dimensionless current density (same units as f)
    width_cm: float = 0.1  # cm
    duration_ms: float = 1.0  # ms
    start_ms: float = 0.0  # ms

    def __post_init__(self) -> None:
        """Validate the stimulus description.

        Raises
        ------
        ValueError
            If the width or duration is negative, or the start time is
            negative.
        """
        if self.width_cm < 0.0:
            raise ValueError(
                f"StimulusParams.width_cm must be non-negative, got {self.width_cm}"
            )
        if self.duration_ms < 0.0:
            raise ValueError(
                f"StimulusParams.duration_ms must be non-negative, "
                f"got {self.duration_ms}"
            )
        if self.start_ms < 0.0:
            raise ValueError(
                f"StimulusParams.start_ms must be non-negative, got {self.start_ms}"
            )


@dataclass(frozen=True)
class SolverParams:
    """Time-integration settings for the hand-coded explicit schemes.

    Attributes
    ----------
    dt_ms : float
        Time step. ms. Default 0.02, giving a safety factor of about 2.3 below
        the computed explicit-Euler limit of 0.04651 ms at ``dx = 0.01 cm`` and
        ``D = 0.001 cm^2/ms``.
    t_end_ms : float
        Total simulated duration. ms.
    method : {"euler", "rk4"}
        Time integrator. Explicit Euler is the primary scheme because part (c)
        of the assignment requires demonstrating *its* stability limit. RK4 is
        available purely as an accuracy comparison. Neither is adaptive: an
        adaptive controller would silently shrink the step near the limit and
        hide the phenomenon being studied.
    f_v_bound : float
        Bound on ``|df/dV| = |1 - V^2|`` used in the stability estimate. Over
        an action potential ``V`` ranges from about ``-1.2`` to ``+1.99``, so
        ``V^2`` reaches about 4 and ``|1 - V^2|`` reaches about 3. This is a
        modelling bound, not an exact constant, which is why it lives in the
        configuration where it can be varied.
    record_every : int
        Store a spatial snapshot every ``record_every`` steps. Snapshots are
        always recorded (they are cheap); the full ``V(x, t)`` history is not.
    store_full_history : bool
        If True, keep every time level of ``V``. Off by default because a
        300 ms run at ``dt = 0.02 ms`` on 201 nodes is 15 000 x 201 floats per
        field, and none of the reported measurements need it.
    """

    dt_ms: float = 0.02  # ms
    t_end_ms: float = 300.0  # ms
    method: IntegratorName = "euler"
    f_v_bound: float = 3.0  # bound on |1 - V^2| over an action potential
    record_every: int = 25  # steps between stored snapshots
    store_full_history: bool = False

    def __post_init__(self) -> None:
        """Validate the solver settings.

        Raises
        ------
        ValueError
            If the step or duration is non-positive, the method is unknown, the
            bound on ``|f_V|`` is negative, or ``record_every`` is not a
            positive integer.
        """
        if self.dt_ms <= 0.0:
            raise ValueError(f"SolverParams.dt_ms must be positive, got {self.dt_ms}")
        if self.t_end_ms <= 0.0:
            raise ValueError(
                f"SolverParams.t_end_ms must be positive, got {self.t_end_ms}"
            )
        if self.method not in ("euler", "rk4"):
            raise ValueError(
                f"SolverParams.method must be 'euler' or 'rk4', got {self.method!r}"
            )
        if self.f_v_bound < 0.0:
            raise ValueError(
                f"SolverParams.f_v_bound must be non-negative, got {self.f_v_bound}"
            )
        if self.record_every < 1:
            raise ValueError(
                f"SolverParams.record_every must be >= 1, got {self.record_every}"
            )

    @property
    def n_steps(self) -> int:
        """Number of time steps needed to reach ``t_end_ms``.

        Rounded up, so the simulation never stops short of the requested end
        time.

        Notes
        -----
        The division is rounded to nine decimal places before the ceiling is
        taken. Without that, ``300.0 / 0.02`` evaluating to 14999.999999999998
        would round up to 15000 correctly, but ``t_end / dt`` landing a hair
        *above* an integer would add one spurious extra step and shift every
        recorded time by ``dt``. Nine places is far below any step size that
        would be sensible here and far above the round-off being suppressed.
        """
        steps_exact = round(self.t_end_ms / self.dt_ms, 9)
        return math.ceil(steps_exact)


@dataclass(frozen=True)
class MeasurementParams:
    """How derived quantities are extracted from a completed simulation.

    Attributes
    ----------
    activation_rule : {"v_zero_crossing", "max_dvdt"}
        Definition of "the wave arrived here". ``v_zero_crossing`` records the
        time at which ``V`` first crosses ``activation_level``;
        ``max_dvdt`` records the time of steepest upstroke. The two differ
        slightly (typically well under a millisecond) and the report must state
        which was used -- hence a configuration option rather than a hard-wired
        choice.
    activation_level : float
        Voltage level defining activation for the crossing rule. Zero, which
        sits between the threshold root ``V2 = -0.786`` and the excited root
        ``V3 = +1.986`` and is therefore crossed exactly once per upstroke.
    cv_fit_skip_start_cm : float
        Distance from the stimulus discarded before fitting conduction
        velocity. cm. The wave is still forming out of the stimulus over
        roughly the first half-centimetre, so including it biases the fit.
    cv_fit_skip_end_cm : float
        Distance from the far boundary discarded before fitting. cm. The sealed
        end reflects and accelerates the front as it arrives.
    block_margin_cm : float
        How far beyond the downstream edge of the gap a node must be before its
        activation counts as successful propagation. cm. Value 0.3, which is
        about sixteen front thicknesses -- comfortably past the region where
        electrotonic spread alone can raise ``V`` without a regenerative
        upstroke. The same offset is used symmetrically on the upstream side to
        place the probe for the conduction-delay measurement, so delay is
        measured between the two points that bracket the region where block is
        judged.
    block_window_ms : float
        Time after the stimulus within which activation must occur for
        propagation to count as successful. ms. Value 200.
    """

    activation_rule: ActivationRule = "v_zero_crossing"
    activation_level: float = 0.0  # dimensionless voltage
    cv_fit_skip_start_cm: float = 0.5  # cm
    cv_fit_skip_end_cm: float = 0.2  # cm
    block_margin_cm: float = 0.3  # cm
    block_window_ms: float = 200.0  # ms

    def __post_init__(self) -> None:
        """Validate the measurement settings.

        Raises
        ------
        ValueError
            If the activation rule is unknown or any distance/time window is
            negative.
        """
        if self.activation_rule not in ("v_zero_crossing", "max_dvdt"):
            raise ValueError(
                f"MeasurementParams.activation_rule must be 'v_zero_crossing' "
                f"or 'max_dvdt', got {self.activation_rule!r}"
            )
        for name in (
            "cv_fit_skip_start_cm",
            "cv_fit_skip_end_cm",
            "block_margin_cm",
            "block_window_ms",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(
                    f"MeasurementParams.{name} must be non-negative, got {value}"
                )


@dataclass(frozen=True)
class RunConfig:
    """Complete description of one simulation run.

    This is what gets serialised alongside every result file, so that any
    figure in the report can be traced back to the exact numbers that produced
    it.

    Attributes
    ----------
    fhn : FHNParams
        Reaction kinetics.
    grid : GridParams
        Spatial discretisation and baseline coupling.
    gap : GapParams
        Reduced-coupling patch.
    stimulus : StimulusParams
        Stimulus current.
    solver : SolverParams
        Time integration.
    measurement : MeasurementParams
        Post-processing definitions.
    seed : int
        Random seed. The model is fully deterministic and uses no random
        numbers, but the brief requires seeds to be fixed and reported, so it
        is set and printed anyway. If randomness is ever added (for example
        heterogeneous coupling), this is where it is controlled from.
    label : str
        Short human-readable name for the run, used in filenames and figure
        titles.
    """

    fhn: FHNParams = field(default_factory=FHNParams)
    grid: GridParams = field(default_factory=GridParams)
    gap: GapParams = field(default_factory=GapParams)
    stimulus: StimulusParams = field(default_factory=StimulusParams)
    solver: SolverParams = field(default_factory=SolverParams)
    measurement: MeasurementParams = field(default_factory=MeasurementParams)
    seed: int = 20260810  # fixed and reported, per the brief
    label: str = "default"

    def to_dict(self) -> dict[str, Any]:
        """Return the whole configuration as a plain nested dictionary.

        Returns
        -------
        dict
            Nested ``{section: {field: value}}`` mapping containing only JSON-
            serialisable types, suitable for writing next to a result file.

        Notes
        -----
        ``dataclasses.asdict`` recurses into the nested frozen dataclasses
        automatically, so adding a field to any section is picked up here with
        no further work -- there is no hand-maintained list to fall out of date.
        """
        return dataclasses.asdict(self)

    def replace(self, **changes: Any) -> RunConfig:
        """Return a copy of this configuration with top-level fields replaced.

        Parameters
        ----------
        **changes
            Top-level field names of :class:`RunConfig` (for example
            ``gap=GapParams(rho=0.2)`` or ``label="my_run"``).

        Returns
        -------
        RunConfig
            A new frozen configuration. The original is untouched, which is the
            whole reason the dataclasses are frozen: sweeps build many
            configurations without any chance of one leaking into another.
        """
        return dataclasses.replace(self, **changes)


# ---------------------------------------------------------------------------
# Analytic reference values quoted in the assignment brief.
#
# These are NOT used by the solver. They exist only so that the verification
# tests and docs/verification_log.md can compare what the code computes from
# first principles against what the brief states, and report the discrepancy
# honestly. Everything the solver uses is derived at run time in fhn.py.
# ---------------------------------------------------------------------------

BRIEF_REST_V: float = -1.199408  # rest potential quoted in the brief
BRIEF_REST_W: float = -0.624260  # rest recovery variable quoted in the brief
BRIEF_JACOBIAN_TRACE: float = -0.5034  # brief's tr(J); see verification log
BRIEF_JACOBIAN_DET: float = 0.1081  # brief's det(J)
BRIEF_CV_PREFACTOR: float = 0.9634  # brief's theta / sqrt(D)
BRIEF_DT_LIMIT_MS: float = 0.04651  # brief's reaction-diffusion Euler limit
BRIEF_BISTABLE_ROOTS: tuple[float, float, float] = (-1.19941, -0.78638, +1.98579)


def default_config() -> RunConfig:
    """Build the standard configuration used by most experiments.

    Returns
    -------
    RunConfig
        A healthy homogeneous strand (``rho = 1``) with the assignment's
        parameter values, integrated with explicit Euler at ``dt = 0.02 ms``.

    Notes
    -----
    Experiments start from this and call :meth:`RunConfig.replace` to vary one
    thing at a time, which keeps every sweep anchored to the same baseline.
    """
    return RunConfig()
