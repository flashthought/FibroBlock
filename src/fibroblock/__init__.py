"""FibroBlock: cardiac action-potential propagation and conduction block.

A 1-D monodomain cable model with FitzHugh-Nagumo kinetics, built to study how a
patch of reduced intercellular coupling (a model of fibrosis) slows and
eventually blocks a propagating action potential.

COE 562 -- Engineering Systems Design and Modelling, Problem 8.

Quick start
-----------
>>> from fibroblock import default_config, run_simulation, measure_velocity
>>> result = run_simulation(default_config())          # doctest: +SKIP
>>> measure_velocity(result).theta_cm_per_ms           # doctest: +SKIP
0.0255336...

The public API below is deliberately small: the configuration types, the two
entry points that run a model, and the handful of measurement functions that
turn a run into a reported number. Everything else -- the operators, the
integrators, the plotting helpers -- is reachable through the submodules, which
is where anyone reading the code in detail should be looking anyway.

Submodules
----------
config
    Every parameter, as frozen dataclasses. The single source of truth.
fhn
    FitzHugh-Nagumo kinetics, rest state, Jacobian, bistable roots, analytic
    conduction velocity.
grid
    Spatial grid, ``D(x)`` profiles, harmonic/arithmetic half-node averaging.
operators
    The conservative divergence operator with sealed (no-flux) ends.
solvers
    Hand-coded explicit Euler and RK4, and the von Neumann stability limit.
stimulus
    Stimulus current profiles and the threshold-bisection helper.
simulate
    ``run_simulation`` -- the single time loop.
measure
    Conduction velocity, block detection, conduction delay, observed order.
plotting
    House figure style; every helper labels axes with units.
utils
    Seeding, timing, provenance stamping, and file output.
"""

from __future__ import annotations

__version__ = "1.0.0"

# --- Configuration ----------------------------------------------------------
from fibroblock.config import (
    FHNParams,
    GapParams,
    GridParams,
    MeasurementParams,
    RunConfig,
    SolverParams,
    StimulusParams,
    default_config,
)

# --- Physics ----------------------------------------------------------------
from fibroblock.fhn import (
    analytic_cv,
    analytic_cv_prefactor,
    bistable_roots,
    excitability,
    rest_state,
)

# --- Measuring a run --------------------------------------------------------
from fibroblock.measure import (
    detect_block,
    measure_delay,
    measure_velocity,
)

# --- Running a model --------------------------------------------------------
from fibroblock.simulate import SimulationResult, run_simulation, run_single_cell

# --- Stability --------------------------------------------------------------
from fibroblock.solvers import explicit_euler_dt_limit, stability_limits

__all__ = [
    "__version__",
    # configuration
    "FHNParams",
    "GapParams",
    "GridParams",
    "MeasurementParams",
    "RunConfig",
    "SolverParams",
    "StimulusParams",
    "default_config",
    # physics
    "analytic_cv",
    "analytic_cv_prefactor",
    "bistable_roots",
    "excitability",
    "rest_state",
    # running
    "SimulationResult",
    "run_simulation",
    "run_single_cell",
    # measuring
    "detect_block",
    "measure_delay",
    "measure_velocity",
    # stability
    "explicit_euler_dt_limit",
    "stability_limits",
]
