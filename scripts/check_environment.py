"""Print the environment provenance for the report appendix.

Run this and paste the output into the report's appendix, or redirect it to a
file. It reports everything an examiner would need to reproduce the numbers
exactly: interpreter, library versions, platform, git commit, and whether the
working tree is clean.

Usage::

    python scripts/check_environment.py
    python scripts/check_environment.py --json > results/environment.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fibroblock import config as cfg  # noqa: E402
from fibroblock import fhn, solvers, utils  # noqa: E402


def print_human_readable() -> None:
    """Print the provenance as a readable block."""
    record = utils.provenance()

    print("=" * 70)
    print("FibroBlock environment report")
    print("=" * 70)

    print("\n-- Provenance ---------------------------------------------------")
    print(f"  generated (UTC) : {record['generated_utc']}")
    print(f"  git commit      : {record['git_commit']}")
    print(f"  project root    : {utils.project_root()}")

    print("\n-- Interpreter and libraries ------------------------------------")
    for name, version in record["libraries"].items():
        print(f"  {name:<12} : {version}")

    print("\n-- Platform ------------------------------------------------------")
    for name, value in record["platform"].items():
        print(f"  {name:<12} : {value}")

    # The baseline configuration and the quantities derived from it, so the
    # appendix records not just the software but the model it was running.
    config = cfg.default_config()
    print("\n-- Baseline configuration ---------------------------------------")
    print(f"  seed             : {config.seed}")
    print(f"  a, b, eps        : {config.fhn.a}, {config.fhn.b}, {config.fhn.eps}")
    print(f"  L, dx, nodes     : {config.grid.length_cm} cm, "
          f"{config.grid.dx_cm} cm, {config.grid.n_nodes}")
    print(f"  D0               : {config.grid.baseline_D} cm^2/ms")
    print(f"  dt, method       : {config.solver.dt_ms} ms, {config.solver.method}")
    print(f"  averaging        : {config.gap.averaging}")
    print(f"  activation rule  : {config.measurement.activation_rule}")
    print(f"  time unit        : 1 FHN time unit = {config.fhn.time_unit_ms} ms "
          f"(assumption A1)")

    print("\n-- Derived quantities --------------------------------------------")
    V_rest, w_rest = fhn.rest_state(config.fhn)
    summary = fhn.excitability(config.fhn)
    limits = solvers.stability_limits(
        config.grid.baseline_D,
        config.grid.dx_cm,
        config.solver.f_v_bound,
        config.solver.dt_ms,
    )
    print(f"  rest state       : V* = {V_rest:.9f}, w* = {w_rest:.9f}")
    print(f"  classification   : {summary.classification} "
          f"(tr = {summary.trace:+.6f}, det = {summary.determinant:+.6f})")
    theta = fhn.analytic_cv(config.grid.baseline_D, config.fhn)
    print(f"  analytic theta   : {theta:.6f} cm/ms at D = {config.grid.baseline_D}")
    print(f"  dt limit         : {limits.reaction_diffusion_dt_ms:.6f} ms "
          f"(reaction-diffusion)")
    print(f"                     {limits.pure_diffusion_dt_ms:.6f} ms "
          f"(pure diffusion, {100 * limits.relative_overestimate:.1f}% too optimistic)")
    print(f"  safety factor    : {limits.safety_factor:.2f}")

    print("\n" + "=" * 70)
    if "dirty" in record["git_commit"]:
        print("WARNING: the working tree has uncommitted changes. The committed")
        print("figures may not correspond exactly to the current source.")
    elif "unavailable" in record["git_commit"]:
        print("NOTE: not a git checkout, so the commit hash could not be read.")
    else:
        print("Working tree is clean; the commit above identifies this code exactly.")
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    """Print the environment report.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.

    Returns
    -------
    int
        Always 0. This script only reports; it never fails a build.
    """
    parser = argparse.ArgumentParser(
        description="Print environment provenance for the report appendix."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the readable block.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Also write results/environment.json.",
    )
    arguments = parser.parse_args(argv)

    if arguments.json:
        print(json.dumps(utils.provenance(cfg.default_config().to_dict()), indent=2))
    else:
        print_human_readable()

    if arguments.save:
        path = utils.save_metadata(
            "environment", utils.provenance(cfg.default_config().to_dict())
        )
        print(f"\nWrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
