"""Regenerate every figure and every results file, from empty.

This is **the reproducibility entry point**. It deletes ``figures/``,
``results/`` and ``report/figures/``, runs all eight experiments in order, and
prints a summary table of what was produced and how long each stage took.

Run it with::

    python scripts/make_all_figures.py

There are no manual steps, no arguments to remember, and no ordering the user
has to know: experiments are independent of one another and each rebuilds
whatever it needs from ``config.py``.

Exit status is 0 only if every stage succeeded, so this script can be used as a
CI check.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

# Put both src/ and experiments/ on the path so this works from a clean clone
# whether or not the package has been pip-installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from fibroblock import utils  # noqa: E402

# Experiments in report order. Each module must expose main() -> dict.
EXPERIMENT_MODULES: tuple[str, ...] = (
    "ex01_single_cell",
    "ex02_stability",
    "ex03_convergence",
    "ex04_pure_diffusion",
    "ex05_cv_vs_D",
    "ex06_block_threshold",
    "ex07_conduction_delay",
    "ex08_sensitivity",
)

# Directories emptied before the run, so that "it regenerates from empty" is
# demonstrated rather than asserted.
GENERATED_DIRECTORIES: tuple[str, ...] = ("figures", "results", "report/figures")


@dataclass
class StageResult:
    """Outcome of one experiment.

    Attributes
    ----------
    name : str
        Module name.
    seconds : float
        Wall-clock duration.
    figures : int
        Number of PNG figures present afterwards that were not present before.
    succeeded : bool
        Whether main() returned without raising.
    error : str
        Formatted traceback if it failed, otherwise empty.
    """

    name: str
    seconds: float
    figures: int
    succeeded: bool
    error: str = ""


def clean_generated_directories(announce: bool = True) -> None:
    """Delete and recreate every generated output directory.

    Parameters
    ----------
    announce : bool, optional
        Print what is being removed.

    Notes
    -----
    This is what makes the reproducibility claim meaningful. If the directories
    were merely overwritten, a figure whose generating code had been deleted
    would survive from a previous run and the examiner would see output that
    the current code cannot produce.
    """
    for relative in GENERATED_DIRECTORIES:
        directory = PROJECT_ROOT / relative
        if directory.exists():
            shutil.rmtree(directory)
            if announce:
                print(f"  removed {relative}/")
        directory.mkdir(parents=True, exist_ok=True)
        # Keep the directory in git even when empty.
        (directory / ".gitkeep").write_text("", encoding="utf-8")


def count_figures() -> int:
    """Count PNG files currently in ``figures/``."""
    return len(list((PROJECT_ROOT / "figures").glob("*.png")))


def run_experiment(module_name: str) -> StageResult:
    """Import an experiment module and run its ``main()``.

    Parameters
    ----------
    module_name : str
        Module name inside ``experiments/``.

    Returns
    -------
    StageResult
        Timing, figure count and success flag.

    Notes
    -----
    Failures are caught and recorded rather than allowed to abort the whole
    pipeline. If experiment 3 breaks, it is far more useful to see that 4 to 8
    still work than to see nothing at all -- and the non-zero exit status still
    reports the failure.
    """
    figures_before = count_figures()
    start = time.perf_counter()

    try:
        module = importlib.import_module(module_name)
        module.main()
        succeeded, error = True, ""
    except Exception:  # noqa: BLE001 -- deliberately broad; see Notes
        succeeded = False
        error = traceback.format_exc()
        print(f"\n!! {module_name} FAILED:\n{error}")

    seconds = time.perf_counter() - start
    return StageResult(
        name=module_name,
        seconds=seconds,
        figures=count_figures() - figures_before,
        succeeded=succeeded,
        error=error,
    )


def copy_figures_to_report() -> int:
    """Copy generated figures into ``report/figures/`` for the report build.

    Returns
    -------
    int
        Number of files copied.
    """
    source = PROJECT_ROOT / "figures"
    destination = utils.ensure_dir(PROJECT_ROOT / "report" / "figures")

    copied = 0
    for pattern in ("*.png", "*.pdf"):
        for path in sorted(source.glob(pattern)):
            shutil.copy2(path, destination / path.name)
            copied += 1
    return copied


def print_summary(stages: list[StageResult], total_seconds: float) -> None:
    """Print the pipeline summary table.

    Parameters
    ----------
    stages : list of StageResult
        Outcomes in run order.
    total_seconds : float
        Total wall-clock time.
    """
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  {'stage':<26} {'status':>9} {'figures':>9} {'time':>12}")
    print("  " + "-" * 60)

    for stage in stages:
        status = "ok" if stage.succeeded else "FAILED"
        print(
            f"  {stage.name:<26} {status:>9} {stage.figures:>9} "
            f"{utils.format_duration(stage.seconds):>12}"
        )

    print("  " + "-" * 60)
    total_figures = sum(stage.figures for stage in stages)
    print(
        f"  {'TOTAL':<26} {'':>9} {total_figures:>9} "
        f"{utils.format_duration(total_seconds):>12}"
    )

    failures = [stage.name for stage in stages if not stage.succeeded]
    if failures:
        print(f"\n  {len(failures)} stage(s) FAILED: {', '.join(failures)}")
    else:
        print("\n  All stages completed successfully.")


def main(argv: list[str] | None = None) -> int:
    """Run the whole pipeline.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status: 0 if every stage succeeded, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Regenerate all FibroBlock figures and results from empty."
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help=(
            "Do not delete figures/ and results/ first. Off by default: "
            "starting from empty is the whole point of this script."
        ),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="MODULE",
        help="Run only the named experiment modules (for development).",
    )
    arguments = parser.parse_args(argv)

    print("=" * 70)
    print("FibroBlock: regenerating all figures and results")
    print("=" * 70)

    provenance = utils.provenance()
    print(f"  git commit : {provenance['git_commit']}")
    print(f"  generated  : {provenance['generated_utc']}")
    print(f"  python     : {provenance['libraries']['python']}")
    print(
        f"  numpy      : {provenance['libraries']['numpy']}   "
        f"scipy {provenance['libraries']['scipy']}   "
        f"matplotlib {provenance['libraries']['matplotlib']}"
    )
    print(f"  platform   : {provenance['platform']['system']} "
          f"{provenance['platform']['release']} ({provenance['platform']['machine']})")

    modules = tuple(arguments.only) if arguments.only else EXPERIMENT_MODULES

    if arguments.keep_existing:
        print("\n  (keeping existing output; not a clean reproducibility test)")
    else:
        print("\n  Clearing generated directories:")
        clean_generated_directories()

    start = time.perf_counter()
    stages = []
    for index, module_name in enumerate(modules, start=1):
        print(f"\n[{index}/{len(modules)}] {module_name}")
        stages.append(run_experiment(module_name))

    copied = copy_figures_to_report()
    total_seconds = time.perf_counter() - start

    print_summary(stages, total_seconds)
    print(f"  Copied {copied} figure files into report/figures/.")

    # Record the pipeline run itself, so the examiner can see when the
    # committed outputs were produced and on what.
    utils.save_metadata(
        "pipeline_run",
        {
            "stages": [
                {
                    "name": stage.name,
                    "seconds": stage.seconds,
                    "figures": stage.figures,
                    "succeeded": stage.succeeded,
                }
                for stage in stages
            ],
            "total_seconds": total_seconds,
            "report_figure_files_copied": copied,
            **utils.provenance(),
        },
    )
    print("  Wrote results/pipeline_run.json.")

    return 0 if all(stage.succeeded for stage in stages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
