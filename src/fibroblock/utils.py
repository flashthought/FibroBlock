"""Housekeeping: seeding, timing, provenance stamping, and file output.

None of this is physics, but the reproducibility requirement lives here. Every
saved array and every figure is accompanied by a provenance record stating the
timestamp, git commit, platform, library versions and the exact configuration
that produced it, so any number in the report can be traced back to the run
that generated it.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import fibroblock

# Name of the file that marks the repository root. Used to locate figures/ and
# results/ no matter which directory an experiment is launched from.
ROOT_MARKER = "pyproject.toml"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def project_root() -> Path:
    """Locate the repository root directory.

    Returns
    -------
    Path
        The directory containing ``pyproject.toml``.

    Raises
    ------
    RuntimeError
        If no such directory is found, which means the package has been
        installed somewhere detached from its source tree and the output
        directories cannot be located.

    Notes
    -----
    Two searches are tried: upwards from this module's own location (correct
    for the editable install the README prescribes), then upwards from the
    current working directory (a fallback for unusual installs). Hard-coding a
    path would break the moment the examiner clones to a different drive.
    """
    candidates = [Path(fibroblock.__file__).resolve(), Path.cwd().resolve()]

    for start in candidates:
        for directory in [start, *start.parents]:
            if (directory / ROOT_MARKER).is_file():
                return directory

    raise RuntimeError(
        f"Could not locate the project root: no {ROOT_MARKER} found above "
        f"{candidates[0]} or {candidates[1]}."
    )


def figures_dir() -> Path:
    """Return the ``figures/`` directory, creating it if necessary."""
    return ensure_dir(project_root() / "figures")


def results_dir() -> Path:
    """Return the ``results/`` directory, creating it if necessary."""
    return ensure_dir(project_root() / "results")


def report_figures_dir() -> Path:
    """Return ``report/figures/``, creating it if necessary."""
    return ensure_dir(project_root() / "report" / "figures")


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it does not exist, and return it.

    Parameters
    ----------
    path : Path
        Directory to create.

    Returns
    -------
    Path
        The same path, now guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def set_seed(seed: int, announce: bool = True) -> np.random.Generator:
    """Fix and report the random seed.

    Parameters
    ----------
    seed : int
        The seed to use.
    announce : bool, optional
        Print the seed to stdout. Default True.

    Returns
    -------
    numpy.random.Generator
        A seeded generator.

    Notes
    -----
    The model is fully deterministic and consumes no random numbers, so this
    changes nothing about the results. It is done anyway because the brief
    requires seeds to be fixed and reported, and because a generator that is
    already threaded through the code is one that cannot be forgotten if
    heterogeneous coupling is added later.

    A ``Generator`` is used rather than the legacy ``numpy.random.seed`` global
    state, so that no library call anywhere can silently disturb the stream.
    """
    if announce:
        print(f"[seed] random seed fixed at {seed} (model is deterministic)")
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class Timer:
    """Context manager measuring wall-clock elapsed time.

    Parameters
    ----------
    label : str
        Name printed alongside the elapsed time.
    announce : bool, optional
        Print on exit. Default True.

    Attributes
    ----------
    seconds : float
        Elapsed time, available after the block exits.

    Examples
    --------
    >>> with Timer("integration") as t:  # doctest: +SKIP
    ...     run_simulation(config)
    >>> t.seconds                        # doctest: +SKIP
    0.83
    """

    def __init__(self, label: str, announce: bool = True) -> None:
        self.label = label
        self.announce = announce
        self.seconds: float = float("nan")
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        """Start the clock."""
        # perf_counter, not time(): it is monotonic and has the highest
        # resolution available, and is unaffected by system clock adjustments.
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stop the clock and optionally report."""
        self.seconds = time.perf_counter() - self._start
        if self.announce:
            print(f"[time] {self.label}: {self.seconds:.2f} s")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def git_commit_hash() -> str:
    """Return the current git commit hash, or a clear marker if unavailable.

    Returns
    -------
    str
        Full 40-character hash, with ``"-dirty"`` appended if the working tree
        has uncommitted changes; or ``"unavailable (not a git checkout)"``.

    Notes
    -----
    Never raises. A missing git installation or a downloaded ZIP rather than a
    clone must not stop the pipeline -- it only makes the provenance less
    precise, which is recorded honestly rather than hidden.
    """
    try:
        root = project_root()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()

        return f"{commit}-dirty" if status else commit
    except (
        subprocess.SubprocessError,
        FileNotFoundError,
        OSError,
        RuntimeError,
    ):
        return "unavailable (not a git checkout)"


def library_versions() -> dict[str, str]:
    """Collect version strings for every library that can affect the numbers.

    Returns
    -------
    dict
        Mapping of package name to version string.
    """
    import matplotlib
    import scipy

    return {
        "fibroblock": fibroblock.__version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def platform_summary() -> dict[str, str]:
    """Collect a description of the machine, for the report appendix.

    Returns
    -------
    dict
        Operating system, release, machine architecture and processor.
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def provenance(config_dict: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the provenance record attached to every result and figure.

    Parameters
    ----------
    config_dict : mapping, optional
        Output of :meth:`fibroblock.config.RunConfig.to_dict`. Omitted for
        artefacts that are not tied to one particular run.

    Returns
    -------
    dict
        JSON-serialisable record containing an ISO-8601 UTC timestamp, the git
        commit, library versions, platform description and the configuration.
    """
    record: dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit_hash(),
        "libraries": library_versions(),
        "platform": platform_summary(),
    }
    if config_dict is not None:
        record["config"] = dict(config_dict)
    return record


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SavedArtefact:
    """Record of one file written by an experiment.

    Attributes
    ----------
    path : Path
        Where the file was written.
    kind : str
        ``"figure"``, ``"array"``, ``"table"`` or ``"metadata"``.
    caption : str
        For figures, the takeaway caption stored in the metadata.
    """

    path: Path
    kind: str
    caption: str = ""


def save_arrays(
    name: str,
    arrays: Mapping[str, np.ndarray],
    config_dict: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save named arrays to ``results/<name>.npz`` with a provenance sidecar.

    Parameters
    ----------
    name : str
        Base filename, without extension.
    arrays : mapping of str to ndarray
        Arrays to store.
    config_dict : mapping, optional
        Configuration that produced them.
    extra_metadata : mapping, optional
        Any additional scalars worth recording (measured velocities, fit
        quality, and so on).

    Returns
    -------
    npz_path, json_path : Path
        The two files written.

    Notes
    -----
    The provenance is written to a *separate* JSON file rather than stuffed
    into the ``.npz``. NumPy would have to pickle a dictionary to store it,
    and loading pickled data requires ``allow_pickle=True``, which is both a
    security foot-gun and a portability risk. A plain JSON sidecar is readable
    by anything, including the examiner's text editor.
    """
    npz_path = results_dir() / f"{name}.npz"
    json_path = results_dir() / f"{name}.json"

    np.savez_compressed(npz_path, **arrays)

    record = provenance(config_dict)
    record["arrays"] = {key: list(value.shape) for key, value in arrays.items()}
    if extra_metadata is not None:
        record["measurements"] = _jsonable(extra_metadata)

    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return npz_path, json_path


def save_table(
    name: str,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> Path:
    """Save a summary table to ``results/<name>.csv``.

    Parameters
    ----------
    name : str
        Base filename, without extension.
    header : sequence of str
        Column names. Include units in the names, for example
        ``"theta_cm_per_ms"``.
    rows : iterable of sequences
        Row values.

    Returns
    -------
    Path
        The file written.

    Notes
    -----
    ``newline=""`` is required on Windows: without it, Python's text layer and
    the ``csv`` module each append a carriage return and every row ends up
    separated by a blank line.
    """
    path = results_dir() / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
    return path


def save_metadata(name: str, record: Mapping[str, Any]) -> Path:
    """Save a standalone JSON metadata record to ``results/<name>.json``.

    Parameters
    ----------
    name : str
        Base filename, without extension.
    record : mapping
        Content to write. Converted to JSON-safe types first.

    Returns
    -------
    Path
        The file written.
    """
    path = results_dir() / f"{name}.json"
    path.write_text(json.dumps(_jsonable(record), indent=2), encoding="utf-8")
    return path


def _jsonable(value: Any) -> Any:
    """Recursively convert NumPy scalars and arrays into plain Python types.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    Any
        The same structure using only ``dict``, ``list``, ``str``, ``float``,
        ``int``, ``bool`` and ``None``.

    Notes
    -----
    ``json.dumps`` refuses ``numpy.float64`` and ``numpy.bool_``, which is
    exactly what measurement dictionaries are full of. Converting here means no
    caller has to remember to cast. NaN is preserved rather than replaced,
    because a NaN activation time is meaningful data: it means the wave never
    arrived.
    """
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def format_duration(seconds: float) -> str:
    """Format an elapsed time for the pipeline summary table.

    Parameters
    ----------
    seconds : float
        Elapsed time.

    Returns
    -------
    str
        For example ``"0.83 s"`` or ``"2 m 07 s"``.
    """
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    remainder = seconds - 60 * minutes
    return f"{minutes} m {remainder:04.1f} s"
