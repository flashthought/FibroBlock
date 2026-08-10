"""House figure style and figure-saving with provenance.

One style, defined once, applied everywhere. Every helper here labels axes with
units, because a figure whose axes lack units is not evidence of anything.

Colour
------
The palette is **Okabe-Ito**, an eight-colour set designed to remain
distinguishable under the three common forms of colour-vision deficiency. It
replaces matplotlib's default cycle, whose first two colours (a blue and an
orange of similar luminance) are a poor choice for print and a bad one for
deuteranopic readers.

Every series that carries meaning is also distinguished by line style or
marker, so the figures survive being printed in greyscale.

References
----------
Okabe, M., Ito, K. (2008). Color Universal Design (CUD): how to make figures
and presentations that are friendly to colorblind people.
https://jfly.uni-koeln.de/color/
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

# Force the non-interactive Agg backend BEFORE pyplot is imported. Experiments
# run head-less from make_all_figures.py, and without this matplotlib may try
# to open a GUI window (or fail outright on a machine with no display).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402

from fibroblock import utils  # noqa: E402

# --- Okabe-Ito colour-blind-safe palette ------------------------------------
PALETTE: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

# Cycle order chosen so the first three series are maximally separated in both
# hue and luminance -- most figures here have two or three series.
COLOUR_CYCLE: list[str] = [
    PALETTE["blue"],
    PALETTE["vermillion"],
    PALETTE["green"],
    PALETTE["orange"],
    PALETTE["purple"],
    PALETTE["sky"],
    PALETTE["black"],
    PALETTE["yellow"],
]

# Semantic colours, so that "measured" is the same colour in every figure.
COLOUR_MEASURED: str = PALETTE["blue"]
COLOUR_ANALYTIC: str = PALETTE["vermillion"]
COLOUR_REFERENCE: str = PALETTE["black"]
COLOUR_GAP: str = PALETTE["orange"]
COLOUR_BLOCKED: str = PALETTE["vermillion"]
COLOUR_PROPAGATED: str = PALETTE["green"]

# Resolution required for the raster copies. 300 dpi is the usual minimum for
# printed figures; below it, axis labels visibly degrade.
FIGURE_DPI: int = 300

# Default figure size in inches, sized for a single column of an A4 report.
DEFAULT_FIGSIZE: tuple[float, float] = (6.5, 4.2)
WIDE_FIGSIZE: tuple[float, float] = (9.0, 4.2)
TALL_FIGSIZE: tuple[float, float] = (6.5, 7.5)


def use_house_style() -> None:
    """Apply the project's matplotlib settings.

    Notes
    -----
    Called automatically when this module is imported, so no experiment can
    forget it and quietly produce an off-style figure. It can also be called
    again by hand from a notebook after any experimentation with rcParams.
    """
    plt.rcParams.update(
        {
            # --- Colour ---
            "axes.prop_cycle": plt.cycler(color=COLOUR_CYCLE),
            # --- Type ---
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            # --- Lines and markers ---
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            # --- Axes ---
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            # --- Legend ---
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            # --- Output ---
            "figure.dpi": 110,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "figure.autolayout": False,
        }
    )


# Applied at import: see use_house_style's Notes.
use_house_style()


def new_figure(
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Any,
) -> tuple[plt.Figure, Any]:
    """Create a figure and axes in the house style.

    Parameters
    ----------
    figsize : tuple of float, optional
        Width and height in inches.
    nrows, ncols : int, optional
        Subplot grid.
    **kwargs
        Passed through to ``matplotlib.pyplot.subplots``.

    Returns
    -------
    fig : Figure
    axes : Axes or ndarray of Axes
    """
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)


def label_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str = "") -> None:
    """Label a set of axes. Both labels must carry units.

    Parameters
    ----------
    ax : Axes
        Axes to label.
    xlabel, ylabel : str
        Axis labels **including units**, for example
        ``"position $x$ (cm)"``.
    title : str, optional
        Axes title.

    Raises
    ------
    ValueError
        If either label is empty.

    Notes
    -----
    This function exists to make the units requirement structural. Every plot
    in the project routes through it, so an unlabelled axis would have to be
    created deliberately rather than by omission.

    Dimensionless quantities should say so explicitly -- ``"(dimensionless)"``
    -- rather than carry no unit at all, so that the reader can tell a
    deliberate choice from an oversight.
    """
    if not xlabel.strip():
        raise ValueError("x-axis label must not be empty; include units.")
    if not ylabel.strip():
        raise ValueError("y-axis label must not be empty; include units.")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def shade_gap(
    ax: plt.Axes,
    gap_start_cm: float,
    gap_end_cm: float,
    label: str = "reduced-coupling gap",
) -> None:
    """Shade the reduced-coupling region on a position axis.

    Parameters
    ----------
    ax : Axes
        Axes whose x-axis is position in cm.
    gap_start_cm, gap_end_cm : float
        Edges of the gap. cm.
    label : str, optional
        Legend label. Pass an empty string to omit from the legend.

    Notes
    -----
    Drawn with low alpha and no edge so it reads as background context rather
    than as a data series. Skipped silently for a zero-width gap, so the same
    plotting code works for homogeneous runs.
    """
    if gap_end_cm <= gap_start_cm:
        return
    ax.axvspan(
        gap_start_cm,
        gap_end_cm,
        color=COLOUR_GAP,
        alpha=0.18,
        linewidth=0,
        label=label if label else None,
        zorder=0,
    )


def set_log_ticks(
    ax: plt.Axes,
    values: np.ndarray,
    axis: str = "x",
    fmt: str = "{:g}",
) -> None:
    """Put explicit, readable ticks on a logarithmic axis.

    Parameters
    ----------
    ax : Axes
        Axes to adjust.
    values : ndarray
        The data values to place ticks at.
    axis : {"x", "y"}, optional
        Which axis. Default ``"x"``.
    fmt : str, optional
        Format string applied to each value.

    Raises
    ------
    ValueError
        If ``axis`` is not ``"x"`` or ``"y"``.

    Notes
    -----
    Matplotlib's default log locator labels minor ticks whenever the axis spans
    less than about one decade per major tick. Over a range like
    ``2e-4`` to ``4e-3`` that produces labels at 2, 3, 4 and 6 times each
    decade, which overlap into an unreadable smear. Placing one tick per
    plotted value and turning the minor labels off fixes it, and has the useful
    side effect of showing the reader exactly where the samples are.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    ticks = np.unique(np.asarray(values, dtype=float))
    labels = [fmt.format(value) for value in ticks]

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    target.set_major_formatter(matplotlib.ticker.FixedFormatter(labels))
    # Minor ticks stay as grid marks but lose their labels.
    target.set_minor_formatter(matplotlib.ticker.NullFormatter())


def annotate_takeaway(ax: plt.Axes, text: str, loc: str = "upper left") -> None:
    """Place a short takeaway note inside the axes.

    Parameters
    ----------
    ax : Axes
        Axes to annotate.
    text : str
        Short statement of what the figure shows.
    loc : {"upper left", "upper right", "lower left", "lower right"}, optional
        Corner to place it in.

    Raises
    ------
    ValueError
        If ``loc`` is not one of the four supported corners.
    """
    positions = {
        "upper left": (0.03, 0.97, "left", "top"),
        "upper right": (0.97, 0.97, "right", "top"),
        "lower left": (0.03, 0.03, "left", "bottom"),
        "lower right": (0.97, 0.03, "right", "bottom"),
    }
    if loc not in positions:
        raise ValueError(f"Unknown location {loc!r}; expected one of {sorted(positions)}")

    x, y, ha, va = positions[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
    )


def save_figure(
    fig: plt.Figure,
    name: str,
    caption: str,
    config_dict: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    close: bool = True,
) -> tuple[Path, Path, Path]:
    """Save a figure as PNG and PDF, with its takeaway caption and provenance.

    Parameters
    ----------
    fig : Figure
        Figure to save.
    name : str
        Base filename, for example ``"fig_ex05_cv_vs_D"``.
    caption : str
        Caption stating the figure's **takeaway**, not a description of what is
        plotted. The brief requires this; "conduction velocity against
        diffusion coefficient" is a description, "velocity scales as the square
        root of coupling, confirming the analytic prediction to within 3 %" is
        a takeaway.
    config_dict : mapping, optional
        Configuration that produced the figure.
    extra_metadata : mapping, optional
        Any measured numbers worth recording with it.
    close : bool, optional
        Close the figure afterwards. Default True; leaving hundreds of figures
        open during a full pipeline run exhausts memory and triggers
        matplotlib's open-figure warning.

    Returns
    -------
    png_path, pdf_path, meta_path : Path
        The three files written.

    Raises
    ------
    ValueError
        If the caption is empty, or reads as a bare description. The check is
        deliberately crude -- it only enforces a minimum length -- but it is
        enough to stop a placeholder reaching the report.

    Notes
    -----
    Both raster and vector copies are written: the PNG at 300 dpi for quick
    viewing and for markdown, the PDF for the typeset report, where vector
    output keeps axis text sharp at any zoom.
    """
    if not caption.strip():
        raise ValueError(f"Figure {name!r} has no caption; state its takeaway.")
    if len(caption.strip()) < 40:
        raise ValueError(
            f"Figure {name!r} has a {len(caption.strip())}-character caption. "
            f"Captions must state a takeaway, which needs a sentence: {caption!r}"
        )

    figures = utils.figures_dir()
    png_path = figures / f"{name}.png"
    pdf_path = figures / f"{name}.pdf"

    fig.savefig(png_path, dpi=FIGURE_DPI)
    fig.savefig(pdf_path)  # vector; dpi is irrelevant

    record: dict[str, Any] = {
        "figure": name,
        "caption": caption.strip(),
        "files": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    record.update(utils.provenance(config_dict))
    if extra_metadata is not None:
        record["measurements"] = extra_metadata

    meta_path = utils.save_metadata(f"{name}_meta", record)

    if close:
        plt.close(fig)

    return png_path, pdf_path, meta_path


def space_time_image(
    ax: plt.Axes,
    x: np.ndarray,
    times: np.ndarray,
    V: np.ndarray,
    colourbar_label: str = "membrane potential $V$ (dimensionless)",
) -> Any:
    """Draw a space-time map of the potential.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    x : ndarray, shape (n_nodes,)
        Node positions. cm.
    times : ndarray, shape (n_snapshots,)
        Snapshot times. ms.
    V : ndarray, shape (n_snapshots, n_nodes)
        Potential.
    colourbar_label : str, optional
        Label for the colour bar, including units.

    Returns
    -------
    QuadMesh
        The mesh artist, so the caller can attach a colour bar.

    Notes
    -----
    ``pcolormesh`` is used rather than ``imshow`` because the snapshot times
    are not necessarily uniformly spaced once a run has been stopped early, and
    ``imshow`` would silently misrepresent them as uniform.

    The ``viridis`` colour map is perceptually uniform and colour-blind-safe,
    unlike ``jet``, which invents banding that is not in the data.
    """
    mesh = ax.pcolormesh(x, times, V, shading="auto", cmap="viridis")
    label_axes(ax, "position $x$ (cm)", "time $t$ (ms)")
    return mesh
