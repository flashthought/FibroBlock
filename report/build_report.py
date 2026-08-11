"""Optional pandoc wrapper for building the report to PDF or DOCX.

Pandoc is **not** required to use this project. The report is written in
Markdown and is perfectly readable as-is on GitHub; this script exists only for
producing a typeset copy for submission.

Usage::

    python report/build_report.py            # PDF via pandoc
    python report/build_report.py --to docx  # Word, if the submission needs it
    python report/build_report.py --check    # report what is available, build nothing

If pandoc is missing the script says so clearly and exits non-zero rather than
producing a confusing error from a subprocess.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parent

SOURCE = REPORT_DIR / "report.md"
BIBLIOGRAPHY = REPORT_DIR / "references.bib"


def pandoc_available() -> str | None:
    """Return the pandoc executable path, or None if it is not installed."""
    return shutil.which("pandoc")


def check() -> int:
    """Report what is present and what is missing.

    Returns
    -------
    int
        0 if a build could proceed, 1 otherwise.
    """
    print("Report build prerequisites")
    print("-" * 50)

    pandoc = pandoc_available()
    print(f"  pandoc         : {pandoc or 'NOT FOUND'}")
    print(f"  report.md      : {'ok' if SOURCE.is_file() else 'MISSING'}")
    print(f"  references.bib : {'ok' if BIBLIOGRAPHY.is_file() else 'MISSING'}")

    figures = sorted((REPORT_DIR / "figures").glob("*.pdf"))
    print(f"  figures        : {len(figures)} PDF files in report/figures/")
    if not figures:
        print("                   (run: python scripts/make_all_figures.py)")

    ready = bool(pandoc) and SOURCE.is_file()
    print("-" * 50)
    print("  Ready to build." if ready else "  Cannot build; see above.")
    return 0 if ready else 1


def build(output_format: str) -> int:
    """Run pandoc to produce the typeset report.

    Parameters
    ----------
    output_format : str
        ``"pdf"``, ``"docx"`` or ``"html"``.

    Returns
    -------
    int
        Pandoc's exit status, or 1 if pandoc is unavailable.
    """
    pandoc = pandoc_available()
    if pandoc is None:
        print(
            "pandoc was not found on PATH.\n"
            "\n"
            "This is optional: report/report.md is readable as Markdown and\n"
            "renders on GitHub. To produce a typeset copy, install pandoc from\n"
            "https://pandoc.org/installing.html (a LaTeX distribution is also\n"
            "needed for PDF output).",
            file=sys.stderr,
        )
        return 1

    if not SOURCE.is_file():
        print(f"Source not found: {SOURCE}", file=sys.stderr)
        return 1

    output = REPORT_DIR / f"report.{output_format}"

    command = [
        pandoc,
        str(SOURCE),
        "--output", str(output),
        # Resolve ../figures/... and figures/... relative to the report folder.
        "--resource-path", f"{REPORT_DIR}{';' if sys.platform == 'win32' else ':'}{PROJECT_ROOT}",
        "--standalone",
        "--toc",
        "--toc-depth=2",
        # Render the LaTeX maths in the report body.
        "--mathml" if output_format == "html" else "--mathjax",
        "--metadata", "title=Cardiac Action-Potential Propagation and Conduction Block",
    ]

    if BIBLIOGRAPHY.is_file():
        command += ["--bibliography", str(BIBLIOGRAPHY), "--citeproc"]

    if output_format == "pdf":
        # A4 with sane margins; the default LaTeX geometry wastes a lot of page.
        command += [
            "--variable", "geometry:a4paper,margin=25mm",
            "--variable", "fontsize=11pt",
        ]

    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=REPORT_DIR, check=False)

    if completed.returncode == 0:
        print(f"Wrote {output}")
    else:
        print(
            f"pandoc exited with status {completed.returncode}. For PDF output "
            f"a LaTeX engine (e.g. MiKTeX or TeX Live) must also be installed.",
            file=sys.stderr,
        )
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and build (or check).

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        description="Build report/report.md into a typeset document via pandoc."
    )
    parser.add_argument(
        "--to",
        default="pdf",
        choices=["pdf", "docx", "html"],
        help="Output format (default: pdf).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what is available and exit without building.",
    )
    arguments = parser.parse_args(argv)

    return check() if arguments.check else build(arguments.to)


if __name__ == "__main__":
    raise SystemExit(main())
