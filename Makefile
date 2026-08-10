# ---------------------------------------------------------------------------
# FibroBlock Makefile
#
# On Windows, GNU make is often unavailable. Every target below is a thin
# wrapper around a single Python command, and the equivalent raw command is
# printed in the comment above each target so it can be pasted directly into
# PowerShell if make is missing. The README gives the make-free route.
# ---------------------------------------------------------------------------

PYTHON ?= python

.PHONY: all figures test clean env lint format help

# Default target: regenerate everything and then verify it.
all: figures test

# Regenerate every figure and every results file from scratch.
#   PowerShell equivalent:  python scripts/make_all_figures.py
figures:
	$(PYTHON) scripts/make_all_figures.py

# Run the verification test suite.
#   PowerShell equivalent:  pytest
test:
	$(PYTHON) -m pytest

# Print environment provenance for the report appendix.
#   PowerShell equivalent:  python scripts/check_environment.py
env:
	$(PYTHON) scripts/check_environment.py

# Remove all GENERATED output so that "make figures" can be proven to rebuild
# from truly empty directories. This is the reproducibility acid test.
#   PowerShell equivalent:
#     Remove-Item -Recurse -Force figures\*, results\* -ErrorAction SilentlyContinue
clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) or p.mkdir(parents=True, exist_ok=True) for p in (pathlib.Path('figures'), pathlib.Path('results'))]"
	$(PYTHON) -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('report/figures'), ignore_errors=True); pathlib.Path('report/figures').mkdir(parents=True, exist_ok=True)"

lint:
	$(PYTHON) -m ruff check src tests experiments scripts

format:
	$(PYTHON) -m black src tests experiments scripts

help:
	@echo "FibroBlock targets:"
	@echo "  make figures  - regenerate all figures and results"
	@echo "  make test     - run the pytest verification suite"
	@echo "  make all      - figures, then test"
	@echo "  make clean    - delete generated figures/ and results/"
	@echo "  make env      - print environment provenance"
	@echo "  make lint     - run ruff"
	@echo "  make format   - run black"
