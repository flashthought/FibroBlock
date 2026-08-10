"""FibroBlock: cardiac action-potential propagation and conduction block.

A 1-D monodomain cable model with FitzHugh-Nagumo kinetics, built to study how
a patch of reduced intercellular coupling (a model of fibrosis) slows and
eventually blocks a propagating action potential.

COE 562 -- Engineering Systems Design and Modelling, Problem 8.

The public API below is deliberately small. Experiments import
:func:`~fibroblock.simulate.run_simulation` and the configuration dataclasses;
everything else is available through the submodules for anyone reading the
code in detail.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
