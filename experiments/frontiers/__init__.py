"""Frontier research modules — pushing into uncharted territory.

This subpackage is a flat collection of one-experiment-per-file research
prototypes. Each frontier exposes an ``XYZExperiment`` class with a
``run(out_dir) -> Dict[str, float]`` method whose returned dict contains
``composite_score``. The shared scaffolding for new experiments lives in:

    - :mod:`experiments.frontiers._base`     — :class:`FrontierExperiment`,
      :func:`seed_all`, :func:`save_summary`, :func:`make_frontier_parser`.
    - :mod:`experiments.frontiers._geometry` — Möbius / Poincaré-ball ops.
    - :mod:`experiments.frontiers._lipschitz` — spectral / empirical Lipschitz.
    - :mod:`experiments.frontiers._metrics`  — return aggregation, normalisers.

Frontiers are deliberately not auto-imported here: importing the package
must remain side-effect-free so callers can enumerate frontier names
without triggering heavy ``torch`` initialisation paths from the
underlying experiment classes.
"""

from ._base import (
    FrontierConfig,
    FrontierExperiment,
    make_frontier_parser,
    run_default_cli,
    save_summary,
    seed_all,
)

__all__ = [
    "FrontierConfig",
    "FrontierExperiment",
    "make_frontier_parser",
    "run_default_cli",
    "save_summary",
    "seed_all",
]
