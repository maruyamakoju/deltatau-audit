"""Metric aggregation helpers for frontier experiments.

Twenty-plus frontier files compute the same handful of statistics from a
list of episode returns and then divide by a hard-coded normalisation
constant (typically ``200`` for CartPole). The normalisation constants are
*environment-dependent*, so a magic ``/ 200.0`` quietly breaks when the
experiment moves to a different env.

This module:

    - Centralises the per-env normalisation table.
    - Provides ``aggregate_returns`` for the boilerplate stats block.
    - Provides ``normalize_score`` for [0, 1] clipping with explicit bounds.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = [
    "aggregate_returns",
    "normalize_score",
    "env_return_ceiling",
]


# Approximate "solved" returns per environment. Used as the default ceiling
# when an experiment does not pass an explicit ``ceiling`` to
# :func:`normalize_score`. Values match the implicit constants previously
# scattered across the frontier files (most divided by 200 for CartPole).
_RETURN_CEILINGS: Dict[str, float] = {
    "CartPole-v1": 200.0,
    "CartPole-v0": 200.0,
    "Acrobot-v1": 0.0,  # negative-return env; caller must override
    "MountainCar-v0": 0.0,  # negative-return env; caller must override
    "LunarLander-v2": 200.0,
    "BipedalWalker-v3": 300.0,
}


def env_return_ceiling(env_id: str, default: float = 200.0) -> float:
    """Return the conventional 'solved' return for ``env_id``.

    Falls back to ``default`` when the environment is unknown. Negative-
    return environments (Acrobot, MountainCar) are stored as ``0.0`` to
    force an explicit override at the call site rather than silently
    miscomputing a normalised score.
    """
    return float(_RETURN_CEILINGS.get(env_id, default))


def aggregate_returns(returns: Sequence[float]) -> Dict[str, float]:
    """Standard summary stats over an episode-return sequence.

    Returns ``{mean_return, std_return, min_return, max_return,
    median_return, n_episodes}``. An empty sequence yields zeros (and
    ``n_episodes=0``) rather than raising — keeps callers tidy when an
    experiment fails before any episode completes.
    """
    if not returns:
        return {
            "mean_return": 0.0,
            "std_return": 0.0,
            "min_return": 0.0,
            "max_return": 0.0,
            "median_return": 0.0,
            "n_episodes": 0,
        }
    arr = np.asarray(list(returns), dtype=float)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "median_return": float(np.median(arr)),
        "n_episodes": int(arr.size),
    }


def normalize_score(value: float, *, baseline: float = 0.0, ceiling: float = 1.0) -> float:
    """Linearly map ``value`` from ``[baseline, ceiling]`` to ``[0, 1]``.

    Clipped at both ends. Returns ``0.0`` when ``ceiling <= baseline``
    (degenerate range) rather than raising.
    """
    span = ceiling - baseline
    if span <= 0:
        return 0.0
    return float(np.clip((value - baseline) / span, 0.0, 1.0))
