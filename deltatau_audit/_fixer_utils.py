"""Shared utilities for fix-sb3 and fix-cleanrl pipelines."""
from __future__ import annotations

from .color import bold, dim, err, ok, warn

# ── Training timestep estimates ───────────────────────────────────────────────

_TIMESTEP_DEFAULTS = {
    "CartPole": 100_000,
    "LunarLander": 300_000,
    "BipedalWalker": 1_000_000,
    "Ant": 1_000_000,
    "HalfCheetah": 500_000,
    "Hopper": 500_000,
    "Walker2d": 1_000_000,
    "Humanoid": 2_000_000,
    "Pendulum": 50_000,
    "MountainCar": 200_000,
    "Acrobot": 100_000,
}


def estimate_timesteps(env_id: str, default: int = 300_000) -> int:
    """Estimate retraining timesteps for a given environment ID.

    Uses the env name (case-insensitive substring match) to pick a sensible
    default. Returns `default` if no match found.
    """
    env_lower = env_id.lower()
    for key, ts in _TIMESTEP_DEFAULTS.items():
        if key.lower() in env_lower:
            return ts
    return default


def print_fix_comparison(before_summary: dict, after_summary: dict) -> None:
    """Print a before/after comparison table to stdout."""

    print(bold("\n  Before/After Comparison"))
    print(dim("  " + "-" * 42))

    rows = [
        ("Deployment", "deployment_rating", "deployment_score"),
        ("Stress",     "stress_rating",     "stress_score"),
        ("Quadrant",   "quadrant",          None),
    ]

    for label, rating_key, score_key in rows:
        b_rating = before_summary.get(rating_key, "?")
        a_rating = after_summary.get(rating_key, "?")

        if score_key:
            b_score = before_summary.get(score_key)
            a_score = after_summary.get(score_key)
            b_str = f"{b_rating} ({b_score:.2f})" if b_score is not None else b_rating
            a_str = f"{a_rating} ({a_score:.2f})" if a_score is not None else a_rating
        else:
            b_str = b_rating.replace("_", " ").title() if isinstance(b_rating, str) else str(b_rating)
            a_str = a_rating.replace("_", " ").title() if isinstance(a_rating, str) else str(a_rating)

        # Color the after rating
        if a_rating in ("PASS",):
            a_colored = ok(a_str)
        elif a_rating in ("MILD", "DEGRADED"):
            a_colored = warn(a_str)
        elif a_rating in ("FAIL",):
            a_colored = err(a_str)
        else:
            a_colored = a_str

        arrow = " -> "
        print(f"  {label:<12} {dim(b_str)}{arrow}{a_colored}")

    print()
