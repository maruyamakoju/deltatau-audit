"""Single source of truth for visual theme: colors, rating thresholds, quadrant labels.

All other modules should import from here rather than defining their own color dicts.
"""

from __future__ import annotations

# ── Rating → color mapping ────────────────────────────────────────────────────
# Used by badge.py, diff.py, report/generator.py, metrics.py.
# IMPORTANT: MILD = amber (#ffc107), NOT green. Green means PASS.

RATING_COLORS: dict[str, str] = {
    "PASS": "#28a745",
    "MILD": "#ffc107",
    "DEGRADED": "#fd7e14",
    "FAIL": "#dc3545",
    "N/A": "#6c757d",
}

# ── Quadrant → label mapping ──────────────────────────────────────────────────
QUADRANT_LABELS: dict[str, str] = {
    "time_aware_robust": "Time-Aware & Robust",
    "time_aware_fragile": "Time-Aware but Fragile",
    "time_blind_fragile": "Time-Blind & Fragile",
    "time_blind_robust": "Time-Blind but Robust",
    "deployment_ready": "Deployment Ready",
    "deployment_fragile": "Deployment Fragile",
}

# ── Quadrant → color mapping ──────────────────────────────────────────────────
QUADRANT_COLORS: dict[str, str] = {
    "time_aware_robust": "#28a745",
    "time_blind_robust": "#28a745",
    "deployment_ready": "#28a745",
    "time_aware_fragile": "#fd7e14",
    "time_blind_fragile": "#dc3545",
    "deployment_fragile": "#dc3545",
}

# ── Reliance rating → color (informational blue spectrum) ─────────────────────
RELIANCE_COLORS: dict[str, str] = {
    "N/A": "#BDBDBD",
    "LOW": "#9E9E9E",
    "MODERATE": "#42A5F5",
    "HIGH": "#1E88E5",
    "VERY_HIGH": "#1565C0",
}

# ── Robustness thresholds (return ratio) ─────────────────────────────────────
# These must match metrics.robustness_rating() exactly.
PASS_THRESHOLD: float = 0.95
MILD_THRESHOLD: float = 0.80
DEGRADED_THRESHOLD: float = 0.50  # Below this = FAIL

# ── Reliance threshold ────────────────────────────────────────────────────────
# RMSE ratio >= this → "time_aware" classification.
RELIANCE_THRESHOLD: float = 2.0

# ── CI / pipeline defaults ────────────────────────────────────────────────────
DEPLOY_THRESHOLD_DEFAULT: float = 0.80
STRESS_THRESHOLD_DEFAULT: float = 0.50


def rating_color(rating: str) -> str:
    """Return hex color for a PASS/MILD/DEGRADED/FAIL/N/A rating string."""
    return RATING_COLORS.get(rating, "#6c757d")


def quadrant_label(quadrant: str) -> str:
    """Return human-readable label for a quadrant key."""
    return QUADRANT_LABELS.get(quadrant, quadrant.replace("_", " ").title())


def quadrant_color(quadrant: str) -> str:
    """Return hex color for a quadrant key."""
    return QUADRANT_COLORS.get(quadrant, "#fd7e14")
