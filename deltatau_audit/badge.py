"""Generate shields.io-style SVG badges from audit results.

CLI::

    deltatau-audit badge audit_report/summary.json --out badges/

Python::

    from deltatau_audit.badge import generate_badges
    generate_badges("audit_report/summary.json", "badges/")
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

# ── SVG template ──────────────────────────────────────────────────────

_SVG_TEMPLATE = """\
<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="20" role="img" \
aria-label="{label}: {value}">
  <title>{label}: {value}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{total_w}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_w}" height="20" fill="#555"/>
    <rect x="{label_w}" width="{value_w}" height="20" fill="{color}"/>
    <rect width="{total_w}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" \
font-family="Verdana,Geneva,DejaVu Sans,sans-serif" \
text-rendering="geometricPrecision" font-size="11">
    <text aria-hidden="true" x="{label_cx}" y="150" fill="#010101" \
fill-opacity=".3" transform="scale(0.1)" textLength="{label_tw}">{label}</text>
    <text x="{label_cx}" y="140" transform="scale(0.1)" \
textLength="{label_tw}">{label}</text>
    <text aria-hidden="true" x="{value_cx}" y="150" fill="#010101" \
fill-opacity=".3" transform="scale(0.1)" textLength="{value_tw}">{value}</text>
    <text x="{value_cx}" y="140" transform="scale(0.1)" \
textLength="{value_tw}">{value}</text>
  </g>
</svg>"""


def _text_width(text: str) -> int:
    """Approximate pixel width of text at 11px Verdana."""
    # Rough character-width table (matches shields.io behavior)
    wide = set("mwMWGHOQUD")
    narrow = set("fijlrt!|:;.,1I ")
    w = 0
    for ch in text:
        if ch in wide:
            w += 9
        elif ch in narrow:
            w += 5
        else:
            w += 7
    return w


def _make_badge(label: str, value: str, color: str) -> str:
    """Render a shields.io-style flat badge SVG string."""
    pad = 10  # horizontal padding per side
    label_tw = _text_width(label) * 10  # textLength in 0.1px coords
    value_tw = _text_width(value) * 10
    label_w = _text_width(label) + 2 * pad
    value_w = _text_width(value) + 2 * pad
    total_w = label_w + value_w
    label_cx = label_w * 5  # center x in 0.1px
    value_cx = (label_w + value_w / 2) * 10

    return _SVG_TEMPLATE.format(
        label=label,
        value=value,
        color=color,
        total_w=total_w,
        label_w=label_w,
        value_w=value_w,
        label_tw=label_tw,
        value_tw=value_tw,
        label_cx=label_cx,
        value_cx=value_cx,
    )


# ── Color helpers ────────────────────────────────────────────────────

def _rating_color(rating: str) -> str:
    """Color for PASS/MILD/DEGRADED/FAIL ratings."""
    return {
        "PASS": "#28a745",
        "MILD": "#ffc107",
        "DEGRADED": "#fd7e14",
        "FAIL": "#dc3545",
    }.get(rating, "#6c757d")


def _quadrant_color(quadrant: str) -> str:
    """Color for quadrant classification."""
    good = {"time_aware_robust", "time_blind_robust", "deployment_ready"}
    bad = {"time_blind_fragile", "deployment_fragile"}
    if quadrant in good:
        return "#28a745"
    if quadrant in bad:
        return "#dc3545"
    return "#fd7e14"


_QUADRANT_LABELS = {
    "time_aware_robust": "Time-Aware & Robust",
    "time_aware_fragile": "Time-Aware but Fragile",
    "time_blind_fragile": "Time-Blind & Fragile",
    "time_blind_robust": "Time-Blind but Robust",
    "deployment_ready": "Deployment Ready",
    "deployment_fragile": "Deployment Fragile",
}


# ── Public API ───────────────────────────────────────────────────────

def badge_deployment(summary: Dict[str, Any]) -> str:
    """Generate deployment robustness badge SVG."""
    score = summary.get("deployment_score", 0.0)
    rating = summary.get("deployment_rating", "?")
    color = _rating_color(rating)
    return _make_badge("deployment", f"{rating} ({score:.2f})", color)


def badge_stress(summary: Dict[str, Any]) -> str:
    """Generate stress robustness badge SVG."""
    score = summary.get("stress_score", 0.0)
    rating = summary.get("stress_rating", "?")
    color = _rating_color(rating)
    return _make_badge("stress", f"{rating} ({score:.2f})", color)


def badge_status(summary: Dict[str, Any]) -> str:
    """Generate overall status badge SVG."""
    quadrant = summary.get("quadrant", "unknown")
    label = _QUADRANT_LABELS.get(quadrant, quadrant.replace("_", " ").title())
    color = _quadrant_color(quadrant)
    return _make_badge("time robustness", label, color)


def generate_badges(
    summary_json: str,
    output_dir: str = ".",
    prefix: str = "badge",
) -> Dict[str, str]:
    """Generate all badge SVGs from a summary.json file.

    Args:
        summary_json: Path to audit summary.json.
        output_dir: Directory to write SVG files.
        prefix: Filename prefix (default: "badge").

    Returns:
        Dict mapping badge name to output file path.
    """
    with open(summary_json) as f:
        data = json.load(f)

    summary = data.get("summary", data)

    os.makedirs(output_dir, exist_ok=True)

    badges = {
        "deployment": badge_deployment(summary),
        "stress": badge_stress(summary),
        "status": badge_status(summary),
    }

    paths: Dict[str, str] = {}
    for name, svg in badges.items():
        path = os.path.join(output_dir, f"{prefix}-{name}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        paths[name] = path

    return paths
