"""Tests for v0.6.2: Badge SVG generation."""

import json
import os

import pytest


# ── Badge rendering tests ──────────────────────────────────────────

def test_badge_module_importable():
    from deltatau_audit import badge
    assert hasattr(badge, "generate_badges")
    assert hasattr(badge, "badge_deployment")
    assert hasattr(badge, "badge_stress")
    assert hasattr(badge, "badge_status")


def _sample_summary():
    return {
        "deployment_score": 0.85,
        "deployment_rating": "MILD",
        "stress_score": 0.45,
        "stress_rating": "FAIL",
        "quadrant": "deployment_ready",
    }


def test_badge_deployment_svg():
    from deltatau_audit.badge import badge_deployment
    svg = badge_deployment(_sample_summary())
    assert svg.startswith("<svg")
    assert "deployment" in svg
    assert "MILD" in svg
    assert "0.85" in svg
    # MILD color = yellow
    assert "#ffc107" in svg


def test_badge_stress_svg():
    from deltatau_audit.badge import badge_stress
    svg = badge_stress(_sample_summary())
    assert svg.startswith("<svg")
    assert "stress" in svg
    assert "FAIL" in svg
    assert "0.45" in svg
    # FAIL color = red
    assert "#dc3545" in svg


def test_badge_status_svg():
    from deltatau_audit.badge import badge_status
    svg = badge_status(_sample_summary())
    assert svg.startswith("<svg")
    assert "time robustness" in svg
    assert "Deployment Ready" in svg
    # deployment_ready = green
    assert "#28a745" in svg


def test_badge_status_fragile():
    from deltatau_audit.badge import badge_status
    summary = {**_sample_summary(), "quadrant": "deployment_fragile"}
    svg = badge_status(summary)
    assert "Deployment Fragile" in svg
    assert "#dc3545" in svg


def test_badge_status_time_aware_robust():
    from deltatau_audit.badge import badge_status
    summary = {**_sample_summary(), "quadrant": "time_aware_robust"}
    svg = badge_status(summary)
    assert "Time-Aware &amp; Robust" in svg or "Time-Aware & Robust" in svg
    assert "#28a745" in svg


def test_badge_deployment_pass():
    from deltatau_audit.badge import badge_deployment
    summary = {"deployment_score": 0.98, "deployment_rating": "PASS"}
    svg = badge_deployment(summary)
    assert "PASS" in svg
    assert "#28a745" in svg


def test_badge_deployment_degraded():
    from deltatau_audit.badge import badge_deployment
    summary = {"deployment_score": 0.65, "deployment_rating": "DEGRADED"}
    svg = badge_deployment(summary)
    assert "DEGRADED" in svg
    assert "#fd7e14" in svg


# ── generate_badges file output ────────────────────────────────────

def test_generate_badges_creates_files(tmp_path):
    from deltatau_audit.badge import generate_badges

    # Write a summary.json
    summary_path = str(tmp_path / "summary.json")
    data = {"summary": _sample_summary()}
    with open(summary_path, "w") as f:
        json.dump(data, f)

    out_dir = str(tmp_path / "badges")
    paths = generate_badges(summary_path, out_dir)

    assert "deployment" in paths
    assert "stress" in paths
    assert "status" in paths

    for name, path in paths.items():
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert content.startswith("<svg")


def test_generate_badges_custom_prefix(tmp_path):
    from deltatau_audit.badge import generate_badges

    summary_path = str(tmp_path / "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"summary": _sample_summary()}, f)

    paths = generate_badges(summary_path, str(tmp_path), prefix="audit")
    for path in paths.values():
        assert "audit-" in os.path.basename(path)


def test_generate_badges_flat_summary(tmp_path):
    """Works when summary.json IS the summary (no nesting)."""
    from deltatau_audit.badge import generate_badges

    summary_path = str(tmp_path / "flat.json")
    with open(summary_path, "w") as f:
        json.dump(_sample_summary(), f)

    paths = generate_badges(summary_path, str(tmp_path))
    assert len(paths) == 3
    for path in paths.values():
        assert os.path.exists(path)


# ── SVG validity ──────────────────────────────────────────────────

def test_svg_valid_xml():
    """Badge SVGs are valid XML."""
    import xml.etree.ElementTree as ET
    from deltatau_audit.badge import badge_deployment
    svg = badge_deployment(_sample_summary())
    # Should parse without errors
    ET.fromstring(svg)


def test_svg_has_aria_label():
    """Badge SVGs have aria-label for accessibility."""
    from deltatau_audit.badge import badge_deployment
    svg = badge_deployment(_sample_summary())
    assert 'aria-label="deployment: MILD (0.85)"' in svg


# ── CLI integration ───────────────────────────────────────────────

def test_badge_cli_subcommand(tmp_path):
    """deltatau-audit badge runs successfully."""
    from deltatau_audit.badge import generate_badges

    summary_path = str(tmp_path / "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"summary": _sample_summary()}, f)

    out_dir = str(tmp_path / "cli_badges")

    import subprocess
    result = subprocess.run(
        ["python", "-m", "deltatau_audit", "badge", summary_path,
         "--out", out_dir, "--prefix", "test"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert os.path.exists(os.path.join(out_dir, "test-deployment.svg"))
    assert os.path.exists(os.path.join(out_dir, "test-stress.svg"))
    assert os.path.exists(os.path.join(out_dir, "test-status.svg"))


# ── text width helper ─────────────────────────────────────────────

def test_text_width_approximation():
    from deltatau_audit.badge import _text_width
    # Empty string
    assert _text_width("") == 0
    # Known widths (approximate)
    assert _text_width("a") == 7  # normal char
    assert _text_width("i") == 5  # narrow
    assert _text_width("M") == 9  # wide
    # Longer text
    w = _text_width("PASS (0.98)")
    assert 60 < w < 100
