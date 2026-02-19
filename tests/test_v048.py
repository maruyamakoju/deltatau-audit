"""Tests for v0.4.8:
  1. color.py: colorize(), colored_rating(), auto-disable
  2. _print_markdown_summary(): output structure
  3. --format markdown CLI flag on audit-sb3 and audit-cleanrl
  4. colored output in auditor._print_summary (no crash, correct text)
"""

import argparse
import os
import sys

import pytest


# ─────────────────────────────────────────────────────────────────────
# 1. color.py
# ─────────────────────────────────────────────────────────────────────

def test_colorize_returns_text_when_no_color_set(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    # Re-import after env change forces re-evaluation
    import importlib
    import deltatau_audit.color as c
    importlib.reload(c)
    result = c.colorize("PASS", "bright_green")
    assert result == "PASS"
    importlib.reload(c)  # restore


def test_colorize_wraps_with_ansi_when_force_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    import importlib
    import deltatau_audit.color as c
    importlib.reload(c)
    result = c.colorize("PASS", "bright_green")
    assert "\033[" in result, "ANSI codes should be present with FORCE_COLOR"
    assert "PASS" in result
    importlib.reload(c)


def test_colored_rating_contains_rating_text(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    import importlib
    import deltatau_audit.color as c
    importlib.reload(c)
    for rating in ("PASS", "MILD", "DEGRADED", "FAIL", "N/A"):
        result = c.colored_rating(rating)
        assert rating in result, f"Rating text '{rating}' missing from output"
    importlib.reload(c)


def test_colored_rating_width_pads_correctly():
    """colored_rating with width should right-pad without ANSI affecting alignment."""
    from deltatau_audit.color import colored_rating, _rj
    # _rj should right-justify based on text length, not ANSI length
    padded = _rj("PASS", 10)
    assert len(padded) == 10
    assert padded.endswith("PASS")


def test_rating_codes_all_ratings_mapped():
    from deltatau_audit.color import rating_codes
    for r in ("PASS", "MILD", "DEGRADED", "FAIL", "N/A"):
        codes = rating_codes(r)
        assert isinstance(codes, tuple) and len(codes) >= 1


# ─────────────────────────────────────────────────────────────────────
# 2. _print_markdown_summary
# ─────────────────────────────────────────────────────────────────────

def _make_minimal_result(dep_score: float = 0.9, dep_rating: str = "PASS") -> dict:
    return {
        "summary": {
            "reliance_rating": "N/A",
            "reliance_score": None,
            "deployment_rating": dep_rating,
            "deployment_score": dep_score,
            "stress_rating": "MILD",
            "stress_score": 0.82,
            "quadrant": "deployment_ready",
            "prescription": "OK",
        },
        "robustness": {
            "per_scenario_scores": {
                "jitter":    {"return_ratio": 0.95, "return_drop_pct": 5,
                              "significant": False},
                "delay":     {"return_ratio": 0.88, "return_drop_pct": 12,
                              "significant": True},
                "obs_noise": {"return_ratio": 0.91, "return_drop_pct": 9,
                              "significant": False},
                "speed_5x":  {"return_ratio": 0.82, "return_drop_pct": 18,
                              "significant": True},
            },
        },
    }


def test_print_markdown_summary_returns_string(capsys):
    from deltatau_audit.cli import _print_markdown_summary
    result = _make_minimal_result()
    md = _print_markdown_summary(result, label="Test")
    assert isinstance(md, str)
    assert len(md) > 100


def test_print_markdown_summary_contains_ratings(capsys):
    from deltatau_audit.cli import _print_markdown_summary
    result = _make_minimal_result(dep_score=0.9, dep_rating="PASS")
    md = _print_markdown_summary(result)
    assert "PASS" in md
    assert "Deployment" in md
    assert "Stress" in md


def test_print_markdown_summary_contains_scenarios(capsys):
    from deltatau_audit.cli import _print_markdown_summary
    result = _make_minimal_result()
    md = _print_markdown_summary(result)
    assert "jitter" in md
    assert "obs_noise" in md
    assert "speed_5x" in md


def test_print_markdown_summary_fail_has_x_icon(capsys):
    from deltatau_audit.cli import _print_markdown_summary
    result = _make_minimal_result(dep_score=0.2, dep_rating="FAIL")
    md = _print_markdown_summary(result)
    assert "❌" in md


def test_print_markdown_summary_pass_has_checkmark(capsys):
    from deltatau_audit.cli import _print_markdown_summary
    result = _make_minimal_result(dep_score=0.9, dep_rating="PASS")
    md = _print_markdown_summary(result)
    assert "✅" in md


def test_print_markdown_summary_no_github_step_summary_no_crash(monkeypatch, capsys):
    """Should not crash even when $GITHUB_STEP_SUMMARY is not set."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    from deltatau_audit.cli import _print_markdown_summary
    result = _make_minimal_result()
    _print_markdown_summary(result)  # should not raise


# ─────────────────────────────────────────────────────────────────────
# 3. --format flag on audit-sb3 and audit-cleanrl parsers
# ─────────────────────────────────────────────────────────────────────

def test_add_format_arg_default_text():
    from deltatau_audit.cli import _add_format_arg
    p = argparse.ArgumentParser()
    _add_format_arg(p)
    args = p.parse_args([])
    assert args.output_format == "text"


def test_add_format_arg_markdown():
    from deltatau_audit.cli import _add_format_arg
    p = argparse.ArgumentParser()
    _add_format_arg(p)
    args = p.parse_args(["--format", "markdown"])
    assert args.output_format == "markdown"


def test_audit_sb3_parser_accepts_format_markdown():
    from deltatau_audit.cli import _add_format_arg, _add_workers_arg, _add_seed_arg
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    sb3 = sub.add_parser("audit-sb3")
    sb3.add_argument("--model", required=True)
    sb3.add_argument("--algo", required=True)
    sb3.add_argument("--env", required=True)
    _add_format_arg(sb3)
    args = p.parse_args([
        "audit-sb3", "--model", "m.zip",
        "--algo", "ppo", "--env", "CartPole-v1",
        "--format", "markdown",
    ])
    assert args.output_format == "markdown"


# ─────────────────────────────────────────────────────────────────────
# 4. colored _print_summary (no crash, correct text content)
# ─────────────────────────────────────────────────────────────────────

def test_print_summary_no_crash(capsys, monkeypatch):
    """auditor._print_summary() must not crash regardless of color support."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    from deltatau_audit.auditor import _print_summary
    summary = {
        "reliance_rating": "N/A",
        "reliance_score": None,
        "deployment_rating": "PASS",
        "deployment_score": 0.9,
        "stress_rating": "FAIL",
        "stress_score": 0.3,
        "sensitivity_mean": None,
        "quadrant": "deployment_ready",
        "prescription": "No action needed.",
    }
    _print_summary(summary)
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "FAIL" in out
    assert "deployment_ready" in out


def test_print_summary_with_reliance(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")  # plain text for easy assertion
    from deltatau_audit.auditor import _print_summary
    import importlib, deltatau_audit.color as c
    importlib.reload(c)
    summary = {
        "reliance_rating": "HIGH",
        "reliance_score": 3.14,
        "deployment_rating": "MILD",
        "deployment_score": 0.85,
        "stress_rating": "DEGRADED",
        "stress_score": 0.6,
        "sensitivity_mean": 0.0123,
        "quadrant": "time_aware_robust",
        "prescription": "Agent actively uses internal timing.",
    }
    _print_summary(summary)
    out = capsys.readouterr().out
    assert "MILD" in out
    assert "DEGRADED" in out
    assert "3.14" in out
    importlib.reload(c)
