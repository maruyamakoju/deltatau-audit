"""Tests for v0.4.7:
  1. ci_summary.json has _version and _timestamp
  2. --compare flag on audit-sb3 and audit-cleanrl subparsers
  3. _maybe_compare helper: graceful handling of missing file
  4. _maybe_compare generates comparison.html when both files exist
"""

import argparse
import json
import pathlib
import tempfile

import pytest


# ─────────────────────────────────────────────────────────────────────
# 1. ci_summary.json _version + _timestamp
# ─────────────────────────────────────────────────────────────────────

def _make_minimal_summary() -> dict:
    return {
        "reliance_rating": "N/A",
        "reliance_score": None,
        "deployment_rating": "PASS",
        "deployment_score": 0.9,
        "stress_rating": "PASS",
        "stress_score": 0.8,
        "quadrant": "deployment_ready",
        "prescription": "OK",
    }


def _make_minimal_robustness() -> dict:
    return {
        "deployment": {
            "return_score": 0.9,
            "rmse_score": 1.0,
            "rating": "PASS",
            "worst_case": {"scenario": "jitter", "return_ratio": 0.9,
                           "return_drop_pct": 10},
        },
        "stress": {
            "return_score": 0.8,
            "rmse_score": 1.1,
            "rating": "PASS",
            "worst_case": {"scenario": "speed_5x", "return_ratio": 0.8,
                           "return_drop_pct": 20},
        },
        "per_scenario_scores": {
            "jitter": {"return_ratio": 0.9, "return_drop_pct": 10},
        },
    }


def test_ci_summary_json_has_version():
    from deltatau_audit.ci import write_ci_summary
    import deltatau_audit
    with tempfile.TemporaryDirectory() as tmp:
        write_ci_summary(_make_minimal_summary(), _make_minimal_robustness(), tmp)
        data = json.loads((pathlib.Path(tmp) / "ci_summary.json").read_text())
        assert "_version" in data, "ci_summary.json must have _version"
        assert data["_version"] == deltatau_audit.__version__


def test_ci_summary_json_has_timestamp():
    from deltatau_audit.ci import write_ci_summary
    with tempfile.TemporaryDirectory() as tmp:
        write_ci_summary(_make_minimal_summary(), _make_minimal_robustness(), tmp)
        data = json.loads((pathlib.Path(tmp) / "ci_summary.json").read_text())
        assert "_timestamp" in data, "ci_summary.json must have _timestamp"
        ts = data["_timestamp"]
        assert "T" in ts or "Z" in ts, f"Unexpected timestamp format: {ts}"


def test_ci_summary_json_status_fields_still_present():
    """Existing fields must still be present after adding version/timestamp."""
    from deltatau_audit.ci import write_ci_summary
    with tempfile.TemporaryDirectory() as tmp:
        write_ci_summary(_make_minimal_summary(), _make_minimal_robustness(), tmp)
        data = json.loads((pathlib.Path(tmp) / "ci_summary.json").read_text())
        for field in ("status", "exit_code", "deployment_score", "stress_score",
                      "thresholds"):
            assert field in data, f"ci_summary.json missing required field: {field}"


# ─────────────────────────────────────────────────────────────────────
# 2. --compare flag on audit-sb3 and audit-cleanrl
# ─────────────────────────────────────────────────────────────────────

def test_audit_sb3_subparser_has_compare():
    from deltatau_audit.cli import _add_compare_arg
    p = argparse.ArgumentParser()
    _add_compare_arg(p)
    args = p.parse_args(["--compare", "before/summary.json"])
    assert args.compare == "before/summary.json"


def test_audit_sb3_subparser_compare_default_none():
    from deltatau_audit.cli import _add_compare_arg
    p = argparse.ArgumentParser()
    _add_compare_arg(p)
    args = p.parse_args([])
    assert args.compare is None


def test_audit_cleanrl_subparser_has_compare():
    """audit-cleanrl full subparser must accept --compare."""
    from deltatau_audit.cli import _add_compare_arg
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    cleanrl_p = sub.add_parser("audit-cleanrl")
    cleanrl_p.add_argument("--checkpoint", required=True)
    cleanrl_p.add_argument("--agent-module", required=True)
    cleanrl_p.add_argument("--env", required=True)
    _add_compare_arg(cleanrl_p)
    args = p.parse_args([
        "audit-cleanrl",
        "--checkpoint", "agent.pt",
        "--agent-module", "agent.py",
        "--env", "CartPole-v1",
        "--compare", "prev/summary.json",
    ])
    assert args.compare == "prev/summary.json"


# ─────────────────────────────────────────────────────────────────────
# 3. _maybe_compare: graceful handling
# ─────────────────────────────────────────────────────────────────────

def test_maybe_compare_missing_file_no_exception(capsys):
    """_maybe_compare should warn but not crash when --compare file is missing."""
    from deltatau_audit.cli import _maybe_compare

    class FakeArgs:
        compare = "/nonexistent/path/summary.json"

    with tempfile.TemporaryDirectory() as tmp:
        # Should not raise
        _maybe_compare(FakeArgs(), tmp)
    out = capsys.readouterr().out
    assert "WARNING" in out or "not found" in out.lower()


def test_maybe_compare_none_compare_no_output(capsys):
    """_maybe_compare should be silent when --compare is None."""
    from deltatau_audit.cli import _maybe_compare

    class FakeArgs:
        compare = None

    with tempfile.TemporaryDirectory() as tmp:
        _maybe_compare(FakeArgs(), tmp)
    out = capsys.readouterr().out
    assert out == ""


# ─────────────────────────────────────────────────────────────────────
# 4. _maybe_compare generates comparison.html
# ─────────────────────────────────────────────────────────────────────

def _write_minimal_summary_json(path: pathlib.Path, dep_score: float):
    """Write minimal summary.json for _maybe_compare integration test."""
    data = {
        "_version": "0.4.7",
        "_timestamp": "2026-02-19T00:00:00Z",
        "summary": {
            "reliance_rating": "N/A",
            "reliance_score": None,
            "deployment_rating": "PASS" if dep_score >= 0.8 else "FAIL",
            "deployment_score": dep_score,
            "stress_rating": "FAIL",
            "stress_score": 0.3,
            "quadrant": "deployment_ready" if dep_score >= 0.8 else "deployment_fragile",
            "prescription": "test",
        },
        "robustness": {
            "per_scenario_scores": {
                "jitter": {"return_ratio": dep_score, "return_drop_pct": (1-dep_score)*100,
                           "rmse_ratio": 1.0, "rmse_increase_pct": 0,
                           "ci_lower": dep_score - 0.1, "ci_upper": dep_score + 0.1,
                           "significant": False},
            },
            "deployment": {
                "return_score": dep_score, "rmse_score": 1.0,
                "rating": "PASS" if dep_score >= 0.8 else "FAIL",
                "worst_case": {"scenario": "jitter", "return_ratio": dep_score,
                               "return_drop_pct": (1-dep_score)*100},
            },
            "stress": {
                "return_score": 0.3, "rmse_score": 1.5, "rating": "FAIL",
                "worst_case": {"scenario": "speed_5x", "return_ratio": 0.3,
                               "return_drop_pct": 70},
            },
        },
    }
    path.write_text(json.dumps(data, indent=2))


def test_maybe_compare_generates_html():
    from deltatau_audit.cli import _maybe_compare

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        before_json = tmp / "before_summary.json"
        after_dir = tmp / "after_audit"
        after_dir.mkdir()
        after_json = after_dir / "summary.json"

        _write_minimal_summary_json(before_json, dep_score=0.3)
        _write_minimal_summary_json(after_json, dep_score=0.9)

        class FakeArgs:
            compare = str(before_json)

        _maybe_compare(FakeArgs(), str(after_dir))

        html_path = after_dir / "comparison.html"
        assert html_path.exists(), "comparison.html should be written by _maybe_compare"
        content = html_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
