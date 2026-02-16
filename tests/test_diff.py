"""Tests for deltatau_audit.diff — comparison.md generation."""

import json
import os
import pytest

from deltatau_audit.diff import generate_comparison


def _write_summary(path, summary, robustness=None):
    """Helper to write a summary.json file."""
    data = {"summary": summary}
    if robustness is not None:
        data["robustness"] = robustness
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


class TestDiff:
    def test_basic_output(self, tmp_path):
        before = tmp_path / "before" / "summary.json"
        after = tmp_path / "after" / "summary.json"

        _write_summary(str(before), {
            "reliance_rating": "N/A",
            "deployment_score": 0.60,
            "deployment_rating": "DEGRADED",
            "stress_score": 0.40,
            "stress_rating": "FAIL",
            "quadrant": "deployment_fragile",
        })
        _write_summary(str(after), {
            "reliance_rating": "N/A",
            "deployment_score": 0.95,
            "deployment_rating": "PASS",
            "stress_score": 0.70,
            "stress_rating": "DEGRADED",
            "quadrant": "deployment_ready",
        })

        md = generate_comparison(str(before), str(after))

        assert "# Audit Comparison" in md
        assert "Deployment" in md
        assert "Stress" in md
        assert "DEGRADED" in md
        assert "PASS" in md
        assert "deployment_fragile" in md
        assert "deployment_ready" in md

    def test_writes_to_file(self, tmp_path):
        before = tmp_path / "b" / "summary.json"
        after = tmp_path / "a" / "summary.json"
        out = tmp_path / "comparison.md"

        _write_summary(str(before), {
            "deployment_score": 0.80, "deployment_rating": "DEGRADED",
            "stress_score": 0.50, "stress_rating": "FAIL",
            "quadrant": "time_blind_fragile",
        })
        _write_summary(str(after), {
            "deployment_score": 0.90, "deployment_rating": "MILD",
            "stress_score": 0.60, "stress_rating": "DEGRADED",
            "quadrant": "time_blind_robust",
        })

        md = generate_comparison(str(before), str(after), str(out))
        assert os.path.exists(str(out))
        written = open(str(out), encoding="utf-8").read()
        assert written == md

    def test_reliance_na_both(self, tmp_path):
        before = tmp_path / "b" / "summary.json"
        after = tmp_path / "a" / "summary.json"

        _write_summary(str(before), {
            "reliance_rating": "N/A",
            "deployment_score": 0.90, "deployment_rating": "MILD",
            "stress_score": 0.80, "stress_rating": "MILD",
            "quadrant": "deployment_ready",
        })
        _write_summary(str(after), {
            "reliance_rating": "N/A",
            "deployment_score": 0.95, "deployment_rating": "PASS",
            "stress_score": 0.90, "stress_rating": "MILD",
            "quadrant": "deployment_ready",
        })

        md = generate_comparison(str(before), str(after))
        assert "| Reliance | N/A | N/A | - |" in md

    def test_reliance_with_scores(self, tmp_path):
        before = tmp_path / "b" / "summary.json"
        after = tmp_path / "a" / "summary.json"

        _write_summary(str(before), {
            "reliance_rating": "HIGH", "reliance_score": 1.5,
            "deployment_score": 0.90, "deployment_rating": "MILD",
            "stress_score": 0.80, "stress_rating": "MILD",
            "quadrant": "time_aware_robust",
        })
        _write_summary(str(after), {
            "reliance_rating": "VERY_HIGH", "reliance_score": 2.5,
            "deployment_score": 0.95, "deployment_rating": "PASS",
            "stress_score": 0.90, "stress_rating": "MILD",
            "quadrant": "time_aware_robust",
        })

        md = generate_comparison(str(before), str(after))
        assert "1.50x" in md
        assert "2.50x" in md
        assert "HIGH -> VERY_HIGH" in md

    def test_per_scenario_detail(self, tmp_path):
        before = tmp_path / "b" / "summary.json"
        after = tmp_path / "a" / "summary.json"

        robustness = {
            "per_scenario_scores": {
                "jitter": {"return_ratio": 0.90, "rmse_ratio": 1.2},
                "speed_5x": {"return_ratio": 0.40, "rmse_ratio": 2.0},
            },
            "deployment": {"worst_case": {"scenario": "jitter", "return_drop_pct": 10}},
            "stress": {"worst_case": {"scenario": "speed_5x", "return_drop_pct": 60}},
        }

        _write_summary(str(before), {
            "deployment_score": 0.90, "deployment_rating": "MILD",
            "stress_score": 0.40, "stress_rating": "FAIL",
            "quadrant": "deployment_fragile",
        }, robustness)

        _write_summary(str(after), {
            "deployment_score": 0.95, "deployment_rating": "PASS",
            "stress_score": 0.80, "stress_rating": "MILD",
            "quadrant": "deployment_ready",
        }, {
            "per_scenario_scores": {
                "jitter": {"return_ratio": 0.95, "rmse_ratio": 1.1},
                "speed_5x": {"return_ratio": 0.80, "rmse_ratio": 1.3},
            },
            "deployment": {"worst_case": {"scenario": "jitter", "return_drop_pct": 5}},
            "stress": {"worst_case": {"scenario": "speed_5x", "return_drop_pct": 20}},
        })

        md = generate_comparison(str(before), str(after))
        assert "Per-Scenario Detail" in md
        assert "jitter" in md
        assert "speed_5x" in md
        assert "Worst Scenarios" in md

    def test_worst_scenario_null(self, tmp_path):
        """Handles null scenario (no drop) gracefully."""
        before = tmp_path / "b" / "summary.json"
        after = tmp_path / "a" / "summary.json"

        _write_summary(str(before), {
            "deployment_score": 1.0, "deployment_rating": "PASS",
            "stress_score": 1.0, "stress_rating": "PASS",
            "quadrant": "deployment_ready",
        }, {
            "per_scenario_scores": {},
            "deployment": {"worst_case": {"scenario": None, "return_drop_pct": 0}},
            "stress": {"worst_case": {"scenario": None, "return_drop_pct": 0}},
        })
        _write_summary(str(after), {
            "deployment_score": 1.0, "deployment_rating": "PASS",
            "stress_score": 1.0, "stress_rating": "PASS",
            "quadrant": "deployment_ready",
        }, {
            "per_scenario_scores": {},
            "deployment": {"worst_case": {"scenario": None, "return_drop_pct": 0}},
            "stress": {"worst_case": {"scenario": None, "return_drop_pct": 0}},
        })

        md = generate_comparison(str(before), str(after))
        assert "none (no drop)" in md
