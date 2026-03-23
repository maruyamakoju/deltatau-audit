"""Tests for deltatau_audit.diff — comparison.md generation."""

import json
import os

from deltatau_audit.diff import generate_comparison, generate_comparison_html


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

    def test_accepts_flat_summary_json(self, tmp_path):
        """`generate_comparison` should accept files with summary at top level."""
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"

        before.write_text(
            json.dumps(
                {
                    "deployment_score": 0.5,
                    "deployment_rating": "DEGRADED",
                    "stress_score": 0.4,
                    "stress_rating": "FAIL",
                    "quadrant": "deployment_fragile",
                }
            ),
            encoding="utf-8",
        )
        after.write_text(
            json.dumps(
                {
                    "deployment_score": 0.9,
                    "deployment_rating": "MILD",
                    "stress_score": 0.8,
                    "stress_rating": "MILD",
                    "quadrant": "deployment_ready",
                }
            ),
            encoding="utf-8",
        )

        md = generate_comparison(str(before), str(after))
        assert "DEGRADED" in md
        assert "MILD" in md
        assert "deployment_ready" in md

    def test_scenario_union_includes_after_only_scenario(self, tmp_path):
        """Per-scenario table should include scenarios that only exist in `after`."""
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"

        before.write_text(
            json.dumps(
                {
                    "summary": {
                        "deployment_score": 0.8,
                        "deployment_rating": "MILD",
                        "stress_score": 0.6,
                        "stress_rating": "DEGRADED",
                        "quadrant": "deployment_ready",
                    },
                    "robustness": {
                        "per_scenario_scores": {
                            "jitter": {"return_ratio": 0.8, "rmse_ratio": 1.2}
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        after.write_text(
            json.dumps(
                {
                    "summary": {
                        "deployment_score": 0.9,
                        "deployment_rating": "PASS",
                        "stress_score": 0.7,
                        "stress_rating": "MILD",
                        "quadrant": "deployment_ready",
                    },
                    "robustness": {
                        "per_scenario_scores": {
                            "speed_5x": {"return_ratio": 0.7, "rmse_ratio": 1.1}
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        md = generate_comparison(str(before), str(after))
        assert "jitter" in md
        assert "speed_5x" in md

    def test_html_escapes_scenario_name(self, tmp_path):
        """HTML diff should escape scenario names from JSON keys."""
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        injected = '<script>alert("x")</script>'

        payload = {
            "summary": {
                "deployment_score": 0.8,
                "deployment_rating": "MILD",
                "stress_score": 0.7,
                "stress_rating": "MILD",
                "quadrant": "deployment_ready",
            },
            "robustness": {
                "per_scenario_scores": {
                    injected: {"return_ratio": 0.9, "rmse_ratio": 1.0, "significant": False}
                }
            },
        }
        before.write_text(json.dumps(payload), encoding="utf-8")
        after.write_text(json.dumps(payload), encoding="utf-8")

        html = generate_comparison_html(str(before), str(after))
        assert injected not in html
        assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html

    def test_html_includes_multi_seed_variance_section(self, tmp_path, monkeypatch):
        import deltatau_audit.diff as diff_mod

        monkeypatch.setattr(diff_mod, "_make_seed_variance_chart", lambda _b, _a: "ZmFrZQ==")

        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        payload = {
            "summary": {
                "deployment_score": 0.8,
                "deployment_rating": "MILD",
                "stress_score": 0.6,
                "stress_rating": "DEGRADED",
                "quadrant": "deployment_ready",
            },
            "robustness": {
                "per_scenario_scores": {
                    "jitter": {"return_ratio": 0.8, "rmse_ratio": 1.2}
                }
            },
            "seed_sweep": {
                "n_seeds": 3,
                "per_seed": [
                    {"seed": 0, "deployment_score": 0.8, "stress_score": 0.6},
                    {"seed": 1, "deployment_score": 0.9, "stress_score": 0.7},
                    {"seed": 2, "deployment_score": 0.7, "stress_score": 0.5},
                ],
                "aggregate": {
                    "pass_rates": {"deployment": 2 / 3, "stress": 2 / 3},
                    "metrics": {
                        "deployment_score": {
                            "mean": 0.8,
                            "ci_lower": 0.7,
                            "ci_upper": 0.9,
                        },
                        "stress_score": {
                            "mean": 0.6,
                            "ci_lower": 0.5,
                            "ci_upper": 0.7,
                        },
                    },
                },
            },
        }
        before.write_text(json.dumps(payload), encoding="utf-8")

        payload_after = dict(payload)
        payload_after["summary"] = dict(payload["summary"])
        payload_after["summary"]["deployment_score"] = 0.9
        payload_after["summary"]["stress_score"] = 0.7
        payload_after["seed_sweep"] = dict(payload["seed_sweep"])
        payload_after["seed_sweep"]["aggregate"] = dict(payload["seed_sweep"]["aggregate"])
        payload_after["seed_sweep"]["aggregate"]["pass_rates"] = {"deployment": 1.0, "stress": 1.0}
        payload_after["seed_sweep"]["aggregate"]["metrics"] = {
            "deployment_score": {"mean": 0.9, "ci_lower": 0.8, "ci_upper": 1.0},
            "stress_score": {"mean": 0.7, "ci_lower": 0.6, "ci_upper": 0.8},
        }
        after.write_text(json.dumps(payload_after), encoding="utf-8")

        html = generate_comparison_html(str(before), str(after))
        assert "Multi-Seed Variance" in html
        assert "deployment pass-rate" in html
