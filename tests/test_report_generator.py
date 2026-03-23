"""Security-focused tests for HTML report generation."""

from __future__ import annotations

import pathlib

from deltatau_audit.report import generate_report


def _minimal_result_with_untrusted_text() -> dict:
    return {
        "speeds": [1, 2, 3],
        "n_episodes": 3,
        "supports_intervention": False,
        "reliance": {
            "per_speed": {},
            "degradation": {},
            "score": None,
            "rating": "N/A",
            "worst_case": {
                "speed": None,
                "intervention": None,
                "rmse_ratio": None,
                "percent": None,
            },
        },
        "robustness": {
            "scenarios": {},
            "per_scenario_scores": {
                "jitter": {
                    "return_ratio": 0.9,
                    "return_drop_pct": 10.0,
                    "rmse_ratio": 1.1,
                    "rmse_increase_pct": 10.0,
                    "ci_lower": 0.8,
                    "ci_upper": 1.0,
                    "significant": True,
                },
                "speed_5x": {
                    "return_ratio": 0.6,
                    "return_drop_pct": 40.0,
                    "rmse_ratio": 1.3,
                    "rmse_increase_pct": 30.0,
                    "ci_lower": 0.5,
                    "ci_upper": 0.7,
                    "significant": True,
                },
            },
            "deployment": {
                "return_score": 0.9,
                "rmse_score": 1.1,
                "rating": "MILD",
                "worst_case": {"scenario": "jitter", "return_ratio": 0.9, "return_drop_pct": 10.0},
            },
            "stress": {
                "return_score": 0.6,
                "rmse_score": 1.3,
                "rating": "DEGRADED",
                "worst_case": {"scenario": "speed_5x", "return_ratio": 0.6, "return_drop_pct": 40.0},
            },
            "return_score": 0.6,
            "rmse_score": 1.2,
            "rating": "DEGRADED",
            "worst_case": {"scenario": "speed_5x", "return_ratio": 0.6, "return_drop_pct": 40.0},
            "n_episodes_used": {"nominal": 3, "jitter": 3, "speed_5x": 3},
            "adaptive": False,
        },
        "sensitivity": None,
        "summary": {
            "reliance_rating": "N/A",
            "reliance_score": None,
            "robustness_rating": "DEGRADED",
            "robustness_score": 0.6,
            "robustness_rmse_score": 1.2,
            "deployment_rating": "MILD",
            "deployment_score": 0.9,
            "stress_rating": "DEGRADED",
            "stress_score": 0.6,
            "sensitivity_mean": None,
            "quadrant": "deployment_ready",
            "prescription": '<img src=x onerror=alert("pwn")>',
        },
        "diagnosis": {
            "status": "warn",
            "failing_scenarios": ["speed_5x"],
            "issues": [
                {
                    "scenario": "speed_5x",
                    "rating": "DEGRADED",
                    "pattern": '<b>Pattern</b>',
                    "cause": '<i>Cause</i>',
                    "fix": '<a href="x">Fix</a>',
                }
            ],
            "primary_pattern": '<b>Pattern</b>',
            "root_cause": '<i>Cause</i>',
            "fix_recommendation": '<a href="x">Fix</a>',
            "summary_line": '<script>alert("diag")</script>',
        },
    }


def test_generate_report_escapes_untrusted_html(tmp_path: pathlib.Path):
    result = _minimal_result_with_untrusted_text()
    generate_report(result, str(tmp_path), title='<script>alert("title")</script>')

    index_path = tmp_path / "index.html"
    html = index_path.read_text(encoding="utf-8")

    # Raw HTML/script payloads must not appear in output.
    assert '<script>alert("title")</script>' not in html
    assert '<script>alert("diag")</script>' not in html
    assert '<img src=x onerror=alert("pwn")>' not in html
    assert "<b>Pattern</b>" not in html
    assert "<i>Cause</i>" not in html
    assert '<a href="x">Fix</a>' not in html

    # Escaped text should be present.
    assert "&lt;script&gt;alert(&quot;title&quot;)&lt;/script&gt;" in html
    assert "&lt;script&gt;alert(&quot;diag&quot;)&lt;/script&gt;" in html
    assert "&lt;img src=x onerror=alert(&quot;pwn&quot;)&gt;" in html
    assert "&lt;b&gt;Pattern&lt;/b&gt;" in html
