"""Tests for multi-seed aggregation utilities."""

from deltatau_audit.seed_sweep import (
    seed_sweep_payload_to_result,
    summarize_seed_sweep,
)


def _result(dep: float, stress: float, rel: float | None, quad: str):
    return {
        "summary": {
            "deployment_score": dep,
            "stress_score": stress,
            "robustness_score": min(dep, stress),
            "reliance_score": rel,
            "quadrant": quad,
            "deploy_threshold": 0.80,
            "stress_threshold": 0.50,
        }
    }


def test_summarize_seed_sweep_empty():
    out = summarize_seed_sweep([])
    assert out["n_seeds"] == 0
    assert out["metrics"] == {}


def test_summarize_seed_sweep_aggregates_scores_and_rates():
    results = [
        _result(0.90, 0.60, 2.1, "time_aware_robust"),
        _result(0.85, 0.40, 1.8, "time_aware_fragile"),
        _result(0.70, 0.55, None, "deployment_fragile"),
    ]
    out = summarize_seed_sweep(results)

    assert out["n_seeds"] == 3
    assert out["thresholds"]["deployment"] == 0.80
    assert out["thresholds"]["stress"] == 0.50

    dep_stats = out["metrics"]["deployment_score"]
    str_stats = out["metrics"]["stress_score"]
    rob_stats = out["metrics"]["robustness_score"]

    assert dep_stats["n"] == 3
    assert dep_stats["mean"] > 0.8
    assert str_stats["mean"] > 0.5
    assert rob_stats["mean"] > 0.4

    # deployment passes: 2 / 3
    assert out["pass_rates"]["deployment"] == 2 / 3
    # stress passes: 2 / 3
    assert out["pass_rates"]["stress"] == 2 / 3

    # reliance exists for only two seeds
    rel_stats = out["metrics"]["reliance_score"]
    assert rel_stats["n"] == 2
    assert out["quadrant_counts"]["time_aware_robust"] == 1
    assert out["quadrant_counts"]["time_aware_fragile"] == 1
    assert out["quadrant_counts"]["deployment_fragile"] == 1


def test_summarize_seed_sweep_includes_scenario_metrics():
    results = [
        {
            "summary": {
                "deployment_score": 0.90,
                "stress_score": 0.60,
                "robustness_score": 0.60,
                "quadrant": "deployment_ready",
                "deploy_threshold": 0.80,
                "stress_threshold": 0.50,
            },
            "robustness": {
                "per_scenario_scores": {
                    "jitter": {
                        "return_ratio": 0.90,
                        "rmse_ratio": 1.1,
                        "cohens_d": -0.2,
                        "significant": False,
                        "significant_change": False,
                    }
                }
            },
        },
        {
            "summary": {
                "deployment_score": 0.85,
                "stress_score": 0.55,
                "robustness_score": 0.55,
                "quadrant": "deployment_ready",
                "deploy_threshold": 0.80,
                "stress_threshold": 0.50,
            },
            "robustness": {
                "per_scenario_scores": {
                    "jitter": {
                        "return_ratio": 0.80,
                        "rmse_ratio": 1.3,
                        "cohens_d": -0.5,
                        "significant": True,
                        "significant_change": True,
                    }
                }
            },
        },
    ]

    out = summarize_seed_sweep(results)
    sc = out["scenario_metrics"]["jitter"]
    assert sc["return_ratio"]["n"] == 2
    assert sc["rmse_ratio"]["mean"] > 1.1
    assert sc["cohens_d"]["mean"] < 0
    assert sc["significant_rate"] == 0.5


def test_seed_sweep_payload_to_result_uses_aggregate_metrics_and_scenarios():
    payload = {
        "aggregate": {
            "metrics": {
                "deployment_score": {"mean": 0.82},
                "stress_score": {"mean": 0.47},
                "robustness_score": {"mean": 0.47},
                "reliance_score": {"mean": 1.6},
            },
            "thresholds": {"deployment": 0.8, "stress": 0.5},
            "scenario_metrics": {
                "jitter": {
                    "return_ratio": {"mean": 1.02, "ci_lower": 0.95, "ci_upper": 1.09},
                    "rmse_ratio": {"mean": 1.1},
                    "cohens_d": {"mean": -0.15},
                    "significant_rate": 0.0,
                    "significant_change_rate": 0.2,
                },
                "delay": {
                    "return_ratio": {"mean": 0.96, "ci_lower": 0.90, "ci_upper": 1.03},
                    "rmse_ratio": {"mean": 1.02},
                    "cohens_d": {"mean": -0.05},
                    "significant_rate": 0.0,
                    "significant_change_rate": 0.0,
                },
                "spike": {
                    "return_ratio": {"mean": 0.81, "ci_lower": 0.71, "ci_upper": 0.91},
                    "rmse_ratio": {"mean": 1.25},
                    "cohens_d": {"mean": -0.45},
                    "significant_rate": 0.6,
                    "significant_change_rate": 0.7,
                },
                "obs_noise": {
                    "return_ratio": {"mean": 1.01, "ci_lower": 0.95, "ci_upper": 1.08},
                    "rmse_ratio": {"mean": 0.99},
                    "cohens_d": {"mean": 0.02},
                    "significant_rate": 0.0,
                    "significant_change_rate": 0.0,
                },
                "speed_5x": {
                    "return_ratio": {"mean": 0.47, "ci_lower": 0.38, "ci_upper": 0.56},
                    "rmse_ratio": {"mean": 1.48},
                    "cohens_d": {"mean": -0.9},
                    "significant_rate": 1.0,
                    "significant_change_rate": 1.0,
                },
            },
            "quadrant_counts": {"deployment_ready": 3, "deployment_fragile": 2},
        },
        "results": [
            {
                "summary": {"sensitivity_mean": None},
                "reliance": {"per_speed": {}, "degradation": {}, "worst_case": {}},
                "robustness": {"scenarios": {}},
                "supports_intervention": False,
                "speeds": [1, 2, 3, 5, 8],
                "n_episodes": 30,
            }
        ],
        "n_seeds": 5,
        "seeds": [0, 1, 2, 3, 4],
    }

    out = seed_sweep_payload_to_result(payload, speeds=[1, 2, 3], n_episodes=10)

    assert out["summary"]["deployment_score"] == 0.82
    assert out["summary"]["stress_score"] == 0.47
    assert out["summary"]["deployment_rating"] == "MILD"
    assert out["summary"]["stress_rating"] == "FAIL"
    assert out["summary"]["quadrant"] == "deployment_ready"
    assert out["summary"]["reliance_rating"] == "HIGH"
    assert out["robustness"]["deployment"]["worst_case"]["scenario"] == "spike"
    assert out["robustness"]["stress"]["worst_case"]["scenario"] == "speed_5x"
    assert out["robustness"]["per_scenario_scores"]["speed_5x"]["significant"] is True
    assert "speed_5x" in out["diagnosis"]["failing_scenarios"]
    assert "results" not in out["seed_sweep"]


def test_seed_sweep_payload_to_result_falls_back_to_template_scores():
    payload = {
        "aggregate": {
            "metrics": {
                "deployment_score": {"mean": 0.75},
                "stress_score": {"mean": 0.70},
                "robustness_score": {"mean": 0.70},
            },
            "thresholds": {"deployment": 0.8, "stress": 0.5},
        },
        "results": [
            {
                "summary": {"quadrant": "deployment_fragile", "sensitivity_mean": 0.1},
                "robustness": {
                    "per_scenario_scores": {
                        "jitter": {"return_ratio": 0.75, "rmse_ratio": 1.2, "significant": True},
                        "speed_5x": {"return_ratio": 0.70, "rmse_ratio": 1.3, "significant": True},
                    }
                },
                "reliance": {"rating": "N/A", "score": None},
                "speeds": [1, 2, 3, 5, 8],
                "n_episodes": 42,
            }
        ],
    }

    out = seed_sweep_payload_to_result(payload, speeds=[1], n_episodes=5)

    assert out["summary"]["quadrant"] == "deployment_fragile"
    assert out["summary"]["reliance_rating"] == "N/A"
    assert out["summary"]["deployment_score"] == 0.75
    assert out["n_episodes"] == 42
    assert out["speeds"] == [1, 2, 3, 5, 8]
    assert out["robustness"]["per_scenario_scores"]["jitter"]["return_ratio"] == 0.75
