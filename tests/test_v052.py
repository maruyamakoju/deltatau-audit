"""Tests for v0.5.2: Richer failure diagnostics."""

import pytest
from deltatau_audit.diagnose import generate_diagnosis, _return_ratio_to_rating


# ─────────────────────────────────────────────────────────────────────
# 1. _return_ratio_to_rating helper
# ─────────────────────────────────────────────────────────────────────

def test_rating_pass():
    assert _return_ratio_to_rating(1.00) == "PASS"
    assert _return_ratio_to_rating(0.95) == "PASS"


def test_rating_mild():
    assert _return_ratio_to_rating(0.94) == "MILD"
    assert _return_ratio_to_rating(0.80) == "MILD"


def test_rating_degraded():
    assert _return_ratio_to_rating(0.79) == "DEGRADED"
    assert _return_ratio_to_rating(0.60) == "DEGRADED"


def test_rating_fail():
    assert _return_ratio_to_rating(0.59) == "FAIL"
    assert _return_ratio_to_rating(0.00) == "FAIL"


# ─────────────────────────────────────────────────────────────────────
# 2. generate_diagnosis — no failures
# ─────────────────────────────────────────────────────────────────────

def _make_summary(dep_rating="PASS", str_rating="PASS"):
    return {
        "deployment_rating": dep_rating,
        "stress_rating": str_rating,
    }


def _make_robustness(per_scenario=None):
    return {"per_scenario_scores": per_scenario or {}}


def test_no_failures_returns_pass():
    summary = _make_summary("PASS", "PASS")
    robustness = _make_robustness({
        "jitter": {"return_ratio": 0.98, "return_drop_pct": 2.0},
        "speed_5x": {"return_ratio": 0.97, "return_drop_pct": 3.0},
    })
    diag = generate_diagnosis(summary, robustness)
    assert diag["status"] == "pass"
    assert diag["failing_scenarios"] == []
    assert diag["issues"] == []
    assert diag["primary_pattern"] is None
    assert diag["fix_recommendation"] is None
    assert "No significant" in diag["summary_line"]


# ─────────────────────────────────────────────────────────────────────
# 3. generate_diagnosis — single FAIL scenario
# ─────────────────────────────────────────────────────────────────────

def test_single_fail_scenario():
    summary = _make_summary("FAIL", "PASS")
    robustness = _make_robustness({
        "jitter": {"return_ratio": 0.40, "return_drop_pct": 60.0},
    })
    diag = generate_diagnosis(summary, robustness)
    assert diag["status"] == "fail"
    assert "jitter" in diag["failing_scenarios"]
    assert len(diag["issues"]) == 1
    assert diag["issues"][0]["rating"] == "FAIL"
    assert diag["primary_pattern"] == "Speed Jitter Sensitivity"
    assert diag["root_cause"] is not None
    assert diag["fix_recommendation"] is not None
    assert "JitterWrapper" in diag["fix_recommendation"] or "jitter" in diag["fix_recommendation"].lower()
    assert "1 FAIL" in diag["summary_line"]
    assert "jitter" in diag["summary_line"]


# ─────────────────────────────────────────────────────────────────────
# 4. generate_diagnosis — multiple scenarios, sorted by severity
# ─────────────────────────────────────────────────────────────────────

def test_multiple_scenarios_sorted_by_severity():
    summary = _make_summary("FAIL", "FAIL")
    robustness = _make_robustness({
        "obs_noise": {"return_ratio": 0.70, "return_drop_pct": 30.0},  # DEGRADED
        "speed_5x": {"return_ratio": 0.30, "return_drop_pct": 70.0},   # FAIL
        "delay": {"return_ratio": 0.65, "return_drop_pct": 35.0},       # DEGRADED
    })
    diag = generate_diagnosis(summary, robustness)
    assert diag["status"] == "fail"
    # FAIL scenario must be first
    assert diag["issues"][0]["scenario"] == "speed_5x"
    assert diag["issues"][0]["rating"] == "FAIL"
    # Both DEGRADED scenarios in the list
    sc_names = [i["scenario"] for i in diag["issues"]]
    assert "obs_noise" in sc_names
    assert "delay" in sc_names
    assert "1 FAIL" in diag["summary_line"]
    assert "2 DEGRADED" in diag["summary_line"]


# ─────────────────────────────────────────────────────────────────────
# 5. generate_diagnosis — known scenario patterns populated
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scenario,expected_pattern", [
    ("jitter", "Speed Jitter Sensitivity"),
    ("delay", "Observation Recency Dependency"),
    ("spike", "Frequency Spike Fragility"),
    ("obs_noise", "Observation Noise Sensitivity"),
    ("speed_5x", "Extreme Frequency Fragility"),
])
def test_known_scenario_patterns(scenario, expected_pattern):
    summary = _make_summary("FAIL", "FAIL")
    robustness = _make_robustness({
        scenario: {"return_ratio": 0.50, "return_drop_pct": 50.0},
    })
    diag = generate_diagnosis(summary, robustness)
    assert diag["primary_pattern"] == expected_pattern


# ─────────────────────────────────────────────────────────────────────
# 6. generate_diagnosis — unknown scenario gets generic pattern
# ─────────────────────────────────────────────────────────────────────

def test_unknown_scenario_gets_generic_pattern():
    summary = _make_summary("FAIL", "PASS")
    robustness = _make_robustness({
        "my_custom_perturbation": {"return_ratio": 0.40, "return_drop_pct": 60.0},
    })
    diag = generate_diagnosis(summary, robustness)
    assert diag["primary_pattern"] is not None
    # Should still produce a valid diagnosis
    assert diag["root_cause"] is not None
    assert diag["fix_recommendation"] is not None


# ─────────────────────────────────────────────────────────────────────
# 7. generate_diagnosis — warn status for DEGRADED deployment
# ─────────────────────────────────────────────────────────────────────

def test_degraded_deployment_gives_warn_status():
    summary = _make_summary("DEGRADED", "PASS")
    robustness = _make_robustness({
        "jitter": {"return_ratio": 0.75, "return_drop_pct": 25.0},
    })
    diag = generate_diagnosis(summary, robustness)
    assert diag["status"] == "warn"
    assert diag["issues"][0]["rating"] == "DEGRADED"


# ─────────────────────────────────────────────────────────────────────
# 8. run_full_audit result includes diagnosis key
# ─────────────────────────────────────────────────────────────────────

def test_run_full_audit_result_has_diagnosis_key():
    """run_full_audit() result dict must include a 'diagnosis' key."""
    import gymnasium as gym
    from deltatau_audit.auditor import run_full_audit
    from deltatau_audit.adapters.base import AgentAdapter
    import numpy as np

    class _ConstAdapter(AgentAdapter):
        supports_intervention = False

        def reset_hidden(self, batch_size=1, device="cpu"):
            return None

        def act(self, obs, hidden, dt=1.0, intervention=None):
            action = self._env_action_space.sample()
            return action, 0.0, None, None

        def rerun_with_dt(self, obs_seq, dt_seq, hidden):
            return [0.0] * len(obs_seq)

    env_factory = lambda: gym.make("CartPole-v1")
    sample_env = env_factory()
    adapter = _ConstAdapter()
    adapter._env_action_space = sample_env.action_space
    sample_env.close()

    result = run_full_audit(
        adapter, env_factory,
        speeds=[1], n_episodes=3, verbose=False,
    )
    assert "diagnosis" in result
    diag = result["diagnosis"]
    assert "status" in diag
    assert "issues" in diag
    assert "summary_line" in diag


# ─────────────────────────────────────────────────────────────────────
# 9. _print_markdown_summary includes diagnosis block when issues exist
# ─────────────────────────────────────────────────────────────────────

def test_markdown_summary_includes_diagnosis():
    from deltatau_audit.cli import _print_markdown_summary

    result = {
        "summary": {
            "deployment_rating": "FAIL",
            "deployment_score": 0.40,
            "stress_rating": "FAIL",
            "stress_score": 0.30,
            "quadrant": "deployment_fragile",
        },
        "robustness": {
            "per_scenario_scores": {
                "jitter": {"return_ratio": 0.40, "return_drop_pct": 60.0,
                           "significant": True},
            }
        },
        "diagnosis": {
            "status": "fail",
            "failing_scenarios": ["jitter"],
            "issues": [{
                "scenario": "jitter",
                "rating": "FAIL",
                "return_ratio": 0.40,
                "return_drop_pct": 60.0,
                "pattern": "Speed Jitter Sensitivity",
                "cause": "Test cause.",
                "fix": "Test fix.",
            }],
            "primary_pattern": "Speed Jitter Sensitivity",
            "root_cause": "Test cause.",
            "fix_recommendation": "Test fix.",
            "summary_line": "1 FAIL scenario: jitter",
        },
    }
    md = _print_markdown_summary(result)
    assert "Failure Analysis" in md
    assert "Speed Jitter Sensitivity" in md
    assert "Test cause." in md
    assert "Test fix." in md


def test_markdown_summary_no_diagnosis_block_when_passing():
    from deltatau_audit.cli import _print_markdown_summary

    result = {
        "summary": {
            "deployment_rating": "PASS",
            "deployment_score": 0.97,
            "stress_rating": "PASS",
            "stress_score": 0.95,
            "quadrant": "deployment_ready",
        },
        "robustness": {"per_scenario_scores": {}},
        "diagnosis": {
            "status": "pass",
            "failing_scenarios": [],
            "issues": [],
            "primary_pattern": None,
            "root_cause": None,
            "fix_recommendation": None,
            "summary_line": "No significant timing failures detected.",
        },
    }
    md = _print_markdown_summary(result)
    assert "Failure Analysis" not in md
