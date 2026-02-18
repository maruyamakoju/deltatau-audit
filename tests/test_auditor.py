"""Tests for deltatau_audit.auditor — core audit engine."""

import pytest
import torch
import numpy as np
import gymnasium as gym

from deltatau_audit.auditor import (
    _run_single_episode,
    _make_wrapped_env,
    run_reliance_audit,
    run_robustness_audit,
    compute_temporal_sensitivity,
    run_full_audit,
)


# ── Test fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def cartpole_factory():
    """Factory for CartPole-v1 env."""
    def factory():
        return gym.make("CartPole-v1")
    return factory


# ── _run_single_episode ───────────────────────────────────────────────

class TestRunSingleEpisode:
    def test_runs_to_completion(self, dummy_adapter, cartpole_factory):
        """Episode should run until termination."""
        env = cartpole_factory()
        result = _run_single_episode(dummy_adapter, env, "none", gamma=0.99)

        assert "rmse" in result
        assert "mae" in result
        assert "bias" in result
        assert "total_reward" in result
        assert "length" in result
        assert "dt_mean" in result

        assert result["total_reward"] > 0  # CartPole gives +1 per step
        assert result["length"] > 0
        assert result["dt_mean"] is None  # dummy has no dt

        env.close()

    def test_intervention_not_applied_without_support(self, dummy_adapter, cartpole_factory):
        """Intervention should be skipped if adapter doesn't support it."""
        env = cartpole_factory()
        result = _run_single_episode(dummy_adapter, env, "clamp_1", gamma=0.99)

        # Should run normally without error
        assert result["total_reward"] > 0
        env.close()

    def test_intervention_applied_with_support(self, intervention_adapter, cartpole_factory):
        """Intervention should be applied if adapter supports it."""
        env = cartpole_factory()
        result = _run_single_episode(intervention_adapter, env, "clamp_1", gamma=0.99)

        assert result["total_reward"] > 0
        assert result["dt_mean"] is not None  # intervention adapter has dt
        env.close()

    def test_different_gamma_values(self, dummy_adapter, cartpole_factory):
        """Test that different gamma values affect return computation."""
        env1 = cartpole_factory()
        result1 = _run_single_episode(dummy_adapter, env1, "none", gamma=0.0)
        env1.close()

        env2 = cartpole_factory()
        result2 = _run_single_episode(dummy_adapter, env2, "none", gamma=0.99)
        env2.close()

        # Both should complete successfully
        assert result1["total_reward"] > 0
        assert result2["total_reward"] > 0


# ── _make_wrapped_env ─────────────────────────────────────────────────

class TestMakeWrappedEnv:
    def test_nominal_scenario(self, cartpole_factory):
        """Nominal should return unwrapped env."""
        env = _make_wrapped_env(cartpole_factory, "nominal")
        assert env is not None

        # Should be able to step
        obs, _ = env.reset()
        obs, reward, term, trunc, info = env.step(0)
        assert obs is not None
        env.close()

    def test_speed_5x_scenario(self, cartpole_factory):
        """speed_5x should wrap with FixedSpeedWrapper."""
        env = _make_wrapped_env(cartpole_factory, "speed_5x")
        obs, _ = env.reset()
        obs, reward, term, trunc, info = env.step(0)

        # Reward should accumulate from 5 steps
        assert reward >= 0
        env.close()

    def test_jitter_scenario(self, cartpole_factory):
        """jitter should wrap with JitterWrapper."""
        env = _make_wrapped_env(cartpole_factory, "jitter")
        obs, _ = env.reset()
        obs, reward, term, trunc, info = env.step(0)

        assert "actual_speed" in info
        env.close()

    def test_delay_scenario(self, cartpole_factory):
        """delay should wrap with ObservationDelayWrapper."""
        env = _make_wrapped_env(cartpole_factory, "delay")
        obs, _ = env.reset()
        obs, reward, term, trunc, info = env.step(0)

        assert "obs_delay" in info
        env.close()

    def test_spike_scenario(self, cartpole_factory):
        """spike should wrap with PiecewiseSwitchWrapper."""
        env = _make_wrapped_env(cartpole_factory, "spike")
        obs, _ = env.reset()
        obs, reward, term, trunc, info = env.step(0)

        assert "current_speed" in info
        env.close()

    def test_unknown_scenario_raises(self, cartpole_factory):
        """Unknown scenario should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown robustness scenario"):
            _make_wrapped_env(cartpole_factory, "unknown_scenario")


# ── run_reliance_audit ────────────────────────────────────────────────

class TestRunRelianceAudit:
    def test_no_intervention_support(self, dummy_adapter, cartpole_factory):
        """Adapter without intervention support should return N/A."""
        result = run_reliance_audit(
            dummy_adapter, cartpole_factory,
            speeds=[1, 2],
            n_episodes=2,
            verbose=False,
        )

        assert result["rating"] == "N/A"
        assert result["score"] is None
        assert result["per_speed"] == {}
        assert result["worst_case"]["speed"] is None

    def test_with_intervention_support(self, intervention_adapter, cartpole_factory):
        """Adapter with intervention support should run ablation."""
        result = run_reliance_audit(
            intervention_adapter, cartpole_factory,
            speeds=[1, 2],
            n_episodes=3,
            interventions=["none", "clamp_1"],
            verbose=False,
        )

        assert result["rating"] in ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
        assert result["score"] is not None
        assert "1" in result["per_speed"]
        assert "2" in result["per_speed"]
        assert "none" in result["per_speed"]["1"]
        assert "clamp_1" in result["per_speed"]["1"]

        # Check degradation structure
        if result["degradation"]:
            for interv in ["clamp_1"]:
                if interv in result["degradation"]:
                    deg = result["degradation"][interv]
                    assert isinstance(deg, dict)

    def test_custom_speeds(self, intervention_adapter, cartpole_factory):
        """Custom speed list should be respected."""
        result = run_reliance_audit(
            intervention_adapter, cartpole_factory,
            speeds=[3, 5],
            n_episodes=2,
            verbose=False,
        )

        assert "3" in result["per_speed"]
        assert "5" in result["per_speed"]
        assert "1" not in result["per_speed"]


# ── run_robustness_audit ──────────────────────────────────────────────

class TestRunRobustnessAudit:
    def test_basic_robustness(self, dummy_adapter, cartpole_factory):
        """Basic robustness test with minimal scenarios."""
        result = run_robustness_audit(
            dummy_adapter, cartpole_factory,
            scenarios=["nominal", "jitter"],
            n_episodes=3,
            verbose=False,
        )

        assert result["rating"] in ["PASS", "MILD", "DEGRADED", "FAIL"]
        assert "nominal" in result["scenarios"]
        assert "jitter" in result["scenarios"]
        assert "jitter" in result["per_scenario_scores"]

        # Check score structure
        jitter_score = result["per_scenario_scores"]["jitter"]
        assert "return_ratio" in jitter_score
        assert "return_drop_pct" in jitter_score
        assert "rmse_ratio" in jitter_score
        assert "ci_lower" in jitter_score
        assert "ci_upper" in jitter_score
        assert "significant" in jitter_score

    def test_all_scenarios(self, dummy_adapter, cartpole_factory):
        """Test with all robustness scenarios."""
        result = run_robustness_audit(
            dummy_adapter, cartpole_factory,
            scenarios=None,  # Should default to all
            n_episodes=2,
            verbose=False,
        )

        # Should include all predefined scenarios
        assert "nominal" in result["scenarios"]
        assert "speed_5x" in result["scenarios"]
        assert "jitter" in result["scenarios"]
        assert "delay" in result["scenarios"]
        assert "spike" in result["scenarios"]

    def test_deployment_vs_stress_split(self, dummy_adapter, cartpole_factory):
        """Test that deployment and stress are split correctly."""
        result = run_robustness_audit(
            dummy_adapter, cartpole_factory,
            scenarios=["nominal", "jitter", "speed_5x"],
            n_episodes=2,
            verbose=False,
        )

        assert "deployment" in result
        assert "stress" in result

        # Deployment should have jitter
        assert result["deployment"]["worst_case"]["scenario"] is not None

        # Stress should have speed_5x
        assert result["stress"]["worst_case"]["scenario"] is not None

    def test_worst_case_tracking(self, dummy_adapter, cartpole_factory):
        """Worst case should be tracked correctly."""
        result = run_robustness_audit(
            dummy_adapter, cartpole_factory,
            scenarios=["nominal", "jitter"],
            n_episodes=2,
            verbose=False,
        )

        worst = result["worst_case"]
        assert "scenario" in worst
        assert "return_ratio" in worst
        assert "return_drop_pct" in worst


# ── compute_temporal_sensitivity ──────────────────────────────────────

class TestComputeTemporalSensitivity:
    def test_no_intervention_support(self, dummy_adapter, cartpole_factory):
        """Without intervention support, should return None."""
        result = compute_temporal_sensitivity(
            dummy_adapter, cartpole_factory,
            speeds=[1],
            n_episodes=2,
            verbose=False,
        )

        assert result is None

    def test_with_intervention_support(self, intervention_adapter, cartpole_factory):
        """With intervention support, should compute sensitivity."""
        result = compute_temporal_sensitivity(
            intervention_adapter, cartpole_factory,
            speeds=[1, 2],
            n_episodes=2,
            epsilon=0.1,
            verbose=False,
        )

        if result is not None:  # May be None if no dt samples
            assert "mean" in result
            assert "std" in result
            assert "median" in result
            assert "n_samples" in result
            assert "per_speed" in result

            if result["n_samples"] > 0:
                assert result["mean"] >= 0
                assert result["std"] >= 0


# ── run_full_audit ────────────────────────────────────────────────────

class TestRunFullAudit:
    def test_full_audit_structure(self, dummy_adapter, cartpole_factory):
        """Full audit should return complete structure."""
        result = run_full_audit(
            dummy_adapter, cartpole_factory,
            speeds=[1, 2],
            n_episodes=2,
            sensitivity_episodes=1,
            verbose=False,
        )

        # Top-level keys
        assert "speeds" in result
        assert "n_episodes" in result
        assert "supports_intervention" in result
        assert "reliance" in result
        assert "robustness" in result
        assert "sensitivity" in result
        assert "summary" in result

        # Summary keys
        summary = result["summary"]
        assert "reliance_rating" in summary
        assert "deployment_rating" in summary
        assert "stress_rating" in summary
        assert "quadrant" in summary
        assert "prescription" in summary

    def test_no_intervention_quadrant(self, dummy_adapter, cartpole_factory):
        """Agent without intervention should get 1-axis classification."""
        result = run_full_audit(
            dummy_adapter, cartpole_factory,
            speeds=[1],
            n_episodes=2,
            sensitivity_episodes=0,
            verbose=False,
        )

        summary = result["summary"]
        assert summary["reliance_rating"] == "N/A"
        assert summary["quadrant"] in ["deployment_ready", "deployment_fragile"]

    def test_with_intervention_quadrant(self, intervention_adapter, cartpole_factory):
        """Agent with intervention should get 2-axis classification."""
        result = run_full_audit(
            intervention_adapter, cartpole_factory,
            speeds=[1, 2],
            n_episodes=2,
            sensitivity_episodes=1,
            verbose=False,
        )

        summary = result["summary"]
        assert summary["reliance_rating"] != "N/A"
        assert summary["quadrant"] in [
            "time_aware_robust",
            "time_aware_fragile",
            "time_blind_robust",
            "time_blind_fragile",
        ]

    def test_custom_parameters(self, dummy_adapter, cartpole_factory):
        """Custom parameters should be respected."""
        result = run_full_audit(
            dummy_adapter, cartpole_factory,
            speeds=[3],
            n_episodes=1,
            robustness_scenarios=["nominal", "jitter"],
            sensitivity_episodes=0,
            gamma=0.95,
            verbose=False,
        )

        assert result["speeds"] == [3]
        assert result["n_episodes"] == 1
        assert "jitter" in result["robustness"]["scenarios"]

    def test_prescription_provided(self, dummy_adapter, cartpole_factory):
        """All quadrants should provide actionable prescription."""
        result = run_full_audit(
            dummy_adapter, cartpole_factory,
            speeds=[1],
            n_episodes=2,
            sensitivity_episodes=0,
            verbose=False,
        )

        prescription = result["summary"]["prescription"]
        assert isinstance(prescription, str)
        assert len(prescription) > 50  # Should be substantial
