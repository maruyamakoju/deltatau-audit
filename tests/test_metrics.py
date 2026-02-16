"""Tests for deltatau_audit.metrics — rating boundaries & compute functions."""

import math
import pytest

from deltatau_audit.metrics import (
    reliance_rating,
    robustness_rating,
    severity_rating,
    reliance_color,
    robustness_color,
    severity_color,
    compute_value_rmse,
    compute_value_bias,
    compute_value_mae,
    compute_discounted_returns,
    compute_degradation,
    compute_return_ratio,
    aggregate_episode_metrics,
)


# ── Reliance rating boundaries ───────────────────────────────────────

class TestRelianceRating:
    def test_low(self):
        assert reliance_rating(1.0) == "LOW"
        assert reliance_rating(1.04) == "LOW"

    def test_low_boundary(self):
        assert reliance_rating(1.049) == "LOW"

    def test_moderate(self):
        assert reliance_rating(1.05) == "MODERATE"
        assert reliance_rating(1.10) == "MODERATE"

    def test_moderate_boundary(self):
        assert reliance_rating(1.199) == "MODERATE"

    def test_high(self):
        assert reliance_rating(1.20) == "HIGH"
        assert reliance_rating(1.50) == "HIGH"

    def test_high_boundary(self):
        assert reliance_rating(1.999) == "HIGH"

    def test_very_high(self):
        assert reliance_rating(2.0) == "VERY_HIGH"
        assert reliance_rating(5.0) == "VERY_HIGH"
        assert reliance_rating(100.0) == "VERY_HIGH"


# ── Robustness rating boundaries ─────────────────────────────────────

class TestRobustnessRating:
    def test_pass(self):
        assert robustness_rating(1.0) == "PASS"
        assert robustness_rating(0.96) == "PASS"

    def test_pass_boundary(self):
        assert robustness_rating(0.951) == "PASS"

    def test_mild(self):
        assert robustness_rating(0.95) == "MILD"
        assert robustness_rating(0.90) == "MILD"

    def test_mild_boundary(self):
        assert robustness_rating(0.81) == "MILD"

    def test_degraded(self):
        assert robustness_rating(0.80) == "DEGRADED"
        assert robustness_rating(0.60) == "DEGRADED"

    def test_degraded_boundary(self):
        assert robustness_rating(0.51) == "DEGRADED"

    def test_fail(self):
        assert robustness_rating(0.50) == "FAIL"
        assert robustness_rating(0.0) == "FAIL"
        assert robustness_rating(-1.0) == "FAIL"


# ── Severity rating boundaries (legacy) ──────────────────────────────

class TestSeverityRating:
    def test_pass(self):
        assert severity_rating(0) == "PASS"
        assert severity_rating(4.9) == "PASS"

    def test_mild(self):
        assert severity_rating(5) == "MILD"
        assert severity_rating(19.9) == "MILD"

    def test_moderate(self):
        assert severity_rating(20) == "MODERATE"
        assert severity_rating(49.9) == "MODERATE"

    def test_severe(self):
        assert severity_rating(50) == "SEVERE"
        assert severity_rating(99.9) == "SEVERE"

    def test_critical(self):
        assert severity_rating(100) == "CRITICAL"
        assert severity_rating(500) == "CRITICAL"


# ── Color functions return valid hex ──────────────────────────────────

class TestColors:
    def test_reliance_colors(self):
        for rating in ["N/A", "LOW", "MODERATE", "HIGH", "VERY_HIGH"]:
            c = reliance_color(rating)
            assert c.startswith("#"), f"Bad color for {rating}: {c}"

    def test_robustness_colors(self):
        for rating in ["PASS", "MILD", "DEGRADED", "FAIL"]:
            c = robustness_color(rating)
            assert c.startswith("#"), f"Bad color for {rating}: {c}"

    def test_severity_colors(self):
        for rating in ["PASS", "MILD", "MODERATE", "SEVERE", "CRITICAL"]:
            c = severity_color(rating)
            assert c.startswith("#"), f"Bad color for {rating}: {c}"

    def test_unknown_fallback(self):
        assert reliance_color("UNKNOWN").startswith("#")
        assert robustness_color("UNKNOWN").startswith("#")
        assert severity_color("UNKNOWN").startswith("#")


# ── Compute functions ─────────────────────────────────────────────────

class TestComputeFunctions:
    def test_rmse_perfect(self):
        vals = [1.0, 2.0, 3.0]
        assert compute_value_rmse(vals, vals) == 0.0

    def test_rmse_nonzero(self):
        vals = [1.0, 2.0, 3.0]
        rets = [1.0, 2.0, 4.0]
        rmse = compute_value_rmse(vals, rets)
        expected = math.sqrt((0 + 0 + 1) / 3)
        assert abs(rmse - expected) < 1e-6

    def test_bias_zero(self):
        vals = [1.0, 2.0, 3.0]
        assert compute_value_bias(vals, vals) == 0.0

    def test_bias_overestimate(self):
        vals = [2.0, 3.0, 4.0]
        rets = [1.0, 2.0, 3.0]
        assert compute_value_bias(vals, rets) == pytest.approx(1.0)

    def test_bias_underestimate(self):
        vals = [0.0, 1.0, 2.0]
        rets = [1.0, 2.0, 3.0]
        assert compute_value_bias(vals, rets) == pytest.approx(-1.0)

    def test_mae_zero(self):
        vals = [1.0, 2.0, 3.0]
        assert compute_value_mae(vals, vals) == 0.0

    def test_mae_nonzero(self):
        vals = [1.0, 3.0, 5.0]
        rets = [2.0, 2.0, 2.0]
        # |1-2| + |3-2| + |5-2| = 1 + 1 + 3 = 5, / 3 = 5/3
        assert compute_value_mae(vals, rets) == pytest.approx(5.0 / 3.0)


# ── Discounted returns ────────────────────────────────────────────────

class TestDiscountedReturns:
    def test_single_reward(self):
        returns = compute_discounted_returns([1.0], gamma=0.99)
        assert len(returns) == 1
        assert returns[0] == pytest.approx(1.0)

    def test_no_discount(self):
        returns = compute_discounted_returns([1.0, 1.0, 1.0], gamma=0.0)
        assert returns == [1.0, 1.0, 1.0]

    def test_full_discount(self):
        returns = compute_discounted_returns([1.0, 1.0, 1.0], gamma=1.0)
        assert returns[0] == pytest.approx(3.0)
        assert returns[1] == pytest.approx(2.0)
        assert returns[2] == pytest.approx(1.0)

    def test_standard_gamma(self):
        returns = compute_discounted_returns([1.0, 0.0], gamma=0.99)
        assert returns[0] == pytest.approx(1.0)  # 1 + 0.99*0
        assert returns[1] == pytest.approx(0.0)

    def test_empty(self):
        assert compute_discounted_returns([]) == []


# ── Degradation ───────────────────────────────────────────────────────

class TestDegradation:
    def test_no_change(self):
        d = compute_degradation(1.0, 1.0)
        assert d["percent_increase"] == pytest.approx(0.0)
        assert d["ratio"] == pytest.approx(1.0)

    def test_doubled_rmse(self):
        d = compute_degradation(1.0, 2.0)
        assert d["percent_increase"] == pytest.approx(100.0)
        assert d["ratio"] == pytest.approx(2.0)
        assert d["absolute_increase"] == pytest.approx(1.0)

    def test_zero_baseline(self):
        d = compute_degradation(0.0, 0.0)
        assert d["percent_increase"] == 0.0
        assert d["ratio"] == 1.0

    def test_zero_baseline_nonzero_intervention(self):
        d = compute_degradation(0.0, 1.0)
        assert d["percent_increase"] == float("inf")
        assert d["ratio"] == float("inf")


# ── Return ratio ──────────────────────────────────────────────────────

class TestReturnRatio:
    def test_equal(self):
        assert compute_return_ratio(100.0, 100.0) == pytest.approx(1.0)

    def test_half(self):
        assert compute_return_ratio(100.0, 50.0) == pytest.approx(0.5)

    def test_zero_nominal(self):
        assert compute_return_ratio(0.0, 0.0) == 1.0

    def test_zero_nominal_nonzero_perturbed(self):
        assert compute_return_ratio(0.0, 5.0) == 0.0


# ── Aggregation ───────────────────────────────────────────────────────

class TestAggregation:
    def test_empty(self):
        agg = aggregate_episode_metrics([])
        assert agg["n_episodes"] == 0

    def test_single_episode(self):
        eps = [{"rmse": 1.0, "mae": 0.5, "bias": 0.1,
                "total_reward": 100.0, "length": 50, "dt_mean": 1.0}]
        agg = aggregate_episode_metrics(eps)
        assert agg["n_episodes"] == 1
        assert agg["rmse_mean"] == pytest.approx(1.0)
        assert agg["dt_mean"] == pytest.approx(1.0)

    def test_two_episodes(self):
        eps = [
            {"rmse": 1.0, "mae": 0.5, "bias": 0.0,
             "total_reward": 100.0, "length": 50, "dt_mean": None},
            {"rmse": 3.0, "mae": 1.5, "bias": 0.2,
             "total_reward": 200.0, "length": 60, "dt_mean": None},
        ]
        agg = aggregate_episode_metrics(eps)
        assert agg["n_episodes"] == 2
        assert agg["rmse_mean"] == pytest.approx(2.0)
        assert agg["total_reward_mean"] == pytest.approx(150.0)
        assert "dt_mean" not in agg  # all None dt
