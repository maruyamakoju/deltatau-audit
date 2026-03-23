"""Tests for _theme.py — ensures single source of truth consistency."""
from __future__ import annotations


class TestRatingColors:
    def test_pass_is_green(self):
        from deltatau_audit._theme import RATING_COLORS
        assert RATING_COLORS["PASS"] == "#28a745"

    def test_mild_is_amber_not_green(self):
        """Critical: MILD must be amber, not Bootstrap green."""
        from deltatau_audit._theme import RATING_COLORS
        assert RATING_COLORS["MILD"] == "#ffc107", \
            f"MILD must be amber #ffc107, not {RATING_COLORS['MILD']}"
        assert RATING_COLORS["MILD"] != "#5cb85c", \
            "MILD must NOT be Bootstrap green #5cb85c"

    def test_fail_is_red(self):
        from deltatau_audit._theme import RATING_COLORS
        assert RATING_COLORS["FAIL"] == "#dc3545"

    def test_all_ratings_covered(self):
        from deltatau_audit._theme import RATING_COLORS
        for rating in ("PASS", "MILD", "DEGRADED", "FAIL", "N/A"):
            assert rating in RATING_COLORS


class TestRatingColorFunction:
    def test_delegates_to_dict(self):
        from deltatau_audit._theme import RATING_COLORS, rating_color
        for k, v in RATING_COLORS.items():
            assert rating_color(k) == v

    def test_unknown_returns_fallback(self):
        from deltatau_audit._theme import rating_color
        result = rating_color("UNKNOWN")
        assert result.startswith("#")  # some color, not empty

    def test_consistent_with_metrics_robustness_color(self):
        """_theme and metrics.py must agree on all rating colors."""
        from deltatau_audit._theme import rating_color
        from deltatau_audit.metrics import robustness_color
        for rating in ("PASS", "MILD", "DEGRADED", "FAIL"):
            assert rating_color(rating) == robustness_color(rating), \
                f"Color mismatch for {rating}: _theme={rating_color(rating)}, metrics={robustness_color(rating)}"


class TestQuadrantLabels:
    def test_all_quadrants_have_labels(self):
        from deltatau_audit._theme import QUADRANT_LABELS
        expected = {"time_aware_robust", "time_aware_fragile", "time_blind_fragile",
                    "time_blind_robust", "deployment_ready", "deployment_fragile"}
        assert set(QUADRANT_LABELS.keys()) >= expected

    def test_quadrant_label_function(self):
        from deltatau_audit._theme import quadrant_label
        assert quadrant_label("deployment_ready") == "Deployment Ready"
        assert quadrant_label("deployment_fragile") == "Deployment Fragile"
        assert "Time-Aware" in quadrant_label("time_aware_robust")

    def test_unknown_quadrant_graceful(self):
        from deltatau_audit._theme import quadrant_label
        result = quadrant_label("some_unknown_key")
        assert isinstance(result, str)
        assert len(result) > 0


class TestConstants:
    def test_deployment_scenarios(self):
        from deltatau_audit._constants import DEPLOYMENT_SCENARIOS
        assert "jitter" in DEPLOYMENT_SCENARIOS
        assert "delay" in DEPLOYMENT_SCENARIOS
        assert "spike" in DEPLOYMENT_SCENARIOS
        assert "obs_noise" in DEPLOYMENT_SCENARIOS

    def test_stress_scenarios(self):
        from deltatau_audit._constants import STRESS_SCENARIOS
        assert "speed_5x" in STRESS_SCENARIOS

    def test_no_overlap_between_scenario_lists(self):
        from deltatau_audit._constants import DEPLOYMENT_SCENARIOS, STRESS_SCENARIOS
        dep = set(DEPLOYMENT_SCENARIOS)
        stress = set(STRESS_SCENARIOS)
        assert dep.isdisjoint(stress), f"Overlap: {dep & stress}"

    def test_consistent_with_auditor(self):
        """_constants must match what auditor.py uses."""
        from deltatau_audit._constants import DEPLOYMENT_SCENARIOS
        from deltatau_audit.auditor import DEPLOYMENT_SCENARIOS as AUDIT_DEP
        assert set(DEPLOYMENT_SCENARIOS) == set(AUDIT_DEP), \
            f"Mismatch: _constants={DEPLOYMENT_SCENARIOS}, auditor={AUDIT_DEP}"
