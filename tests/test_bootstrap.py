"""Tests for bootstrap confidence interval functions."""

import pytest

from deltatau_audit.metrics import bootstrap_ci, bootstrap_return_ratio


class TestBootstrapCI:
    def test_empty(self):
        result = bootstrap_ci([])
        assert result["n"] == 0
        assert result["mean"] == 0.0

    def test_single_value(self):
        result = bootstrap_ci([5.0])
        assert result["mean"] == 5.0
        assert result["ci_lower"] == 5.0
        assert result["ci_upper"] == 5.0
        assert result["n"] == 1

    def test_identical_values(self):
        result = bootstrap_ci([3.0] * 20)
        assert result["mean"] == pytest.approx(3.0)
        assert result["ci_lower"] == pytest.approx(3.0)
        assert result["ci_upper"] == pytest.approx(3.0)

    def test_ci_contains_mean(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = bootstrap_ci(data)
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_ci_width_decreases_with_n(self):
        """Larger sample → narrower CI."""
        import numpy as np
        rng = np.random.RandomState(0)
        small = rng.normal(100, 10, size=10).tolist()
        large = rng.normal(100, 10, size=200).tolist()

        ci_small = bootstrap_ci(small)
        ci_large = bootstrap_ci(large)

        width_small = ci_small["ci_upper"] - ci_small["ci_lower"]
        width_large = ci_large["ci_upper"] - ci_large["ci_lower"]
        assert width_large < width_small


class TestBootstrapReturnRatio:
    def test_identical_returns(self):
        nom = [100.0] * 20
        pert = [100.0] * 20
        result = bootstrap_return_ratio(nom, pert)
        assert result["ratio"] == pytest.approx(1.0)
        assert result["significant"] is False

    def test_clear_drop(self):
        nom = [100.0] * 30
        pert = [50.0] * 30
        result = bootstrap_return_ratio(nom, pert)
        assert result["ratio"] == pytest.approx(0.5)
        assert result["significant"] is True
        assert result["ci_upper"] < 1.0

    def test_no_drop(self):
        nom = [100.0] * 30
        pert = [110.0] * 30
        result = bootstrap_return_ratio(nom, pert)
        assert result["ratio"] > 1.0
        assert result["significant"] is False

    def test_noisy_but_significant(self):
        """Large consistent drop should be significant even with noise."""
        import numpy as np
        rng = np.random.RandomState(42)
        nom = (rng.normal(100, 10, size=50)).tolist()
        pert = (rng.normal(60, 10, size=50)).tolist()
        result = bootstrap_return_ratio(nom, pert)
        assert result["significant"] is True

    def test_empty_inputs(self):
        result = bootstrap_return_ratio([], [100.0])
        assert result["significant"] is False
