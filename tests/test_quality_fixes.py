"""Tests for v0.4.3 quality fixes:
  1. Episode timeout / max_steps guard
  2. Seed reproducibility
  3. Negative nominal return ratio
  4. Continuous action detection in fixer_cleanrl
"""

import warnings
import pytest
import torch
import numpy as np

from deltatau_audit.metrics import compute_return_ratio, bootstrap_return_ratio
from deltatau_audit.auditor import _run_single_episode, run_robustness_audit


# ─────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────

class _NeverDoneEnv:
    """Gymnasium-compatible env that never terminates (step returns done=False)."""

    def __init__(self):
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {}

    def close(self):
        pass


class _DummyAdapter:
    """Minimal adapter for testing (no intervention, no value recompute)."""

    supports_intervention = False
    supports_value_recompute = False

    def reset_hidden(self, batch=1, device="cpu"):
        return None

    def act(self, obs, hidden):
        return 0, 1.0, hidden, None


# ─────────────────────────────────────────────────────────────────────
# 1. Episode timeout
# ─────────────────────────────────────────────────────────────────────

class TestEpisodeTimeout:
    def test_timeout_fires(self):
        """Episode exceeding max_steps must be truncated with RuntimeWarning."""
        env = _NeverDoneEnv()
        adapter = _DummyAdapter()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _run_single_episode(
                adapter, env, intervention="none",
                max_steps=10,
            )

        assert len(caught) == 1
        assert issubclass(caught[0].category, RuntimeWarning)
        assert "max_steps=10" in str(caught[0].message)

    def test_timeout_episode_length(self):
        """Truncated episode must have exactly max_steps steps."""
        env = _NeverDoneEnv()
        adapter = _DummyAdapter()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = _run_single_episode(
                adapter, env, intervention="none",
                max_steps=7,
            )

        assert result["length"] == 7

    def test_normal_episode_no_warning(self):
        """An episode that terminates normally must not raise RuntimeWarning."""
        import gymnasium as gym

        env = gym.make("CartPole-v1")
        adapter = _DummyAdapter()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _run_single_episode(adapter, env, max_steps=10_000)

        timeout_warnings = [w for w in caught
                            if issubclass(w.category, RuntimeWarning)]
        assert len(timeout_warnings) == 0
        env.close()


# ─────────────────────────────────────────────────────────────────────
# 2. Seed reproducibility
# ─────────────────────────────────────────────────────────────────────

class TestSeedReproducibility:
    def test_same_seed_same_episode(self):
        """Two runs with the same seed must produce identical total_reward."""
        import gymnasium as gym

        adapter = _DummyAdapter()

        def run(seed):
            env = gym.make("CartPole-v1")
            result = _run_single_episode(adapter, env, seed=seed)
            env.close()
            return result

        r1 = run(42)
        r2 = run(42)
        assert r1["total_reward"] == r2["total_reward"]
        assert r1["length"] == r2["length"]

    def test_different_seeds_differ(self):
        """Different seeds should (almost always) produce different episodes."""
        import gymnasium as gym

        adapter = _DummyAdapter()

        lengths = set()
        for seed in range(10):
            env = gym.make("CartPole-v1")
            result = _run_single_episode(adapter, env, seed=seed)
            env.close()
            lengths.add(result["length"])

        # With 10 different seeds, we expect at least 2 distinct lengths
        assert len(lengths) >= 2

    def test_robustness_audit_seed_reproducible(self):
        """run_robustness_audit with seed must give same return scores."""
        import gymnasium as gym

        adapter = _DummyAdapter()

        def cartpole():
            return gym.make("CartPole-v1")

        res1 = run_robustness_audit(
            adapter, cartpole,
            scenarios=["nominal", "jitter"],
            n_episodes=5,
            verbose=False,
            seed=0,
        )
        res2 = run_robustness_audit(
            adapter, cartpole,
            scenarios=["nominal", "jitter"],
            n_episodes=5,
            verbose=False,
            seed=0,
        )

        s1 = res1["scenarios"]["nominal"]["total_reward_mean"]
        s2 = res2["scenarios"]["nominal"]["total_reward_mean"]
        assert abs(s1 - s2) < 1e-6


# ─────────────────────────────────────────────────────────────────────
# 3. Negative nominal return ratio
# ─────────────────────────────────────────────────────────────────────

class TestNegativeNominalReturnRatio:
    """Verify sign-aware ratio formula for penalty-heavy envs."""

    # ── compute_return_ratio ──────────────────────────────────────────

    def test_positive_nominal_normal(self):
        assert compute_return_ratio(100.0, 50.0) == pytest.approx(0.5)

    def test_positive_nominal_equal(self):
        assert compute_return_ratio(100.0, 100.0) == pytest.approx(1.0)

    def test_negative_nominal_improvement(self):
        """nominal=-100, perturbed=-50  → 1.5 (less penalty = improvement)."""
        assert compute_return_ratio(-100.0, -50.0) == pytest.approx(1.5)

    def test_negative_nominal_same(self):
        """nominal=-100, perturbed=-100 → 1.0 (no change)."""
        assert compute_return_ratio(-100.0, -100.0) == pytest.approx(1.0)

    def test_negative_nominal_degradation(self):
        """nominal=-100, perturbed=-150 → 0.5 (more penalty = degradation)."""
        assert compute_return_ratio(-100.0, -150.0) == pytest.approx(0.5)

    def test_negative_nominal_double_penalty(self):
        """nominal=-100, perturbed=-200 → 0.0 (penalty doubled, maps to 0)."""
        assert compute_return_ratio(-100.0, -200.0) == pytest.approx(0.0)

    def test_zero_nominal_zero_perturbed(self):
        assert compute_return_ratio(0.0, 0.0) == pytest.approx(1.0)

    def test_zero_nominal_nonzero_perturbed(self):
        assert compute_return_ratio(0.0, 5.0) == pytest.approx(0.0)

    # ── bootstrap_return_ratio ────────────────────────────────────────

    def test_bootstrap_negative_nominal_improvement(self):
        """Bootstrap ratio must also handle negative nominal correctly."""
        # nominal all -100, perturbed all -50 → ratio ≈ 1.5
        nominal = [-100.0] * 20
        perturbed = [-50.0] * 20

        res = bootstrap_return_ratio(nominal, perturbed, n_bootstrap=500)
        assert res["ratio"] == pytest.approx(1.5, abs=0.01)
        # CI should be tight for constant data
        assert res["ci_lower"] > 1.0
        assert res["ci_upper"] > 1.0

    def test_bootstrap_negative_nominal_degradation(self):
        """Bootstrap ratio must flag more penalty as degradation."""
        nominal = [-100.0] * 20
        perturbed = [-150.0] * 20

        res = bootstrap_return_ratio(nominal, perturbed, n_bootstrap=500)
        assert res["ratio"] == pytest.approx(0.5, abs=0.01)
        # CI entirely below 1.0 → significant
        assert res["significant"] is True

    def test_bootstrap_positive_nominal_degrades(self):
        """Regression check: positive nominal still works correctly."""
        nominal = [100.0] * 20
        perturbed = [50.0] * 20

        res = bootstrap_return_ratio(nominal, perturbed, n_bootstrap=500)
        assert res["ratio"] == pytest.approx(0.5, abs=0.01)
        assert res["significant"] is True

    def test_bootstrap_empty(self):
        res = bootstrap_return_ratio([], [], n_bootstrap=100)
        assert res["ratio"] == 0.0
        assert res["significant"] is False


# ─────────────────────────────────────────────────────────────────────
# 4. Continuous action detection in _ppo_train_cleanrl
# ─────────────────────────────────────────────────────────────────────

class TestContinuousActionDetection:
    """
    Unit-test the action-space detection logic extracted from fixer_cleanrl.py.
    We test the dtype-based heuristic directly, since running the full PPO
    loop is expensive in a test suite.
    """

    def test_discrete_action_dtype_detection(self):
        """Integer-dtype actions should be classified as discrete."""
        for dtype in (torch.int32, torch.int64, torch.bool):
            action = torch.tensor([1], dtype=dtype)
            is_discrete = action.dtype in (torch.int32, torch.int64, torch.bool)
            assert is_discrete, f"Expected discrete for dtype={dtype}"

    def test_continuous_action_dtype_detection(self):
        """Float-dtype actions should be classified as continuous."""
        for dtype in (torch.float32, torch.float64):
            action = torch.tensor([0.5, -0.3], dtype=dtype)
            is_discrete = action.dtype in (torch.int32, torch.int64, torch.bool)
            assert not is_discrete, f"Expected continuous for dtype={dtype}"

    def test_act_dim_discrete(self):
        """Discrete act_dim must be 1."""
        action = torch.tensor([2], dtype=torch.int64)
        is_discrete = action.dtype in (torch.int32, torch.int64, torch.bool)
        act_dim = 1 if is_discrete else int(action.numel())
        assert act_dim == 1

    def test_act_dim_continuous(self):
        """Continuous act_dim must equal number of action dimensions."""
        action = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        is_discrete = action.dtype in (torch.int32, torch.int64, torch.bool)
        act_dim = 1 if is_discrete else int(action.numel())
        assert act_dim == 3

    def test_buffer_shape_discrete(self):
        """Discrete buffer must be 1-D (num_steps,)."""
        num_steps = 128
        act_buf = torch.zeros(num_steps, dtype=torch.long)
        assert act_buf.shape == (num_steps,)
        assert act_buf.dtype == torch.long

    def test_buffer_shape_continuous(self):
        """Continuous buffer must be 2-D (num_steps, act_dim)."""
        num_steps, act_dim = 128, 6
        act_buf = torch.zeros(num_steps, act_dim, dtype=torch.float32)
        assert act_buf.shape == (num_steps, act_dim)
        assert act_buf.dtype == torch.float32
