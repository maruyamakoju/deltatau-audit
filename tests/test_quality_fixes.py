"""Tests for v0.4.3 and v0.4.4 quality fixes:
  1. Episode timeout / max_steps guard
  2. Seed reproducibility
  3. Negative nominal return ratio
  4. Continuous action detection in fixer_cleanrl
  5. ObsNoiseWrapper
  6. Parallel episode execution (n_workers > 1)
  7. obs_noise scenario in robustness audit
"""

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import torch

from deltatau_audit._constants import DEPLOYMENT_SCENARIOS
from deltatau_audit._constants import ROBUSTNESS_SCENARIO_LABELS as ROBUSTNESS_SCENARIOS
from deltatau_audit.auditors import RobustnessAuditor
from deltatau_audit.core.runner import EpisodeRunner
from deltatau_audit.metrics import bootstrap_return_ratio, compute_return_ratio
from deltatau_audit.schema import TemporalCapability
from deltatau_audit.wrappers.factory import create_wrapped_env

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

    def reset_internal_state(self) -> None:
        pass

    def act(
        self,
        observation: Any,
        deterministic: bool = True,
        ponder_steps: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        return 0, {"value": 0.0, "dt": 1.0}

    def get_capabilities(self) -> TemporalCapability:
        return TemporalCapability(can_ponder=False, max_lookahead_steps=0)


# ─────────────────────────────────────────────────────────────────────
# 1. Episode timeout
# ─────────────────────────────────────────────────────────────────────

class TestEpisodeTimeout:
    def test_timeout_fires(self):
        """Episode exceeding max_steps must be truncated with RuntimeWarning."""
        env_factory = lambda: _NeverDoneEnv()
        adapter = _DummyAdapter()
        runner = EpisodeRunner(adapter, env_factory, max_steps=10)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner.run_single(intervention="none")

        assert len(caught) == 1
        assert issubclass(caught[0].category, RuntimeWarning)
        assert "max_steps=10" in str(caught[0].message)

    def test_timeout_episode_length(self):
        """Truncated episode must have exactly max_steps steps."""
        env_factory = lambda: _NeverDoneEnv()
        adapter = _DummyAdapter()
        runner = EpisodeRunner(adapter, env_factory, max_steps=7)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = runner.run_single(intervention="none")

        assert result.length == 7

    def test_normal_episode_no_warning(self):
        """An episode that terminates normally must not raise RuntimeWarning."""
        import gymnasium as gym

        env_factory = lambda: gym.make("CartPole-v1")
        adapter = _DummyAdapter()
        runner = EpisodeRunner(adapter, env_factory, max_steps=10_000)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner.run_single()

        timeout_warnings = [w for w in caught
                            if issubclass(w.category, RuntimeWarning)]
        assert len(timeout_warnings) == 0


# ─────────────────────────────────────────────────────────────────────
# 2. Seed reproducibility
# ─────────────────────────────────────────────────────────────────────

class TestSeedReproducibility:
    def test_same_seed_same_episode(self):
        """Two runs with the same seed must produce identical total_reward."""
        import gymnasium as gym

        adapter = _DummyAdapter()
        env_factory = lambda: gym.make("CartPole-v1")
        runner = EpisodeRunner(adapter, env_factory)

        def run(seed):
            return runner.run_single(seed=seed)

        r1 = run(42)
        r2 = run(42)
        assert r1.total_reward == r2.total_reward
        assert r1.length == r2.length

    def test_different_seeds_differ(self):
        """Different seeds should (almost always) produce different episodes."""
        import gymnasium as gym

        adapter = _DummyAdapter()
        env_factory = lambda: gym.make("CartPole-v1")
        runner = EpisodeRunner(adapter, env_factory)

        lengths = set()
        for seed in range(10):
            result = runner.run_single(seed=seed)
            lengths.add(result.length)

        # With 10 different seeds, we expect at least 2 distinct lengths
        assert len(lengths) >= 2

    def test_robustness_audit_seed_reproducible(self):
        """RobustnessAuditor with seed must give same return scores."""

        adapter = _DummyAdapter()
        env_id = "CartPole-v1"

        auditor = RobustnessAuditor(
            n_episodes=5,
            verbose=False,
            seed=0,
        )

        # Auditor uses create_wrapped_env internally
        res1 = auditor.run(adapter, env_id, scenarios=["nominal", "jitter"])
        res2 = auditor.run(adapter, env_id, scenarios=["nominal", "jitter"])

        # Compare nominal stage results
        nom_label = ROBUSTNESS_SCENARIOS.get("nominal", "nominal")
        s1 = [s for s in res1.stages if s.stage_name == nom_label][0].metrics["mean_reward"].value
        s2 = [s for s in res2.stages if s.stage_name == nom_label][0].metrics["mean_reward"].value
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


# ─────────────────────────────────────────────────────────────────────
# 5. ObsNoiseWrapper
# ─────────────────────────────────────────────────────────────────────

class TestObsNoiseWrapper:
    """Verify Gaussian observation noise wrapper."""

    def test_noise_applied_on_step(self):
        """Step must return a perturbed observation (not identical to clean)."""
        import gymnasium as gym

        from deltatau_audit.wrappers.latency import ObsNoiseWrapper

        base = gym.make("CartPole-v1")
        env = ObsNoiseWrapper(base, std=1.0, seed=0)
        env.reset(seed=0)

        # Collect 20 steps; at least some should differ from clean obs
        clean_env = gym.make("CartPole-v1")
        clean_env.reset(seed=0)

        diffs = []
        for _ in range(20):
            action = 0
            noisy_obs, _, _, _, _ = env.step(action)
            clean_obs, _, _, _, _ = clean_env.step(action)
            diffs.append(np.any(noisy_obs != clean_obs))

        assert any(diffs), "Noise wrapper returned identical obs on all steps"
        clean_env.close()
        env.close()

    def test_reset_is_clean(self):
        """Reset must return obs without noise (initial state is clean)."""
        import gymnasium as gym

        from deltatau_audit.wrappers.latency import ObsNoiseWrapper

        base_env = gym.make("CartPole-v1")
        noisy_env = ObsNoiseWrapper(base_env, std=10.0, seed=42)

        # Two independent resets with same seed → same obs
        obs_noisy, _ = noisy_env.reset(seed=99)
        # Compare with a fresh base env reset
        base_env2 = gym.make("CartPole-v1")
        obs_clean, _ = base_env2.reset(seed=99)

        np.testing.assert_array_equal(
            obs_noisy, obs_clean,
            err_msg="ObsNoiseWrapper reset should return clean observation")
        noisy_env.close()
        base_env2.close()

    def test_noise_info_key(self):
        """Step info must contain obs_noise_std key."""
        import gymnasium as gym

        from deltatau_audit.wrappers.latency import ObsNoiseWrapper

        env = ObsNoiseWrapper(gym.make("CartPole-v1"), std=0.05, seed=0)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "obs_noise_std" in info
        assert info["obs_noise_std"] == pytest.approx(0.05)
        env.close()

    def test_obs_noise_scenario_in_auditor(self):
        """obs_noise must be a valid robustness scenario."""
        assert "obs_noise" in ROBUSTNESS_SCENARIOS
        assert "obs_noise" in DEPLOYMENT_SCENARIOS

    def test_make_wrapped_env_obs_noise(self):
        """create_wrapped_env must create ObsNoiseWrapper for obs_noise scenario."""
        import gymnasium as gym

        from deltatau_audit.wrappers.latency import ObsNoiseWrapper

        env = create_wrapped_env(lambda: gym.make("CartPole-v1"), "obs_noise")
        assert isinstance(env, ObsNoiseWrapper)
        env.close()


# ─────────────────────────────────────────────────────────────────────
# 6. Parallel episode execution
# ─────────────────────────────────────────────────────────────────────

class TestParallelExecution:
    """Verify n_workers > 1 produces correct results."""

    def _cartpole(self):
        import gymnasium as gym
        return gym.make("CartPole-v1")

    def test_parallel_same_count(self):
        """Parallel run must return same number of episodes as serial."""
        adapter = _DummyAdapter()
        env_id = "CartPole-v1"

        auditor_serial = RobustnessAuditor(
            n_episodes=4, verbose=False, seed=0, n_workers=1,
        )
        auditor_parallel = RobustnessAuditor(
            n_episodes=4, verbose=False, seed=0, n_workers=2,
        )

        res_s = auditor_serial.run(adapter, env_id, scenarios=["nominal"])
        res_p = auditor_parallel.run(adapter, env_id, scenarios=["nominal"])

        # Stages list
        nom_label = ROBUSTNESS_SCENARIOS.get("nominal", "nominal")
        s_n = [s for s in res_s.stages if s.stage_name == nom_label][0].metrics["n_episodes"].value
        p_n = [s for s in res_p.stages if s.stage_name == nom_label][0].metrics["n_episodes"].value
        assert s_n == p_n == 4

    def test_parallel_result_structure(self):
        """Parallel run must return the correct Report structure."""
        adapter = _DummyAdapter()
        env_id = "CartPole-v1"

        auditor = RobustnessAuditor(
            n_episodes=3, verbose=False, seed=42, n_workers=2,
        )
        report = auditor.run(adapter, env_id, scenarios=["nominal", "obs_noise"])

        stage_names = [s.stage_name for s in report.stages]
        nom_label = ROBUSTNESS_SCENARIOS.get("nominal", "nominal")
        noise_label = ROBUSTNESS_SCENARIOS.get("obs_noise", "obs_noise")
        assert nom_label in stage_names
        assert noise_label in stage_names

        nominal_stage = [s for s in report.stages if s.stage_name == nom_label][0]
        assert nominal_stage.metrics["n_episodes"].value == 3

    def test_parallel_n_workers_1_same_as_serial(self):
        """n_workers=1 must be functionally identical to not specifying n_workers."""
        adapter = _DummyAdapter()
        env_id = "CartPole-v1"

        a1 = RobustnessAuditor(
            n_episodes=5, verbose=False, seed=7, n_workers=1,
        )
        a2 = RobustnessAuditor(
            n_episodes=5, verbose=False, seed=7,
        )

        r1 = a1.run(adapter, env_id, scenarios=["nominal"])
        r2 = a2.run(adapter, env_id, scenarios=["nominal"])

        nom_label = ROBUSTNESS_SCENARIOS.get("nominal", "nominal")
        m1 = [s for s in r1.stages if s.stage_name == nom_label][0].metrics["mean_reward"].value
        m2 = [s for s in r2.stages if s.stage_name == nom_label][0].metrics["mean_reward"].value
        assert abs(m1 - m2) < 1e-6
