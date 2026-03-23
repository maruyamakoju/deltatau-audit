"""Tests for timing environment wrappers.

Covers: FixedSpeedWrapper, JitterWrapper, PiecewiseSwitchWrapper,
        ObservationDelayWrapper, ObsNoiseWrapper, ActionRepeatWrapper,
        TimeFeatureWrapper
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────

def _cartpole():
    """Create a fresh CartPole-v1 environment."""
    return gym.make("CartPole-v1")


# ── FixedSpeedWrapper ─────────────────────────────────────────────────────────

class TestFixedSpeedWrapper:
    def test_import(self):
        from deltatau_audit.wrappers.speed import FixedSpeedWrapper
        assert FixedSpeedWrapper is not None

    def test_wraps_env(self):
        from deltatau_audit.wrappers.speed import FixedSpeedWrapper
        env = FixedSpeedWrapper(_cartpole(), speed=2)
        assert env is not None
        obs, _ = env.reset(seed=42)
        assert obs is not None
        env.close()

    def test_action_space_unchanged(self):
        from deltatau_audit.wrappers.speed import FixedSpeedWrapper
        base = _cartpole()
        wrapped = FixedSpeedWrapper(base, speed=3)
        assert wrapped.action_space == base.action_space
        wrapped.close()

    def test_observation_space_unchanged(self):
        from deltatau_audit.wrappers.speed import FixedSpeedWrapper
        base = _cartpole()
        wrapped = FixedSpeedWrapper(base, speed=2)
        assert wrapped.observation_space == base.observation_space
        wrapped.close()

    def test_speed_1_same_as_base(self):
        """Speed=1 wrapper should give same step count behavior as base."""
        from deltatau_audit.wrappers.speed import FixedSpeedWrapper
        env = FixedSpeedWrapper(_cartpole(), speed=1)
        obs, _ = env.reset(seed=0)
        # Should be able to step
        obs2, rew, term, trunc, info = env.step(0)
        assert obs2 is not None
        env.close()

    def test_speed_2_accumulates_reward(self):
        """Speed=2 should accumulate reward over 2 base steps."""
        from deltatau_audit.wrappers.speed import FixedSpeedWrapper
        env = FixedSpeedWrapper(_cartpole(), speed=2)
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(0)
        # Reward should be >= 1.0 (at least one step survived at r=1.0 each)
        assert reward >= 1.0
        env.close()


# ── JitterWrapper ──────────────────────────────────────────────────────────────

class TestJitterWrapper:
    def test_import(self):
        from deltatau_audit.wrappers.speed import JitterWrapper
        assert JitterWrapper is not None

    def test_wraps_env(self):
        from deltatau_audit.wrappers.speed import JitterWrapper
        env = JitterWrapper(_cartpole(), base_speed=1, jitter=1)
        obs, _ = env.reset(seed=42)
        assert obs is not None
        env.close()

    def test_observation_space_unchanged(self):
        from deltatau_audit.wrappers.speed import JitterWrapper
        base = _cartpole()
        wrapped = JitterWrapper(base, base_speed=2, jitter=1)
        assert wrapped.observation_space == base.observation_space
        wrapped.close()

    def test_zero_jitter_is_deterministic(self):
        """JitterWrapper with jitter=0 should give identical results each reset."""
        from deltatau_audit.wrappers.speed import JitterWrapper
        env = JitterWrapper(_cartpole(), base_speed=1, jitter=0)
        env.reset(seed=0)
        obs1, r1, _, _, _ = env.step(0)
        env.reset(seed=0)
        obs2, r2, _, _, _ = env.step(0)
        np.testing.assert_array_almost_equal(obs1, obs2)
        env.close()


# ── ObsNoiseWrapper ────────────────────────────────────────────────────────────

class TestObsNoiseWrapper:
    def test_import(self):
        from deltatau_audit.wrappers.latency import ObsNoiseWrapper
        assert ObsNoiseWrapper is not None

    def test_wraps_env(self):
        from deltatau_audit.wrappers.latency import ObsNoiseWrapper
        env = ObsNoiseWrapper(_cartpole(), std=0.1)
        obs, _ = env.reset(seed=42)
        assert obs is not None
        env.close()

    def test_observation_space_unchanged(self):
        from deltatau_audit.wrappers.latency import ObsNoiseWrapper
        base = _cartpole()
        wrapped = ObsNoiseWrapper(base, std=0.1)
        assert wrapped.observation_space == base.observation_space
        wrapped.close()

    def test_zero_std_returns_original(self):
        """With std=0, observations should be identical to base env."""
        from deltatau_audit.wrappers.latency import ObsNoiseWrapper
        # Create two envs with same seed
        base = _cartpole()
        noisy = ObsNoiseWrapper(_cartpole(), std=0.0)

        obs_base, _ = base.reset(seed=42)
        obs_noisy, _ = noisy.reset(seed=42)
        np.testing.assert_array_almost_equal(obs_base, obs_noisy, decimal=5)

        base.close()
        noisy.close()

    def test_nonzero_std_adds_noise(self):
        """With std=0.5, observations should differ from the base after a step."""
        from deltatau_audit.wrappers.latency import ObsNoiseWrapper
        env = ObsNoiseWrapper(_cartpole(), std=0.5, seed=0)
        base = _cartpole()

        # ObsNoiseWrapper returns clean obs on reset; noise is applied on step.
        env.reset(seed=42)
        base.reset(seed=42)

        obs_noisy, _, _, _, _ = env.step(0)
        obs_base, _, _, _, _ = base.step(0)

        # They won't be exactly equal (noise added)
        assert not np.allclose(obs_noisy, obs_base, atol=1e-6), \
            "Expected noise to change at least one observation element"

        env.close()
        base.close()

    def test_seeded_noise_is_reproducible(self):
        """Same seed should give same noisy observations."""
        from deltatau_audit.wrappers.latency import ObsNoiseWrapper
        env1 = ObsNoiseWrapper(_cartpole(), std=0.1, seed=99)
        env2 = ObsNoiseWrapper(_cartpole(), std=0.1, seed=99)

        env1.reset(seed=42)
        env2.reset(seed=42)

        # Take a few steps
        for _ in range(5):
            a = env1.action_space.sample()
            o1, _, _, _, _ = env1.step(a)
            o2, _, _, _, _ = env2.step(a)

        np.testing.assert_array_almost_equal(o1, o2, decimal=5)
        env1.close()
        env2.close()


# ── ObservationDelayWrapper ────────────────────────────────────────────────────

class TestObservationDelayWrapper:
    def test_import(self):
        from deltatau_audit.wrappers.latency import ObservationDelayWrapper
        assert ObservationDelayWrapper is not None

    def test_wraps_env(self):
        from deltatau_audit.wrappers.latency import ObservationDelayWrapper
        env = ObservationDelayWrapper(_cartpole(), delay=1)
        obs, _ = env.reset(seed=42)
        assert obs is not None
        env.close()

    def test_observation_space_unchanged(self):
        from deltatau_audit.wrappers.latency import ObservationDelayWrapper
        base = _cartpole()
        wrapped = ObservationDelayWrapper(base, delay=2)
        assert wrapped.observation_space == base.observation_space
        wrapped.close()

    def test_zero_delay_same_as_base(self):
        """delay=0 should pass observations through unchanged."""
        from deltatau_audit.wrappers.latency import ObservationDelayWrapper
        env = ObservationDelayWrapper(_cartpole(), delay=0)
        base = _cartpole()

        obs_w, _ = env.reset(seed=42)
        obs_b, _ = base.reset(seed=42)

        np.testing.assert_array_almost_equal(obs_w, obs_b)
        env.close()
        base.close()

    def test_delay_returns_stale_obs(self):
        """delay=1 should return the previous step's observation.

        On reset, the buffer is filled with (delay+1) copies of reset_obs.
        After the first step, the buffer becomes [reset_obs, step1_obs] and
        buffer[0] (the reset obs) is returned as the delayed observation.
        """
        from deltatau_audit.wrappers.latency import ObservationDelayWrapper
        env_delayed = ObservationDelayWrapper(_cartpole(), delay=1)
        env_base = _cartpole()

        obs_delayed_reset, _ = env_delayed.reset(seed=42)
        obs_base_reset, _ = env_base.reset(seed=42)

        # First step with delayed env should return the reset observation (stale by 1)
        obs_after_step_delayed, _, _, _, _ = env_delayed.step(0)
        # The delayed wrapper should return the reset obs as the "current" obs
        np.testing.assert_array_almost_equal(obs_after_step_delayed, obs_base_reset)

        env_delayed.close()
        env_base.close()


# ── TimeFeatureWrapper ────────────────────────────────────────────────────────

class TestTimeFeatureWrapper:
    def test_import(self):
        from deltatau_audit.wrappers.time_feature import TimeFeatureWrapper
        assert TimeFeatureWrapper is not None

    def test_observation_space_expands_by_3(self):
        from deltatau_audit.wrappers.time_feature import TimeFeatureWrapper
        base = _cartpole()
        wrapped = TimeFeatureWrapper(base)
        assert wrapped.observation_space.shape[0] == base.observation_space.shape[0] + 3
        wrapped.close()

    def test_features_change_after_step(self):
        from deltatau_audit.wrappers.time_feature import TimeFeatureWrapper
        env = TimeFeatureWrapper(_cartpole(), phase_period=20)
        obs0, _ = env.reset(seed=42)
        obs1, _, _, _, _ = env.step(0)

        # last 3 dims are [dt, elapsed, phase]
        assert obs0.shape[0] == obs1.shape[0]
        assert np.isclose(obs0[-3], 1.0)
        assert obs1[-2] > obs0[-2]
        assert obs1[-1] > obs0[-1]
        env.close()

    def test_dt_tracks_outer_fixed_speed_wrapper(self):
        from deltatau_audit.wrappers.speed import FixedSpeedWrapper
        from deltatau_audit.wrappers.time_feature import TimeFeatureWrapper

        env = FixedSpeedWrapper(TimeFeatureWrapper(_cartpole(), phase_period=20), speed=5)
        env.reset(seed=42)
        obs, _, _, _, info = env.step(0)

        assert np.isclose(obs[-3], 0.2, atol=1e-6)
        assert np.isclose(float(info.get("current_speed", 0.0)), 5.0)
        env.close()
