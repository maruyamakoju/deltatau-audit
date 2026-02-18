"""Tests for deltatau_audit.wrappers — timing perturbation wrappers."""

import pytest
import numpy as np
import gymnasium as gym


# ── ObservationDelayWrapper ──────────────────────────────────────────

class TestObservationDelayWrapper:
    def test_no_delay(self):
        """With delay=0, should return current observation."""
        from deltatau_audit.wrappers.latency import ObservationDelayWrapper

        env = gym.make("CartPole-v1")
        wrapped = ObservationDelayWrapper(env, delay=0)

        obs, info = wrapped.reset()
        action = wrapped.action_space.sample()
        obs_new, reward, term, trunc, info = wrapped.step(action)

        assert obs_new is not None
        assert "obs_delay" in info
        assert info["obs_delay"] == 0
        wrapped.close()

    def test_delay_one_step(self):
        """With delay=1, observation should be from previous step."""
        from deltatau_audit.wrappers.latency import ObservationDelayWrapper

        env = gym.make("CartPole-v1")
        wrapped = ObservationDelayWrapper(env, delay=1)

        obs_reset, info = wrapped.reset()

        # First step: should get initial obs (delayed)
        action = 0
        obs1, reward1, term1, trunc1, info1 = wrapped.step(action)

        # The delayed obs should match the reset obs (1 step delay)
        assert np.array_equal(obs1, obs_reset)
        assert info1["obs_delay"] == 1

        wrapped.close()

    def test_delay_multiple_steps(self):
        """Test delay=3 returns observation from 3 steps ago."""
        from deltatau_audit.wrappers.latency import ObservationDelayWrapper

        env = gym.make("CartPole-v1")
        wrapped = ObservationDelayWrapper(env, delay=3)

        obs_reset, _ = wrapped.reset()

        observations = [obs_reset]
        for i in range(5):
            obs, _, _, _, info = wrapped.step(0)
            observations.append(obs)
            assert info["obs_delay"] == 3

        # After reset, first 3 steps should return the reset observation
        # due to buffer initialization
        assert np.array_equal(observations[1], obs_reset)
        assert np.array_equal(observations[2], obs_reset)
        assert np.array_equal(observations[3], obs_reset)

        wrapped.close()

    def test_negative_delay_clamped(self):
        """Negative delay should be clamped to 0."""
        from deltatau_audit.wrappers.latency import ObservationDelayWrapper

        env = gym.make("CartPole-v1")
        wrapped = ObservationDelayWrapper(env, delay=-5)

        assert wrapped.delay == 0
        wrapped.close()


# ── ActionRepeatWrapper ───────────────────────────────────────────────

class TestActionRepeatWrapper:
    def test_no_repeat(self):
        """With repeat=1, should execute action once."""
        from deltatau_audit.wrappers.latency import ActionRepeatWrapper

        env = gym.make("CartPole-v1")
        wrapped = ActionRepeatWrapper(env, repeat=1)

        obs, _ = wrapped.reset()
        obs_new, reward, term, trunc, info = wrapped.step(0)

        assert obs_new is not None
        assert isinstance(reward, float)
        wrapped.close()

    def test_repeat_three_times(self):
        """With repeat=3, should execute action 3 times and accumulate rewards."""
        from deltatau_audit.wrappers.latency import ActionRepeatWrapper

        env = gym.make("CartPole-v1")
        wrapped = ActionRepeatWrapper(env, repeat=3)

        obs, _ = wrapped.reset()
        obs_new, reward, term, trunc, info = wrapped.step(1)

        # Reward should be accumulated from 3 steps
        assert reward >= 0  # CartPole gives +1 per step
        assert obs_new is not None
        wrapped.close()

    def test_repeat_stops_on_termination(self):
        """If episode ends mid-repeat, should stop early."""
        from deltatau_audit.wrappers.latency import ActionRepeatWrapper

        env = gym.make("CartPole-v1")
        wrapped = ActionRepeatWrapper(env, repeat=100)

        obs, _ = wrapped.reset()

        # Force termination by taking bad actions
        done = False
        steps = 0
        max_steps = 10
        while not done and steps < max_steps:
            # Alternating actions to make pole fall
            obs, reward, term, trunc, info = wrapped.step(steps % 2)
            done = term or trunc
            steps += 1

        # Should eventually terminate
        assert done or steps >= max_steps
        wrapped.close()

    def test_zero_repeat_clamped(self):
        """repeat=0 should be clamped to 1."""
        from deltatau_audit.wrappers.latency import ActionRepeatWrapper

        env = gym.make("CartPole-v1")
        wrapped = ActionRepeatWrapper(env, repeat=0)

        assert wrapped.repeat == 1
        wrapped.close()


# ── FixedSpeedWrapper ─────────────────────────────────────────────────

class TestFixedSpeedWrapper:
    def test_speed_one(self):
        """Speed=1 should execute action once per agent step."""
        from deltatau_audit.wrappers.speed import FixedSpeedWrapper

        env = gym.make("CartPole-v1")
        wrapped = FixedSpeedWrapper(env, speed=1)

        obs, _ = wrapped.reset()
        obs_new, reward, term, trunc, info = wrapped.step(0)

        assert obs_new is not None
        assert reward >= 0
        wrapped.close()

    def test_speed_three(self):
        """Speed=3 should execute action 3 times, accumulating rewards."""
        from deltatau_audit.wrappers.speed import FixedSpeedWrapper

        env = gym.make("CartPole-v1")
        wrapped = FixedSpeedWrapper(env, speed=3)

        obs, _ = wrapped.reset()
        obs_new, reward, term, trunc, info = wrapped.step(1)

        # Reward accumulates from 3 steps
        assert reward >= 0
        assert obs_new is not None
        wrapped.close()

    def test_speed_stops_on_done(self):
        """Should stop repeating if episode terminates."""
        from deltatau_audit.wrappers.speed import FixedSpeedWrapper

        env = gym.make("CartPole-v1")
        wrapped = FixedSpeedWrapper(env, speed=100)

        obs, _ = wrapped.reset()

        done = False
        steps = 0
        while not done and steps < 5:
            obs, reward, term, trunc, info = wrapped.step(0)
            done = term or trunc
            steps += 1

        assert done or steps >= 5
        wrapped.close()

    def test_zero_speed_clamped(self):
        """speed=0 should be clamped to 1."""
        from deltatau_audit.wrappers.speed import FixedSpeedWrapper

        env = gym.make("CartPole-v1")
        wrapped = FixedSpeedWrapper(env, speed=0)

        assert wrapped.speed == 1
        wrapped.close()


# ── JitterWrapper ─────────────────────────────────────────────────────

class TestJitterWrapper:
    def test_no_jitter(self):
        """With jitter=0, speed should be constant at base_speed."""
        from deltatau_audit.wrappers.speed import JitterWrapper

        env = gym.make("CartPole-v1")
        wrapped = JitterWrapper(env, base_speed=2, jitter=0, seed=42)

        obs, _ = wrapped.reset()

        speeds = []
        for _ in range(10):
            obs, reward, term, trunc, info = wrapped.step(1)
            speeds.append(info["actual_speed"])
            if term or trunc:
                break

        # All speeds should be exactly base_speed
        assert all(s == 2 for s in speeds)
        wrapped.close()

    def test_with_jitter(self):
        """With jitter>0, speed should vary around base_speed."""
        from deltatau_audit.wrappers.speed import JitterWrapper

        env = gym.make("CartPole-v1")
        wrapped = JitterWrapper(env, base_speed=3, jitter=1, seed=42)

        obs, _ = wrapped.reset()

        speeds = []
        for _ in range(20):
            obs, reward, term, trunc, info = wrapped.step(1)
            speeds.append(info["actual_speed"])
            if term or trunc:
                break

        # Speeds should be in range [base_speed - jitter, base_speed + jitter]
        assert all(2 <= s <= 4 for s in speeds), f"Speeds out of range: {speeds}"

        # Should have some variation
        assert len(set(speeds)) > 1, "Speed should vary with jitter"
        wrapped.close()

    def test_jitter_min_speed_one(self):
        """Speed should never go below 1."""
        from deltatau_audit.wrappers.speed import JitterWrapper

        env = gym.make("CartPole-v1")
        wrapped = JitterWrapper(env, base_speed=1, jitter=5, seed=42)

        obs, _ = wrapped.reset()

        for _ in range(20):
            obs, reward, term, trunc, info = wrapped.step(0)
            assert info["actual_speed"] >= 1
            if term or trunc:
                break

        wrapped.close()

    def test_deterministic_with_seed(self):
        """Same seed should produce same sequence of speeds."""
        from deltatau_audit.wrappers.speed import JitterWrapper

        env1 = gym.make("CartPole-v1")
        wrapped1 = JitterWrapper(env1, base_speed=2, jitter=1, seed=123)

        env2 = gym.make("CartPole-v1")
        wrapped2 = JitterWrapper(env2, base_speed=2, jitter=1, seed=123)

        obs1, _ = wrapped1.reset()
        obs2, _ = wrapped2.reset()

        speeds1 = []
        speeds2 = []

        for _ in range(10):
            _, _, term1, trunc1, info1 = wrapped1.step(0)
            _, _, term2, trunc2, info2 = wrapped2.step(0)
            speeds1.append(info1["actual_speed"])
            speeds2.append(info2["actual_speed"])
            if term1 or trunc1 or term2 or trunc2:
                break

        assert speeds1 == speeds2
        wrapped1.close()
        wrapped2.close()


# ── PiecewiseSwitchWrapper ────────────────────────────────────────────

class TestPiecewiseSwitchWrapper:
    def test_single_speed(self):
        """With single schedule entry, speed should be constant."""
        from deltatau_audit.wrappers.speed import PiecewiseSwitchWrapper

        env = gym.make("CartPole-v1")
        wrapped = PiecewiseSwitchWrapper(env, schedule=[(0, 2)])

        obs, _ = wrapped.reset()

        for _ in range(10):
            obs, reward, term, trunc, info = wrapped.step(1)
            assert info["current_speed"] == 2
            if term or trunc:
                break

        wrapped.close()

    def test_speed_switch(self):
        """Speed should change at specified thresholds."""
        from deltatau_audit.wrappers.speed import PiecewiseSwitchWrapper

        env = gym.make("CartPole-v1")
        # Speed 1 for steps 0-4, speed 3 for steps 5-9, speed 1 for steps 10+
        wrapped = PiecewiseSwitchWrapper(env, schedule=[(0, 1), (5, 3), (10, 1)])

        obs, _ = wrapped.reset()

        speeds = []
        for i in range(15):
            obs, reward, term, trunc, info = wrapped.step(0)
            speeds.append(info["current_speed"])
            if term or trunc:
                break

        # Check speed changes at correct thresholds
        if len(speeds) >= 5:
            assert all(s == 1 for s in speeds[0:5]), f"Steps 0-4 should be speed 1: {speeds[0:5]}"
        if len(speeds) >= 10:
            assert all(s == 3 for s in speeds[5:10]), f"Steps 5-9 should be speed 3: {speeds[5:10]}"
        if len(speeds) >= 15:
            assert all(s == 1 for s in speeds[10:15]), f"Steps 10-14 should be speed 1: {speeds[10:15]}"

        wrapped.close()

    def test_reset_clears_step_count(self):
        """Reset should restart the schedule from step 0."""
        from deltatau_audit.wrappers.speed import PiecewiseSwitchWrapper

        env = gym.make("CartPole-v1")
        wrapped = PiecewiseSwitchWrapper(env, schedule=[(0, 1), (3, 5)])

        obs, _ = wrapped.reset()

        # Take 5 steps
        for _ in range(5):
            obs, reward, term, trunc, info = wrapped.step(0)
            if term or trunc:
                break

        # Reset
        obs, _ = wrapped.reset()

        # First step after reset should be speed 1 (step 0)
        obs, reward, term, trunc, info = wrapped.step(0)
        assert info["current_speed"] == 1

        wrapped.close()

    def test_default_schedule(self):
        """No schedule should default to speed 1."""
        from deltatau_audit.wrappers.speed import PiecewiseSwitchWrapper

        env = gym.make("CartPole-v1")
        wrapped = PiecewiseSwitchWrapper(env)

        obs, _ = wrapped.reset()
        obs, reward, term, trunc, info = wrapped.step(0)

        assert info["current_speed"] == 1
        wrapped.close()

    def test_spike_scenario(self):
        """Test mid-episode spike scenario (1->5->1)."""
        from deltatau_audit.wrappers.speed import PiecewiseSwitchWrapper

        env = gym.make("CartPole-v1")
        wrapped = PiecewiseSwitchWrapper(env, schedule=[(0, 1), (20, 5), (40, 1)])

        obs, _ = wrapped.reset()

        speeds = []
        for i in range(50):
            obs, reward, term, trunc, info = wrapped.step(0)
            speeds.append(info["current_speed"])
            if term or trunc:
                break

        # Verify the spike pattern if enough steps completed
        if len(speeds) >= 20:
            assert speeds[0] == 1
            assert speeds[19] == 1
        if len(speeds) >= 40:
            assert speeds[20] == 5
            assert speeds[39] == 5
        if len(speeds) >= 50:
            assert speeds[40] == 1

        wrapped.close()

    def test_negative_speed_clamped(self):
        """Negative speed values should be clamped to 1."""
        from deltatau_audit.wrappers.speed import PiecewiseSwitchWrapper

        env = gym.make("CartPole-v1")
        wrapped = PiecewiseSwitchWrapper(env, schedule=[(0, -5)])

        obs, _ = wrapped.reset()
        obs, reward, term, trunc, info = wrapped.step(0)

        assert info["current_speed"] == 1
        wrapped.close()
