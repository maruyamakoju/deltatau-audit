"""Observation wrapper that appends explicit timing features."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class TimeFeatureWrapper(gym.ObservationWrapper):
    """Append explicit timing features to observations.

    Added features:
    - dt: inverse of effective speed (1 / speed), inferred from info if present
    - elapsed: normalized elapsed steps since reset
    - phase: normalized phase in [0, 1) over a fixed period
    """

    def __init__(self, env: gym.Env, *, phase_period: int = 200):
        super().__init__(env)
        self.phase_period = max(1, int(phase_period))
        self._step_count = 0
        self._dt_feature = 1.0

        base_space = getattr(self.env, "observation_space", None)
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError(
                "TimeFeatureWrapper requires Box observation_space, "
                f"got: {type(base_space).__name__}"
            )

        low = np.asarray(base_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base_space.high, dtype=np.float32).reshape(-1)
        extra_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        extra_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low, extra_low]).astype(np.float32),
            high=np.concatenate([high, extra_high]).astype(np.float32),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        self._step_count = 0
        self._dt_feature = 1.0
        try:
            root = getattr(self.env, "unwrapped", self.env)
            setattr(root, "actual_speed", 1.0)
            setattr(root, "current_speed", 1.0)
        except Exception:
            pass
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        speed = 1.0
        if isinstance(info, dict):
            if isinstance(info.get("actual_speed"), (int, float)):
                speed = float(info["actual_speed"])
            elif isinstance(info.get("current_speed"), (int, float)):
                speed = float(info["current_speed"])
        if speed <= 1.0:
            root = getattr(self.env, "unwrapped", self.env)
            raw_actual = getattr(root, "actual_speed", None)
            raw_current = getattr(root, "current_speed", None)
            if isinstance(raw_actual, (int, float)):
                speed = float(raw_actual)
            elif isinstance(raw_current, (int, float)):
                speed = float(raw_current)
        speed = max(1.0, speed)

        self._dt_feature = 1.0 / speed
        self._step_count += 1
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        elapsed = min(1.0, float(self._step_count) / float(self.phase_period))
        phase = float(self._step_count % self.phase_period) / float(self.phase_period)
        features = np.array([self._dt_feature, elapsed, phase], dtype=np.float32)
        return np.concatenate([obs, features]).astype(np.float32)
