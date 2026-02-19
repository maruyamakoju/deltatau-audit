"""Latency and noise wrappers.

Simulates real-world deployment issues like sensor lag,
network latency, dropped action packets, and noisy sensors.
"""

from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np


class ObservationDelayWrapper(gym.Wrapper):
    """Delays observations by N steps (agent sees stale data).

    Simulates sensor lag, network latency, or processing delay.
    """

    def __init__(self, env: gym.Env, delay: int = 1):
        super().__init__(env)
        self.delay = max(0, int(delay))
        self._obs_buffer: deque = deque(maxlen=self.delay + 1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.clear()
        for _ in range(self.delay + 1):
            self._obs_buffer.append(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs_buffer.append(obs)
        delayed_obs = self._obs_buffer[0]
        info["obs_delay"] = self.delay
        return delayed_obs, reward, terminated, truncated, info


class ObsNoiseWrapper(gym.Wrapper):
    """Adds i.i.d. Gaussian noise to every observation (simulates noisy sensors).

    Models sensor imperfections, quantization noise, or partial observability.
    The noise is re-sampled at every step; the initial reset observation is
    returned clean so the agent starts from a known state.

    Args:
        env:  The base gymnasium env.
        std:  Standard deviation of the noise (in observation units).
              Default 0.1 — roughly 10% of a unit-normalized observation.
        seed: Optional integer seed for a thread-local RNG.
    """

    def __init__(self, env: gym.Env, std: float = 0.1,
                 seed: Optional[int] = None):
        super().__init__(env)
        self.std = float(std)
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info  # Clean observation on reset

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        noise = self._rng.normal(0.0, self.std, size=obs.shape).astype(obs.dtype)
        info["obs_noise_std"] = self.std
        return obs + noise, reward, terminated, truncated, info


class ActionRepeatWrapper(gym.Wrapper):
    """Repeats each action for N steps (simulates dropped action packets).

    With repeat=3, the agent's action is executed 3 times before the
    next observation is returned. Rewards accumulate.
    """

    def __init__(self, env: gym.Env, repeat: int = 1):
        super().__init__(env)
        self.repeat = max(1, int(repeat))

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
