"""Latency wrappers: observation delay and action repeat.

Simulates real-world deployment issues like sensor lag,
network latency, and dropped action packets.
"""

from collections import deque

import gymnasium as gym
import numpy as np


class ObservationDelayWrapper(gym.Wrapper):
    """Delays observations by N steps (agent sees stale data).

    Simulates sensor lag, network latency, or processing delay.
    """

    def __init__(self, env: gym.Env, delay: int = 1):
        super().__init__(env)
        self.delay = max(0, int(delay))
        self._obs_buffer = deque(maxlen=self.delay + 1)

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
