"""Speed/timing wrappers for injecting temporal variations into any Gym env.

These wrappers simulate what happens in real deployments when the
control frequency varies: frame drops, variable inference latency,
sensor rate changes, etc.
"""

from typing import Any, List, Optional

import gymnasium as gym
import numpy as np


class FixedSpeedWrapper(gym.Wrapper):
    """Run the underlying env at a fixed speed multiplier.

    speed=1: normal (1 env step per agent step)
    speed=3: 3 env steps per agent step (fast environment)

    The agent sees every `speed`-th observation but rewards accumulate.
    This simulates frame-skipping / action repeat.
    """

    def __init__(self, env: gym.Env, speed: int = 1):
        super().__init__(env)
        self.speed = max(1, int(speed))

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.speed):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class JitterWrapper(gym.Wrapper):
    """Add random timing jitter: each agent step takes speed ± jitter env steps.

    Simulates variable inference latency or sensor rate fluctuations.
    """

    def __init__(self, env: gym.Env, base_speed: int = 1,
                 jitter: int = 1, seed: Optional[int] = None):
        super().__init__(env)
        self.base_speed = max(1, int(base_speed))
        self.jitter = max(0, int(jitter))
        self.rng = np.random.RandomState(seed)

    def step(self, action):
        actual_speed = max(1, self.base_speed + self.rng.randint(
            -self.jitter, self.jitter + 1))

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(actual_speed):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        info["actual_speed"] = actual_speed
        return obs, total_reward, terminated, truncated, info


class PiecewiseSwitchWrapper(gym.Wrapper):
    """Speed changes at predetermined step boundaries.

    Simulates scenarios like: normal operation → sudden load spike → recovery.
    Schedule is a list of (step_threshold, speed) tuples.
    """

    def __init__(self, env: gym.Env,
                 schedule: Optional[List[Any]] = None):
        """
        Args:
            schedule: List of (step, speed) tuples. Speed changes when
                      agent_step >= step. E.g. [(0, 1), (20, 5), (40, 1)]
                      means speed=1 for steps 0-19, speed=5 for 20-39, etc.
        """
        super().__init__(env)
        self.schedule = schedule or [(0, 1)]
        self.agent_step = 0

    def reset(self, **kwargs):
        self.agent_step = 0
        return self.env.reset(**kwargs)

    def _current_speed(self) -> int:
        speed = self.schedule[0][1]
        for threshold, s in self.schedule:
            if self.agent_step >= threshold:
                speed = s
        return max(1, int(speed))

    def step(self, action):
        speed = self._current_speed()
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(speed):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        info["current_speed"] = speed
        self.agent_step += 1
        return obs, total_reward, terminated, truncated, info
