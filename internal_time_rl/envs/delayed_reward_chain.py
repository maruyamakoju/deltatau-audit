"""Delayed Reward Chain Environment.

A corridor where the agent starts at position 0 and must reach position N-1.
Upon reaching the goal, the reward is delayed by D additional steps.

This environment specifically tests the hypothesis that internal time
helps agents handle long-term temporal dependencies.

Layout:  [0] - [1] - [2] - ... - [N-1=GOAL]
Actions: 0=left, 1=right
Reward:  +1.0 delivered D steps after reaching goal, -0.01 per step
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DelayedRewardChainEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        length: int = 20,
        delay: int = 10,
        max_steps: int = 200,
        noise: float = 0.0,
    ):
        super().__init__()
        self.length = length
        self.delay = delay
        self.max_steps = max_steps
        self.noise = noise

        # Observation: one-hot position + normalized step counter + reward countdown indicator
        self.obs_size = length + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(self.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.position = 0
        self.steps = 0
        self.reward_countdown = -1
        self.reached_goal = False

    def _get_obs(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        obs[self.position] = 1.0
        # Normalized step counter
        obs[self.length] = self.steps / self.max_steps
        # Reward countdown indicator (1 if reward is pending)
        obs[self.length + 1] = 1.0 if self.reward_countdown >= 0 else 0.0
        if self.noise > 0:
            obs += np.random.normal(0, self.noise, size=obs.shape).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        self.steps = 0
        self.reward_countdown = -1
        self.reached_goal = False
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        # Move
        if action == 0:
            self.position = max(0, self.position - 1)
        elif action == 1:
            self.position = min(self.length - 1, self.position + 1)

        reward = 0.0
        terminated = False
        truncated = False

        # Check if reached goal for the first time
        if self.position == self.length - 1 and not self.reached_goal:
            self.reached_goal = True
            self.reward_countdown = self.delay

        # Process delayed reward countdown
        if self.reward_countdown > 0:
            self.reward_countdown -= 1
        elif self.reward_countdown == 0:
            reward = 1.0
            terminated = True

        # Step penalty
        reward -= 0.01

        # Truncation
        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {
            "position": self.position,
            "reward_countdown": self.reward_countdown,
        }

    def render(self):
        line = ["."] * self.length
        if self.position < self.length:
            line[self.position] = "A"
        line[-1] = "G" if self.position != self.length - 1 else "A"
        status = f" step={self.steps}"
        if self.reward_countdown >= 0:
            status += f" countdown={self.reward_countdown}"
        return "[" + "][".join(line) + "]" + status
