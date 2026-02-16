"""Variable Frequency Environments for testing temporal adaptability.

Core idea: the environment's "speed" (action repeat / control frequency)
changes between episodes (or within episodes). The agent must adapt its
internal clock to maintain performance.

Key experimental protocol:
  Train on speeds {1, 2, 3}
  Test on speeds  {1, 2, 3, 5, 8}  (includes unseen speeds)
  → Measure generalization gap

This is the strongest test for "Adaptive Temporal Reparameterization":
internal time Δτ should track environment speed.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class VariableFrequencyChainEnv(gym.Env):
    """Delayed reward chain with variable action repeat.

    Each episode, a speed (action repeat) is sampled. The agent's action
    is repeated `speed` times in the base environment. This means:
    - speed=1: standard (19 steps to reach goal in chain-20)
    - speed=2: action repeated twice (~10 steps)
    - speed=5: action repeated 5x (~4 steps)

    The agent can observe the current speed (optionally), testing whether
    internal time helps even when speed info is available.

    speed_schedule options:
    - "constant" (default): speed stays fixed for the whole episode
    - "switch": speed switches mid-episode (e.g., 1→8 at switch_step)
      Used to demonstrate dynamic Δτ adaptation within a single episode.

    Observation: [position_one_hot, step_frac, speed_normalized, reward_pending]
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        chain_length: int = 20,
        delay: int = 10,
        max_agent_steps: int = 100,
        train_speeds: tuple = (1, 2, 3),
        speed_in_obs: bool = True,
        noise: float = 0.0,
        fixed_speed: int = None,
        speed_schedule: str = "constant",
        switch_speeds: tuple = (1, 8),
        switch_step: int = None,
    ):
        super().__init__()
        self.chain_length = chain_length
        self.delay = delay
        self.max_agent_steps = max_agent_steps
        self.train_speeds = list(train_speeds)
        self.speed_in_obs = speed_in_obs
        self.noise = noise
        self.fixed_speed = fixed_speed
        self.speed_schedule = speed_schedule
        self.switch_speeds = switch_speeds
        # Default switch at 40% of episode (before goal typically)
        self.switch_step = switch_step if switch_step is not None else max_agent_steps * 2 // 5

        # Observation size: chain_length (position) + step_frac + speed + reward_pending
        self.obs_size = chain_length + 3
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(self.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # left, right

        self._reset_state()

    def _reset_state(self):
        self.position = 0
        self.agent_steps = 0
        self.env_steps = 0
        self.current_speed = 1
        self.reward_countdown = -1
        self.reached_goal = False

    def _get_obs(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        obs[self.position] = 1.0
        obs[self.chain_length] = self.agent_steps / self.max_agent_steps
        if self.speed_in_obs:
            obs[self.chain_length + 1] = self.current_speed / 5.0  # normalized
        obs[self.chain_length + 2] = 1.0 if self.reward_countdown >= 0 else 0.0
        if self.noise > 0:
            obs += np.random.normal(0, self.noise, size=obs.shape).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        if self.speed_schedule == "switch":
            # Start with first switch speed; will change at switch_step
            self.current_speed = self.switch_speeds[0]
        elif self.fixed_speed is not None:
            self.current_speed = self.fixed_speed
        else:
            self.current_speed = self.np_random.choice(self.train_speeds)

        return self._get_obs(), {"speed": self.current_speed}

    def _update_speed_schedule(self):
        """Update speed based on schedule and current step."""
        if self.speed_schedule == "switch":
            if self.agent_steps == self.switch_step:
                self.current_speed = self.switch_speeds[1]

    def step(self, action):
        self.agent_steps += 1
        reward = 0.0
        terminated = False
        truncated = False

        # Update speed schedule (may change speed mid-episode)
        self._update_speed_schedule()

        # Execute action `speed` times
        for _repeat in range(self.current_speed):
            if action == 0:
                self.position = max(0, self.position - 1)
            elif action == 1:
                self.position = min(self.chain_length - 1, self.position + 1)
            self.env_steps += 1

            # Check goal
            if self.position == self.chain_length - 1 and not self.reached_goal:
                self.reached_goal = True
                self.reward_countdown = self.delay

            # Process delayed reward
            if self.reward_countdown > 0:
                self.reward_countdown -= 1
            elif self.reward_countdown == 0:
                reward = 1.0
                terminated = True
                break

        # Step penalty (per agent step, not env step)
        reward -= 0.01

        if self.agent_steps >= self.max_agent_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {
            "speed": self.current_speed,
            "position": self.position,
            "env_steps": self.env_steps,
            "agent_steps": self.agent_steps,
        }


class IntervalTimingEnv(gym.Env):
    """Interval timing: wait T steps then act.

    At episode start, a target time T is presented as a cue.
    The agent must wait approximately T agent-steps, then press "go".
    Reward is based on accuracy: |press_time - T|.

    With variable speed, the relationship between agent-steps and
    "real time" changes. The agent must learn temporal abstraction,
    not just step counting.

    Observation: [target_time(normalized), steps_elapsed(normalized), speed(normalized)]
    Actions: 0=wait, 1=press
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        min_target: int = 5,
        max_target: int = 30,
        max_steps: int = 50,
        tolerance: int = 2,
        train_speeds: tuple = (1, 2, 3),
        speed_in_obs: bool = True,
        fixed_speed: int = None,
    ):
        super().__init__()
        self.min_target = min_target
        self.max_target = max_target
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.train_speeds = list(train_speeds)
        self.speed_in_obs = speed_in_obs
        self.fixed_speed = fixed_speed

        # Obs: target_time_norm, steps_elapsed_norm, speed_norm
        self.obs_size = 3
        self.observation_space = spaces.Box(
            low=0.0, high=2.0, shape=(self.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # wait, press

        self._reset_state()

    def _reset_state(self):
        self.target_time = 10
        self.current_step = 0
        self.current_speed = 1
        self.pressed = False
        self.elapsed_real_time = 0.0  # accumulated "real time"

    def _get_obs(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        obs[0] = self.target_time / self.max_target  # normalized target
        # Give raw step count (NOT real time) — agent must internally
        # account for speed to know when real_time ≈ target
        obs[1] = self.current_step / self.max_steps  # normalized step count
        if self.speed_in_obs:
            obs[2] = self.current_speed / 5.0
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        # Sample target time and speed
        self.target_time = self.np_random.integers(self.min_target, self.max_target + 1)
        if self.fixed_speed is not None:
            self.current_speed = self.fixed_speed
        else:
            self.current_speed = int(self.np_random.choice(self.train_speeds))

        return self._get_obs(), {
            "target_time": self.target_time,
            "speed": self.current_speed,
        }

    def step(self, action):
        self.current_step += 1
        # Each agent step advances "real time" by `speed` units
        self.elapsed_real_time += self.current_speed

        reward = 0.0
        terminated = False
        truncated = False

        if action == 1 and not self.pressed:  # press
            self.pressed = True
            # How close was the press to the target?
            error = abs(self.elapsed_real_time - self.target_time)
            if error <= self.tolerance:
                reward = 1.0  # perfect timing
            elif error <= self.tolerance * 3:
                reward = 0.5 * (1.0 - error / (self.tolerance * 3))  # partial credit
            else:
                reward = -0.5  # too far off
            terminated = True

        # Small wait penalty (encourages pressing eventually)
        reward -= 0.005

        if self.current_step >= self.max_steps:
            if not self.pressed:
                reward = -1.0  # never pressed
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {
            "target_time": self.target_time,
            "elapsed_real_time": self.elapsed_real_time,
            "speed": self.current_speed,
            "pressed": self.pressed,
        }
