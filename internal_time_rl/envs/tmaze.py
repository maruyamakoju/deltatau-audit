"""T-Maze Environment: A memory-dependent task with delayed rewards.

The T-maze is a classic test for memory in RL agents:

    [L] --- [T] --- [R]     <- top junction (decision point)
             |
             |
             |
            [S]              <- start (cue given here)

1. At step 0, the agent sees a cue (LEFT or RIGHT) in its observation
2. The agent must walk up the corridor (length steps)
3. At the junction, it must choose LEFT or RIGHT based on the remembered cue
4. Correct choice gives reward +1 (after optional delay), wrong gives -1
5. The cue is only visible at step 0 (POMDP!)

This is an ideal test for internal time because:
- The agent must maintain a memory trace of the cue
- The corridor walk is "boring" (no new info) - internal time should speed up
- At the cue and junction, internal time should slow down (critical moments)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TMazeEnv(gym.Env):
    """T-Maze with corridor, memory cue, and delayed reward."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        corridor_length: int = 10,
        delay: int = 0,
        max_steps: int = 100,
        noise: float = 0.0,
    ):
        super().__init__()
        self.corridor_length = corridor_length
        self.delay = delay
        self.max_steps = max_steps
        self.noise = noise

        # Observation: [position_in_corridor(1), at_junction(1), cue_signal(1), step_frac(1)]
        # cue_signal is only nonzero at step 0
        self.obs_size = 4
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.obs_size,), dtype=np.float32
        )
        # Actions: 0=forward, 1=left, 2=right
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    def _reset_state(self):
        self.position = 0  # 0 to corridor_length (corridor_length = junction)
        self.steps = 0
        self.cue = 0  # 0=left, 1=right
        self.at_junction = False
        self.chose = False
        self.correct = False
        self.reward_countdown = -1

    def _get_obs(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        # Normalized position along corridor
        obs[0] = self.position / self.corridor_length
        # At junction indicator
        obs[1] = 1.0 if self.at_junction else 0.0
        # Cue signal: only at step 0
        if self.steps == 0:
            obs[2] = 1.0 if self.cue == 1 else -1.0
        # Step fraction
        obs[3] = self.steps / self.max_steps

        if self.noise > 0:
            obs += np.random.normal(0, self.noise, size=obs.shape).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        # Random cue
        self.cue = self.np_random.integers(0, 2)
        return self._get_obs(), {"cue": self.cue}

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False

        if not self.chose:
            if not self.at_junction:
                # In the corridor
                if action == 0:  # forward
                    self.position = min(self.position + 1, self.corridor_length)
                # left/right in corridor: no movement (stays in place)

                if self.position >= self.corridor_length:
                    self.at_junction = True
            else:
                # At junction - must choose
                if action == 1:  # left
                    self.chose = True
                    self.correct = (self.cue == 0)
                    if self.delay > 0:
                        self.reward_countdown = self.delay
                    else:
                        reward = 1.0 if self.correct else -1.0
                        terminated = True
                elif action == 2:  # right
                    self.chose = True
                    self.correct = (self.cue == 1)
                    if self.delay > 0:
                        self.reward_countdown = self.delay
                    else:
                        reward = 1.0 if self.correct else -1.0
                        terminated = True
                # action 0 (forward) at junction: no-op

        # Process delayed reward
        if self.reward_countdown > 0:
            self.reward_countdown -= 1
        elif self.reward_countdown == 0:
            reward = 1.0 if self.correct else -1.0
            terminated = True

        # Small step penalty
        reward -= 0.005

        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {
            "position": self.position,
            "at_junction": self.at_junction,
            "chose": self.chose,
            "cue": self.cue,
        }

    def render(self):
        corridor = ["." for _ in range(self.corridor_length)]
        if self.position < self.corridor_length:
            corridor[self.position] = "A"
        junction = "A" if self.at_junction and not self.chose else "T"
        cue_str = "L" if self.cue == 0 else "R"
        return f"[{cue_str}] [{']['.join(corridor)}] [{junction}]"
