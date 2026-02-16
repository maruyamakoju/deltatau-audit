"""Environment wrappers for testing temporal adaptability.

FlickeringWrapper: Randomly masks observations, creating a POMDP.
    Tests whether internal time helps maintain state under missing info.

VariableSpeedWrapper: Changes effective environment speed over time.
    Tests whether the agent adapts its internal clock to match env dynamics.
"""

import numpy as np
import gymnasium as gym


class FlickeringWrapper(gym.ObservationWrapper):
    """Randomly replaces observations with zeros.

    With probability flicker_prob, the agent receives a blank observation,
    forcing it to rely on its recurrent memory. The internal time module
    should learn to slow down when observations are missing.
    """

    def __init__(self, env: gym.Env, flicker_prob: float = 0.3):
        super().__init__(env)
        self.flicker_prob = flicker_prob
        self._flickered = False

    def observation(self, obs):
        self._flickered = np.random.random() < self.flicker_prob
        if self._flickered:
            return np.zeros_like(obs)
        return obs


class VariableSpeedWrapper(gym.Wrapper):
    """Changes the effective environment speed over time.

    Periodically changes how many environment steps are executed per
    agent decision step. When speed > 1, multiple env steps are taken
    for each agent action (fast environment). When speed = 1, normal.

    The agent's internal time should adapt: when the environment moves
    fast, internal time should speed up to process more change per step.
    """

    def __init__(
        self,
        env: gym.Env,
        min_repeat: int = 1,
        max_repeat: int = 4,
        change_interval: int = 50,
    ):
        super().__init__(env)
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat
        self.change_interval = change_interval
        self.current_repeat = 1
        self.step_count = 0

    def reset(self, **kwargs):
        self.current_repeat = 1
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.step_count += 1
        if self.step_count % self.change_interval == 0:
            self.current_repeat = np.random.randint(
                self.min_repeat, self.max_repeat + 1
            )

        total_reward = 0.0
        for _ in range(self.current_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        info["env_speed"] = self.current_repeat
        return obs, total_reward, terminated, truncated, info
