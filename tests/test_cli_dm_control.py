"""Tests for dm_control handling helpers in CLI."""

from __future__ import annotations

from argparse import Namespace

import gymnasium as gym
import numpy as np

from deltatau_audit.cli import _is_dm_control_env_id, _wrap_external_eval_env


class _DictObsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "a": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "b": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {"a": np.zeros(2, dtype=np.float32), "b": np.zeros(1, dtype=np.float32)}, {}

    def step(self, action):
        obs = {"a": np.zeros(2, dtype=np.float32), "b": np.zeros(1, dtype=np.float32)}
        return obs, 0.0, False, False, {}


def test_is_dm_control_env_id_recognizes_aliases():
    assert _is_dm_control_env_id("dm_control/walker-walk-v0")
    assert _is_dm_control_env_id("dm_control/reacher-easy")
    assert not _is_dm_control_env_id("CartPole-v1")


def test_wrap_external_eval_env_flattens_dict_observation():
    env = _DictObsEnv()
    args = Namespace(
        env_wrap_time_feature=False,
        env_wrap_phase_period=200,
        env_wrap_frame_stack=0,
        env_wrap_flatten_obs=False,
    )

    wrapped = _wrap_external_eval_env(env, args)
    assert isinstance(wrapped.observation_space, gym.spaces.Box)
