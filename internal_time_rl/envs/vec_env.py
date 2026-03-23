import numpy as np
from typing import List, Callable, Tuple, Any, Dict

class SyncVectorEnv:
    """Standardized Synchronous Vectorized Environment.
    
    This implementation follows the Gymnasium VectorEnv API structure but remains
    lightweight for research reproducibility.
    """

    def __init__(self, env_fns: List[Callable]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, seed: int = None) -> np.ndarray:
        obs_list = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, _ = env.reset(seed=env_seed)
            obs_list.append(obs)
        return np.stack(obs_list)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        obs_list, rew_list, done_list = [], [], []
        infos = []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if done:
                # Proper handling of terminal observations for bootstrap
                info["terminal_reward"] = reward
                info["terminal_obs"] = obs.copy()
                obs, _ = env.reset()
            obs_list.append(obs)
            rew_list.append(reward)
            done_list.append(float(done))
            infos.append(info)
        
        return (
            np.stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(done_list, dtype=np.float32),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()
