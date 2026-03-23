"""
Action Chunking: Predicting a sequence of future actions.
Standard in DeepMind Robotics (ACT, ALOHA).

This module provides wrappers and agent modifications to handle action chunks
and temporal ensembling.
"""

import gymnasium as gym
import numpy as np
from typing import List, Optional

class ActionChunkWrapper(gym.Wrapper):
    """
    Wraps an environment to accept a 'chunk' of actions.
    The environment executes the chunk sequentially.
    """
    def __init__(self, env: gym.Env, chunk_size: int = 5):
        super().__init__(env)
        self.chunk_size = chunk_size

    def step(self, action_chunk: np.ndarray):
        """
        action_chunk: (chunk_size, action_dim)
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for i in range(self.chunk_size):
            # If action_chunk is flat but should be (L, D)
            if action_chunk.ndim == 1 and self.env.action_space.shape:
                 # fallback/error handling
                 a = action_chunk
            elif action_chunk.ndim == 2:
                 a = action_chunk[i]
            else:
                 a = action_chunk
                 
            obs, reward, term, trunc, info = self.env.step(a)
            total_reward += reward
            if term or trunc:
                terminated = term
                truncated = trunc
                break
        
        return obs, total_reward, terminated, truncated, info

class TemporalEnsembler:
    """
    Handles temporal ensembling of overlapping action chunks.
    Used to smooth trajectories in high-latency deployments.
    """
    def __init__(self, chunk_size: int, action_dim: int):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        # Buffer to store overlapping predictions
        self.buffer = [] # List of (start_step, chunk)

    def add_chunk(self, current_step: int, chunk: np.ndarray):
        self.buffer.append((current_step, chunk))
        # Cleanup old chunks
        self.buffer = [b for b in self.buffer if b[0] + self.chunk_size > current_step]

    def get_ensembled_action(self, current_step: int):
        if not self.buffer:
            return None
            
        actions = []
        weights = []
        
        for start_step, chunk in self.buffer:
            idx = current_step - start_step
            if 0 <= idx < self.chunk_size:
                actions.append(chunk[idx])
                # Exponential weight for more recent chunks
                weights.append(np.exp(-0.1 * (current_step - start_step)))
        
        if not actions:
            return None
            
        weights = np.array(weights)
        weights /= weights.sum()
        
        ensembled = np.sum([a * w for a, w in zip(actions, weights)], axis=0)
        return ensembled
