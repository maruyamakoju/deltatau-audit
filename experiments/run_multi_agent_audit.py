"""
Experiment: Multi-Agent Temporal Desynchronization Audit.

Audits a team of two agents in a cooperative environment.
We measure how the team performance degrades when one agent's 
communication latency increases relative to the other.
"""

import gymnasium as gym
import numpy as np
import torch
from typing import List, Tuple, Dict

from internal_time_rl.models.policy import InternalTimeAgent
from deltatau_audit.adapters.internal_time import InternalTimeAdapter
from deltatau_audit.adapters.multi_agent import MultiAgentAdapter
from deltatau_audit.wrappers.desync import TemporalDesyncWrapper

class MockMultiAgentEnv(gym.Env):
    """Simple 2-agent cooperative env for testing desync."""
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Box(low=-1, high=1, shape=(4,)),
            gym.spaces.Box(low=-1, high=1, shape=(4,))
        ])
        self.action_space = gym.spaces.Tuple([
            gym.spaces.Discrete(2),
            gym.spaces.Discrete(2)
        ])
        self.step_count = 0

    def reset(self, seed=None, options=None):
        return [np.zeros(4), np.zeros(4)], {}

    def step(self, actions):
        self.step_count += 1
        # Success if both agents take the same action
        reward = 1.0 if actions[0] == actions[1] else 0.0
        done = self.step_count >= 20
        return [np.random.randn(4), np.random.randn(4)], [reward, reward], done, False, {}

def run_multi_agent_experiment():
    print("Initializing Multi-Agent Temporal Audit...")
    
    # 1. Create two agents
    agent1 = InternalTimeAgent(4, 2)
    agent2 = InternalTimeAgent(4, 2)
    
    adapter1 = InternalTimeAdapter(agent1)
    adapter2 = InternalTimeAdapter(agent2)
    
    # 2. Team Adapter
    team_adapter = MultiAgentAdapter([adapter1, adapter2])
    
    # 3. Environments with different desync levels
    lags = [0, 2, 5, 10]
    
    print("
--- Desync Performance Audit ---")
    print(f"{'Lag (Agent 2)':<15} | {'Team Reward':<15}")
    print("-" * 35)
    
    for lag in lags:
        env = MockMultiAgentEnv()
        env = TemporalDesyncWrapper(env, agent_speeds=[1, 1], agent_lags=[0, lag])
        
        obs, _ = env.reset()
        hiddens = team_adapter.reset_hidden(batch=1)
        total_reward = 0
        done = False
        
        while not done:
            actions, values, hiddens, dts = team_adapter.act(obs, hiddens)
            obs, rewards, done, trunc, _ = env.step(actions)
            total_reward += rewards[0]
            
        print(f"{lag:<15} | {total_reward:<15.2f}")

if __name__ == "__main__":
    run_multi_agent_experiment()
