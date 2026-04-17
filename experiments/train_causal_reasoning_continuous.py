"""Axis 10 (Continuous): Causal Reasoning Training for MuJoCo.

Trains the CausalResolutionAgentContinuous on HalfCheetah-v5.
Jointly optimizes:
1. Continuous Action Policy (PPO)
2. Causal Latent Transition Model (World Model)
3. Dynamic Resolution (Subjective Uncertainty)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import gymnasium as gym
from typing import List, Dict

from internal_time_rl.envs.vec_env import SyncVectorEnv
from internal_time_rl.models.causal_reasoning_continuous import CausalResolutionAgentContinuous
from internal_time_rl.algorithms.ppo_resolution import RolloutBuffer
from experiments.train_causal_reasoning import PPOCausal

def train_causal_mujoco():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Axis 10 Causal MuJoCo Training on {device}...")
    
    # 1. Environment (HalfCheetah-v5)
    def env_factory():
        return gym.make("HalfCheetah-v5")
    
    num_envs = 4
    envs = SyncVectorEnv([env_factory for _ in range(num_envs)])
    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]
    
    # 2. Agent & Optimizer
    # Note: Use larger hidden dim for MuJoCo
    agent = CausalResolutionAgentContinuous(
        obs_dim, act_dim, hidden_dim=256, max_ponder_base=4, tau_scale=2.0
    ).to(device)
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    ppo = PPOCausal(agent, causal_coef=1.0)
    
    num_steps = 2048 # Larger steps for MuJoCo PPO
    buffer = RolloutBuffer(
        num_steps, num_envs, obs_dim, agent.hidden_dim, device,
        action_dim=act_dim, discrete_actions=False,
    )
    
    total_timesteps = 500_000
    global_step = 0
    obs = envs.reset()
    hidden = agent.get_initial_hidden(num_envs, device)
    
    pbar = tqdm.tqdm(total=total_timesteps, desc="[MuJoCo Causal Training]")
    
    while global_step < total_timesteps:
        # Rollout
        for _ in range(num_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                # Continuous agent returns Normal distribution
                action_dist, log_prob, _, value, hidden_new, dt, diag = agent.get_action_and_value(obs_t, hidden)
                action = action_dist # Already sampled in get_action_and_value? 
                # Let's check get_action_and_value in base class
            
            # Note: get_action_and_value in base class needs to be compatible with Continuous
            # Re-running logic here for safety
            with torch.no_grad():
                dist, value, hidden_new, dt, diag = agent.forward(obs_t, hidden)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            
            next_obs, reward, done, _ = envs.step(action.cpu().numpy())
            
            buffer.add(obs_t, action, torch.as_tensor(reward, device=device), 
                       torch.as_tensor(done, device=device), log_prob, value, 
                       hidden, dt)
            
            obs = next_obs
            hidden = hidden_new
            for i, d in enumerate(done):
                if d > 0.5: hidden[i] = 0.0
            
            global_step += num_envs
            pbar.update(num_envs)

        # Update
        obs_t_last = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, last_value, _, _, _ = agent.forward(obs_t_last, hidden)
        
        buffer.compute_gae(last_value, ppo.gamma, ppo.gae_lambda)
        ppo.update(buffer, optimizer)
        buffer.reset()
        
        # Periodic Save
        if global_step % 100_000 < num_envs:
            torch.save(agent.state_dict(), f"checkpoints/agent_causal_mujoco_{global_step}.pt")
        
    pbar.close()
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(agent.state_dict(), "checkpoints/agent_causal_mujoco_final.pt")
    print("Training Complete. Model saved to checkpoints/agent_causal_mujoco_final.pt")

if __name__ == "__main__":
    train_causal_mujoco()
