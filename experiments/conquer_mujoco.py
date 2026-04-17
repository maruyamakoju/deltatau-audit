"""Unified Causal MuJoCo Conquest.

Trains CausalResolutionAgentContinuous on any specified MuJoCo environment.
Usage:
    python experiments/conquer_mujoco.py --env HalfCheetah-v5 --steps 500000
    python experiments/conquer_mujoco.py --env Walker2d-v5 --steps 500000
    python experiments/conquer_mujoco.py --env Ant-v5 --steps 1000000
"""

import argparse
import os
import torch
import numpy as np
import tqdm
import gymnasium as gym
from internal_time_rl.envs.vec_env import SyncVectorEnv
from internal_time_rl.models.causal_reasoning_continuous import CausalResolutionAgentContinuous
from internal_time_rl.algorithms.ppo_resolution import RolloutBuffer
from experiments.train_causal_reasoning import PPOCausal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Conquering {args.env} on {device}...")
    
    # 1. Setup Envs
    def env_factory():
        return gym.make(args.env)
    
    envs = SyncVectorEnv([env_factory for _ in range(args.envs)])
    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]
    
    # 2. Agent & Algorithm
    agent = CausalResolutionAgentContinuous(
        obs_dim, act_dim, hidden_dim=args.hidden, max_ponder_base=4, tau_scale=2.0
    ).to(device)
    
    ppo = PPOCausal(agent, causal_coef=1.0)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    
    num_steps = 2048
    buffer = RolloutBuffer(
        num_steps, args.envs, obs_dim, agent.hidden_dim, device,
        action_dim=act_dim, discrete_actions=False,
    )
    
    # 3. Training Loop
    obs = envs.reset()
    hidden = agent.get_initial_hidden(args.envs, device)
    global_step = 0
    pbar = tqdm.tqdm(total=args.steps, desc=f"[{args.env}]")
    
    while global_step < args.steps:
        for _ in range(num_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
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
            
            global_step += args.envs
            pbar.update(args.envs)

        # Update
        obs_t_last = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, last_value, _, _, _ = agent.forward(obs_t_last, hidden)
        
        buffer.compute_gae(last_value, ppo.gamma, ppo.gae_lambda)
        ppo.update(buffer, optimizer)
        buffer.reset()
        
    pbar.close()
    envs.close()
    
    # Save Final
    os.makedirs("checkpoints", exist_ok=True)
    save_name = f"checkpoints/agent_causal_{args.env.lower().replace('-v5', '')}_final.pt"
    torch.save(agent.state_dict(), save_name)
    print(f"Success! Model saved to {save_name}")

if __name__ == "__main__":
    main()
