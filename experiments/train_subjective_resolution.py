"""Debug-Enhanced Comparative Study: Fixed vs Subjective Resolution.

Includes:
1. Try-except blocks for error logging.
2. NaN checks in training loop.
3. Periodic progress logging to file.
"""

import os
import sys
import torch
import numpy as np
import tqdm
import traceback
import matplotlib.pyplot as plt
from typing import List, Dict

from internal_time_rl.envs.variable_frequency import VariableFrequencyChainEnv
from internal_time_rl.envs.vec_env import SyncVectorEnv
from internal_time_rl.models.subjective_resolution import SubjectiveResolutionAgent
from internal_time_rl.algorithms.ppo_resolution import PPOResolution, RolloutBuffer

def log_error(msg):
    with open("error_log.txt", "a") as f:
        f.write(msg + "\n")
    print(msg)

def train_agent(
    agent_type: str, 
    env_factory, 
    total_timesteps: int = 50_000, # Reduced for debug
    num_envs: int = 4, 
    num_steps: int = 128,
    device: str = "auto"
):
    try:
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
            
        log_error(f"\nTraining {agent_type} on {device}...")
        
        envs = SyncVectorEnv([env_factory for _ in range(num_envs)])
        obs_dim = envs.observation_space.shape[0]
        act_dim = 2
        
        if agent_type == "fixed":
            agent = SubjectiveResolutionAgent(obs_dim, act_dim, max_ponder_base=3, tau_scale=0.0).to(device)
            ponder_override = 3
        else:
            agent = SubjectiveResolutionAgent(obs_dim, act_dim, max_ponder_base=4, tau_scale=2.0).to(device)
            ponder_override = None

        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
        ppo = PPOResolution(agent, ponder_coef=0.005 if agent_type == "dynamic" else 0.0)
        buffer = RolloutBuffer(num_steps, num_envs, obs_dim, agent.hidden_dim, device)
        
        obs = envs.reset()
        hidden = agent.get_initial_hidden(num_envs, device)
        
        pbar = tqdm.tqdm(total=total_timesteps, desc=f"[{agent_type}]")
        global_step = 0
        
        while global_step < total_timesteps:
            for _ in range(num_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    action, log_prob, _, value, hidden_new, dt, diag = agent.get_action_and_value(
                        obs_t, hidden, ponder_override=ponder_override
                    )
                
                # Check for NaNs in action/value
                if torch.isnan(value).any() or torch.isnan(dt).any():
                    raise ValueError(f"NaN detected in agent output at step {global_step}")

                next_obs, reward, done, infos = envs.step(action.cpu().numpy())
                
                buffer.add(
                    obs_t, action, torch.as_tensor(reward, device=device), 
                    torch.as_tensor(done, device=device), log_prob, value, 
                    hidden, dt
                )
                
                obs = next_obs
                hidden = hidden_new
                for i, d in enumerate(done):
                    if d > 0.5:
                        hidden[i] = 0.0
                
                global_step += num_envs
                pbar.update(num_envs)

            # Update PPO
            obs_t_last = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                _, _, _, last_value, _, _, _ = agent.get_action_and_value(obs_t_last, hidden, ponder_override=ponder_override)
                
            buffer.compute_gae(last_value, ppo.gamma, ppo.gae_lambda)
            avg_loss = ppo.update(buffer, optimizer)
            
            if np.isnan(avg_loss):
                raise ValueError(f"NaN detected in PPO loss at step {global_step}")
                
            buffer.reset()
            
        pbar.close()
        envs.close()
        return agent
    except Exception as e:
        log_error(f"Error in training {agent_type}: {str(e)}")
        log_error(traceback.format_exc())
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    
    def env_factory():
        return VariableFrequencyChainEnv(
            chain_length=20, train_speeds=(1, 2, 3), speed_in_obs=False
        )
    
    # Run Fixed
    fixed_agent = train_agent("fixed", env_factory, total_timesteps=50_000, device=device)
    if fixed_agent:
        torch.save(fixed_agent.state_dict(), "checkpoints/agent_fixed_debug.pt")
    
    # Run Dynamic
    dyn_agent = train_agent("dynamic", env_factory, total_timesteps=50_000, device=device)
    if dyn_agent:
        torch.save(dyn_agent.state_dict(), "checkpoints/agent_dynamic_debug.pt")

    if fixed_agent and dyn_agent:
        print("\nDebug training complete. Both agents trained successfully.")
    else:
        print("\nDebug training failed for one or more agents. Check error_log.txt")

if __name__ == "__main__":
    main()
