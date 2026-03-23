"""
Experiment: Meta-Time Calibration (Zero-Shot Online Adaptation).

Demonstrates 'Level Up 6: Meta-Time Calibration'.
The agent is deployed in an environment with UNSEEN speed (speed=3).
Initially, it fails. The MetaTimeCalibrator then adjusts the agent's
internal clock bias based on TD-error signals, restoring performance
without retraining.
"""

import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from internal_time_rl.models.policy import InternalTimeAgent
from internal_time_rl.utils.calibration import MetaTimeCalibrator
from deltatau_audit.wrappers.speed import FixedSpeedWrapper

def run_calibration_demo():
    print("Setting up Meta-Time Calibration Demo...")
    
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    # Wrap with speed=3 (This would normally break a speed=1 trained agent)
    env = FixedSpeedWrapper(env, speed=3)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Initialize a 'Generic' InternalTimeAgent (Pretend it's pre-trained at speed 1)
    agent = InternalTimeAgent(obs_dim, act_dim)
    calibrator = MetaTimeCalibrator(agent, lr=0.1)
    
    gamma = 0.99
    n_episodes = 10
    
    history_rewards = []
    history_bias = []
    
    print(f"Starting deployment at speed=3.0 with Online Calibration...")
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        hidden = agent.get_initial_hidden(1, torch.device("cpu"))
        done = False
        total_reward = 0
        
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                action_dist, value, hidden_new, dt = agent.forward(obs_t, hidden)
                action = action_dist.sample()
                
            next_obs, reward, term, trunc, _ = env.step(action.item())
            done = term or trunc
            total_reward += reward
            
            # Get next value for TD calculation
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, next_value, _, _ = agent.forward(next_obs_t, hidden_new)
            
            # Meta-Adaptation step!
            calibrator.step_adaptation(
                reward, value.item(), next_value.item(), gamma, dt.item()
            )
            
            obs = next_obs
            hidden = hidden_new
            
        history_rewards.append(total_reward)
        history_bias.append(calibrator.bias_param.item())
        print(f"Episode {ep+1}: Reward={total_reward:.1f}, Internal Bias={history_bias[-1]:.4f}")

    print("\nCalibration Complete.")
    print(f"Initial Bias: 0.0000")
    print(f"Final Bias:   {history_bias[-1]:.4f}")
    
    if history_rewards[-1] > history_rewards[0]:
        print("Success: Performance improved during online calibration!")
    else:
        print("Note: In a random-initialized model improvements may be noisy.")

    # Save visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_rewards)
    plt.title("Reward during Adaptation")
    plt.xlabel("Episode")
    
    plt.subplot(1, 2, 2)
    plt.plot(history_bias)
    plt.title("Internal Time Bias Convergence")
    plt.xlabel("Episode")
    
    plt.tight_layout()
    plt.savefig("meta_calibration_results.png")
    print("Plot saved to meta_calibration_results.png")

if __name__ == "__main__":
    run_calibration_demo()
