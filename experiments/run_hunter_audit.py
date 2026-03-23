"""
Experiment: Red-Teaming with 'The Hunter'.

The Hunter is an AI that learns to exploit temporal vulnerabilities in another AI.
This represents 'Stage 1: Autonomous Temporal Red-Teaming' of the Singularity Phase.
"""

import gymnasium as gym
import torch
import numpy as np
from typing import List, Dict

from internal_time_rl.models.policy import InternalTimeAgent
from deltatau_audit.adapters.internal_time import InternalTimeAdapter
from deltatau_audit.adversarial.hunter import HunterAgent, HunterTrainer
from deltatau_audit.wrappers.speed import FixedSpeedWrapper

def run_hunter_training():
    print("🏹 Initiating Hunter-Target Duel...")
    
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    
    # 1. The Target: A standard RL agent (vulnerable to timing)
    target = InternalTimeAgent(4, 2, use_internal_time=False)
    target_adapter = InternalTimeAdapter(target)
    
    # 2. The Hunter: Learns which speed [1, 2, 3, 5, 8] to apply at each step
    possible_speeds = [1, 2, 3, 5, 8]
    hunter = HunterAgent(target_obs_dim=4, target_hidden_dim=target.hidden_dim, n_speeds=len(possible_speeds))
    trainer = HunterTrainer(hunter)
    
    n_training_episodes = 50
    print(f"Hunter training for {n_training_episodes} episodes...")
    
    for ep in range(n_training_episodes):
        obs, _ = env.reset()
        target_hidden = target_adapter.reset_hidden(1)
        done = False
        
        log_probs = []
        rewards = []
        
        while not done:
            # Hunter observes target and chooses timing attack
            obs_t = torch.tensor(obs, dtype=torch.float32)
            speed_idx, log_prob = hunter.select_attack(obs_t, target_hidden)
            chosen_speed = possible_speeds[speed_idx]
            
            # Apply chosen timing perturbation
            total_reward = 0
            for _ in range(chosen_speed):
                action, _, h_new, _ = target_adapter.act(obs_t, target_hidden)
                obs, r, term, trunc, _ = env.step(action)
                total_reward += r
                if term or trunc:
                    done = True
                    break
                target_hidden = h_new
                obs_t = torch.tensor(obs, dtype=torch.float32)
            
            log_probs.append(log_prob)
            rewards.append(total_reward)
            
        # Update Hunter
        loss = trainer.update(log_probs, rewards)
        
        if (ep + 1) % 10 == 0:
            avg_reward = np.sum(rewards)
            print(f"  Episode {ep+1:3d}: Target Reward={avg_reward:4.1f} (Hunter is learning to minimize this)")

    print("\n🏹 Hunter training complete.")
    print("The Hunter has discovered temporal bottlenecks where the target agent collapses.")

if __name__ == "__main__":
    run_hunter_training()
