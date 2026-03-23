"""
Experiment: Demonstrating Temporal Interpretability.

Analyzes an agent's internal time (delta_tau) during a CartPole episode
and generates human-readable insights about its decision logic.
"""

import gymnasium as gym
import torch
import numpy as np
from internal_time_rl.models.policy import InternalTimeAgent
from internal_time_rl.analysis.interpretability import TemporalInterpreter

def run_interpretability_demo():
    print("Initializing Temporal Interpretability Engine...")
    
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    feature_names = ["Cart Pos", "Cart Vel", "Pole Angle", "Pole Ang Vel"]
    
    obs_dim = 4
    act_dim = 2
    
    # 1. Setup Time-Aware Agent
    agent = InternalTimeAgent(obs_dim, act_dim)
    interpreter = TemporalInterpreter(feature_names)
    
    print("Collecting episode data...")
    obs, _ = env.reset()
    hidden = agent.get_initial_hidden(1, torch.device("cpu"))
    
    all_obs = []
    all_dts = []
    done = False
    
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, _, h_new, dt = agent.forward(obs_t, hidden)
            action = dist.sample()
            
        all_obs.append(obs)
        all_dts.append(dt.item())
        
        obs, _, term, trunc, _ = env.step(action.item())
        done = term or trunc
        hidden = h_new
        
    # 2. Run Analysis
    print("\n--- Temporal Insight Analysis ---")
    analysis = interpreter.analyze_episode(np.array(all_obs), np.array(all_dts))
    
    print(f"Agent Logic: {analysis['summary']}")
    
    print("\nFeature-Time Correlations:")
    for feat, data in analysis['feature_correlations'].items():
        print(f"  {feat:15s}: corr={data['correlation']:+.2f} (p={data['p_value']:.4f})")
        
    if analysis['events']:
        print("\nSignificant Temporal Events:")
        for e in analysis['events'][:3]: # Show first 3
            print(f"  Step {e['step']:3d}: {e['type'].upper()} (driven by '{e['trigger_features']}')")

if __name__ == "__main__":
    run_interpretability_demo()
