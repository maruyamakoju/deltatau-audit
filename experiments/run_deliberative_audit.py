"""
Experiment: Auditing the Deliberative Reasoning Process.

This script audits a 'Deliberative' agent that can take multiple 'thinking steps'
per environment step. We measure:
1. Performance vs Thinking Steps: Does more thinking lead to better results?
2. Thinking Efficiency: How much performance gain per unit of computation (dt)?
3. Temporal Jitter Resilience of the Thinking Process.
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Tuple, Any, Optional

from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
from deltatau_audit.adapters.base import AgentAdapter
from deltatau_audit.auditor import run_full_audit
from deltatau_audit.report import generate_report

class DeliberativeAdapter(AgentAdapter):
    """Adapter for the DeliberativeInternalTimeAgent."""
    def __init__(self, agent: DeliberativeInternalTimeAgent):
        self.agent = agent

    def reset_hidden(self, batch=1, device="cpu"):
        return self.agent.get_initial_hidden(batch, torch.device(device))

    def act(self, obs: torch.Tensor, hidden: torch.Tensor):
        # Ensure batch dim
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        with torch.no_grad():
            action, log_prob, entropy, value, hidden_new, delta_tau = self.agent.get_action_and_value(obs, hidden)
            
        return action[0].item(), value[0].item(), hidden_new, delta_tau[0].item()

def run_deliberative_experiment():
    print("Setting up Deliberative Reasoning Audit...")
    
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Create agent with max 5 thinking steps
    agent = DeliberativeInternalTimeAgent(obs_dim, act_dim, max_thinking_steps=5)
    adapter = DeliberativeAdapter(agent)
    
    print(f"Auditing agent on {env_name}...")
    
    # Standard audit to see how the thinking process handles timing perturbations
    result = run_full_audit(
        adapter,
        lambda: gym.make(env_name),
        speeds=[1, 2, 3],
        n_episodes=5,
        verbose=True
    )
    
    # Custom Thinking Analysis
    print("\n--- Thinking Efficiency Analysis ---")
    nominal_dt = result['robustness']['scenarios']['nominal']['dt_mean']
    nominal_reward = result['robustness']['scenarios']['nominal']['total_reward_mean']
    
    print(f"Avg subjective time spent thinking (dt): {nominal_dt:.4f}")
    print(f"Average Reward: {nominal_reward:.2f}")
    
    # Level Up 4 Impact:
    # If dt > 1.0, the agent is using its internal clock to 'think more' than the standard step.
    efficiency = nominal_reward / nominal_dt if nominal_dt > 0 else 0
    print(f"Thinking Efficiency (Reward/dt): {efficiency:.4f}")

    report_dir = "deliberative_audit_report"
    generate_report(result, report_dir, title="Deliberative Agent Audit")
    print(f"\nAudit complete. Detailed report in {report_dir}/index.html")

if __name__ == "__main__":
    run_deliberative_experiment()
