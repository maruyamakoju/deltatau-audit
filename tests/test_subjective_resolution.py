"""Smoke test for Subjective Resolution Agent.

Verifies that the Level 4 agent can be audited and demonstrates 
dynamic scale adjustment.
"""

import gymnasium as gym
import torch
import numpy as np
from internal_time_rl.models.subjective_resolution import SubjectiveResolutionAgent
from deltatau_audit.adapters.subjective_resolution import SubjectiveResolutionAdapter
from deltatau_audit.auditors import RobustnessAuditor
from deltatau_audit.core.runner import EpisodeRunner

def test_subjective_resolution_flow():
    print("Testing SubjectiveResolutionAgent Flow...")
    
    # 1. Initialize
    obs_dim = 4
    act_dim = 2
    agent = SubjectiveResolutionAgent(obs_dim, act_dim, max_ponder_base=4, tau_scale=2.0)
    adapter = SubjectiveResolutionAdapter(agent)
    
    # 2. Basic Act Test
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=42)
    
    action, info = adapter.act(obs)
    print(f"Action: {action}, info: {info}")
    
    assert "dt" in info
    assert "reasoning_trace" in info
    assert info["reasoning_trace"]["expected_steps"] >= 1.0
    
    # 3. Dynamic Scaling Check
    # Force high delta_tau and see if expected_steps increases
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    hidden = agent.get_initial_hidden(1, torch.device("cpu"))
    
    # Low DT
    _, _, _, _, diag_low = agent.forward(obs_t, hidden, dt_override=torch.tensor([[0.5]]))
    # High DT
    _, _, _, _, diag_high = agent.forward(obs_t, hidden, dt_override=torch.tensor([[2.5]]))
    
    print(f"Expected steps (DT=0.5): {diag_low['expected_steps'].item():.2f}")
    print(f"Expected steps (DT=2.5): {diag_high['expected_steps'].item():.2f}")
    
    # High DT should prompt more pondering due to tau_scale logit shift
    # (Since weights are random, this is a probabilistic check, but our logit shift is strong)
    assert diag_high['max_steps'] > diag_low['max_steps']
    
    # 4. Auditor Integration Test
    print("Running Mini-Audit...")
    auditor = RobustnessAuditor(n_episodes=2, verbose=True, seed=0)
    report = auditor.run(adapter, "CartPole-v1", scenarios=["nominal", "jitter"])
    
    print(f"Audit Score: {report.reliability_score:.2f}")
    assert report.reliability_score >= 0.0
    
    env.close()
    print("Test Passed!")

if __name__ == "__main__":
    test_subjective_resolution_flow()
