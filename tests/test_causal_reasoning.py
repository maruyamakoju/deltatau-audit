"""Smoke test for Causal Resolution Agent (Axis 10).

Verifies that the agent can fall back to System 2 (Causal Reasoning)
when subjective uncertainty (delta_tau) is high.
"""

import torch

from internal_time_rl.models.causal_reasoning import CausalResolutionAgent


def test_causal_resolution():
    print("Testing CausalResolutionAgent Flow...")

    B = 2
    obs_dim = 4
    act_dim = 2
    device = torch.device("cpu")

    # Initialize Agent
    agent = CausalResolutionAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        max_ponder_base=4,
        tau_scale=3.0,
        causal_depth=1
    ).to(device)

    obs = torch.randn(B, obs_dim)
    hidden = agent.get_initial_hidden(B, device)

    # 1. System 1 (Fast Thinking) - Low Uncertainty
    print("\n--- Low Uncertainty (System 1) ---")
    dt_low = torch.tensor([[0.5], [0.5]])
    dist1, val1, _, _, diag1 = agent.forward(obs, hidden, dt_override=dt_low)
    print(f"Expected Steps: {diag1['expected_steps'].mean().item():.2f}")
    print(f"Causal Engaged: {diag1.get('causal_engaged', False)}")

    # 2. System 2 (Slow Causal Thinking) - High Uncertainty
    print("\n--- High Uncertainty (System 2) ---")
    dt_high = torch.tensor([[2.5], [2.5]])
    dist2, val2, _, _, diag2 = agent.forward(obs, hidden, dt_override=dt_high)
    print(f"Expected Steps: {diag2['expected_steps'].mean().item():.2f}")
    print(f"Causal Engaged: {diag2.get('causal_engaged', False)}")

    # Assertions
    assert diag2['expected_steps'].mean().item() > diag1['expected_steps'].mean().item()
    assert not diag1.get('causal_engaged', False)
    assert diag2.get('causal_engaged', False)

    print("\nCausal Reasoning Test Passed!")

if __name__ == "__main__":
    test_causal_resolution()
