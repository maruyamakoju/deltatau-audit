"""Tests for the Proper ACT deliberative agent (PHASE 3)."""
from __future__ import annotations

import torch
import pytest


# ── DeliberativeInternalTimeAgent tests ──────────────────────────────────────

def test_deliberative_agent_forward_returns_five_values():
    """forward() must return (dist, value, hidden, cumulative_halt, ponder_cost)."""
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    agent = DeliberativeInternalTimeAgent(obs_dim=4, act_dim=2, hidden_dim=16, latent_dim=8, max_thinking_steps=3)
    obs = torch.randn(2, 4)
    hidden = torch.zeros(2, 16)
    result = agent.forward(obs, hidden)
    assert len(result) == 5, f"Expected 5 return values, got {len(result)}"
    dist, value, hidden_new, cumulative_halt, ponder_cost = result
    assert value.shape == (2,)
    assert hidden_new.shape == (2, 16)
    assert cumulative_halt.shape == (2, 1)
    assert ponder_cost.shape == (2, 1)


def test_deliberative_ponder_cost_bounded():
    """Ponder cost must be <= max_thinking_steps."""
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    max_steps = 5
    agent = DeliberativeInternalTimeAgent(obs_dim=4, act_dim=2, hidden_dim=16, latent_dim=8,
                                          max_thinking_steps=max_steps)
    obs = torch.randn(4, 4)
    hidden = torch.zeros(4, 16)
    _, _, _, _, ponder_cost = agent.forward(obs, hidden)
    assert (ponder_cost <= max_steps + 1e-6).all(), (
        f"Ponder cost {ponder_cost} exceeds max_thinking_steps={max_steps}"
    )


def test_deliberative_ponder_cost_at_least_one():
    """Agent must think at least once (ponder_cost >= 1)."""
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    agent = DeliberativeInternalTimeAgent(obs_dim=4, act_dim=2, hidden_dim=16, latent_dim=8,
                                          max_thinking_steps=3)
    obs = torch.randn(2, 4)
    hidden = torch.zeros(2, 16)
    _, _, _, _, ponder_cost = agent.forward(obs, hidden)
    assert (ponder_cost >= 1.0 - 1e-6).all(), (
        f"Agent must think at least 1 step, got ponder_cost={ponder_cost}"
    )


def test_deliberative_cumulative_halt_in_unit_interval():
    """cumulative_halt must be in [0, 1]."""
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    agent = DeliberativeInternalTimeAgent(obs_dim=4, act_dim=2, hidden_dim=16, latent_dim=8)
    obs = torch.randn(3, 4)
    hidden = torch.zeros(3, 16)
    _, _, _, cumulative_halt, _ = agent.forward(obs, hidden)
    assert (cumulative_halt >= -1e-6).all(), "cumulative_halt < 0"
    assert (cumulative_halt <= 1.0 + 1e-6).all(), "cumulative_halt > 1"


def test_deliberative_weighted_hidden_differs_from_initial():
    """Weighted hidden state should differ from initial zeros."""
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    agent = DeliberativeInternalTimeAgent(obs_dim=4, act_dim=2, hidden_dim=16, latent_dim=8)
    obs = torch.randn(1, 4)
    hidden = torch.zeros(1, 16)
    _, _, hidden_new, _, _ = agent.forward(obs, hidden)
    assert not torch.allclose(hidden_new, hidden), (
        "weighted_hidden should differ from initial zero hidden state"
    )


def test_deliberative_gradients_flow():
    """Gradients must flow through ACT weighted sum (no hard threshold breaks)."""
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    agent = DeliberativeInternalTimeAgent(obs_dim=4, act_dim=2, hidden_dim=16, latent_dim=8)
    obs = torch.randn(2, 4)
    hidden = torch.zeros(2, 16)
    dist, value, _, _, ponder_cost = agent.forward(obs, hidden)
    loss = value.mean() + 0.01 * ponder_cost.mean()
    loss.backward()
    # Check that halting_net got gradients
    for name, param in agent.halting_net.named_parameters():
        if param.grad is not None:
            return  # at least one param has gradient
    pytest.fail("No gradients in halting_net — ACT is not differentiable")


def test_ponder_loss_positive():
    """compute_ponder_loss() must return a positive scalar."""
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    ponder = torch.tensor([[2.5], [3.0]])
    loss = DeliberativeInternalTimeAgent.compute_ponder_loss(ponder, lambda_p=0.01)
    assert float(loss.item()) > 0.0
    assert loss.shape == ()  # scalar


def test_ponder_loss_scales_with_lambda():
    """compute_ponder_loss should scale linearly with lambda_p."""
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    ponder = torch.tensor([[3.0]])
    loss_01 = float(DeliberativeInternalTimeAgent.compute_ponder_loss(ponder, lambda_p=0.01).item())
    loss_10 = float(DeliberativeInternalTimeAgent.compute_ponder_loss(ponder, lambda_p=0.10).item())
    assert abs(loss_10 / loss_01 - 10.0) < 1e-4


def test_temporal_uncertainty_estimator_output_shape():
    """TemporalUncertaintyEstimator must return correct output shapes."""
    from internal_time_rl.models.deliberative import TemporalUncertaintyEstimator
    est = TemporalUncertaintyEstimator(hidden_dim=16, latent_dim=8)
    hidden = torch.zeros(2, 16)
    encoded = torch.randn(2, 8)
    result = est.estimate_timing_uncertainty(encoded, hidden, n_samples=5)
    assert "mean_value" in result
    assert "std_value" in result
    assert "recommended_ponder_steps" in result
    assert result["mean_value"].shape == (2, 1)
    assert result["std_value"].shape == (2, 1)
    assert isinstance(result["recommended_ponder_steps"], int)
    assert result["recommended_ponder_steps"] >= 1


# ── DeliberativeAgentAdapter tests ───────────────────────────────────────────

def test_deliberative_adapter_act_returns_four_values():
    """DeliberativeAgentAdapter.act() must return (action, value, hidden, dt)."""
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    from deltatau_audit.adapters.deliberative_adapter import DeliberativeAgentAdapter
    import gymnasium as gym

    agent = DeliberativeInternalTimeAgent(obs_dim=4, act_dim=2, hidden_dim=16, latent_dim=8)
    adapter = DeliberativeAgentAdapter(agent)

    obs = torch.randn(4)
    hidden = adapter.reset_hidden(1)
    action, value, hidden_new, dt = adapter.act(obs, hidden)

    assert isinstance(action, int)
    assert isinstance(value, float)
    assert hidden_new is not None
    assert dt is not None and dt >= 0


def test_deliberative_adapter_deliberation_stats():
    """get_deliberation_stats() must return ponder tracking metrics."""
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    from deltatau_audit.adapters.deliberative_adapter import DeliberativeAgentAdapter

    agent = DeliberativeInternalTimeAgent(obs_dim=4, act_dim=2, hidden_dim=16, latent_dim=8,
                                          max_thinking_steps=4)
    adapter = DeliberativeAgentAdapter(agent)

    # Run a few steps
    for _ in range(5):
        obs = torch.randn(4)
        hidden = adapter.reset_hidden(1)
        adapter.act(obs, hidden)

    stats = adapter.get_deliberation_stats()
    assert "mean_ponder_steps" in stats
    assert "ponder_utilization" in stats
    assert "halt_efficiency" in stats
    assert 0.0 <= stats["ponder_utilization"] <= 1.0 + 1e-6


def test_deliberative_adapter_reset_episode():
    """reset_episode() must clear ponder tracking."""
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    from deltatau_audit.adapters.deliberative_adapter import DeliberativeAgentAdapter

    agent = DeliberativeInternalTimeAgent(obs_dim=4, act_dim=2, hidden_dim=16, latent_dim=8)
    adapter = DeliberativeAgentAdapter(agent)

    obs = torch.randn(4)
    hidden = adapter.reset_hidden(1)
    adapter.act(obs, hidden)
    assert len(adapter._episode_ponder_steps) > 0

    adapter.reset_episode()
    assert len(adapter._episode_ponder_steps) == 0
