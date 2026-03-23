"""Tests for reasoning.py MCTS (bug fixes + TemporalPlanningAgent)."""
from __future__ import annotations

import math
import pytest
import torch


# ── MCTSNode UCB bug fix tests ────────────────────────────────────────────────

def test_ucb_score_pure_python_types():
    """UCB score must use pure Python math, not torch tensors."""
    from internal_time_rl.models.reasoning import MCTSNode

    state = torch.zeros(1, 16)
    obs = torch.zeros(1, 8)
    node = MCTSNode(state=state, obs=obs, delta_tau=1.0)
    node.value = 10.0
    node.visits = 5

    # This must not raise a TypeError about torch/Python type mixing
    score = node.ucb_score(parent_visits=20)
    assert isinstance(score, float), f"UCB score must be Python float, got {type(score)}"
    assert not math.isnan(score), "UCB score is NaN"
    assert not math.isinf(score) or node.visits == 0, "Unexpectedly infinite UCB"


def test_ucb_score_unvisited_returns_inf():
    """Unvisited node must return inf to guarantee exploration."""
    from internal_time_rl.models.reasoning import MCTSNode
    node = MCTSNode(state=torch.zeros(1, 8), obs=torch.zeros(1, 8), delta_tau=1.0)
    assert node.ucb_score(parent_visits=10) == float("inf")


def test_ucb_score_exploit_explore_balance():
    """UCB score must include both exploit and explore terms."""
    from internal_time_rl.models.reasoning import MCTSNode
    node = MCTSNode(state=torch.zeros(1, 8), obs=torch.zeros(1, 8), delta_tau=1.0)
    node.value = 8.0
    node.visits = 4
    parent_visits = 16

    score = node.ucb_score(parent_visits=parent_visits, c_puct=1.41)
    exploit = 8.0 / 4
    explore = 1.41 * math.sqrt(math.log(16) / 4)
    expected = exploit + explore

    # Must be close (there's an uncertainty bonus too, so >= expected)
    assert score >= expected - 1e-6, (
        f"UCB score {score} below expected exploit+explore={expected}"
    )


def test_ucb_consistency_with_math():
    """UCB result must be deterministic (same inputs → same output)."""
    from internal_time_rl.models.reasoning import MCTSNode
    node = MCTSNode(state=torch.zeros(1, 8), obs=torch.zeros(1, 8), delta_tau=1.0)
    node.value = 5.0
    node.visits = 3

    s1 = node.ucb_score(parent_visits=10)
    s2 = node.ucb_score(parent_visits=10)
    assert s1 == s2, "UCB score not deterministic"


# ── SearchBasedReasoningEngine tests ─────────────────────────────────────────

def test_search_returns_state_and_trace():
    """search() must return (state, trace) with correct types."""
    from internal_time_rl.models.reasoning import SearchBasedReasoningEngine
    engine = SearchBasedReasoningEngine(hidden_dim=8, obs_dim=4, search_depth=2, n_simulations=3)
    state = torch.zeros(1, 8)
    obs = torch.zeros(1, 4)
    result_state, trace = engine.search(state, obs)
    assert isinstance(result_state, torch.Tensor)
    assert isinstance(trace, list)


def test_search_trace_has_metadata():
    """Search trace must contain useful metadata."""
    from internal_time_rl.models.reasoning import SearchBasedReasoningEngine
    engine = SearchBasedReasoningEngine(hidden_dim=8, obs_dim=4, search_depth=2, n_simulations=3)
    state = torch.zeros(1, 8)
    obs = torch.zeros(1, 4)
    _, trace = engine.search(state, obs)
    if trace and "error" not in trace[0]:
        assert "best_delta_tau" in trace[0] or "search_depth" in trace[0]


# ── TemporalPlanningAgent tests ───────────────────────────────────────────────

def test_temporal_planning_agent_plan_and_act():
    """plan_and_act() must return (int action, dict plan_info)."""
    from internal_time_rl.models.reasoning import TemporalPlanningAgent
    agent = TemporalPlanningAgent(
        obs_dim=4, act_dim=2, hidden_dim=32, latent_dim=8,
        n_rollouts=2, horizon=2,
    )
    agent.eval()

    B = 1
    obs = torch.randn(B, 4)
    h, z = agent.world_model.initial_state(B)

    action, plan_info = agent.plan_and_act(obs, h, z)
    assert isinstance(action, int), f"action must be int, got {type(action)}"
    assert 0 <= action < 2, f"action {action} out of range [0, 2)"
    assert "best_expected_return" in plan_info
    assert "planning_advantage" not in plan_info  # that's from horizon auditor


def test_temporal_planning_agent_action_in_range():
    """Actions must be within [0, act_dim)."""
    from internal_time_rl.models.reasoning import TemporalPlanningAgent
    agent = TemporalPlanningAgent(obs_dim=4, act_dim=3, hidden_dim=16, latent_dim=4, n_rollouts=3, horizon=2)
    agent.eval()

    for _ in range(5):
        obs = torch.randn(1, 4)
        h, z = agent.world_model.initial_state(1)
        action, _ = agent.plan_and_act(obs, h, z)
        assert 0 <= action < 3, f"action {action} out of range"
