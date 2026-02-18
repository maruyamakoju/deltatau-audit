"""Tests for CleanRLAdapter."""

import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ── Minimal CleanRL Agent fixtures ─────────────────────────────────────────────

class MinimalAgent(nn.Module):
    """Minimal MLP CleanRL agent."""

    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.critic = nn.Linear(obs_dim, 1)

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)


class MinimalLSTMAgent(nn.Module):
    """Minimal LSTM CleanRL agent."""

    def __init__(self, obs_dim=4, act_dim=2, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=False)
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def get_action_and_value(self, obs, lstm_state, done):
        # obs: (seq, batch, obs_dim) or (batch, obs_dim)
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        h, c = lstm_state
        x, (h_new, c_new) = self.lstm(obs, (h, c))
        x = x.squeeze(0)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return (action, probs.log_prob(action), probs.entropy(),
                self.critic_head(x), (h_new, c_new))


# ── Import adapter ─────────────────────────────────────────────────────────────

from deltatau_audit.adapters.cleanrl import CleanRLAdapter


# ── Tests: MLP ─────────────────────────────────────────────────────────────────

class TestCleanRLAdapterMLP:

    def test_init(self):
        agent = MinimalAgent()
        adapter = CleanRLAdapter(agent, lstm=False)
        assert adapter is not None
        assert not adapter._lstm

    def test_reset_hidden_none_for_mlp(self):
        agent = MinimalAgent()
        adapter = CleanRLAdapter(agent)
        h = adapter.reset_hidden(batch=1)
        assert h is None

    def test_act_returns_tuple(self):
        agent = MinimalAgent()
        adapter = CleanRLAdapter(agent)
        obs = torch.zeros(4)
        action, value, hidden_new, dt = adapter.act(obs, hidden=None)
        assert isinstance(action, int)
        assert isinstance(value, float)
        assert hidden_new is None
        assert dt is None

    def test_act_discrete_action_in_range(self):
        agent = MinimalAgent(obs_dim=4, act_dim=2)
        adapter = CleanRLAdapter(agent)
        for _ in range(20):
            obs = torch.randn(4)
            action, _, _, _ = adapter.act(obs, hidden=None)
            assert action in (0, 1)

    def test_act_batched_obs(self):
        """Adapter should handle 2D obs (already batched)."""
        agent = MinimalAgent()
        adapter = CleanRLAdapter(agent)
        obs = torch.zeros(1, 4)  # already (1, obs_dim)
        action, value, _, _ = adapter.act(obs, hidden=None)
        assert isinstance(action, int)

    def test_supports_intervention_false(self):
        agent = MinimalAgent()
        adapter = CleanRLAdapter(agent)
        assert not adapter.supports_intervention

    def test_agent_in_eval_mode(self):
        agent = MinimalAgent()
        adapter = CleanRLAdapter(agent)
        assert not adapter.agent.training


# ── Tests: LSTM ────────────────────────────────────────────────────────────────

class TestCleanRLAdapterLSTM:

    def test_reset_hidden_returns_tuple(self):
        agent = MinimalLSTMAgent(obs_dim=4, act_dim=2, hidden_dim=32)
        adapter = CleanRLAdapter(agent, lstm=True)
        h = adapter.reset_hidden(batch=1)
        assert isinstance(h, tuple)
        assert len(h) == 2
        assert h[0].shape == (1, 1, 32)  # (n_layers, batch, hidden)
        assert h[1].shape == (1, 1, 32)

    def test_act_returns_new_hidden(self):
        agent = MinimalLSTMAgent(obs_dim=4, act_dim=2, hidden_dim=32)
        adapter = CleanRLAdapter(agent, lstm=True)
        hidden = adapter.reset_hidden(batch=1)
        obs = torch.zeros(4)
        action, value, new_hidden, dt = adapter.act(obs, hidden=hidden)
        assert isinstance(action, int)
        assert isinstance(value, float)
        assert new_hidden is not None
        assert isinstance(new_hidden, tuple)

    def test_hidden_state_updates(self):
        """Hidden state should differ after processing different obs."""
        agent = MinimalLSTMAgent(obs_dim=4, act_dim=2, hidden_dim=32)
        adapter = CleanRLAdapter(agent, lstm=True)
        h0 = adapter.reset_hidden(batch=1)
        _, _, h1, _ = adapter.act(torch.randn(4), h0)
        _, _, h2, _ = adapter.act(torch.randn(4), h1)
        # h1 and h2 should differ (state updated)
        assert not torch.allclose(h1[0], h2[0])


# ── Tests: from_checkpoint ─────────────────────────────────────────────────────

class TestCleanRLFromCheckpoint:

    def test_from_checkpoint(self, tmp_path):
        agent = MinimalAgent(obs_dim=4, act_dim=2)
        ckpt = tmp_path / "agent.pt"
        torch.save(agent.state_dict(), ckpt)

        adapter = CleanRLAdapter.from_checkpoint(
            str(ckpt), agent_class=MinimalAgent,
            agent_kwargs={"obs_dim": 4, "act_dim": 2},
        )
        obs = torch.zeros(4)
        action, _, _, _ = adapter.act(obs, hidden=None)
        assert isinstance(action, int)

    def test_from_checkpoint_model_state_dict_format(self, tmp_path):
        agent = MinimalAgent()
        ckpt = tmp_path / "agent.pt"
        torch.save({"model_state_dict": agent.state_dict()}, ckpt)

        adapter = CleanRLAdapter.from_checkpoint(
            str(ckpt), agent_class=MinimalAgent,
            agent_kwargs={"obs_dim": 4, "act_dim": 2},
        )
        assert adapter is not None

    def test_from_checkpoint_state_dict_format(self, tmp_path):
        agent = MinimalAgent()
        ckpt = tmp_path / "agent.pt"
        torch.save({"state_dict": agent.state_dict()}, ckpt)

        adapter = CleanRLAdapter.from_checkpoint(
            str(ckpt), agent_class=MinimalAgent,
            agent_kwargs={"obs_dim": 4, "act_dim": 2},
        )
        assert adapter is not None


# ── Tests: from_module_path ────────────────────────────────────────────────────

class TestCleanRLFromModulePath:

    def test_from_module_path(self, tmp_path):
        # Write a minimal agent module to tmp dir
        agent_code = """
import torch.nn as nn
from torch.distributions import Categorical

class Agent(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.critic = nn.Linear(obs_dim, 1)

    def get_action_and_value(self, obs, action=None):
        from torch.distributions import Categorical
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)
"""
        import torch
        module_file = tmp_path / "my_agent.py"
        module_file.write_text(agent_code)

        ckpt = tmp_path / "agent.pt"
        import importlib.util
        spec = importlib.util.spec_from_file_location("_tmp_agent", str(module_file))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        agent = mod.Agent(obs_dim=4, act_dim=2)
        torch.save(agent.state_dict(), ckpt)

        adapter = CleanRLAdapter.from_module_path(
            checkpoint_path=str(ckpt),
            agent_module_path=str(module_file),
            agent_class_name="Agent",
            agent_kwargs={"obs_dim": 4, "act_dim": 2},
        )
        import torch as t
        action, _, _, _ = adapter.act(t.zeros(4), None)
        assert isinstance(action, int)

    def test_from_module_path_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CleanRLAdapter.from_module_path(
                checkpoint_path="dummy.pt",
                agent_module_path=str(tmp_path / "nonexistent.py"),
            )

    def test_from_module_path_wrong_class(self, tmp_path):
        agent_code = "class Foo: pass\n"
        module_file = tmp_path / "agent.py"
        module_file.write_text(agent_code)

        with pytest.raises(AttributeError, match="not found"):
            CleanRLAdapter.from_module_path(
                checkpoint_path="dummy.pt",
                agent_module_path=str(module_file),
                agent_class_name="Agent",
            )
