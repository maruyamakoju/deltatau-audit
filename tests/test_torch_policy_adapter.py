"""Tests for TorchPolicyAdapter."""

import pytest
import torch
import torch.nn as nn

from deltatau_audit.adapters.torch_policy import TorchPolicyAdapter


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_discrete_actor_critic(obs_dim=4, act_dim=2):
    actor = nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh(), nn.Linear(32, act_dim))
    critic = nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh(), nn.Linear(32, 1))
    return actor, critic


def make_continuous_actor_critic(obs_dim=4, act_dim=2):
    actor = nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh(), nn.Linear(32, act_dim))
    critic = nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh(), nn.Linear(32, 1))
    return actor, critic


# ── Tests: callable API ────────────────────────────────────────────────────────

class TestTorchPolicyCallable:

    def test_init(self):
        def act_fn(obs):
            return torch.tensor([0]), torch.tensor(0.0)

        adapter = TorchPolicyAdapter(act_fn)
        assert adapter is not None

    def test_reset_hidden_none(self):
        adapter = TorchPolicyAdapter(lambda obs: (torch.tensor([0]), torch.tensor(0.0)))
        h = adapter.reset_hidden(batch=1)
        assert h is None

    def test_act_discrete(self):
        def act_fn(obs):
            return torch.tensor([1], dtype=torch.int64), torch.tensor(0.5)

        adapter = TorchPolicyAdapter(act_fn)
        obs = torch.zeros(4)
        action, value, hidden, dt = adapter.act(obs, hidden=None)
        assert action == 1
        assert abs(value - 0.5) < 1e-5
        assert hidden is None
        assert dt is None

    def test_act_continuous(self):
        def act_fn(obs):
            return torch.tensor([0.3, -0.7]), torch.tensor(1.5)

        adapter = TorchPolicyAdapter(act_fn)
        obs = torch.zeros(6)
        action, value, hidden, dt = adapter.act(obs, hidden=None)
        assert hasattr(action, '__len__')
        assert abs(value - 1.5) < 1e-5

    def test_act_numpy_action(self):
        import numpy as np

        def act_fn(obs):
            return np.array([0]), torch.tensor(0.0)

        adapter = TorchPolicyAdapter(act_fn)
        obs = torch.zeros(4)
        action, value, _, _ = adapter.act(obs, hidden=None)
        # Single-element numpy → int
        assert isinstance(action, (int, float))

    def test_supports_intervention_false(self):
        adapter = TorchPolicyAdapter(lambda obs: (torch.tensor([0]), torch.tensor(0.0)))
        assert not adapter.supports_intervention

    def test_obs_auto_unsqueeze(self):
        """1D obs should be automatically unsqueezed to (1, obs_dim)."""
        received_shapes = []

        def act_fn(obs):
            received_shapes.append(obs.shape)
            return torch.tensor([0]), torch.tensor(0.0)

        adapter = TorchPolicyAdapter(act_fn)
        adapter.act(torch.zeros(4), hidden=None)  # 1D obs
        assert received_shapes[-1] == torch.Size([1, 4])


# ── Tests: from_actor_critic ───────────────────────────────────────────────────

class TestTorchPolicyFromActorCritic:

    def test_discrete(self):
        actor, critic = make_discrete_actor_critic()
        adapter = TorchPolicyAdapter.from_actor_critic(actor, critic, is_discrete=True)
        obs = torch.zeros(4)
        action, value, hidden, dt = adapter.act(obs, hidden=None)
        assert isinstance(action, int)
        assert action in (0, 1)
        assert isinstance(value, float)

    def test_continuous(self):
        actor, critic = make_continuous_actor_critic(obs_dim=6, act_dim=3)
        adapter = TorchPolicyAdapter.from_actor_critic(actor, critic, is_discrete=False)
        obs = torch.zeros(6)
        action, value, _, _ = adapter.act(obs, hidden=None)
        assert hasattr(action, '__len__') or isinstance(action, float)

    def test_actor_in_eval_mode(self):
        actor, critic = make_discrete_actor_critic()
        TorchPolicyAdapter.from_actor_critic(actor, critic)
        assert not actor.training
        assert not critic.training

    def test_no_critic(self):
        """from_actor_critic requires critic; actor-only via from_checkpoint."""
        actor, _ = make_discrete_actor_critic()
        # No critic: from_checkpoint with critic=None
        adapter = TorchPolicyAdapter.from_actor_critic(actor, nn.Identity(),
                                                        is_discrete=True)
        assert adapter is not None


# ── Tests: from_checkpoint ─────────────────────────────────────────────────────

class TestTorchPolicyFromCheckpoint:

    def test_raw_state_dict(self, tmp_path):
        actor, critic = make_discrete_actor_critic()
        ckpt = tmp_path / "model.pt"
        torch.save(actor.state_dict(), ckpt)

        new_actor, new_critic = make_discrete_actor_critic()
        adapter = TorchPolicyAdapter.from_checkpoint(
            str(ckpt), actor=new_actor, critic=new_critic,
            is_discrete=True,
        )
        obs = torch.zeros(4)
        action, _, _, _ = adapter.act(obs, None)
        assert action in (0, 1)

    def test_separate_actor_critic_keys(self, tmp_path):
        actor, critic = make_discrete_actor_critic()
        ckpt = tmp_path / "model.pt"
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, ckpt)

        new_actor, new_critic = make_discrete_actor_critic()
        adapter = TorchPolicyAdapter.from_checkpoint(
            str(ckpt), actor=new_actor, critic=new_critic,
            is_discrete=True,
        )
        obs = torch.zeros(4)
        action, _, _, _ = adapter.act(obs, None)
        assert action in (0, 1)

    def test_model_state_dict_with_prefixes(self, tmp_path):
        actor, critic = make_discrete_actor_critic()
        # Build state dict with actor.* / critic.* prefixes (RSL-RL style)
        combined = {}
        for k, v in actor.state_dict().items():
            combined[f"actor.{k}"] = v
        for k, v in critic.state_dict().items():
            combined[f"critic.{k}"] = v
        ckpt = tmp_path / "model.pt"
        torch.save({"model_state_dict": combined}, ckpt)

        new_actor, new_critic = make_discrete_actor_critic()
        adapter = TorchPolicyAdapter.from_checkpoint(
            str(ckpt), actor=new_actor, critic=new_critic,
            is_discrete=True,
        )
        obs = torch.zeros(4)
        action, _, _, _ = adapter.act(obs, None)
        assert action in (0, 1)

    def test_actor_only(self, tmp_path):
        actor, _ = make_discrete_actor_critic()
        ckpt = tmp_path / "actor.pt"
        torch.save(actor.state_dict(), ckpt)

        new_actor, _ = make_discrete_actor_critic()
        adapter = TorchPolicyAdapter.from_checkpoint(
            str(ckpt), actor=new_actor, critic=None,
            is_discrete=True,
        )
        obs = torch.zeros(4)
        action, value, _, _ = adapter.act(obs, None)
        assert action in (0, 1)
        assert value == 0.0  # dummy value when no critic

    def test_invalid_checkpoint_format(self, tmp_path):
        ckpt = tmp_path / "bad.pt"
        torch.save([1, 2, 3], ckpt)  # list, not dict

        actor, _ = make_discrete_actor_critic()
        with pytest.raises(ValueError, match="Unexpected checkpoint format"):
            TorchPolicyAdapter.from_checkpoint(str(ckpt), actor=actor)


# ── Tests: CLI parse_kwargs helper ────────────────────────────────────────────

class TestParseKwargs:
    """Test the _parse_kwargs helper used by audit-cleanrl CLI."""

    def test_import(self):
        from deltatau_audit.cli import _parse_kwargs
        assert callable(_parse_kwargs)

    def test_empty(self):
        from deltatau_audit.cli import _parse_kwargs
        assert _parse_kwargs("") == {}
        assert _parse_kwargs(None) == {}

    def test_int_values(self):
        from deltatau_audit.cli import _parse_kwargs
        result = _parse_kwargs("obs_dim=4,act_dim=2")
        assert result == {"obs_dim": 4, "act_dim": 2}

    def test_float_values(self):
        from deltatau_audit.cli import _parse_kwargs
        result = _parse_kwargs("lr=0.001")
        assert abs(result["lr"] - 0.001) < 1e-9

    def test_string_values(self):
        from deltatau_audit.cli import _parse_kwargs
        result = _parse_kwargs("name=agent,device=cpu")
        assert result == {"name": "agent", "device": "cpu"}

    def test_mixed(self):
        from deltatau_audit.cli import _parse_kwargs
        result = _parse_kwargs("obs_dim=4,lr=0.001,name=ppo")
        assert result["obs_dim"] == 4
        assert abs(result["lr"] - 0.001) < 1e-9
        assert result["name"] == "ppo"
