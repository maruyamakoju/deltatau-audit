"""Tests for deltatau_audit.adapters.sb3 — Stable-Baselines3 adapter."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_discrete_action_space():
    """Mock discrete action space."""
    space = Mock()
    space.n = 4  # 4 discrete actions
    return space


@pytest.fixture
def mock_continuous_action_space():
    """Mock continuous action space."""
    space = Mock()
    del space.n  # Continuous spaces don't have .n
    space.shape = (2,)  # 2D action
    return space


@pytest.fixture
def mock_sb3_model_discrete(mock_discrete_action_space):
    """Mock SB3 model with discrete actions."""
    model = Mock()
    model.action_space = mock_discrete_action_space
    model.device = torch.device("cpu")

    # Mock predict method
    def predict_fn(obs, deterministic=False):
        # Return action and state (None for non-recurrent)
        return np.array([2]), None  # Always predict action 2

    model.predict = Mock(side_effect=predict_fn)

    # Mock policy value prediction
    def predict_values_fn(obs):
        # Return tensor of values
        batch_size = obs.shape[0]
        return torch.tensor([1.5] * batch_size)

    model.policy = Mock()
    model.policy.predict_values = Mock(side_effect=predict_values_fn)

    return model


@pytest.fixture
def mock_sb3_model_continuous(mock_continuous_action_space):
    """Mock SB3 model with continuous actions."""
    model = Mock()
    model.action_space = mock_continuous_action_space
    model.device = torch.device("cpu")

    # Mock predict method
    def predict_fn(obs, deterministic=False):
        # Return continuous action array
        return np.array([0.5, -0.3]), None

    model.predict = Mock(side_effect=predict_fn)

    # Mock policy value prediction
    def predict_values_fn(obs):
        batch_size = obs.shape[0]
        return torch.tensor([2.5] * batch_size)

    model.policy = Mock()
    model.policy.predict_values = Mock(side_effect=predict_values_fn)

    return model


# ── SB3Adapter tests ──────────────────────────────────────────────────

class TestSB3Adapter:
    def test_reset_hidden_returns_none(self, mock_sb3_model_discrete):
        """Non-recurrent models should return None for hidden state."""
        from deltatau_audit.adapters.sb3 import SB3Adapter

        adapter = SB3Adapter(mock_sb3_model_discrete)
        hidden = adapter.reset_hidden()

        assert hidden is None

    def test_supports_intervention_false(self, mock_sb3_model_discrete):
        """SB3Adapter should not support intervention."""
        from deltatau_audit.adapters.sb3 import SB3Adapter

        adapter = SB3Adapter(mock_sb3_model_discrete)

        assert adapter.supports_intervention is False

    def test_supports_value_recompute_false(self, mock_sb3_model_discrete):
        """SB3Adapter should not support value recompute."""
        from deltatau_audit.adapters.sb3 import SB3Adapter

        adapter = SB3Adapter(mock_sb3_model_discrete)

        assert adapter.supports_value_recompute is False

    def test_act_discrete_1d_obs(self, mock_sb3_model_discrete):
        """Test act with 1D observation (discrete actions)."""
        from deltatau_audit.adapters.sb3 import SB3Adapter

        adapter = SB3Adapter(mock_sb3_model_discrete)
        obs = torch.randn(4)  # 1D observation

        action, value, hidden, dt = adapter.act(obs, None)

        # Check return types
        assert isinstance(action, int)
        assert isinstance(value, float)
        assert hidden is None
        assert dt is None

        # Check values
        assert action == 2  # Mocked to return 2
        assert value == pytest.approx(1.5)

    def test_act_discrete_2d_obs(self, mock_sb3_model_discrete):
        """Test act with 2D observation batch (discrete actions)."""
        from deltatau_audit.adapters.sb3 import SB3Adapter

        adapter = SB3Adapter(mock_sb3_model_discrete)
        obs = torch.randn(1, 4)  # Batch of 1

        action, value, hidden, dt = adapter.act(obs, None)

        assert isinstance(action, int)
        assert value == pytest.approx(1.5)

    def test_act_continuous_actions(self, mock_sb3_model_continuous):
        """Test act with continuous action space."""
        from deltatau_audit.adapters.sb3 import SB3Adapter

        adapter = SB3Adapter(mock_sb3_model_continuous)
        obs = torch.randn(4)

        action, value, hidden, dt = adapter.act(obs, None)

        # Continuous actions return numpy array
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert isinstance(value, float)
        assert value == pytest.approx(2.5)

    def test_model_predict_called(self, mock_sb3_model_discrete):
        """Test that model.predict is called with correct args."""
        from deltatau_audit.adapters.sb3 import SB3Adapter

        adapter = SB3Adapter(mock_sb3_model_discrete)
        obs = torch.randn(4)

        adapter.act(obs, None)

        # Check predict was called
        mock_sb3_model_discrete.predict.assert_called_once()

        # Check it was called with numpy array
        call_args = mock_sb3_model_discrete.predict.call_args
        obs_arg = call_args[0][0]
        assert isinstance(obs_arg, np.ndarray)
        assert obs_arg.shape == (1, 4)  # Reshaped to batch

    def test_value_prediction_called(self, mock_sb3_model_discrete):
        """Test that policy.predict_values is called."""
        from deltatau_audit.adapters.sb3 import SB3Adapter

        adapter = SB3Adapter(mock_sb3_model_discrete)
        obs = torch.randn(4)

        adapter.act(obs, None)

        # Check predict_values was called
        mock_sb3_model_discrete.policy.predict_values.assert_called_once()

    def test_from_path_ppo(self, tmp_path, monkeypatch):
        """Test from_path class method for PPO."""
        from deltatau_audit.adapters.sb3 import SB3Adapter

        # Mock stable_baselines3 import
        mock_sb3 = Mock()
        mock_ppo = Mock()

        # Mock PPO.load to return a mock model
        mock_model = Mock()
        mock_model.action_space = Mock()
        mock_model.action_space.n = 2
        mock_model.device = torch.device("cpu")

        mock_ppo.load = Mock(return_value=mock_model)
        mock_sb3.PPO = mock_ppo
        mock_sb3.SAC = Mock()
        mock_sb3.TD3 = Mock()
        mock_sb3.A2C = Mock()

        # Mock the import
        import sys
        sys.modules['stable_baselines3'] = mock_sb3

        try:
            # Create a fake model file
            model_path = tmp_path / "model.zip"
            model_path.touch()

            adapter = SB3Adapter.from_path(str(model_path), algo="ppo")

            # Check that PPO.load was called
            mock_ppo.load.assert_called_once()
            assert adapter.model == mock_model

        finally:
            # Cleanup
            if 'stable_baselines3' in sys.modules:
                del sys.modules['stable_baselines3']

    def test_from_path_unknown_algo(self):
        """Test from_path with unknown algorithm raises error."""
        from deltatau_audit.adapters.sb3 import SB3Adapter

        # Mock stable_baselines3
        import sys
        sys.modules['stable_baselines3'] = Mock()

        try:
            with pytest.raises(ValueError, match="Unknown algo"):
                SB3Adapter.from_path("model.zip", algo="unknown_algo")

        finally:
            if 'stable_baselines3' in sys.modules:
                del sys.modules['stable_baselines3']

    def test_from_path_missing_sb3(self, monkeypatch):
        """Test from_path raises ImportError if SB3 not installed."""
        from deltatau_audit.adapters.sb3 import SB3Adapter

        # Ensure stable_baselines3 is not importable
        import sys
        if 'stable_baselines3' in sys.modules:
            del sys.modules['stable_baselines3']

        # Mock import to fail
        def mock_import(name, *args, **kwargs):
            if name == 'stable_baselines3':
                raise ModuleNotFoundError("No module named 'stable_baselines3'")
            return orig_import(name, *args, **kwargs)

        orig_import = __builtins__.__import__
        monkeypatch.setattr(__builtins__, '__import__', mock_import)

        with pytest.raises(ImportError, match="stable-baselines3 is required"):
            SB3Adapter.from_path("model.zip", algo="ppo")

    def test_device_parameter(self, mock_sb3_model_discrete):
        """Test device parameter is stored."""
        from deltatau_audit.adapters.sb3 import SB3Adapter

        adapter = SB3Adapter(mock_sb3_model_discrete, device="cuda")

        assert adapter.device == "cuda"

    def test_adapter_with_real_tensors(self, mock_sb3_model_discrete):
        """Test adapter works with real PyTorch tensors."""
        from deltatau_audit.adapters.sb3 import SB3Adapter

        adapter = SB3Adapter(mock_sb3_model_discrete)

        # Create real tensors
        obs = torch.tensor([1.0, 2.0, 3.0, 4.0])

        action, value, hidden, dt = adapter.act(obs, None)

        assert action == 2
        assert value == pytest.approx(1.5)
        assert hidden is None
        assert dt is None
