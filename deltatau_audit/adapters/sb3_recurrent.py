"""Adapter for Stable-Baselines3 Contrib RecurrentPPO models.

Wraps an SB3 RecurrentPPO model for robustness auditing.
Intervention is not supported (Reliance = N/A).

Requires: sb3-contrib >= 2.0

Usage:
    from sb3_contrib import RecurrentPPO
    from deltatau_audit.adapters.sb3_recurrent import SB3RecurrentAdapter

    model = RecurrentPPO.load("my_model.zip")
    adapter = SB3RecurrentAdapter(model)

    # Then use with the auditor
    from deltatau_audit.auditor import run_full_audit
    result = run_full_audit(adapter, env_factory, ...)
"""

from typing import Any, Optional, Tuple

import numpy as np
import torch

from .base import AgentAdapter


class SB3RecurrentAdapter(AgentAdapter):
    """Adapter for SB3 Contrib RecurrentPPO.

    This adapter does NOT support intervention (no dt parameter).
    The audit will run robustness tests only (Reliance = N/A).
    """

    def __init__(self, model, device: str = "cpu"):
        """
        Args:
            model: An sb3_contrib.RecurrentPPO instance (already loaded).
            device: Device string (default: "cpu").
        """
        self.model = model
        self.device = device
        self._n_envs = 1

    def reset_hidden(self, batch: int = 1,
                     device: str = "cpu") -> Any:
        self._n_envs = batch
        # SB3 RecurrentPPO uses LSTM states as numpy arrays
        # Shape: (num_layers, batch, hidden_dim)
        lstm = self.model.policy.lstm_actor
        n_layers = lstm.num_layers
        hidden_dim = lstm.hidden_size
        h = np.zeros((n_layers, batch, hidden_dim), dtype=np.float32)
        c = np.zeros((n_layers, batch, hidden_dim), dtype=np.float32)
        return (h, c)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, hidden: Any
            ) -> Tuple[int, float, Any, Optional[float]]:
        # Convert obs to numpy for SB3
        if obs.dim() == 1:
            obs_np = obs.cpu().numpy().reshape(1, -1)
        else:
            obs_np = obs.cpu().numpy()

        # SB3 episode_start flag (False = continuing episode)
        episode_starts = np.array([False] * self._n_envs)

        # Get action and next hidden state
        action, hidden_new = self.model.predict(
            obs_np,
            state=hidden,
            episode_start=episode_starts,
            deterministic=False,
        )

        # Get value estimate
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32,
                                device=self.model.device)
        lstm_states_t = (
            torch.as_tensor(hidden[0], device=self.model.device),
            torch.as_tensor(hidden[1], device=self.model.device),
        )
        ep_starts_t = torch.as_tensor(episode_starts, dtype=torch.float32,
                                      device=self.model.device)

        value = self.model.policy.predict_values(
            obs_t,
            lstm_states_t,
            ep_starts_t,
        )
        value_scalar = value.item()

        action_int = int(action[0]) if hasattr(action, '__len__') else int(action)

        return (action_int, value_scalar, hidden_new, None)

    # No rerun_with_dt → supports_intervention = False

    @classmethod
    def from_path(cls, path: str, device: str = "cpu") -> "SB3RecurrentAdapter":
        """Load an SB3 RecurrentPPO model from a .zip file.

        Args:
            path: Path to the saved model (.zip)
            device: Device string

        Returns:
            SB3RecurrentAdapter instance
        """
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError:
            raise ImportError(
                "sb3_contrib is required for SB3 adapter. "
                "Install with: pip install sb3-contrib"
            )

        model = RecurrentPPO.load(path, device=device)
        return cls(model, device=device)
