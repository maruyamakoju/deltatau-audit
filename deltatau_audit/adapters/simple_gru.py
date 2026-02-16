"""Adapter for SimpleGRUPolicy — demonstrates external model auditing.

This adapter wraps a GRU actor-critic that does NOT have internal time.
Only robustness testing is supported (no dt intervention).
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .base import AgentAdapter


class SimpleGRUPolicy(nn.Module):
    """Minimal GRU actor-critic (standalone, no InternalTimeAgent dependency)."""

    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(obs_dim, hidden_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, obs, hidden):
        h = self.gru(obs, hidden)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return Categorical(logits=logits), value, h

    def get_initial_hidden(self, batch, device="cpu"):
        return torch.zeros(batch, self.hidden_dim, device=device)


class SimpleGRUAdapter(AgentAdapter):
    """Adapter for SimpleGRUPolicy.

    This adapter does NOT support intervention (no dt parameter).
    The audit will run robustness tests only.
    """

    def __init__(self, model: SimpleGRUPolicy, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def reset_hidden(self, batch: int = 1,
                     device: str = "cpu") -> Any:
        return self.model.get_initial_hidden(batch, device or self.device)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, hidden: Any
            ) -> Tuple[int, float, Any, Optional[float]]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)

        dist, value, hidden_new = self.model(obs, hidden)
        action = dist.sample()

        return (
            action.item(),
            value.item(),
            hidden_new,
            None,  # No dt — this agent doesn't have internal time
        )

    # No rerun_with_dt → supports_intervention = False
    # No recompute_value → supports_value_recompute = False

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, obs_dim: int = 4,
                        act_dim: int = 2, hidden_dim: int = 64,
                        device: str = "cpu") -> "SimpleGRUAdapter":
        """Load from a saved checkpoint."""
        model = SimpleGRUPolicy(obs_dim, act_dim, hidden_dim)
        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        return cls(model, device=device)
