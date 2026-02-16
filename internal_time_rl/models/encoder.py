"""Observation encoder: raw observations -> latent representations."""

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """MLP encoder mapping raw observations to a latent space."""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
