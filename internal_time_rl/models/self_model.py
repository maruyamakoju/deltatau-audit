"""Self-Model and Prediction-Error Driven Time Modulation.

Phase 2 Extension: The agent predicts its own next hidden state,
and the prediction error modulates internal time.

Key insight from the paper:
    delta_tau_t = g(h_t, x_t, h_hat_{t+1})

where h_hat_{t+1} = f_self(h_t) is the agent's prediction of its own
next hidden state.

When prediction error is HIGH (surprise/novelty):
    -> internal time slows down (process more carefully)
When prediction error is LOW (predictable situation):
    -> internal time speeds up (skip through boring parts)

This connects to:
- Predictive coding theory
- Self-referential loops
- Subjective time generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfModel(nn.Module):
    """Predicts the agent's own next hidden state: h_hat_{t+1} = f_self(h_t).

    A simple forward model of the agent's own internal dynamics.
    The prediction error |h_{t+1} - h_hat_{t+1}| serves as a
    surprise/novelty signal.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Predict next hidden state from current hidden state."""
        return self.predictor(hidden)

    def prediction_error(
        self, hidden_current: torch.Tensor, hidden_next: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction error (MSE per sample)."""
        h_pred = self.forward(hidden_current)
        error = (h_pred - hidden_next.detach()).pow(2).mean(dim=-1, keepdim=True)
        return error


class PredictionErrorTimeModule(nn.Module):
    """Time module that incorporates self-model prediction error.

    delta_tau_t = g(h_t, x_t, prediction_error_t)

    The prediction error from the self-model modulates the time step:
    - High prediction error (surprise) -> slower time (careful processing)
    - Low prediction error (predictable) -> faster time (skip ahead)
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        time_hidden_dim: int = 32,
        init_bias: float = 0.0,
    ):
        super().__init__()
        # Input: hidden + encoded_obs + prediction_error (scalar)
        input_dim = hidden_dim + latent_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, time_hidden_dim),
            nn.Tanh(),
            nn.Linear(time_hidden_dim, 1),
        )
        nn.init.constant_(self.net[-1].bias, -0.76 + init_bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(
        self,
        hidden: torch.Tensor,
        encoded_obs: torch.Tensor,
        pred_error: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden: current hidden state (batch, hidden_dim)
            encoded_obs: encoded observation (batch, latent_dim)
            pred_error: self-model prediction error (batch, 1)

        Returns:
            delta_tau: internal time step (batch, 1)
        """
        combined = torch.cat([hidden, encoded_obs, pred_error], dim=-1)
        raw = self.net(combined)
        dt_min, dt_max = 0.3, 2.5
        delta_tau = dt_min + (dt_max - dt_min) * torch.sigmoid(raw)
        return delta_tau
