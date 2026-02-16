"""Internal time module and time-aware recurrent cells.

Core idea: the agent learns an internal time step delta_tau that modulates
how much its hidden state changes at each environment step.

- delta_tau > 1: subjective time runs fast (large state updates)
- delta_tau < 1: subjective time runs slow (small state updates)
- delta_tau = 1: equivalent to standard RNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeModule(nn.Module):
    """Computes internal time step delta_tau = g(h_t, x_t).

    Output is strictly positive via softplus.
    Initialized so that delta_tau starts near 1.0.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        time_hidden_dim: int = 32,
        init_bias: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, time_hidden_dim),
            nn.Tanh(),
            nn.Linear(time_hidden_dim, 1),
        )
        # With sigmoid parameterization dt = 0.3 + 2.2*sigmoid(x),
        # we want dt ≈ 1.0: sigmoid(x) = 0.7/2.2 = 0.318, x ≈ -0.76
        nn.init.constant_(self.net[-1].bias, -0.76 + init_bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, hidden: torch.Tensor, encoded_obs: torch.Tensor) -> torch.Tensor:
        """Returns delta_tau of shape (batch, 1), strictly positive.

        Uses sigmoid-based parameterization: Δτ ∈ [dt_min, dt_max].
        This prevents destabilization of the GRU at extreme values.
        Default range [0.3, 2.5] allows meaningful adaptation while
        keeping the time-modulated gate z_eff = 1-(1-z)^{Δτ} well-behaved.
        """
        combined = torch.cat([hidden, encoded_obs], dim=-1)
        raw = self.net(combined)
        # Sigmoid maps to [0,1], then scale to [dt_min, dt_max]
        dt_min, dt_max = 0.3, 2.5
        delta_tau = dt_min + (dt_max - dt_min) * torch.sigmoid(raw)
        return delta_tau


class TimeAwareGRUCell(nn.Module):
    """GRU cell with time-modulated gating.

    Standard GRU:
        z = sigmoid(W_z [x, h])
        r = sigmoid(W_r [x, h])
        h_tilde = tanh(W_h [x, r*h])
        h_new = (1 - z) * h + z * h_tilde

    Time-modulated GRU:
        z_eff = 1 - (1 - z)^{delta_tau}
        h_new = (1 - z_eff) * h + z_eff * h_tilde

    Properties:
        - delta_tau = 1 => standard GRU
        - delta_tau -> 0 => h_new -> h (time freezes)
        - delta_tau -> inf => h_new -> h_tilde (instant transition)

    This is mathematically well-defined since z in (0,1) and delta_tau > 0,
    making (1-z)^{delta_tau} always valid.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, delta_tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: encoded observation (batch, input_dim)
            h: hidden state (batch, hidden_dim)
            delta_tau: internal time step (batch, 1), positive

        Returns:
            h_new: updated hidden state (batch, hidden_dim)
        """
        combined = torch.cat([x, h], dim=-1)

        z = torch.sigmoid(self.W_z(combined))       # update gate
        r = torch.sigmoid(self.W_r(combined))       # reset gate

        combined_r = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_r))  # candidate state

        # Time-modulated update: z_eff = 1 - (1 - z)^{delta_tau}
        # Clamp (1-z) away from 0 for numerical stability with pow
        z_eff = 1.0 - torch.pow((1.0 - z).clamp(min=1e-8), delta_tau)

        h_new = (1.0 - z_eff) * h + z_eff * h_tilde
        return h_new


class NeuralODETransition(nn.Module):
    """Neural ODE-style transition via Euler integration.

    h_{t+1} = h_t + delta_tau * f(h_t, x_t)

    The dynamics function f is a small MLP with Tanh activations
    to keep derivatives bounded (implicit stability constraint).
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dynamics = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, delta_tau: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([x, h], dim=-1)
        dh = self.dynamics(combined)
        h_new = h + delta_tau * dh
        return h_new
