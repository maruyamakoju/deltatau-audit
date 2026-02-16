"""Baseline agents for comparison.

1. SkipRNNAgent (ACT/Skip-RNN style):
   - Binary (soft) gate decides whether to update hidden state
   - gate ≈ 1: update with GRU
   - gate ≈ 0: copy previous hidden state
   - Trained with ponder cost (penalizes updates)
   - Difference from Internal Time: binary skip vs continuous scaling

2. ExternalDtAgent (ODE-RNN style):
   - Uses Neural ODE transition: h_{t+1} = h_t + dt * f(h_t, x_t)
   - dt is GIVEN by the environment (action repeat, observation interval)
   - dt is NOT learned
   - Difference from Internal Time: dt is observed vs learned
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .encoder import ObservationEncoder
from .time_module import NeuralODETransition


class SkipGRUCell(nn.Module):
    """GRU cell with learnable binary skip gate.

    h_candidate = GRU(h_t, x_t)
    gate = sigmoid(W_gate @ [h_t, x_t])
    h_{t+1} = gate * h_candidate + (1 - gate) * h_t

    When gate → 0: skip update (copy h_t)
    When gate → 1: full GRU update
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.gate_net = nn.Linear(input_dim + hidden_dim, 1)
        # Initialize gate bias high so initial behavior is "always update"
        nn.init.constant_(self.gate_net.bias, 2.0)

    def forward(self, x, h, delta_tau=None):
        """delta_tau is ignored (compatibility interface). Gate is internal."""
        h_candidate = self.gru_cell(x, h)
        gate_input = torch.cat([x, h], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_input))  # (batch, 1)
        h_new = gate * h_candidate + (1 - gate) * h
        return h_new, gate


class SkipRNNAgent(nn.Module):
    """Agent with Skip-RNN (ACT-like update gating).

    Comparable to InternalTimeAgent but with binary skip/update decision
    instead of continuous time scaling.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        ponder_cost: float = 0.01,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.use_internal_time = True  # for compatibility
        self.ponder_cost = ponder_cost

        self.encoder = ObservationEncoder(obs_dim, latent_dim)
        self.rnn = SkipGRUCell(latent_dim, hidden_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, act_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def get_initial_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs, hidden):
        encoded = self.encoder(obs)
        hidden_new, gate = self.rnn(encoded, hidden)

        logits = self.policy_head(hidden_new)
        value = self.value_head(hidden_new).squeeze(-1)
        dist = Categorical(logits=logits)

        # Report gate as "delta_tau" for logging compatibility
        return dist, value, hidden_new, gate

    def get_action_and_value(self, obs, hidden, action=None):
        dist, value, hidden_new, gate = self.forward(obs, hidden)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, hidden_new, gate


class ExternalDtAgent(nn.Module):
    """Agent with ODE-RNN using external (environment-given) dt.

    Uses Neural ODE transition: h_{t+1} = h_t + dt * f(h_t, x_t)
    where dt comes from the environment (e.g., action repeat value).

    This tests: "is it enough to know dt, or do you need to learn it?"
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.use_internal_time = False  # dt is external

        self.encoder = ObservationEncoder(obs_dim, latent_dim)
        self.rnn = NeuralODETransition(latent_dim, hidden_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, act_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self._external_dt = None

    def set_external_dt(self, dt: torch.Tensor):
        """Set the external dt for the next forward pass."""
        self._external_dt = dt

    def get_initial_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs, hidden):
        encoded = self.encoder(obs)

        # Use external dt if available, else default to 1.0
        if self._external_dt is not None:
            dt = self._external_dt
        else:
            dt = torch.ones(obs.shape[0], 1, device=obs.device)

        hidden_new = self.rnn(encoded, hidden, dt)

        logits = self.policy_head(hidden_new)
        value = self.value_head(hidden_new).squeeze(-1)
        dist = Categorical(logits=logits)

        return dist, value, hidden_new, dt

    def get_action_and_value(self, obs, hidden, action=None):
        dist, value, hidden_new, dt = self.forward(obs, hidden)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, hidden_new, dt
