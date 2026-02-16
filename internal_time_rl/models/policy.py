"""Internal Time Agent: RL policy with learnable temporal dynamics.

Architecture:
    Observation x_t
         |
    [Encoder]  ->  encoded_x
         |
    [TimeModule]  ->  delta_tau_t = g(h_t, encoded_x)
         |
    [TimeAwareGRU]  ->  h_{t+1} = f(h_t, encoded_x; delta_tau_t)
         |
    +----+----+
    |         |
 [Policy]  [Value]
    |         |
  pi(a|s)   V(s)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .encoder import ObservationEncoder
from .time_module import TimeModule, TimeAwareGRUCell, NeuralODETransition


class InternalTimeAgent(nn.Module):
    """RL agent with learnable internal time dynamics."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        time_hidden_dim: int = 32,
        use_internal_time: bool = True,
        transition_type: str = "gru",
        time_init_bias: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.use_internal_time = use_internal_time

        # Observation encoder
        self.encoder = ObservationEncoder(obs_dim, latent_dim)

        # Internal time module (optional)
        if use_internal_time:
            self.time_module = TimeModule(
                hidden_dim, latent_dim, time_hidden_dim, time_init_bias
            )
        else:
            self.time_module = None

        # Recurrent transition
        if transition_type == "gru":
            self.rnn = TimeAwareGRUCell(latent_dim, hidden_dim)
        elif transition_type == "ode":
            self.rnn = NeuralODETransition(latent_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown transition type: {transition_type}")

        # Policy head (discrete actions)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        """Full forward pass.

        Args:
            obs: raw observation (batch, obs_dim)
            hidden: recurrent hidden state (batch, hidden_dim)

        Returns:
            dist: Categorical action distribution
            value: state value scalar (batch,)
            hidden_new: updated hidden state (batch, hidden_dim)
            delta_tau: internal time step (batch, 1)
        """
        encoded = self.encoder(obs)

        # Compute internal time step
        if self.use_internal_time:
            delta_tau = self.time_module(hidden, encoded)
        else:
            delta_tau = torch.ones(obs.shape[0], 1, device=obs.device)

        # Time-modulated state transition
        hidden_new = self.rnn(encoded, hidden, delta_tau)

        # Policy and value from updated state
        logits = self.policy_head(hidden_new)
        value = self.value_head(hidden_new).squeeze(-1)
        dist = Categorical(logits=logits)

        return dist, value, hidden_new, delta_tau

    def get_action_and_value(self, obs, hidden, action=None):
        """Convenience method for PPO rollout and update.

        Returns:
            action, log_prob, entropy, value, hidden_new, delta_tau
        """
        dist, value, hidden_new, delta_tau = self.forward(obs, hidden)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value, hidden_new, delta_tau
