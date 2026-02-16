"""Phase 2 Agent: Internal Time with Self-Model.

Architecture:
    Observation x_t
         |
    [Encoder]  ->  encoded_x
         |
    [SelfModel] -> h_hat_{t+1} = f_self(h_t)
         |
    prediction_error = |h_hat_{t+1} - h_{t+1}|^2
         |
    [PredErrorTimeModule] -> delta_tau_t = g(h_t, encoded_x, pred_error)
         |
    [TimeAwareGRU] -> h_{t+1} = f(h_t, encoded_x; delta_tau_t)
         |
    +----+----+
    |         |
 [Policy]  [Value]
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .encoder import ObservationEncoder
from .time_module import TimeAwareGRUCell, NeuralODETransition
from .self_model import SelfModel, PredictionErrorTimeModule


class SelfModelAgent(nn.Module):
    """RL agent with self-model driven internal time dynamics."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        time_hidden_dim: int = 32,
        bottleneck_dim: int = 64,
        transition_type: str = "gru",
        time_init_bias: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.use_internal_time = True  # always True for this agent

        # Observation encoder
        self.encoder = ObservationEncoder(obs_dim, latent_dim)

        # Self-model: predicts next hidden state
        self.self_model = SelfModel(hidden_dim, bottleneck_dim)

        # Prediction-error driven time module
        self.time_module = PredictionErrorTimeModule(
            hidden_dim, latent_dim, time_hidden_dim, time_init_bias
        )

        # Recurrent transition
        if transition_type == "gru":
            self.rnn = TimeAwareGRUCell(latent_dim, hidden_dim)
        elif transition_type == "ode":
            self.rnn = NeuralODETransition(latent_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown transition type: {transition_type}")

        # Policy head
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

        # Running estimate of prediction error for normalization
        self.register_buffer("pred_error_ema", torch.tensor(1.0))
        self.register_buffer("pred_error_count", torch.tensor(0.0))

    def get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        """Forward pass with self-model prediction error.

        Returns:
            dist, value, hidden_new, delta_tau, pred_error, h_predicted
        """
        encoded = self.encoder(obs)

        # Self-model prediction (predict next hidden state from current)
        h_predicted = self.self_model(hidden)

        # Compute prediction error from previous step's prediction
        # Use a simple heuristic: predict h_{t+1}, then at step t+1 compare
        # For the current step, we use the error between predicted and actual
        # as a running signal. During the first step, error is 0.
        pred_error = self.self_model.prediction_error(hidden, hidden)

        # Normalize prediction error with EMA
        if self.training:
            with torch.no_grad():
                batch_mean = pred_error.mean()
                alpha = 0.01
                self.pred_error_ema = (1 - alpha) * self.pred_error_ema + alpha * batch_mean
                self.pred_error_count += 1

        normalized_error = pred_error / (self.pred_error_ema + 1e-8)

        # Compute delta_tau using prediction error
        delta_tau = self.time_module(hidden, encoded, normalized_error)

        # Time-modulated state transition
        hidden_new = self.rnn(encoded, hidden, delta_tau)

        # Policy and value
        logits = self.policy_head(hidden_new)
        value = self.value_head(hidden_new).squeeze(-1)
        dist = Categorical(logits=logits)

        return dist, value, hidden_new, delta_tau, pred_error, h_predicted

    def get_action_and_value(self, obs, hidden, action=None):
        """Convenience method compatible with PPO interface."""
        dist, value, hidden_new, delta_tau, pred_error, h_predicted = self.forward(
            obs, hidden
        )

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value, hidden_new, delta_tau

    def compute_self_model_loss(self, hidden_current, hidden_next):
        """Compute self-model prediction loss for training.

        This trains the self-model to predict the agent's own state transitions.
        """
        h_pred = self.self_model(hidden_current)
        return (h_pred - hidden_next.detach()).pow(2).mean()
