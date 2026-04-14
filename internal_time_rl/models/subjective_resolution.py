r"""Level 4 Agent: Subjective Resolution Agent (Axis 8 & 9 Integration).

The "Temporal Singularity" agent: decouples environment dt from agent
subjective time AND scales its "thinking resolution" (pondering steps) 
based on internal subjective uncertainty.

Core Idea:
    1. delta_tau = g(h_t, x_t) represents "Subjective Uncertainty".
    2. Pondering iterations N are modulated by delta_tau.
    3. High delta_tau (fast subjective time/high change) => Deeper pondering.
    4. KL-Regularization balances "Thinking Cost" vs "Temporal Accuracy".

References:
    - Graves (2016): Adaptive Computation Time.
    - Axis 8: Subjective Time (Neural ODE / Time-Aware GRU).
    - Axis 9: Recursive Deliberation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from act_utils import apply_act_step, halt_distribution_stats, stack_halt_weights

from .policy import InternalTimeAgent
from .encoder import ObservationEncoder
from .time_module import TimeModule, TimeAwareGRUCell
from .deliberative import GeometricHaltingPrior


class SubjectiveResolutionAgent(nn.Module):
    """Integrated Axis 8/9 Agent: Subjective Uncertainty-based Deliberation.

    Args:
        obs_dim: Observation dimensionality.
        act_dim: Action dimensionality.
        hidden_dim: Recursive hidden state size.
        latent_dim: Encoded observation size.
        max_ponder_base: Base maximum pondering steps.
        discrete_actions: Whether action space is discrete.
        lambda_geo: Geometric prior parameter for halting.
        tau_scale: Scaling factor for delta_tau's effect on pondering.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        max_ponder_base: int = 8,
        discrete_actions: bool = True,
        lambda_geo: float = 0.5,
        tau_scale: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.max_ponder_base = max_ponder_base
        self.discrete_actions = discrete_actions
        self.tau_scale = tau_scale

        self.encoder = ObservationEncoder(obs_dim, latent_dim)
        self.time_module = TimeModule(hidden_dim, latent_dim)
        
        # Recursive Cell (Subjective-Time Aware)
        self.cell = TimeAwareGRUCell(latent_dim, hidden_dim)

        # Halting Network (Predicts p_halt at each pondering step)
        self.halt_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            # No sigmoid here; applied in act loop with delta_tau shift
        )

        # Geometric prior for ACT regularization
        self.prior = GeometricHaltingPrior(lambda_geo=lambda_geo, max_steps=max_ponder_base * 2)

        # Policy & Value Heads (applied to aggregated hidden state)
        if discrete_actions:
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, act_dim),
            )
        else:
            self.policy_mean = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, act_dim),
            )
            self.policy_log_std = nn.Parameter(torch.zeros(1, act_dim))

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self, 
        obs: torch.Tensor, 
        hidden: torch.Tensor,
        ponder_override: Optional[int] = None,
        dt_override: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Integrated Forward Pass with Subjective-Pondering.

        Returns:
            Tuple of (dist, value, hidden_new, delta_tau, diagnostics).
        """
        B = obs.shape[0]
        device = obs.device
        encoded = self.encoder(obs)

        # 1. Subjective Uncertainty Detection
        delta_tau = dt_override if dt_override is not None else self.time_module(hidden, encoded) # (B, 1)

        # 2. Dynamic Pondering Resolution
        # If delta_tau is high, we allow more pondering steps
        max_steps = ponder_override or int(self.max_ponder_base * delta_tau.mean().item() + 1)
        max_steps = max(1, min(max_steps, self.max_ponder_base * 3))

        # ACT Loop
        cumulative_halt = torch.zeros(B, 1, device=device)
        remainder = torch.ones(B, 1, device=device)
        still_running = torch.ones(B, 1, device=device)
        
        h_n = hidden
        aggregated_h = torch.zeros_like(hidden)
        step_weights = []
        
        for n in range(max_steps):
            # One step of "subjective" recurrence
            # We use delta_tau / n_steps or similar to scale the internal step?
            # Actually, delta_tau is the TOTAL subjective change for this env step.
            # We distribute it across pondering steps.
            dt_step = delta_tau / max_steps
            h_n = self.cell(encoded, h_n, dt_step)

            # Predict halting probability
            # Key Innovation: delta_tau shifts the halting logit. 
            # High delta_tau => lower halting prob => more pondering.
            halt_logit = self.halt_net(torch.cat([h_n, encoded], dim=-1))
            halt_logit = halt_logit - self.tau_scale * (delta_tau - 1.0)
            p_halt = torch.sigmoid(halt_logit)

            # Apply ACT logic
            force_halt = (n == max_steps - 1)
            lambda_n, cumulative_halt, remainder, halted = apply_act_step(
                cumulative_halt, remainder, p_halt, still_running, 
                force_halt=torch.tensor([force_halt], device=device).expand(B, 1)
            )

            aggregated_h = aggregated_h + lambda_n * h_n
            step_weights.append(lambda_n)
            still_running = still_running * (1.0 - halted.float().unsqueeze(-1))
            
            if still_running.sum() == 0 and not self.training:
                break

        # Post-pondering diagnostics
        weight_matrix, _ = stack_halt_weights(step_weights, B, device)
        diag = halt_distribution_stats(weight_matrix)
        diag["delta_tau"] = delta_tau
        diag["max_steps"] = torch.tensor([float(max_steps)], device=device)

        # 3. Policy & Value
        if self.discrete_actions:
            logits = self.policy_head(aggregated_h)
            dist = Categorical(logits=logits)
        else:
            mean = self.policy_mean(aggregated_h)
            std = self.policy_log_std.exp().expand_as(mean)
            dist = Normal(mean, std)

        value = self.value_head(aggregated_h).squeeze(-1)
        
        return dist, value, aggregated_h, delta_tau, diag

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        ponder_override: Optional[int] = None,
    ) -> Tuple:
        dist, value, hidden_new, dt, diag = self.forward(obs, hidden, ponder_override=ponder_override)
        if action is None:
            action = dist.sample()

        if self.discrete_actions:
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value, hidden_new, dt, diag
