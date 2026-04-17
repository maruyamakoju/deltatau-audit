r"""Level 5 Agent (Continuous): Causal Resolution Agent for MuJoCo.

Extends Axis 10 logic to continuous action spaces using MPPI-lite
(Model Predictive Path Integral) style counterfactual unrolling.

Core Idea:
1. When delta_tau is high (uncertainty spike), sample K candidate actions
   near the current policy mean.
2. Unroll these K candidates into the future using the Causal Transition Model.
3. Select the candidate that maximizes imagined future value.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .subjective_resolution import SubjectiveResolutionAgent


class CausalResolutionAgentContinuous(SubjectiveResolutionAgent):
    """Integrated Axis 10 Agent for Continuous Control (MuJoCo)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        max_ponder_base: int = 4,
        tau_scale: float = 2.0,
        causal_depth: int = 2,
        num_samples: int = 8, # K samples for MPPI-lite
    ):
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_ponder_base=max_ponder_base,
            discrete_actions=False, # Continuous!
            tau_scale=tau_scale,
        )
        self.causal_depth = causal_depth
        self.num_samples = num_samples
        
        # Causal Transition Model (latent state evolution)
        self.causal_transition = nn.Sequential(
            nn.Linear(hidden_dim + act_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def simulate_counterfactuals_continuous(
        self, 
        h_start: torch.Tensor, 
        base_dist: Normal
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample candidate actions and simulate their futures."""
        B = h_start.shape[0]
        K = self.num_samples
        device = h_start.device
        
        # 1. Sample K candidate actions from current policy
        samples = base_dist.sample((K,)) # (K, B, act_dim)
        samples = samples.permute(1, 0, 2) # (B, K, act_dim)
        
        # 2. Flatten for parallel unroll
        h_expanded = h_start.unsqueeze(1).expand(-1, K, -1).reshape(B*K, -1)
        samples_flat = samples.reshape(B*K, -1)
        
        # 3. Predict futures (1-step unroll)
        h_next = self.causal_transition(torch.cat([h_expanded, samples_flat], dim=-1))
        v_future = self.value_head(h_next).squeeze(-1) # (B*K,)
        v_future = v_future.reshape(B, K)
        
        return samples, v_future

    def forward(
        self, 
        obs: torch.Tensor, 
        hidden: torch.Tensor,
        ponder_override: Optional[int] = None,
        dt_override: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Causal Forward Pass for Continuous Actions."""
        
        # 1. System 1 (Intuitive Pondering)
        dist, value, aggregated_h, delta_tau, diag = super().forward(
            obs, hidden, ponder_override, dt_override
        )
        
        # 2. System 2 (Continuous Causal Check)
        B = obs.shape[0]
        expected_steps = diag["expected_steps"]
        # Threshold to engage System 2
        causal_mask = (expected_steps > (self.max_ponder_base * 0.5))
        
        if causal_mask.any():
            samples, imagined_values = self.simulate_counterfactuals_continuous(aggregated_h, dist)
            
            best_indices = torch.argmax(imagined_values, dim=-1) # (B,)
            best_actions = samples[torch.arange(B), best_indices]
            
            # alpha: blending factor shape (B, 1)
            alpha = torch.clamp((expected_steps - 1.0) / self.max_ponder_base, 0.0, 1.0).unsqueeze(-1)
            
            # Shift the mean of the distribution
            new_mean = (1 - alpha) * dist.mean + alpha * best_actions
            dist = Normal(new_mean, dist.stddev)
            
            # Update value estimate
            best_v = imagined_values[torch.arange(B), best_indices]
            value = (1 - alpha.squeeze(-1)) * value + alpha.squeeze(-1) * best_v
            
            diag["causal_engaged"] = True
            diag["imagined_value_gain"] = (best_v - imagined_values.mean(dim=-1)).mean().item()

        return dist, value, aggregated_h, delta_tau, diag
