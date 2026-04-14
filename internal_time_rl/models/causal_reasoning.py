r"""Level 5 Agent: Causal Resolution Agent (Axis 10 Integration).

Combines Subjective Resolution (Axis 8+9) with Causal Temporal Reasoning (Axis 10).

Core Idea: "Counterfactual Pondering"
1. Fast Thinking (System 1): When subjective uncertainty (delta_tau) is low,
   the agent acts reactively using the base policy.
2. Slow Thinking (System 2): When delta_tau spikes, the agent increases its 
   pondering steps. However, instead of just updating its hidden state linearly, 
   it uses a World Model to simulate "What if?" (Counterfactuals) for different 
   actions into the future, and selects the action that maximizes expected 
   future value (MCTS-lite).

References:
    - Axis 8: Subjective Time
    - Axis 9: Recursive Deliberation
    - Axis 10: Causal Temporal Reasoning
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .subjective_resolution import SubjectiveResolutionAgent


class CausalResolutionAgent(SubjectiveResolutionAgent):
    """Integrated Axis 10 Agent: Counterfactual Unroll during Pondering."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        max_ponder_base: int = 4,
        tau_scale: float = 2.0,
        causal_depth: int = 3, # How many steps into the future to imagine
    ):
        # Initialize the base Subjective Resolution agent
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_ponder_base=max_ponder_base,
            discrete_actions=True, # Require discrete for simple counterfactual branching
            tau_scale=tau_scale,
        )
        self.causal_depth = causal_depth
        
        # Causal Transition Model (Simplified World Model for "What if?")
        # Predicts next latent state given (current latent, action)
        self.causal_transition = nn.Sequential(
            nn.Linear(hidden_dim + act_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def simulate_counterfactuals(
        self, 
        h_start: torch.Tensor, 
        depth: int
    ) -> torch.Tensor:
        """Simulate future values for all possible initial actions (1-step lookahead for now).
        
        Returns:
            Expected values for each action: shape (B, act_dim)
        """
        B = h_start.shape[0]
        device = h_start.device
        action_values = torch.zeros(B, self.act_dim, device=device)
        
        for a in range(self.act_dim):
            # One-hot encode the action
            act_one_hot = F.one_hot(torch.tensor([a]*B, device=device), num_classes=self.act_dim).float()
            
            # Predict "What if I take action `a`?"
            h_next = self.causal_transition(torch.cat([h_start, act_one_hot], dim=-1))
            
            # Evaluate the imagined future state
            v_future = self.value_head(h_next).squeeze(-1)
            action_values[:, a] = v_future
            
        return action_values

    def forward(
        self, 
        obs: torch.Tensor, 
        hidden: torch.Tensor,
        ponder_override: Optional[int] = None,
        dt_override: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Causal Forward Pass.
        
        Overrides the base forward pass to inject causal reasoning when
        uncertainty (delta_tau) demands it.
        """
        # 1. Base Unroll (System 1 + standard ACT pondering)
        dist, value, aggregated_h, delta_tau, diag = super().forward(
            obs, hidden, ponder_override, dt_override
        )
        
        # 2. Causal Override (System 2)
        # If the agent decided to ponder deeply (e.g., expected_steps > threshold),
        # we engage the Causal Transition Model to double-check the policy.
        B = obs.shape[0]
        device = obs.device
        
        expected_steps = diag["expected_steps"]
        # Threshold: If it thought for more than 1.5x its base capacity
        causal_mask = (expected_steps > (self.max_ponder_base * 0.5)).float().unsqueeze(-1)
        
        if causal_mask.sum() > 0:
            # Imagine futures
            causal_q_values = self.simulate_counterfactuals(aggregated_h, self.causal_depth)
            
            # Convert Q-values to a policy via Softmax (Temperature = 1.0)
            causal_logits = causal_q_values
            
            # Blend the intuitive policy (System 1) with the causal policy (System 2)
            # based on how much it pondered.
            alpha = torch.clamp((expected_steps.unsqueeze(-1) - 1.0) / self.max_ponder_base, 0.0, 1.0)
            
            blended_logits = (1 - alpha) * dist.logits + alpha * causal_logits
            dist = Categorical(logits=blended_logits)
            
            # Update value estimate based on causal tree
            # V = max_a Q(s, a) for the causal part
            causal_v, _ = torch.max(causal_q_values, dim=-1)
            value = (1 - alpha.squeeze(-1)) * value + alpha.squeeze(-1) * causal_v
            
            diag["causal_engaged"] = causal_mask.sum() > 0

        return dist, value, aggregated_h, delta_tau, diag
