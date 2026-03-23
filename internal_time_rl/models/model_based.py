"""
Model-Based Internal Time Agent.

Uses a Temporal World Model to plan actions by simulating future 
trajectories with potential latency spikes.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .world_model import TemporalWorldModel

class PredictiveAgent(nn.Module):
    """
    An agent that 'dreams' about future timing.
    It uses the Temporal World Model to evaluate action sequences 
    under predicted environment jitter.
    """
    def __init__(self, obs_dim: int, act_dim: int, world_model: TemporalWorldModel):
        super().__init__()
        self.world_model = world_model
        self.act_dim = act_dim
        
        # Policy head (e.g. for CEM planning or as a learned prior)
        self.actor = nn.Sequential(
            nn.Linear(world_model.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def plan(self, obs: torch.Tensor, horizon: int = 5, n_samples: int = 10) -> torch.Tensor:
        """
        Simple Random Shooting planner using the world model.
        Evaluates future reward AND predicted future latency.
        """
        batch_size = obs.shape[0]
        z_init = self.world_model.get_latent(obs) # (B, latent)
        
        # Expand for samples
        z = z_init.repeat_interleave(n_samples, dim=0) # (B*N, latent)
        
        best_actions = None
        best_rewards = torch.full((batch_size,), -float('inf'), device=obs.device)
        
        # Sample action sequences
        # (B*N, H, act_dim)
        action_seqs = torch.randn(batch_size * n_samples, horizon, self.act_dim, device=obs.device)
        
        total_sample_rewards = torch.zeros(batch_size * n_samples, device=obs.device)
        
        curr_z = z
        for t in range(horizon):
            a = action_seqs[:, t, :]
            # World model predict s', r, dt
            obs_pred, r_pred, dt_pred = self.world_model(curr_z, a)
            
            # Penalize sequences with high predicted latency/uncertainty
            # (Risk-averse temporal planning)
            effective_reward = r_pred.squeeze(-1) - 0.1 * dt_pred.squeeze(-1)
            
            total_sample_rewards += (0.99 ** t) * effective_reward
            
            # Update latent state (this is a simplification, usually you update z directly)
            curr_z = self.world_model.dynamics(torch.cat([curr_z, a], dim=-1))
            
        # Select best sequences
        # ... logic to find best action for each batch element
        return action_seqs[:, 0, :] # Placeholder: return first action of some sequence

    def forward(self, obs: torch.Tensor):
        z = self.world_model.get_latent(obs)
        return self.actor(z)
