"""
The Hunter: Autonomous Temporal Red-Teaming AI.

An RL agent that learns to manipulate the environment's timing (delta_tau)
to minimize the target agent's cumulative reward.
Used for discovering non-trivial temporal failure modes.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class HunterAgent(nn.Module):
    """
    The Adversarial Agent.
    Input: target_obs, target_hidden_state
    Output: speed_multiplier choice
    Reward: -target_reward
    """

    def __init__(self, target_obs_dim: int, target_hidden_dim: int, n_speeds: int = 5):
        super().__init__()
        self.n_speeds = n_speeds

        # State space: concatenation of what the target sees and its internal state
        self.net = nn.Sequential(
            nn.Linear(target_obs_dim + target_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_speeds),
        )

    def select_attack(self, obs: torch.Tensor, hidden: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Choose a speed multiplier to apply.
        Returns: (speed_index, log_prob)
        """
        # Ensure flat hidden state
        if isinstance(hidden, (list, tuple)):
            # Handle multi-agent or complex hidden states
            hidden_flat = torch.cat([h.flatten() for h in hidden])
        else:
            hidden_flat = hidden.flatten()

        combined = torch.cat([obs.flatten(), hidden_flat])
        logits = self.net(combined.unsqueeze(0))
        dist = Categorical(logits=logits)

        speed_idx = dist.sample()
        return speed_idx.item(), dist.log_prob(speed_idx)


class HunterTrainer:
    """Trains the Hunter agent via simple Policy Gradient."""

    def __init__(self, hunter: HunterAgent, lr: float = 1e-3):
        self.hunter = hunter
        self.optimizer = torch.optim.Adam(hunter.parameters(), lr=lr)

    def update(self, log_probs: List[torch.Tensor], target_rewards: List[float]):
        """
        Update hunter to maximize -target_rewards.
        """
        # Hunter reward is negative of target reward
        hunter_rewards = [-r for r in target_rewards]

        # Compute discounted returns (simple)
        returns = []
        G = 0
        for r in reversed(hunter_rewards):
            G = r + 0.99 * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for lp, G in zip(log_probs, returns):
            loss -= lp * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
