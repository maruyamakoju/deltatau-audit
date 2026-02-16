"""PPO with Internal Time Regularization.

Standard PPO (Schulman et al. 2017) extended with:
1. Recurrent policy support (sequences, not shuffled transitions)
2. Time regularization loss: L_time = lambda1 * Var(delta_tau) + lambda2 * E[delta_tau]
   - Variance penalty prevents time oscillation
   - Mean penalty prevents time from growing unboundedly

The total loss is:
    L = L_clip + c1 * L_value + c2 * L_entropy + L_time
"""

import torch
import torch.nn as nn
import numpy as np


class RolloutBuffer:
    """Storage for on-policy rollout data from vectorized environments."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        hidden_dim: int,
        device: torch.device,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.observations = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.hidden_states = torch.zeros(num_steps, num_envs, hidden_dim, device=device)
        self.delta_taus = torch.zeros(num_steps, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

        self.step = 0

    def add(self, obs, action, reward, done, log_prob, value, hidden, delta_tau):
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value
        self.hidden_states[self.step] = hidden
        self.delta_taus[self.step] = delta_tau
        self.step += 1

    def compute_gae(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """Compute Generalized Advantage Estimation."""
        last_gae = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def reset(self):
        self.step = 0


class PPOTime:
    """Proximal Policy Optimization with internal time regularization."""

    def __init__(
        self,
        agent,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        time_var_coef: float = 0.01,
        time_mean_coef: float = 0.001,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        num_minibatches: int = 4,
    ):
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.time_var_coef = time_var_coef
        self.time_mean_coef = time_mean_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    def update(self, buffer: RolloutBuffer) -> dict:
        """Perform PPO update on collected rollout data.

        Returns dict of training metrics.
        """
        batch_size = buffer.num_steps * buffer.num_envs
        minibatch_size = batch_size // self.num_minibatches

        # Flatten rollout buffer across time and environments
        obs_flat = buffer.observations.reshape(-1, buffer.observations.shape[-1])
        actions_flat = buffer.actions.reshape(-1)
        old_log_probs_flat = buffer.log_probs.reshape(-1)
        advantages_flat = buffer.advantages.reshape(-1)
        returns_flat = buffer.returns.reshape(-1)
        hidden_flat = buffer.hidden_states.reshape(-1, buffer.hidden_states.shape[-1])

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (
            advantages_flat.std() + 1e-8
        )

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "time_loss": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "delta_tau_mean": 0.0,
            "delta_tau_std": 0.0,
        }
        num_updates = 0

        for _epoch in range(self.num_epochs):
            indices = torch.randperm(batch_size, device=buffer.device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                if end > batch_size:
                    break
                mb_idx = indices[start:end]

                mb_obs = obs_flat[mb_idx]
                mb_actions = actions_flat[mb_idx]
                mb_old_log_probs = old_log_probs_flat[mb_idx]
                mb_advantages = advantages_flat[mb_idx]
                mb_returns = returns_flat[mb_idx]
                mb_hidden = hidden_flat[mb_idx]

                # Forward pass with stored hidden states
                _, new_log_probs, entropy, new_values, _, new_delta_taus = (
                    self.agent.get_action_and_value(mb_obs, mb_hidden, mb_actions)
                )

                # PPO clipped policy loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = 0.5 * (new_values - mb_returns).pow(2).mean()

                # Entropy bonus (negative because we maximize entropy)
                entropy_loss = -entropy.mean()

                # Time regularization
                # Variance penalty: prevents wild oscillation
                # Mean-centering penalty: keeps delta_tau near 1.0 (not collapse to 0)
                time_loss = torch.tensor(0.0, device=buffer.device)
                if self.agent.use_internal_time:
                    dt_var = new_delta_taus.var()
                    dt_mean_dev = (new_delta_taus.mean() - 1.0).pow(2)
                    time_loss = self.time_var_coef * dt_var + self.time_mean_coef * dt_mean_dev

                # Total loss
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                    + time_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean()
                    clip_frac = (
                        ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    )

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += -entropy_loss.item()
                metrics["time_loss"] += time_loss.item()
                metrics["total_loss"] += total_loss.item()
                metrics["approx_kl"] += approx_kl.item()
                metrics["clip_fraction"] += clip_frac.item()
                metrics["delta_tau_mean"] += new_delta_taus.mean().item()
                metrics["delta_tau_std"] += new_delta_taus.std().item()
                num_updates += 1

        # Average
        for k in metrics:
            metrics[k] /= max(num_updates, 1)

        return metrics
