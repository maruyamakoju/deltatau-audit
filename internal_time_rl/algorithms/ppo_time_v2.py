"""PPO with Full Temporal Reparameterization (Phase 1 Complete).

Key improvements over ppo_time.py:
1. Time-dependent discount: γ_t = γ^{Δτ_t}  (continuous-time consistent)
2. Time-dependent GAE lambda: λ_t = λ^{Δτ_t}
3. Smoothness regularization: E[(Δτ_t - Δτ_{t-1})^2] instead of Var(Δτ)
   - Smoothness prevents oscillation while allowing state-dependent adaptation
   - Var penalty suppresses useful adaptation
4. Mean-centering: (E[Δτ] - 1)^2 kept to prevent discount-cheating

Supports 4 variants via flags:
- Variant 0 (baseline):  no internal time
- Variant 1 (state only): Δτ modulates hidden state only
- Variant 2 (full reparam): Δτ modulates hidden state + discount + GAE
- Variant 3 (discount only): Δτ modulates discount + GAE only (ablation)
"""

import torch
import torch.nn as nn
import numpy as np


class RolloutBufferV2:
    """Rollout buffer with time-dependent GAE support."""

    def __init__(self, num_steps, num_envs, obs_dim, hidden_dim, device):
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
        # Store external dt for ODE-RNN baseline
        self.external_dts = torch.zeros(num_steps, num_envs, 1, device=device)

        self.step = 0

    def add(self, obs, action, reward, done, log_prob, value, hidden, delta_tau,
            external_dt=None):
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value
        self.hidden_states[self.step] = hidden
        self.delta_taus[self.step] = delta_tau
        if external_dt is not None:
            self.external_dts[self.step] = external_dt
        self.step += 1

    def compute_gae(self, last_value, gamma, gae_lambda, use_time_discount=False):
        """Compute GAE with optional time-dependent discounting.

        When use_time_discount=True:
            γ_t = γ^{Δτ_t}   (more subjective time → more discount)
            λ_t = λ^{Δτ_t}
        When use_time_discount=False:
            Standard fixed γ, λ
        """
        last_gae = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]

            if use_time_discount:
                dt = self.delta_taus[t].squeeze(-1)  # (num_envs,)
                gamma_t = gamma ** dt
                lambda_t = gae_lambda ** dt
            else:
                gamma_t = gamma
                lambda_t = gae_lambda

            delta = self.rewards[t] + gamma_t * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma_t * lambda_t * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def compute_smoothness_loss(self):
        """Compute temporal smoothness of Δτ: E[(Δτ_t - Δτ_{t-1})^2].

        This penalizes rapid changes in internal time while allowing
        state-dependent adaptation (unlike variance penalty).
        """
        if self.num_steps < 2:
            return torch.tensor(0.0, device=self.device)
        dt_diff = self.delta_taus[1:self.step] - self.delta_taus[:self.step - 1]
        return dt_diff.pow(2).mean()

    def reset(self):
        self.step = 0


class PPOTimeV2:
    """PPO with full temporal reparameterization.

    Supports multiple variants for ablation:
    - use_time_discount: whether Δτ affects γ and λ in GAE
    - use_smoothness: whether to use smoothness instead of variance penalty
    """

    def __init__(
        self,
        agent,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        time_smooth_coef=0.02,
        time_mean_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=4,
        num_minibatches=4,
        use_time_discount=False,
        use_smoothness=True,
    ):
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.time_smooth_coef = time_smooth_coef
        self.time_mean_coef = time_mean_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.use_time_discount = use_time_discount
        self.use_smoothness = use_smoothness

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    def update(self, buffer: RolloutBufferV2) -> dict:
        batch_size = buffer.num_steps * buffer.num_envs
        minibatch_size = batch_size // self.num_minibatches

        # Flatten
        obs_flat = buffer.observations.reshape(-1, buffer.observations.shape[-1])
        actions_flat = buffer.actions.reshape(-1)
        old_log_probs_flat = buffer.log_probs.reshape(-1)
        advantages_flat = buffer.advantages.reshape(-1)
        returns_flat = buffer.returns.reshape(-1)
        hidden_flat = buffer.hidden_states.reshape(-1, buffer.hidden_states.shape[-1])
        external_dts_flat = buffer.external_dts.reshape(-1, 1)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (
            advantages_flat.std() + 1e-8
        )

        # Pre-compute smoothness loss from buffer (before shuffling)
        smoothness_loss_val = buffer.compute_smoothness_loss().detach()

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "time_loss": 0.0,
            "smoothness_loss": 0.0,
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

                # Set external dt for ODE-RNN baseline (if applicable)
                if hasattr(self.agent, 'set_external_dt'):
                    self.agent.set_external_dt(external_dts_flat[mb_idx])

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

                # Value loss
                value_loss = 0.5 * (new_values - mb_returns).pow(2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Time regularization
                time_loss = torch.tensor(0.0, device=buffer.device)
                if hasattr(self.agent, 'use_internal_time') and self.agent.use_internal_time:
                    # Mean-centering: (E[Δτ] - 1)^2
                    dt_mean_dev = (new_delta_taus.mean() - 1.0).pow(2)
                    time_loss = self.time_mean_coef * dt_mean_dev

                    # Smoothness loss (from buffer, applied as regularizer)
                    if self.use_smoothness:
                        time_loss = time_loss + self.time_smooth_coef * smoothness_loss_val

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

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += -entropy_loss.item()
                metrics["time_loss"] += time_loss.item()
                metrics["smoothness_loss"] += smoothness_loss_val.item()
                metrics["total_loss"] += total_loss.item()
                metrics["approx_kl"] += approx_kl.item()
                metrics["clip_fraction"] += clip_frac.item()
                metrics["delta_tau_mean"] += new_delta_taus.mean().item()
                metrics["delta_tau_std"] += new_delta_taus.std().item()
                num_updates += 1

        for k in metrics:
            metrics[k] /= max(num_updates, 1)

        return metrics
