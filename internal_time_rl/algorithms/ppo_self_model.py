"""PPO with Self-Model Training (Phase 2).

Extends PPO with:
1. Internal time regularization (from Phase 1)
2. Self-model prediction loss: trains the agent to predict its own state
3. Prediction error correlation logging

Total loss:
    L = L_clip + c1*L_value + c2*L_entropy + L_time + c3*L_self_model
"""

import torch
import torch.nn as nn
import numpy as np

from .ppo_time import RolloutBuffer


class PPOSelfModel:
    """PPO with self-model training."""

    def __init__(
        self,
        agent,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        time_var_coef: float = 0.005,
        time_mean_coef: float = 0.01,
        self_model_coef: float = 0.1,
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
        self.self_model_coef = self_model_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    def update(self, buffer: RolloutBuffer) -> dict:
        batch_size = buffer.num_steps * buffer.num_envs
        minibatch_size = batch_size // self.num_minibatches

        obs_flat = buffer.observations.reshape(-1, buffer.observations.shape[-1])
        actions_flat = buffer.actions.reshape(-1)
        old_log_probs_flat = buffer.log_probs.reshape(-1)
        advantages_flat = buffer.advantages.reshape(-1)
        returns_flat = buffer.returns.reshape(-1)
        hidden_flat = buffer.hidden_states.reshape(-1, buffer.hidden_states.shape[-1])

        # For self-model: create pairs of (h_t, h_{t+1})
        # We use consecutive hidden states from the buffer
        h_current = buffer.hidden_states[:-1].reshape(-1, buffer.hidden_states.shape[-1])
        h_next = buffer.hidden_states[1:].reshape(-1, buffer.hidden_states.shape[-1])

        advantages_flat = (advantages_flat - advantages_flat.mean()) / (
            advantages_flat.std() + 1e-8
        )

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "time_loss": 0.0,
            "self_model_loss": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "delta_tau_mean": 0.0,
            "delta_tau_std": 0.0,
            "pred_error_mean": 0.0,
        }
        num_updates = 0

        for _epoch in range(self.num_epochs):
            indices = torch.randperm(batch_size, device=buffer.device)
            sm_indices = torch.randperm(h_current.shape[0], device=buffer.device)

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

                # Time regularization (center around 1.0)
                dt_var = new_delta_taus.var()
                dt_mean_dev = (new_delta_taus.mean() - 1.0).pow(2)
                time_loss = self.time_var_coef * dt_var + self.time_mean_coef * dt_mean_dev

                # Self-model prediction loss
                sm_end = min(start + minibatch_size, h_current.shape[0])
                sm_mb_idx = sm_indices[start:sm_end]
                if len(sm_mb_idx) > 0:
                    sm_h_curr = h_current[sm_mb_idx]
                    sm_h_next = h_next[sm_mb_idx]
                    self_model_loss = self.agent.compute_self_model_loss(
                        sm_h_curr, sm_h_next
                    )
                else:
                    self_model_loss = torch.tensor(0.0, device=buffer.device)

                # Total loss
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                    + time_loss
                    + self.self_model_coef * self_model_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean()
                    clip_frac = (
                        ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    )

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += -entropy_loss.item()
                metrics["time_loss"] += time_loss.item()
                metrics["self_model_loss"] += self_model_loss.item()
                metrics["total_loss"] += total_loss.item()
                metrics["approx_kl"] += approx_kl.item()
                metrics["clip_fraction"] += clip_frac.item()
                metrics["delta_tau_mean"] += new_delta_taus.mean().item()
                metrics["delta_tau_std"] += new_delta_taus.std().item()
                num_updates += 1

        for k in metrics:
            metrics[k] /= max(num_updates, 1)

        return metrics
