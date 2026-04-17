"""PPO with Subjective Resolution Regularization.

Extends standard PPO with:
1. Subjective Time Regularization (Variance & Mean of delta_tau).
2. Pondering Cost (Penalty on expected pondering steps).
3. Adaptive Computation Time (ACT) halting probability training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .ppo_time import PPOTime, RolloutBuffer

class PPOResolution(PPOTime):
    """PPO for SubjectiveResolutionAgent with multi-objective regularization."""

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
        ponder_coef: float = 0.005,  # Penalty per expected pondering step
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        num_minibatches: int = 4,
    ):
        super().__init__(
            agent, lr, gamma, gae_lambda, clip_epsilon, value_coef, 
            entropy_coef, time_var_coef, time_mean_coef, max_grad_norm, 
            num_epochs, num_minibatches
        )
        self.ponder_coef = ponder_coef

    def update(self, buffer, optimizer):
        """Perform PPO update with resolution and time penalties."""
        inds = np.arange(buffer.num_steps * buffer.num_envs)
        losses = []

        for _ in range(self.num_epochs):
            np.random.shuffle(inds)
            for start in range(0, len(inds), len(inds) // self.num_minibatches):
                end = start + len(inds) // self.num_minibatches
                mb_inds = inds[start:end]

                # Flatten and slice batch
                b_obs = buffer.observations.reshape(-1, buffer.observations.shape[-1])[mb_inds]
                if buffer.discrete_actions:
                    b_actions = buffer.actions.reshape(-1)[mb_inds]
                else:
                    b_actions = buffer.actions.reshape(-1, buffer.action_dim)[mb_inds]
                b_log_probs = buffer.log_probs.reshape(-1)[mb_inds]
                b_advantages = buffer.advantages.reshape(-1)[mb_inds]
                b_returns = buffer.returns.reshape(-1)[mb_inds]
                b_values = buffer.values.reshape(-1)[mb_inds]
                b_hidden = buffer.hidden_states.reshape(-1, buffer.hidden_states.shape[-1])[mb_inds]

                # Forward pass with gradient
                # SubjectiveResolutionAgent.forward returns (dist, value, hidden_new, dt, diag)
                dist, value, _, dt, diag = self.agent.forward(b_obs, b_hidden)

                # 1. PPO Clipping Loss
                new_log_prob = dist.log_prob(b_actions)
                if not self.agent.discrete_actions:
                    new_log_prob = new_log_prob.sum(-1)
                
                ratio = torch.exp(new_log_prob - b_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 2. Value Loss
                value_loss = F.mse_loss(value, b_returns)

                # 3. Entropy Loss
                entropy_loss = dist.entropy().mean()

                # 4. Subjective Time Regularization (from PPOTime)
                # dt is (B, 1)
                time_var_loss = torch.var(dt)
                time_mean_loss = torch.mean(torch.abs(dt - 1.0))

                # 5. Pondering Cost (Dynamic Resolution Penalty)
                # expected_steps is (B,)
                expected_steps = diag["expected_steps"]
                ponder_loss = expected_steps.mean()

                # Total Loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                    + self.time_var_coef * time_var_loss
                    + self.time_mean_coef * time_mean_loss
                    + self.ponder_coef * ponder_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                optimizer.step()
                losses.append(loss.item())

        return np.mean(losses)
