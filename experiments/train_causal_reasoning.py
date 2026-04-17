"""Axis 10: Joint Training of Policy and Causal World Model.

Trains the CausalResolutionAgent to both:
1. Act effectively (PPO).
2. Predict future latent states based on actions (Causal Transition).

This enables System 2 thinking: simulating counterfactuals during 
pondering spikes.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
from typing import List, Dict

from internal_time_rl.envs.variable_frequency import VariableFrequencyChainEnv
from internal_time_rl.envs.vec_env import SyncVectorEnv
from internal_time_rl.models.causal_reasoning import CausalResolutionAgent
from internal_time_rl.algorithms.ppo_resolution import PPOResolution, RolloutBuffer

class PPOCausal(PPOResolution):
    """PPO extended with Causal Transition Loss for World Model training."""
    
    def __init__(self, agent, causal_coef: float = 1.0, **kwargs):
        super().__init__(agent, **kwargs)
        self.causal_coef = causal_coef

    def update(self, buffer, optimizer):
        inds = np.arange(buffer.num_steps * buffer.num_envs)
        losses = []

        for _ in range(self.num_epochs):
            np.random.shuffle(inds)
            for start in range(0, len(inds), len(inds) // self.num_minibatches):
                end = start + len(inds) // self.num_minibatches
                mb_inds = inds[start:end]

                # Current step data
                b_obs = buffer.observations.reshape(-1, buffer.observations.shape[-1])[mb_inds]
                if buffer.discrete_actions:
                    b_actions = buffer.actions.reshape(-1)[mb_inds]
                else:
                    b_actions = buffer.actions.reshape(-1, buffer.action_dim)[mb_inds]
                b_log_probs = buffer.log_probs.reshape(-1)[mb_inds]
                b_advantages = buffer.advantages.reshape(-1)[mb_inds]
                b_returns = buffer.returns.reshape(-1)[mb_inds]
                b_hidden = buffer.hidden_states.reshape(-1, buffer.hidden_states.shape[-1])[mb_inds]

                # Next step hidden state (target for causal transition)
                # We need to compute the causal loss: predicting next_hidden from (current_hidden, action)
                # For this, we'll use the 'next' entries from the buffer if available,
                # or re-run the forward pass.

                # Forward Pass
                dist, value, aggregated_h, dt, diag = self.agent.forward(b_obs, b_hidden)

                # 1. Standard PPO Losses (Policy, Value, Entropy, Time, Ponder)
                new_log_prob = dist.log_prob(b_actions)
                if not buffer.discrete_actions:
                    new_log_prob = new_log_prob.sum(-1)
                ratio = torch.exp(new_log_prob - b_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, b_returns)
                entropy_loss = dist.entropy().mean()
                time_loss = torch.var(dt) + torch.mean(torch.abs(dt - 1.0))
                ponder_loss = diag["expected_steps"].mean()

                # 2. Causal Transition Loss (Axis 10 Core)
                # We want the causal model to predict the *actual* resulting hidden state
                # after the ACT unroll.
                # (Simplified: predict the aggregated_h of the NEXT step)
                # In a real implementation, we'd slice mb_inds + 1.
                # For this prototype, we'll train it to be self-consistent.
                if buffer.discrete_actions:
                    act_features = F.one_hot(b_actions, num_classes=self.agent.act_dim).float()
                else:
                    act_features = b_actions
                predicted_h_next = self.agent.causal_transition(torch.cat([b_hidden, act_features], dim=-1))
                # Target: the actual next hidden state (we'll approximate with aggregated_h for now)
                causal_loss = F.mse_loss(predicted_h_next, aggregated_h.detach())

                # Total Loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                    + self.time_var_coef * time_loss
                    + self.ponder_coef * ponder_loss
                    + self.causal_coef * causal_loss
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        return np.mean(losses)

def train_causal():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Axis 10 Causal Training on {device}...")
    
    # 1. Environment (Variable speed to trigger pondering)
    def env_factory():
        return VariableFrequencyChainEnv(chain_length=20, train_speeds=(1, 2, 3), speed_in_obs=False)
    
    num_envs = 4
    envs = SyncVectorEnv([env_factory for _ in range(num_envs)])
    obs_dim = envs.observation_space.shape[0]
    act_dim = 2
    
    # 2. Agent & Optimizer
    agent = CausalResolutionAgent(obs_dim, act_dim, max_ponder_base=4, tau_scale=2.0).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    ppo = PPOCausal(agent, causal_coef=1.0)
    
    num_steps = 128
    buffer = RolloutBuffer(num_steps, num_envs, obs_dim, agent.hidden_dim, device)
    
    total_timesteps = 100_000
    global_step = 0
    obs = envs.reset()
    hidden = agent.get_initial_hidden(num_envs, device)
    
    pbar = tqdm.tqdm(total=total_timesteps, desc="[Causal Training]")
    
    while global_step < total_timesteps:
        # Rollout
        for _ in range(num_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_prob, _, value, hidden_new, dt, diag = agent.get_action_and_value(obs_t, hidden)
            
            next_obs, reward, done, _ = envs.step(action.cpu().numpy())
            
            buffer.add(obs_t, action, torch.as_tensor(reward, device=device), 
                       torch.as_tensor(done, device=device), log_prob, value, 
                       hidden, dt)
            
            obs = next_obs
            hidden = hidden_new
            for i, d in enumerate(done):
                if d > 0.5: hidden[i] = 0.0
            
            global_step += num_envs
            pbar.update(num_envs)

        # Update
        obs_t_last = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, _, _, last_value, _, _, _ = agent.get_action_and_value(obs_t_last, hidden)
        buffer.compute_gae(last_value, ppo.gamma, ppo.gae_lambda)
        ppo.update(buffer, optimizer)
        buffer.reset()
        
    pbar.close()
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(agent.state_dict(), "checkpoints/agent_causal_axis10.pt")
    print("Training Complete. Model saved to checkpoints/agent_causal_axis10.pt")

if __name__ == "__main__":
    train_causal()
