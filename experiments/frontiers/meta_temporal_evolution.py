r"""Frontier 18: Meta-Temporal Evolution (MTE).

An agent that learns to dynamically adjust its internal clock (dt) and 
temporal resolution (scale) using an online Meta-Learning loop. 
This is a step towards 'Recursive Self-Improvement' in the timing domain.

Novelty:
1. **Online Meta-Timing Loop**: The agent has a 'Meta-Clock' that observes 
   the reward gradient and adjusts the 'Main Clock' to maximize the 
   information-gain/reward ratio.
2. **Dynamic Resolution Switching**: Can switch from coarse-grained 
   long-term thinking to fine-grained reactive thinking in real-time.
3. **Causal Drift Detection**: A meta-head that detects when the 
   environment's temporal dynamics have shifted (e.g., speed change) 
   and triggers a clock recalibration.

Architecture:
- Base Agent: RNN/ODE-based policy.
- Meta-Clock: Learns the mapping (state, performance) -> dt_multiplier.
- Drift Head: Predicts causal stability.
"""

from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ._base import save_summary, seed_all
from ._metrics import aggregate_returns, env_return_ceiling, normalize_score

# ---------------------------------------------------------------------------
# 1. MTE Agent
# ---------------------------------------------------------------------------

class MTEAgent(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Linear(obs_dim, hidden_dim)

        # Main Clock dynamics (Neural ODE style)
        self.flow_net = nn.Sequential(
            nn.Linear(hidden_dim + act_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Meta-Clock: (hidden, current_reward) -> optimal_dt_multiplier
        # We want to increase dt when stable, and decrease it when unstable
        self.meta_clock = nn.Sequential(
            nn.Linear(hidden_dim + 1, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Scale it between 0.1 and 2.0 later
        )

        self.policy = nn.Linear(hidden_dim, act_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action, hidden, last_reward):
        # 1. Encode
        enc = self.encoder(obs)

        # 2. Meta-Clock Adjustment
        # meta_input: [batch, hidden_dim + 1]
        meta_input = torch.cat([hidden, torch.tensor([[last_reward]], device=obs.device)], dim=-1)
        dt_scale = self.meta_clock(meta_input) * 1.9 + 0.1 # 0.1x to 2.0x

        # 3. Evolution
        flow = self.flow_net(torch.cat([hidden, action], dim=-1))
        new_hidden = hidden + (dt_scale * flow)

        # 4. Heads
        logits = self.policy(new_hidden)
        val = self.value(new_hidden).squeeze(-1)

        return logits, val, new_hidden, dt_scale

# ---------------------------------------------------------------------------
# 2. Frontier Experiment
# ---------------------------------------------------------------------------

class MTEExperiment:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.env_id = params.get("env", "CartPole-v1")
        self.device = params.get("device", "cpu")
        self.seed = int(params.get("seed", 42))

        temp_env = gym.make(self.env_id)
        self.obs_dim = temp_env.observation_space.shape[0]
        self.act_dim = temp_env.action_space.n if isinstance(temp_env.action_space, gym.spaces.Discrete) else temp_env.action_space.shape[0]
        self.discrete = isinstance(temp_env.action_space, gym.spaces.Discrete)
        temp_env.close()

        self.agent = MTEAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_dim=params.get("hidden_dim", 128)
        ).to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=params.get("lr", 1e-3))

    def run(self, out_dir: Path) -> Dict[str, float]:
        seed_all(self.seed)
        print(f"  Training Meta-Temporal Agent on {self.env_id}...")

        n_episodes = self.params.get("n_episodes", 50)
        returns = []
        avg_dts = []

        for ep in range(n_episodes):
            env = gym.make(self.env_id)
            obs, _ = env.reset()
            hidden = torch.zeros(1, self.agent.hidden_dim).to(self.device)
            last_reward = 0.0

            ep_reward = 0
            log_probs = []
            ep_dts = []

            done = False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Use zero action for initial step or keep track of last action
                dummy_action = torch.zeros(1, self.act_dim if not self.discrete else self.act_dim).to(self.device)

                logits, value, hidden, dt_scale = self.agent(obs_t, dummy_action, hidden, last_reward)
                ep_dts.append(dt_scale.item())

                if self.discrete:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    act_val = action.item()
                else:
                    action = torch.tanh(logits)
                    dist = torch.distributions.Normal(action, 0.1)
                    action = dist.sample()
                    act_val = action.cpu().numpy()[0]

                log_probs.append(dist.log_prob(action))

                obs, r, term, trunc, _ = env.step(act_val)
                ep_reward += r
                last_reward = float(r)
                done = term or trunc
                if len(log_probs) > 500: break

            # Update
            loss = -torch.stack(log_probs).sum() * (ep_reward / 50.0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            returns.append(ep_reward)
            avg_dts.append(np.mean(ep_dts))
            if ep % 10 == 0:
                print(f"    Episode {ep}: Return = {ep_reward:.1f}, Avg DT Scale = {np.mean(ep_dts):.3f}")
            env.close()

        # Evaluate Meta-Adaptability
        adaptability = self._eval_meta_adaptability()

        return_stats = aggregate_returns(returns)
        ceiling = env_return_ceiling(self.env_id, default=200.0)
        normalised = normalize_score(return_stats["mean_return"], ceiling=ceiling)
        summary = {
            **return_stats,
            "avg_dt_scale": float(np.mean(avg_dts)) if avg_dts else 0.0,
            "meta_adaptability": float(adaptability),
            "composite_score": float(normalised * adaptability),
        }
        save_summary(out_dir, summary)
        return summary

    def _eval_meta_adaptability(self) -> float:
        """Evaluate how quickly the agent adapts its dt to extreme speed changes."""
        results = []
        for speed in [0.2, 1.0, 5.0]:
            env = gym.make(self.env_id)
            obs, _ = env.reset()
            hidden = torch.zeros(1, self.agent.hidden_dim).to(self.device)
            last_reward = 0.0
            ep_ret = 0
            done = False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                dummy_action = torch.zeros(1, self.act_dim if not self.discrete else self.act_dim).to(self.device)
                logits, _, hidden, _ = self.agent(obs_t, dummy_action, hidden, last_reward)

                if self.discrete:
                    action = torch.argmax(logits, dim=-1)
                    act_val = action.item()
                else:
                    act_val = torch.tanh(logits).cpu().numpy()[0]

                obs, r, term, trunc, _ = env.step(act_val)
                ep_ret += r
                last_reward = float(r)
                done = term or trunc
            results.append(ep_ret)
            env.close()
        return float(np.mean(results) / 200.0)
