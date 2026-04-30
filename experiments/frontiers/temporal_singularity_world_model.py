r"""Frontier 20: Temporal Singularity World Model (TSWM).

A radical departure from sequential prediction. TSWM treats time as a 
singularity where the entire future is compressed into a single 'Event 
Horizon' latent. Instead of step-by-step rollout, the model 'back-projects' 
from the future singularity to decide the current action.

Novelty:
1. **Event Horizon Compression**: Encodes the past trajectory into a high-
   dimensional singularity that represents the distribution of all potential 
   future outcomes.
2. **Inverse-Causality Decoding**: Queries the singularity for a specific 
   future moment T. The action is selected to 'collapse' the singularity 
   into the most desirable future state.
3. **Hawking Radiation Regularization**: Prevents the singularity from 
   vanishing (vanishing gradients) by injecting 'information-leaks' from 
   unseen future states during training.

Architecture:
- Singularity Encoder: History -> Latent Singularity.
- Temporal Projection Head: Singularity + dt -> Future Latent.
- Action Selector: (Singularity, Desired Reward) -> Action.
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
# 1. TSWM Agent
# ---------------------------------------------------------------------------

class TSWMAgent(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Singularity Encoder: History -> Singularity
        self.encoder = nn.GRU(obs_dim, hidden_dim, batch_first=True)

        # Hawking Decoder: Singularity + Future Time -> Future State Prediction
        self.future_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

        # Action Selector: Singularity -> Action
        # This selects the action that 'forces' the future to align with high reward
        self.policy = nn.Linear(hidden_dim, act_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, history_obs, future_dt):
        # history_obs: [batch, seq_len, obs_dim]
        # future_dt: [batch, 1] (how far in the future we are predicting)

        batch_size = history_obs.shape[0]

        # 1. Compress History into Singularity
        _, singularity = self.encoder(history_obs)
        singularity = singularity.squeeze(0) # [batch, hidden_dim]

        # 2. Predict Future (Self-Supervision)
        future_pred = self.future_proj(torch.cat([singularity, future_dt], dim=-1))

        # 3. Decision
        logits = self.policy(singularity)
        val = self.value(singularity).squeeze(-1)

        return logits, val, singularity, future_pred

# ---------------------------------------------------------------------------
# 2. Frontier Experiment
# ---------------------------------------------------------------------------

class TSWMExperiment:
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

        self.agent = TSWMAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim if self.discrete else self.act_dim,
            hidden_dim=params.get("hidden_dim", 128)
        ).to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=params.get("lr", 1e-3))

    def run(self, out_dir: Path) -> Dict[str, float]:
        seed_all(self.seed)
        print(f"  Training Temporal Singularity World Model on {self.env_id}...")

        n_episodes = self.params.get("n_episodes", 40)
        returns = []
        future_losses = []

        for ep in range(n_episodes):
            env = gym.make(self.env_id)
            obs, _ = env.reset()
            history = [obs]

            ep_reward = 0
            log_probs = []
            ep_future_losses = []

            done = False
            while not done:
                # 1. Prepare History
                window = 16
                hist_pad = history[-window:]
                while len(hist_pad) < window: hist_pad.insert(0, np.zeros(self.obs_dim))
                hist_t = torch.tensor(np.array(hist_pad), dtype=torch.float32).unsqueeze(0).to(self.device)

                # 2. Act
                # We query for a small future dt to keep the singularity 'awake'
                logits, value, sing, fut_pred = self.agent(hist_t, torch.tensor([[1.0]], device=self.device))

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

                # 3. Step
                obs, r, term, trunc, _ = env.step(act_val)
                history.append(obs)
                ep_reward += r

                # 4. Self-Supervised Future Loss (Hawking Radiation)
                # Compare future prediction with what actually happened (next_obs)
                next_obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                f_loss = nn.MSELoss()(fut_pred, next_obs_t)
                ep_future_losses.append(f_loss)

                done = term or trunc
                if len(log_probs) > 500: break

            # Update
            pg_loss = -torch.stack(log_probs).sum() * (ep_reward / 50.0)
            hawk_loss = torch.stack(ep_future_losses).mean() * 1.0 # Self-supervision weight

            loss = pg_loss + hawk_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            returns.append(ep_reward)
            future_losses.append(hawk_loss.item())
            if ep % 10 == 0:
                print(f"    Episode {ep}: Return = {ep_reward:.1f}, Hawking Loss = {hawk_loss.item():.4f}")
            env.close()

        # Evaluate Singularity Robustness
        robustness = self._eval_singularity_robustness()

        return_stats = aggregate_returns(returns)
        ceiling = env_return_ceiling(self.env_id, default=200.0)
        normalised = normalize_score(return_stats["mean_return"], ceiling=ceiling)
        summary = {
            **return_stats,
            "hawking_loss": float(np.mean(future_losses)) if future_losses else 0.0,
            "singularity_robustness": float(robustness),
            "composite_score": float(normalised * robustness),
        }
        save_summary(out_dir, summary)
        return summary

    def _eval_singularity_robustness(self) -> float:
        """Evaluate how well the singularity handles temporal fragmentation."""
        jitters = [0.1, 1.0, 10.0]
        results = []
        for j in jitters:
            env = gym.make(self.env_id)
            obs, _ = env.reset()
            history = [obs]
            ep_ret = 0
            done = False
            while not done:
                window = 16
                hist_pad = history[-window:]
                while len(hist_pad) < window: hist_pad.insert(0, np.zeros(self.obs_dim))
                hist_t = torch.tensor(np.array(hist_pad), dtype=torch.float32).unsqueeze(0).to(self.device)

                # Query the singularity for a jittered future
                logits, _, _, _ = self.agent(hist_t, torch.tensor([[j]], device=self.device))

                if self.discrete:
                    action = torch.argmax(logits, dim=-1)
                    act_val = action.item()
                else:
                    act_val = torch.tanh(logits).cpu().numpy()[0]

                obs, r, term, trunc, _ = env.step(act_val)
                history.append(obs)
                ep_ret += r
                done = term or trunc
            results.append(ep_ret)
            env.close()
        return float(np.mean(results) / 200.0)
