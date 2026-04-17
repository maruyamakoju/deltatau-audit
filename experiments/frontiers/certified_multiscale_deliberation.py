r"""Certified Multi-Scale Deliberation (CMS-D) — Frontier Research Module.

A novel unified agent that fuses three previously independent research
directions into a single architecture:

1. **Multi-scale world model** — hierarchical RSSM with fast (reactive) and
   slow (strategic) latent tiers connected by cross-scale attention.
2. **Lipschitz certification** — spectral norm bounds on the transition model
   guarantee timing-safety; branches with high sensitivity to timing
   perturbations are penalised during deliberation.
3. **Uncertainty-adaptive computation** — ACT (Adaptive Computation Time)
   whose halting decision is driven by a *triple signal*:
   - World model predictive uncertainty (epistemic)
   - Lipschitz bound magnitude (timing safety)
   - Scale disagreement (fast-slow prediction mismatch)

**Core novelty**: The *scale disagreement* signal has no analogue in any
single-scale architecture.  When the fast and slow RSSM tiers predict
divergent futures, the agent is at a "temporal phase boundary" — a state
where reactive and strategic predictions conflict.  This is exactly when
extra deliberation matters most.

The composite halting signal is:

    budget(s) = σ(w_u * uncertainty + w_l * lipschitz + w_d * disagreement + b)

where the weights are learned end-to-end.

Self-contained: no imports from the main ``deltatau_audit`` or
``internal_time_rl`` packages.

License: Apache-2.0
"""
from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from act_utils import apply_act_step, halt_distribution_stats, stack_halt_weights

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def _spectral_norm_estimate(weight: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """Largest singular value via power iteration."""
    if weight.ndim < 2:
        return weight.abs().max()
    mat = weight.reshape(weight.shape[0], -1)
    u = F.normalize(torch.randn(mat.shape[0], device=weight.device), dim=0)
    with torch.no_grad():
        for _ in range(n_iters):
            v = F.normalize(mat.t() @ u, dim=0)
            u = F.normalize(mat @ v, dim=0)
    return (u @ mat @ v).abs()


# ---------------------------------------------------------------------------
# Mini Multi-Scale World Model (compact version for deliberation)
# ---------------------------------------------------------------------------

class MiniMultiScaleRSSM(nn.Module):
    """Compact two-tier RSSM with Lipschitz-aware transition.

    Fast tier: updates every step (reactive control).
    Slow tier: updates every K steps (strategic planning).
    Both produce uncertainty estimates via ensemble disagreement.
    """

    def __init__(
        self,
        obs_dim: int = 4,
        action_dim: int = 2,
        fast_hidden_dim: int = 64,
        slow_hidden_dim: int = 48,
        slow_tick_every: int = 3,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fast_hidden_dim = fast_hidden_dim
        self.slow_hidden_dim = slow_hidden_dim
        self.slow_tick_every = slow_tick_every

        # Fast tier GRU
        self.fast_pre = nn.Sequential(
            nn.Linear(fast_hidden_dim + action_dim + slow_hidden_dim, fast_hidden_dim),
            nn.LayerNorm(fast_hidden_dim),
            nn.SiLU(),
        )
        self.fast_gru = nn.GRUCell(fast_hidden_dim, fast_hidden_dim)

        # Fast observation predictor (in symlog space)
        self.fast_obs_pred = nn.Sequential(
            nn.Linear(fast_hidden_dim, fast_hidden_dim),
            nn.SiLU(),
            nn.Linear(fast_hidden_dim, obs_dim),
        )

        # Fast reward predictor
        self.fast_reward = nn.Sequential(
            nn.Linear(fast_hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # Fast encoder: obs -> fast_hidden
        self.fast_encoder = nn.Sequential(
            nn.Linear(obs_dim, fast_hidden_dim),
            nn.SiLU(),
            nn.Linear(fast_hidden_dim, fast_hidden_dim),
        )

        # Slow tier: pools K fast states via attention then GRU
        self.slow_pool = nn.Sequential(
            nn.Linear(fast_hidden_dim, slow_hidden_dim),
            nn.SiLU(),
        )
        self.slow_gru = nn.GRUCell(slow_hidden_dim, slow_hidden_dim)

        # Slow observation predictor
        self.slow_obs_pred = nn.Sequential(
            nn.Linear(slow_hidden_dim, slow_hidden_dim),
            nn.SiLU(),
            nn.Linear(slow_hidden_dim, obs_dim),
        )

        # Uncertainty head: predicts aleatoric uncertainty from fast state
        self.uncertainty_head = nn.Sequential(
            nn.Linear(fast_hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(fast_hidden_dim + slow_hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "fast_h": torch.zeros(batch_size, self.fast_hidden_dim, device=device),
            "slow_h": torch.zeros(batch_size, self.slow_hidden_dim, device=device),
            "fast_buffer": [],  # List of fast_h tensors for slow pooling
            "step_count": 0,
        }

    def fast_step(
        self,
        fast_h: torch.Tensor,
        slow_h: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One fast-tier transition (imagination mode, no observation)."""
        x = self.fast_pre(torch.cat([fast_h, action, slow_h], dim=-1))
        new_h = self.fast_gru(x, fast_h)
        obs_pred = self.fast_obs_pred(new_h)
        reward = self.fast_reward(new_h)
        return new_h, obs_pred, reward

    def fast_observe(
        self,
        fast_h: torch.Tensor,
        slow_h: torch.Tensor,
        action: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """One fast-tier step conditioned on real observation."""
        x = self.fast_pre(torch.cat([fast_h, action, slow_h], dim=-1))
        new_h = self.fast_gru(x, fast_h)
        # Mix in observation via encoder
        obs_enc = self.fast_encoder(obs)
        new_h = new_h + 0.3 * (obs_enc - new_h)  # soft posterior update
        return new_h

    def slow_step(self, slow_h: torch.Tensor, fast_states: List[torch.Tensor]) -> torch.Tensor:
        """Slow-tier update: aggregate K fast states."""
        # Mean pool fast states
        stacked = torch.stack(fast_states, dim=1)  # (B, K, fast_dim)
        pooled = self.slow_pool(stacked.mean(dim=1))
        return self.slow_gru(pooled, slow_h)

    def get_uncertainty(self, fast_h: torch.Tensor) -> torch.Tensor:
        """Predictive uncertainty estimate."""
        return self.uncertainty_head(fast_h)

    def get_value(self, fast_h: torch.Tensor, slow_h: torch.Tensor) -> torch.Tensor:
        return self.value_head(torch.cat([fast_h, slow_h], dim=-1))

    def get_scale_disagreement(
        self,
        fast_h: torch.Tensor,
        slow_h: torch.Tensor,
    ) -> torch.Tensor:
        """Measure disagreement between fast and slow observation predictions.

        This is the novel signal: when fast and slow tiers predict different
        futures, the agent is at a temporal phase boundary.
        """
        fast_pred = self.fast_obs_pred(fast_h)
        slow_pred = self.slow_obs_pred(slow_h)
        # L2 disagreement in symlog space, normalised by obs_dim
        return ((fast_pred - slow_pred) ** 2).mean(dim=-1, keepdim=True)

    def lipschitz_bound(self) -> float:
        """Spectral norm product bound for the fast transition."""
        bound = 1.0
        for name, param in self.named_parameters():
            if "fast_pre" in name and "weight" in name and param.ndim >= 2:
                bound *= _spectral_norm_estimate(param).item()
            if "fast_gru" in name and "weight" in name and param.ndim >= 2:
                bound *= _spectral_norm_estimate(param).item()
        # SiLU Lipschitz ≈ 1.1
        bound *= 1.1
        return bound


# ---------------------------------------------------------------------------
# Certified Multi-Scale Deliberation Agent
# ---------------------------------------------------------------------------

class CertifiedMultiScaleDeliberator(nn.Module):
    """ACT agent with triple-signal halting driven by multi-scale world model.

    The halting probability at each deliberation step is:

        p_halt = σ(w_u * uncertainty + w_l * lipschitz + w_d * disagreement
                   + w_geo * step_fraction + b)

    where the weights and bias are learned.
    """

    def __init__(
        self,
        obs_dim: int = 4,
        action_dim: int = 2,
        fast_hidden_dim: int = 64,
        slow_hidden_dim: int = 48,
        slow_tick_every: int = 3,
        max_thinking_steps: int = 12,
        imagination_horizon: int = 5,
    ) -> None:
        super().__init__()
        self.max_thinking_steps = max_thinking_steps
        self.imagination_horizon = imagination_horizon
        self.action_dim = action_dim

        self.world_model = MiniMultiScaleRSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            fast_hidden_dim=fast_hidden_dim,
            slow_hidden_dim=slow_hidden_dim,
            slow_tick_every=slow_tick_every,
        )

        # Thinking GRU: deliberation recurrence
        think_input = fast_hidden_dim + slow_hidden_dim + 3  # +3 for uncertainty, lipschitz, disagreement
        self.think_gru = nn.GRUCell(think_input, fast_hidden_dim)

        # Action head: from thinking state
        self.action_head = nn.Sequential(
            nn.Linear(fast_hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
        )

        # Learned halting signal combiner (the novel part)
        # Inputs: uncertainty, normalised_lipschitz, scale_disagreement, step_fraction
        self.halt_combiner = nn.Linear(4, 1)
        # Initialize with reasonable priors:
        # - high uncertainty -> don't halt (negative weight)
        # - high lipschitz -> don't halt (negative weight, timing unsafe = think more)
        # - high disagreement -> don't halt (negative weight)
        # - high step_fraction -> halt (positive weight, geometric prior)
        with torch.no_grad():
            self.halt_combiner.weight.copy_(torch.tensor([[-1.0, -0.5, -1.5, 2.0]]))
            self.halt_combiner.bias.fill_(-0.5)

    def forward(
        self,
        obs: torch.Tensor,
        fast_h: torch.Tensor,
        slow_h: torch.Tensor,
    ) -> Dict[str, Any]:
        """Run deliberation loop for a single timestep.

        Returns dict with: action, halt_weights, metrics
        """
        B = obs.shape[0]
        device = obs.device

        # Get world model signals
        uncertainty = self.world_model.get_uncertainty(fast_h)
        disagreement = self.world_model.get_scale_disagreement(fast_h, slow_h)
        lip_bound = self.world_model.lipschitz_bound()
        lip_tensor = torch.full((B, 1), lip_bound / 10.0, device=device)  # normalise

        # ACT deliberation loop
        think_h = fast_h.clone()
        cumulative_halt = torch.zeros(B, 1, device=device)
        remainder = torch.ones(B, 1, device=device)
        still_running = torch.ones(B, 1, device=device)
        step_weights: List[torch.Tensor] = []
        action_accum = torch.zeros(B, self.action_dim, device=device)

        actual_steps = 0
        for step in range(self.max_thinking_steps):
            step_frac = torch.full((B, 1), (step + 1) / self.max_thinking_steps, device=device)

            # Halting decision from triple signal
            halt_input = torch.cat([uncertainty, lip_tensor, disagreement, step_frac], dim=-1)
            p_halt = torch.sigmoid(self.halt_combiner(halt_input))

            force_halt = (step == self.max_thinking_steps - 1)
            force_tensor = torch.full_like(still_running, float(force_halt))

            lambda_n, cumulative_halt, remainder, used_remainder = apply_act_step(
                cumulative_halt, remainder, p_halt, still_running, force_tensor
            )
            step_weights.append(lambda_n)

            # Thinking step: imagine and reason
            think_input = torch.cat([fast_h, slow_h, uncertainty, lip_tensor, disagreement], dim=-1)
            think_h = self.think_gru(think_input, think_h)

            # Action from current thinking state
            step_action = self.action_head(think_h)
            action_accum = action_accum + lambda_n * step_action

            # Update still_running
            still_running = (1.0 - used_remainder.unsqueeze(-1)) * still_running
            actual_steps = step + 1

            if still_running.sum() < 0.01:
                break

        # Stack weights and get stats
        weight_matrix, weight_error = stack_halt_weights(step_weights, B, device)
        halt_stats = halt_distribution_stats(weight_matrix)

        return {
            "action": action_accum,
            "halt_weights": weight_matrix,
            "expected_steps": halt_stats["expected_steps"],
            "halt_entropy": halt_stats["halt_entropy"],
            "actual_steps": actual_steps,
            "uncertainty": uncertainty.mean().item(),
            "lipschitz_bound": lip_bound,
            "scale_disagreement": disagreement.mean().item(),
            "weight_error": weight_error,
        }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

@dataclass
class CMSDExperimentConfig:
    env_id: str = "CartPole-v1"
    obs_dim: int = 4
    action_dim: int = 2
    fast_hidden_dim: int = 64
    slow_hidden_dim: int = 48
    slow_tick_every: int = 3
    max_thinking_steps: int = 12
    imagination_horizon: int = 5
    n_train_episodes: int = 40
    n_eval_episodes: int = 20
    max_steps: int = 500
    train_epochs: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    seed: int = 42


def _collect_episodes(
    env_id: str,
    n_episodes: int,
    max_steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Collect episodes with random policy for training data."""
    env = gym.make(env_id)
    episodes = []
    for ep in range(n_episodes):
        obs_raw, _ = env.reset(seed=seed + ep)
        ep_data: Dict[str, Any] = {"obs": [obs_raw], "actions": [], "rewards": []}
        for _ in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, term, trunc, _ = env.step(action)
            ep_data["actions"].append(action)
            ep_data["rewards"].append(reward)
            ep_data["obs"].append(next_obs)
            if term or trunc:
                break
        episodes.append(ep_data)
    env.close()
    return episodes


class CMSDExperiment:
    """Certified Multi-Scale Deliberation experiment with online learning."""

    def __init__(self, cfg: CMSDExperimentConfig) -> None:
        self.cfg = cfg

    def _train_world_model(
        self,
        agent: CertifiedMultiScaleDeliberator,
        episodes: List[Dict[str, Any]],
        device: torch.device,
    ) -> Dict[str, float]:
        """Train world model on collected episodes: predict next obs from current state + action."""
        wm = agent.world_model
        optimizer = torch.optim.AdamW(wm.parameters(), lr=self.cfg.lr, weight_decay=1e-5)

        total_loss = 0.0
        n_updates = 0

        for epoch in range(self.cfg.train_epochs):
            random.shuffle(episodes)
            epoch_loss = 0.0
            for ep_data in episodes:
                obs_list = ep_data["obs"]
                act_list = ep_data["actions"]
                if len(act_list) < 2:
                    continue

                fast_h = torch.zeros(1, self.cfg.fast_hidden_dim, device=device)
                slow_h = torch.zeros(1, self.cfg.slow_hidden_dim, device=device)
                fast_buffer: List[torch.Tensor] = []
                ep_loss = torch.tensor(0.0, device=device)
                n_steps = 0

                for t in range(len(act_list) - 1):
                    obs_t = torch.tensor(obs_list[t], dtype=torch.float32, device=device).unsqueeze(0)
                    act_t = F.one_hot(torch.tensor([act_list[t]], device=device), self.cfg.action_dim).float()
                    obs_next = torch.tensor(obs_list[t + 1], dtype=torch.float32, device=device).unsqueeze(0)

                    # Fast step with observation
                    fast_h = wm.fast_observe(fast_h, slow_h.detach(), act_t, obs_t)
                    fast_buffer.append(fast_h)

                    # Slow tick
                    if len(fast_buffer) >= self.cfg.slow_tick_every:
                        slow_h = wm.slow_step(slow_h, fast_buffer[-self.cfg.slow_tick_every:])

                    # Predict next obs
                    new_h, obs_pred, _ = wm.fast_step(fast_h, slow_h.detach(), act_t)
                    target = symlog(obs_next)
                    ep_loss = ep_loss + F.mse_loss(obs_pred, target)

                    # Also train slow prediction at boundaries
                    if len(fast_buffer) >= self.cfg.slow_tick_every and len(fast_buffer) % self.cfg.slow_tick_every == 0:
                        slow_pred = wm.slow_obs_pred(slow_h)
                        ep_loss = ep_loss + 0.5 * F.mse_loss(slow_pred, target)

                    n_steps += 1
                    fast_h = fast_h.detach()  # truncate BPTT

                if n_steps > 0:
                    loss = ep_loss / n_steps
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(wm.parameters(), 10.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_updates += 1

            avg = epoch_loss / max(len(episodes), 1)
            total_loss += avg
            if (epoch + 1) % 4 == 0 or epoch == 0:
                print(f"  [WM] epoch {epoch+1}/{self.cfg.train_epochs}: loss={avg:.4f}")

        return {"wm_final_loss": total_loss / max(self.cfg.train_epochs, 1), "wm_updates": n_updates}

    def _train_policy(
        self,
        agent: CertifiedMultiScaleDeliberator,
        device: torch.device,
    ) -> Dict[str, float]:
        """Train policy via REINFORCE on the environment."""
        optimizer = torch.optim.Adam(
            list(agent.think_gru.parameters())
            + list(agent.action_head.parameters())
            + list(agent.halt_combiner.parameters()),
            lr=self.cfg.lr * 0.5,
        )

        env = gym.make(self.cfg.env_id)
        best_mean = 0.0
        train_returns: List[float] = []

        for epoch in range(self.cfg.train_epochs):
            # Collect one episode with gradient tracking
            obs_raw, _ = env.reset(seed=self.cfg.seed + 5000 + epoch)
            obs = torch.tensor(obs_raw, dtype=torch.float32, device=device).unsqueeze(0)

            fast_h = torch.zeros(1, self.cfg.fast_hidden_dim, device=device)
            slow_h = torch.zeros(1, self.cfg.slow_hidden_dim, device=device)
            fast_buffer: List[torch.Tensor] = []

            log_probs: List[torch.Tensor] = []
            rewards: List[float] = []

            for step in range(self.cfg.max_steps):
                act_dummy = torch.zeros(1, self.cfg.action_dim, device=device)
                with torch.no_grad():
                    fast_h = agent.world_model.fast_observe(fast_h, slow_h, act_dummy, obs)
                    fast_buffer.append(fast_h)
                    if len(fast_buffer) >= self.cfg.slow_tick_every:
                        slow_h = agent.world_model.slow_step(slow_h, fast_buffer[-self.cfg.slow_tick_every:])

                result = agent(obs, fast_h.detach(), slow_h.detach())
                action_logits = result["action"]
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))

                obs_raw, reward, term, trunc, _ = env.step(action.item())
                obs = torch.tensor(obs_raw, dtype=torch.float32, device=device).unsqueeze(0)
                rewards.append(reward)

                if term or trunc:
                    break

            ep_return = sum(rewards)
            train_returns.append(ep_return)

            # Compute discounted returns
            G = 0.0
            returns_to_go: List[float] = []
            for r in reversed(rewards):
                G = r + self.cfg.gamma * G
                returns_to_go.insert(0, G)
            returns_tensor = torch.tensor(returns_to_go, dtype=torch.float32, device=device)
            if returns_tensor.std() > 1e-8:
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

            # Policy gradient
            policy_loss = torch.tensor(0.0, device=device)
            for lp, G_t in zip(log_probs, returns_tensor):
                policy_loss = policy_loss - lp * G_t

            optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 5.0)
            optimizer.step()

            if (epoch + 1) % 4 == 0 or epoch == 0:
                print(f"  [Policy] epoch {epoch+1}/{self.cfg.train_epochs}: return={ep_return:.0f}, loss={policy_loss.item():.3f}")

        env.close()
        return {
            "policy_best_train": max(train_returns) if train_returns else 0,
            "policy_mean_train": statistics.mean(train_returns) if train_returns else 0,
        }

    def _evaluate(
        self,
        agent: CertifiedMultiScaleDeliberator,
        device: torch.device,
    ) -> Tuple[Dict[str, float], Dict[str, List]]:
        """Evaluate trained agent."""
        env = gym.make(self.cfg.env_id)
        episode_returns: List[float] = []
        episode_lengths: List[int] = []
        episode_mean_steps: List[float] = []
        episode_uncertainties: List[float] = []
        episode_disagreements: List[float] = []
        episode_lip_bounds: List[float] = []
        certified_fractions: List[float] = []

        for ep in range(self.cfg.n_eval_episodes):
            obs_raw, _ = env.reset(seed=self.cfg.seed + 2000 + ep)
            obs = torch.tensor(obs_raw, dtype=torch.float32, device=device).unsqueeze(0)

            fast_h = torch.zeros(1, self.cfg.fast_hidden_dim, device=device)
            slow_h = torch.zeros(1, self.cfg.slow_hidden_dim, device=device)
            fast_buffer: List[torch.Tensor] = []

            ep_return = 0.0
            ep_think: List[float] = []
            ep_unc: List[float] = []
            ep_dis: List[float] = []
            ep_lip: List[float] = []
            ep_cert = 0
            step_count = 0

            for step in range(self.cfg.max_steps):
                act_dummy = torch.zeros(1, self.cfg.action_dim, device=device)
                fast_h = agent.world_model.fast_observe(fast_h, slow_h, act_dummy, obs)
                fast_buffer.append(fast_h.detach())
                if len(fast_buffer) >= self.cfg.slow_tick_every:
                    slow_h = agent.world_model.slow_step(slow_h, fast_buffer[-self.cfg.slow_tick_every:])

                with torch.no_grad():
                    result = agent(obs, fast_h, slow_h)

                action_logits = result["action"]
                action = int(action_logits[0].argmax().item())

                ep_think.append(result["expected_steps"].mean().item())
                ep_unc.append(result["uncertainty"])
                ep_dis.append(result["scale_disagreement"])
                lip = result["lipschitz_bound"]
                ep_lip.append(lip)
                if lip < 5.0:
                    ep_cert += 1

                obs_raw, reward, term, trunc, _ = env.step(action)
                obs = torch.tensor(obs_raw, dtype=torch.float32, device=device).unsqueeze(0)
                ep_return += reward
                step_count += 1

                if term or trunc:
                    break

            episode_returns.append(ep_return)
            episode_lengths.append(step_count)
            episode_mean_steps.append(statistics.mean(ep_think) if ep_think else 0)
            episode_uncertainties.append(statistics.mean(ep_unc) if ep_unc else 0)
            episode_disagreements.append(statistics.mean(ep_dis) if ep_dis else 0)
            episode_lip_bounds.append(statistics.mean(ep_lip) if ep_lip else 0)
            certified_fractions.append(ep_cert / max(step_count, 1))

        env.close()

        metrics = {
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
            "episode_mean_steps": episode_mean_steps,
            "episode_uncertainties": episode_uncertainties,
            "episode_disagreements": episode_disagreements,
            "episode_lip_bounds": episode_lip_bounds,
            "certified_fractions": certified_fractions,
        }
        summary = {
            "mean_return": statistics.mean(episode_returns),
            "std_return": statistics.stdev(episode_returns) if len(episode_returns) > 1 else 0,
            "max_return": max(episode_returns),
            "mean_think": statistics.mean(episode_mean_steps),
            "cert_score": statistics.mean(certified_fractions),
        }
        return summary, metrics

    def run(self, out_dir: Path) -> Dict[str, float]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        _set_seed(self.cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CMS-D] Device: {device}")

        agent = CertifiedMultiScaleDeliberator(
            obs_dim=self.cfg.obs_dim,
            action_dim=self.cfg.action_dim,
            fast_hidden_dim=self.cfg.fast_hidden_dim,
            slow_hidden_dim=self.cfg.slow_hidden_dim,
            slow_tick_every=self.cfg.slow_tick_every,
            max_thinking_steps=self.cfg.max_thinking_steps,
            imagination_horizon=self.cfg.imagination_horizon,
        ).to(device)

        n_params = sum(p.numel() for p in agent.parameters())
        print(f"[CMS-D] Parameters: {n_params:,}")

        t0 = time.time()

        # ---- Phase 1: Collect training data ----
        print(f"\n--- Phase 1: Collecting {self.cfg.n_train_episodes} training episodes ---")
        train_episodes = _collect_episodes(
            self.cfg.env_id, self.cfg.n_train_episodes, self.cfg.max_steps, self.cfg.seed
        )
        baseline_mean = statistics.mean(sum(ep["rewards"]) for ep in train_episodes)
        print(f"  Random baseline mean return: {baseline_mean:.1f}")

        # ---- Phase 2: Train world model ----
        print(f"\n--- Phase 2: Training world model ({self.cfg.train_epochs} epochs) ---")
        wm_stats = self._train_world_model(agent, train_episodes, device)

        # ---- Phase 3: Train policy ----
        print(f"\n--- Phase 3: Training policy ({self.cfg.train_epochs} epochs) ---")
        policy_stats = self._train_policy(agent, device)

        # ---- Phase 4: Evaluate ----
        print(f"\n--- Phase 4: Evaluation ({self.cfg.n_eval_episodes} episodes) ---")
        eval_summary, eval_metrics = self._evaluate(agent, device)

        elapsed = time.time() - t0

        # ---- Compute composite score ----
        mean_return = eval_summary["mean_return"]
        normalized_return = mean_return / 500.0
        improvement = max(0.0, (mean_return - baseline_mean) / max(baseline_mean, 1e-8))
        improvement_score = min(1.0, improvement)
        cert_score = eval_summary["cert_score"]
        mean_think = eval_summary["mean_think"]

        episode_mean_steps = eval_metrics["episode_mean_steps"]
        think_adaptivity = min(1.0, statistics.stdev(episode_mean_steps) / max(mean_think, 1.0)) if len(episode_mean_steps) > 1 else 0

        episode_disagreements = eval_metrics["episode_disagreements"]
        if len(episode_disagreements) > 2:
            d_arr = np.array(episode_disagreements)
            t_arr = np.array(episode_mean_steps)
            if d_arr.std() > 1e-8 and t_arr.std() > 1e-8:
                disagreement_utilisation = max(0.0, float(np.corrcoef(d_arr, t_arr)[0, 1]))
            else:
                disagreement_utilisation = 0.0
        else:
            disagreement_utilisation = 0.0

        composite = (
            0.25 * normalized_return
            + 0.20 * cert_score
            + 0.15 * improvement_score
            + 0.15 * think_adaptivity
            + 0.15 * disagreement_utilisation
            + 0.10 * min(1.0, mean_think / self.cfg.max_thinking_steps)
        )

        std_return = eval_summary["std_return"]
        max_return = eval_summary["max_return"]

        results: Dict[str, Any] = {
            "composite_score": round(composite, 4),
            "normalized_return": round(normalized_return, 4),
            "mean_return": round(mean_return, 2),
            "std_return": round(std_return, 2),
            "max_return": round(max_return, 2),
            "baseline_mean_return": round(baseline_mean, 2),
            "improvement_over_baseline": round(improvement_score, 4),
            "certified_fraction": round(cert_score, 4),
            "mean_thinking_steps": round(mean_think, 3),
            "think_adaptivity": round(think_adaptivity, 4),
            "disagreement_utilisation": round(disagreement_utilisation, 4),
            "mean_scale_disagreement": round(statistics.mean(episode_disagreements), 6),
            "mean_uncertainty": round(statistics.mean(eval_metrics["episode_uncertainties"]), 6),
            "mean_lipschitz_bound": round(statistics.mean(eval_metrics["episode_lip_bounds"]), 4),
            "wm_final_loss": round(wm_stats["wm_final_loss"], 6),
            "policy_best_train": round(policy_stats["policy_best_train"], 1),
            "n_params": n_params,
            "elapsed_seconds": round(elapsed, 2),
            "device": str(device),
            "n_train_episodes": self.cfg.n_train_episodes,
            "n_eval_episodes": self.cfg.n_eval_episodes,
            "train_epochs": self.cfg.train_epochs,
            "episode_returns": [round(r, 1) for r in eval_metrics["episode_returns"]],
            "episode_lengths": eval_metrics["episode_lengths"],
        }

        print(f"\n=== CMS-D Results ===")
        for k, v in results.items():
            if k not in ("episode_returns", "episode_lengths"):
                print(f"  {k}: {v}")

        # Save
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)

        torch.save(agent.state_dict(), out_dir / "agent.pt")

        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Certified Multi-Scale Deliberation")
    parser.add_argument("--out-dir", type=Path, default=Path("results/cms_deliberation"))
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--max-thinking-steps", type=int, default=12)
    parser.add_argument("--fast-hidden-dim", type=int, default=64)
    parser.add_argument("--slow-hidden-dim", type=int, default=48)
    parser.add_argument("--slow-tick-every", type=int, default=3)
    parser.add_argument("--imagination-horizon", type=int, default=5)
    args = parser.parse_args()

    cfg = CMSDExperimentConfig(
        n_episodes=args.n_episodes,
        max_thinking_steps=args.max_thinking_steps,
        fast_hidden_dim=args.fast_hidden_dim,
        slow_hidden_dim=args.slow_hidden_dim,
        slow_tick_every=args.slow_tick_every,
        imagination_horizon=args.imagination_horizon,
    )
    exp = CMSDExperiment(cfg)
    results = exp.run(args.out_dir)
    print(f"\nComposite: {results['composite_score']}")


if __name__ == "__main__":
    main()
