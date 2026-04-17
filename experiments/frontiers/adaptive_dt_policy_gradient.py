"""Frontier 11: Adaptive-dt Policy Gradient.

Claude-proposed frontier (2026-04-18). Tests whether learning a per-step dt
as part of the policy improves RETURNS on variable-speed environments,
not just reconstruction. Fills the gap left by multiscale_temporal_wm
(reconstruction-only) and temporal_subjectification (CartPole, saturates
trivially).

Setup:
    Train on VariableFrequencyChainEnv with speeds {1, 2, 3}.
    Eval on held-out speeds {1, 2, 3, 5, 8}.
    Same arch for adaptive and fixed baselines; the only difference is
    whether the dt head is active or clamped to 1.0.

Composite:
    0.35 * normalized_return        # raw return on eval speeds
  + 0.30 * dt_adaptation_score      # |corr(dt, speed)| over eval
  + 0.20 * improvement_over_fixed_dt # (adaptive - fixed) / (|fixed| + eps)
  + 0.15 * speed_generalization     # return on unseen speeds / on train speeds
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from internal_time_rl.envs.variable_frequency import VariableFrequencyChainEnv


TRAIN_SPEEDS: Tuple[int, ...] = (1, 2, 3)
EVAL_SPEEDS: Tuple[int, ...] = (1, 2, 3, 5, 8)
UNSEEN_SPEEDS: Tuple[int, ...] = (5, 8)


class AdaptiveDTAgent(nn.Module):
    """Policy+dt+value on a TimeAwareGRU-style recurrent state.

    dt is predicted per step from the hidden state. When
    ``adaptive_dt=False`` the dt head is detached and clamped to 1.0,
    matching the fixed-dt baseline.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        n_actions: int,
        dt_min: float = 0.3,
        dt_max: float = 2.5,
        adaptive_dt: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adaptive_dt = adaptive_dt
        self.dt_min = dt_min
        self.dt_max = dt_max

        self.encoder = nn.Linear(obs_dim, hidden_dim)
        # TimeAwareGRU-style: z gate + candidate, coupled via (1-z)^dt
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cand = nn.Linear(hidden_dim * 2, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.dt_head = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.tanh(self.encoder(obs))
        fused = torch.cat([hidden, x], dim=-1)
        z = torch.sigmoid(self.gate(fused))
        cand = torch.tanh(self.cand(fused))

        if self.adaptive_dt:
            dt_raw = torch.sigmoid(self.dt_head(hidden))
            dt = self.dt_min + dt_raw * (self.dt_max - self.dt_min)
        else:
            dt = torch.ones(obs.shape[0], 1, device=obs.device)

        # alpha = 1 - (1-z)^dt — TimeAwareGRU kernel. Forces proper behavior
        # at dt=1 (alpha=z) and respects dt>1 (more decay) / dt<1 (less).
        alpha = 1.0 - (1.0 - z).clamp(min=1e-6) ** dt
        hidden_new = (1.0 - alpha) * hidden + alpha * cand

        logits = self.policy_head(hidden_new)
        value = self.value_head(hidden_new).squeeze(-1)
        return logits, value, hidden_new, dt.squeeze(-1)


def _rollout(
    agent: AdaptiveDTAgent,
    env: VariableFrequencyChainEnv,
    device: torch.device,
    max_steps: int = 100,
    deterministic: bool = False,
) -> Dict[str, Any]:
    """One episode, returns total reward + logprobs + observed (dt, speed)."""
    obs, info = env.reset()
    hidden = agent.initial_hidden(1, device)
    total_reward = 0.0
    log_probs: List[torch.Tensor] = []
    dts: List[float] = []
    speeds: List[int] = []

    for _ in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, _value, hidden, dt = agent(obs_t, hidden)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = int(torch.argmax(probs, dim=-1).item())
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action_t = dist.sample()
            log_probs.append(dist.log_prob(action_t))
            action = int(action_t.item())

        dts.append(float(dt.item()))
        speeds.append(int(env.current_speed))

        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "log_probs": log_probs,
        "dts": dts,
        "speeds": speeds,
    }


def _train(
    agent: AdaptiveDTAgent,
    device: torch.device,
    n_episodes: int,
    lr: float,
    chain_length: int,
    noise: float,
) -> List[float]:
    """REINFORCE training on randomized-speed chain."""
    optim = torch.optim.Adam(agent.parameters(), lr=lr)
    env = VariableFrequencyChainEnv(
        chain_length=chain_length,
        train_speeds=TRAIN_SPEEDS,
        speed_in_obs=True,
        noise=noise,
    )
    returns: List[float] = []
    for ep in range(n_episodes):
        roll = _rollout(agent, env, device)
        ret = roll["total_reward"]
        returns.append(ret)

        if not roll["log_probs"]:
            continue
        log_probs = torch.stack(roll["log_probs"])
        # Normalize returns across episode as a simple baseline
        advantage = ret  # scalar; treat as return-to-go proxy
        loss = -(log_probs * advantage).sum()
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optim.step()
    return returns


def _evaluate(
    agent: AdaptiveDTAgent,
    device: torch.device,
    speeds: Tuple[int, ...],
    n_per_speed: int,
    chain_length: int,
    noise: float,
) -> Dict[str, Any]:
    """Deterministic eval at each fixed speed; returns per-speed mean and
    the flat (dt, speed) sample for correlation."""
    per_speed: Dict[int, List[float]] = {}
    all_dts: List[float] = []
    all_speeds: List[float] = []
    with torch.no_grad():
        for s in speeds:
            env = VariableFrequencyChainEnv(
                chain_length=chain_length,
                fixed_speed=int(s),
                speed_in_obs=True,
                noise=noise,
            )
            rets: List[float] = []
            for _ in range(n_per_speed):
                roll = _rollout(agent, env, device, deterministic=True)
                rets.append(roll["total_reward"])
                # Sample per-step dt/speed for adaptation correlation
                all_dts.extend(roll["dts"])
                all_speeds.extend([float(sp) for sp in roll["speeds"]])
            per_speed[int(s)] = rets
    return {"per_speed": per_speed, "dts": all_dts, "speeds": all_speeds}


def _safe_correlation(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


class AdaptiveDTExperiment:
    """Entry point consumed by autonomous_research.FRONTIER_REGISTRY."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.device = torch.device(
            "cuda" if params.get("device_policy", "cpu") == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.hidden_dim = int(params.get("hidden_dim", 64))
        self.lr = float(params.get("lr", 3e-3))
        self.n_episodes = int(params.get("n_episodes", 80))
        self.chain_length = int(params.get("chain_length", 20))
        self.noise = float(params.get("noise", 0.05))
        self.eval_per_speed = int(params.get("eval_per_speed", 20))
        self.seed = int(params.get("seed", 0))

    def _build(self, adaptive_dt: bool) -> AdaptiveDTAgent:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # obs = chain_length + step_frac + speed + reward_pending
        obs_dim = self.chain_length + 3
        return AdaptiveDTAgent(
            obs_dim=obs_dim,
            hidden_dim=self.hidden_dim,
            n_actions=2,
            adaptive_dt=adaptive_dt,
        ).to(self.device)

    def run(self, out_dir: Path) -> Dict[str, float]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print("  [adaptive_dt_pg] training adaptive agent...")
        adaptive_agent = self._build(adaptive_dt=True)
        adaptive_train_returns = _train(
            adaptive_agent, self.device, self.n_episodes, self.lr,
            self.chain_length, self.noise,
        )

        print("  [adaptive_dt_pg] training fixed-dt baseline...")
        fixed_agent = self._build(adaptive_dt=False)
        fixed_train_returns = _train(
            fixed_agent, self.device, self.n_episodes, self.lr,
            self.chain_length, self.noise,
        )

        print("  [adaptive_dt_pg] evaluating on held-out speeds...")
        adaptive_eval = _evaluate(
            adaptive_agent, self.device, EVAL_SPEEDS, self.eval_per_speed,
            self.chain_length, self.noise,
        )
        fixed_eval = _evaluate(
            fixed_agent, self.device, EVAL_SPEEDS, self.eval_per_speed,
            self.chain_length, self.noise,
        )

        # --- Sub-metrics ---
        # normalized_return: eval_return / max_possible (≈1.0 for successful chain)
        MAX_POSSIBLE_RETURN = 1.0
        adaptive_all_returns = [r for rs in adaptive_eval["per_speed"].values() for r in rs]
        fixed_all_returns = [r for rs in fixed_eval["per_speed"].values() for r in rs]
        norm_return = float(np.mean(adaptive_all_returns)) / MAX_POSSIBLE_RETURN
        norm_return = float(np.clip(norm_return, -1.0, 1.0))

        # dt_adaptation_score: |corr(dt, speed)| over all adaptive eval steps
        dt_corr_raw = _safe_correlation(adaptive_eval["dts"], adaptive_eval["speeds"])
        dt_adaptation_score = float(np.clip(abs(dt_corr_raw), 0.0, 1.0))

        # improvement_over_fixed_dt: (adaptive - fixed) / (|fixed| + eps), clipped [0, 1]
        adaptive_mean = float(np.mean(adaptive_all_returns))
        fixed_mean = float(np.mean(fixed_all_returns))
        eps = 0.05
        improvement = (adaptive_mean - fixed_mean) / (abs(fixed_mean) + eps)
        improvement = float(np.clip(improvement, 0.0, 1.0))

        # speed_generalization: unseen/train
        train_returns = [
            r for s, rs in adaptive_eval["per_speed"].items() if s in TRAIN_SPEEDS for r in rs
        ]
        unseen_returns = [
            r for s, rs in adaptive_eval["per_speed"].items() if s in UNSEEN_SPEEDS for r in rs
        ]
        if train_returns and unseen_returns and abs(np.mean(train_returns)) > eps:
            gen = float(np.mean(unseen_returns)) / float(np.mean(train_returns))
        else:
            gen = 0.0
        speed_generalization = float(np.clip(gen, 0.0, 1.0))

        composite = (
            0.35 * norm_return
            + 0.30 * dt_adaptation_score
            + 0.20 * improvement
            + 0.15 * speed_generalization
        )

        metrics = {
            "composite_score": float(composite),
            "normalized_return": norm_return,
            "dt_adaptation_score": dt_adaptation_score,
            "dt_speed_corr_raw": float(dt_corr_raw),
            "improvement_over_fixed_dt": improvement,
            "speed_generalization": speed_generalization,
            "adaptive_return_mean": adaptive_mean,
            "fixed_return_mean": fixed_mean,
            "adaptive_train_last10_mean": float(np.mean(adaptive_train_returns[-10:])) if adaptive_train_returns else 0.0,
            "fixed_train_last10_mean": float(np.mean(fixed_train_returns[-10:])) if fixed_train_returns else 0.0,
            "n_eval_samples": float(len(adaptive_all_returns)),
        }

        per_speed_summary = {
            str(s): {
                "adaptive_mean": float(np.mean(adaptive_eval["per_speed"][s])),
                "fixed_mean": float(np.mean(fixed_eval["per_speed"][s])),
                "adaptive_std": float(np.std(adaptive_eval["per_speed"][s])),
                "n": len(adaptive_eval["per_speed"][s]),
            }
            for s in EVAL_SPEEDS
        }
        (out_dir / "per_speed_eval.json").write_text(
            json.dumps(per_speed_summary, indent=2), encoding="utf-8"
        )
        return metrics
