"""Frontier 7: Meta-Policy Distillation (The Universal Adversary).

Distills specialized timing attackers into a cross-environment meta-policy
that predicts 'Temporal Resonant Frequencies' to induce zero-shot failure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deltatau_audit.protocols import AgentAdapter

from ._base import save_summary, seed_all

logger = logging.getLogger("deltatau-audit")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. The Meta-Attacker: Transformer-based Cross-Environment Policy
# ═══════════════════════════════════════════════════════════════════════════════

class MetaTimingAdversary(nn.Module):
    """Transformer-based adversary that encodes (obs, act, reward, dt) context."""
    def __init__(self, obs_dim: int = 16, act_dim: int = 4, hidden_dim: int = 128, n_speeds: int = 6):
        super().__init__()
        # Project heterogeneous obs/act dims to a common space
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.act_proj = nn.Linear(act_dim, hidden_dim)
        self.dt_proj = nn.Linear(1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.policy_head = nn.Linear(hidden_dim, n_speeds)

    def forward(self, obs: torch.Tensor, acts: torch.Tensor, dts: torch.Tensor):
        # inputs: (batch, seq_len, dim)
        o = self.obs_proj(obs)
        a = self.act_proj(acts)
        d = self.dt_proj(dts)

        feat = o + a + d # Summing projected features
        out = self.transformer(feat)
        logits = self.policy_head(out[:, -1, :])
        return logits

# ═══════════════════════════════════════════════════════════════════════════════
# 2. The Meta-Experiment: Multi-Task Distillation
# ═══════════════════════════════════════════════════════════════════════════════

class MetaPolicyDistillationExperiment:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.envs = ["CartPole-v1", "Pendulum-v1", "Acrobot-v1"]
        self.device = params.get("device", "cpu")
        self.seed = int(params.get("seed", 42))
        self.possible_speeds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

        self.meta_attacker = MetaTimingAdversary(
            obs_dim=16, # Max obs dim across envs (zero-padded)
            act_dim=4,  # Max act dim (zero-padded)
            hidden_dim=128
        ).to(self.device)
        self.optimizer = optim.Adam(self.meta_attacker.parameters(), lr=1e-4)

    def run(self, out_dir: Path) -> Dict[str, float]:
        seed_all(self.seed)
        from stable_baselines3 import PPO

        from deltatau_audit.adapters.sb3 import SB3Adapter

        print(f"  Training Universal Adversary on {self.envs}...")

        total_transfer_score = 0.0

        for env_id in self.envs:
            # 1. Load or train a specialized victim for this env
            model = PPO("MlpPolicy", env_id, verbose=0)
            model.learn(total_timesteps=5000)
            adapter = SB3Adapter(model)

            # 2. Distill knowledge: Train meta-attacker to kill this victim
            ep_rets = []
            for ep in range(20):
                rewards, log_probs = self._run_meta_attack(adapter, env_id)
                loss = sum(lp * sum(rewards) for lp in log_probs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ep_rets.append(sum(rewards))

            env_vulnerability = float(np.mean(ep_rets))
            print(f"    Env {env_id} vulnerability detected: {env_vulnerability:.1f}")
            # Victim normal score is ~200-500, lower is better for attacker
            total_transfer_score += max(0.0, 500.0 - env_vulnerability)

        # Step 3: Zero-shot test on a NEW environment (MountainCar)
        test_env = "MountainCar-v0"
        print(f"  Zero-shot testing on {test_env}...")
        test_model = PPO("MlpPolicy", test_env, verbose=0)
        test_adapter = SB3Adapter(test_model)

        test_rets, _ = self._run_meta_attack(test_adapter, test_env, train=False)
        zero_shot_impact = float(sum(test_rets))

        print(f"    Zero-shot impact return: {zero_shot_impact:.1f}")

        # MountainCar ranges from -200 (fail) to -100 (success). Lower is better for attacker.
        zero_shot_score = max(0.0, -100.0 - zero_shot_impact)

        summary = {
            "meta_generalization_score": float(total_transfer_score),
            "zero_shot_kill_rate": float(zero_shot_score),
            "composite_score": float((total_transfer_score / 1500.0) * zero_shot_score),
        }
        save_summary(out_dir, summary)
        return summary

    def _run_meta_attack(self, adapter: AgentAdapter, env_id: str, train: bool = True) -> Tuple[List[float], List[torch.Tensor]]:
        env = gym.make(env_id)
        obs, _ = env.reset()
        adapter.reset_internal_state()

        rewards = []
        log_probs = []
        obs_seq = []
        act_seq = []
        dt_seq = []

        done = False
        while not done:
            # Pad obs/act to fixed meta-dims
            p_obs = np.zeros(16)
            p_obs[:len(obs)] = obs
            p_act = np.zeros(4)
            # handle discrete/box
            if isinstance(env.action_space, gym.spaces.Discrete):
                p_act[0] = 0 # discrete
            else:
                p_act[:len(env.action_space.sample())] = env.action_space.sample()

            obs_seq.append(p_obs)
            act_seq.append(p_act)
            dt_seq.append([1.0])

            if len(obs_seq) > 20:
                obs_seq.pop(0)
                act_seq.pop(0)
                dt_seq.pop(0)

            # Meta-Attacker selects speed
            o_t = torch.tensor(np.array(obs_seq), dtype=torch.float32).unsqueeze(0).to(self.device)
            a_t = torch.tensor(np.array(act_seq), dtype=torch.float32).unsqueeze(0).to(self.device)
            d_t = torch.tensor(np.array(dt_seq), dtype=torch.float32).unsqueeze(0).to(self.device)

            logits = self.meta_attacker(o_t, a_t, d_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            if train:
                action_idx = dist.sample()
                log_probs.append(dist.log_prob(action_idx))
            else:
                action_idx = torch.argmax(probs, dim=-1)

            speed = self.possible_speeds[action_idx.item()]

            # Victim acts
            agent_action, _ = adapter.act(torch.tensor(obs, dtype=torch.float32))

            # Execute
            n_repeats = max(1, int(round(speed)))
            step_reward = 0
            for _ in range(n_repeats):
                obs, r, term, trunc, _ = env.step(agent_action)
                step_reward += r
                if term or trunc: break

            rewards.append(step_reward)
            done = term or trunc
            if len(rewards) > 300: break

        env.close()
        return rewards, log_probs
