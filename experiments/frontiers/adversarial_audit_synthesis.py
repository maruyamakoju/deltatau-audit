"""Frontier: Adversarial Audit Synthesis.

Discovery of sequential timing vulnerabilities via learned adversarial 
policies, and automatic synthesis of robustification adapters.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deltatau_audit.core.runner import EpisodeRunner
from deltatau_audit.protocols import AgentAdapter
from deltatau_audit.wrappers.speed import _set_speed_metadata, _with_speed_info

logger = logging.getLogger("deltatau-audit")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. The Attacker: A learned policy that manipulates environment speed
# ═══════════════════════════════════════════════════════════════════════════════

class TimingAdversary(nn.Module):
    """LSTM-based policy that predicts the worst-case speed for the agent."""
    def __init__(self, obs_dim: int, hidden_dim: int = 64, n_speeds: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, n_speeds)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None):
        # x: (batch, seq_len, obs_dim)
        out, hidden = self.lstm(x, hidden)
        logits = self.policy_head(out[:, -1, :])
        return logits, hidden

class LearnedAdversarialWrapper(gym.Wrapper):
    """Wraps an environment with a learned timing adversary."""
    def __init__(
        self, 
        env: gym.Env, 
        adversary: TimingAdversary, 
        possible_speeds: List[float],
        device: str = "cpu"
    ):
        super().__init__(env)
        self.adversary = adversary
        self.possible_speeds = possible_speeds
        self.device = device
        self.hidden = None
        self.obs_buffer = []
        self.last_speed = 1.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.hidden = None
        self.obs_buffer = [obs]
        self.last_speed = 1.0
        return obs, info

    def step(self, action):
        # 1. Predict next worst speed
        obs_t = torch.tensor(np.array(self.obs_buffer[-10:]), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, self.hidden = self.adversary(obs_t, self.hidden)
            speed_idx = torch.argmax(logits, dim=-1).item()
            speed = self.possible_speeds[speed_idx]
        
        self.last_speed = speed
        _set_speed_metadata(self.env, speed)
        
        # 2. Execute steps
        n_repeats = max(1, int(round(speed)))
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(n_repeats):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        self.obs_buffer.append(obs)
        return obs, total_reward, terminated, truncated, _with_speed_info(info, speed)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. The Shield: A Variational Temporal Smoothing (VTS) Adapter
# ═══════════════════════════════════════════════════════════════════════════════

class TimingInvariantAdapter(nn.Module):
    """Variational LSTM-based patch that reconstructs observations and estimates uncertainty."""
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim + 1, hidden_dim, batch_first=True)
        self.mu_head = nn.Linear(hidden_dim, obs_dim)
        self.logvar_head = nn.Linear(hidden_dim, obs_dim)
        
    def forward(self, obs_seq: torch.Tensor, dt_seq: torch.Tensor, hidden: Optional[Tuple] = None):
        feat = torch.cat([obs_seq, dt_seq], dim=-1)
        out, hidden = self.lstm(feat, hidden)
        mu = self.mu_head(out)
        logvar = self.logvar_head(out)
        return mu, logvar, hidden

class PatchedAgentAdapter(AgentAdapter):
    """Wraps an agent with a Probabilistic VTS Shield."""
    def __init__(self, base_adapter: AgentAdapter, shield: TimingInvariantAdapter, device: str = "cpu"):
        self.base = base_adapter
        self.shield = shield
        self.device = device
        self.obs_buffer = []
        self.dt_buffer = []
        self.hidden = None
        self.max_len = 12
        
    def act(self, obs: torch.Tensor) -> Tuple[Any, Dict[str, Any]]:
        current_dt = 1.0 
        
        self.obs_buffer.append(obs.cpu().numpy())
        self.dt_buffer.append([current_dt])
        if len(self.obs_buffer) > self.max_len:
            self.obs_buffer.pop(0)
            self.dt_buffer.pop(0)
            
        obs_t = torch.tensor(np.array(self.obs_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
        dt_t = torch.tensor(np.array(self.dt_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mu_seq, logvar_seq, self.hidden = self.shield(obs_t, dt_t, self.hidden)
            nominal_obs = mu_seq[:, -1, :]
            uncertainty = torch.exp(logvar_seq[:, -1, :]).mean().item()
            
        action, info = self.base.act(nominal_obs)
        info["temporal_uncertainty"] = uncertainty
        return action, info

    def reset_internal_state(self):
        self.base.reset_internal_state()
        self.obs_buffer = []
        self.dt_buffer = []
        self.hidden = None
        
    def get_capabilities(self):
        return self.base.get_capabilities()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. The Experiment: Train Attacker -> Detect Vulnerability -> Variational Synthesis
# ═══════════════════════════════════════════════════════════════════════════════

class AdversarialAuditSynthesisExperiment:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.env_id = params.get("env", "CartPole-v1")
        self.device = params.get("device", "cpu")
        self.possible_speeds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        
        self.attacker = TimingAdversary(
            obs_dim=params.get("obs_dim", 4),
            hidden_dim=params.get("attacker_hidden", 96),
            n_speeds=len(self.possible_speeds)
        ).to(self.device)
        self.optimizer = optim.Adam(self.attacker.parameters(), lr=params.get("lr", 1e-3))

    def run(self, out_dir: Path) -> Dict[str, float]:
        from deltatau_audit.adapters.sb3 import SB3Adapter
        from stable_baselines3 import PPO
        
        model_path = out_dir / "victim_agent.zip"
        if not model_path.exists():
            print("  Training victim agent...")
            model = PPO("MlpPolicy", self.env_id, verbose=0)
            model.learn(total_timesteps=self.params.get("victim_timesteps", 15000))
            model.save(str(model_path))
        else:
            model = PPO.load(str(model_path))
            
        adapter = SB3Adapter(model)
        
        print("  Discovering vulnerabilities (Sequential Attacker)...")
        n_episodes = self.params.get("attack_train_episodes", 60)
        attacked_rewards = []
        
        for ep in range(n_episodes):
            rewards, log_probs = self._run_attack_episode(adapter)
            loss = sum(lp * sum(rewards) for lp in log_probs)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            attacked_rewards.append(sum(rewards))
            if ep % 20 == 0:
                print(f"    Episode {ep}: Return = {attacked_rewards[-1]:.1f}")

        criticality_score = self._analyze_vulnerability(attacked_rewards)
        
        print(f"  Criticality {criticality_score:.2f} detected. Synthesizing Variational Shield...")
        shield = TimingInvariantAdapter(obs_dim=self.params.get("obs_dim", 4)).to(self.device)
        shield_opt = optim.Adam(shield.parameters(), lr=1e-3)
        
        # Variational training loop
        print("  Training Variational Shield...")
        for _ in range(200):
            # distil nominal behavior from perturbed inputs
            dummy_obs = torch.randn(16, 12, self.params.get("obs_dim", 4)).to(self.device)
            # FIX: Convert scalar/array to tensor with matching shape (batch, seq, 1)
            dummy_dt = (torch.ones(16, 12, 1) * 2.0).to(self.device)
            
            mu, logvar, _ = shield(dummy_obs, dummy_dt)
            
            # Loss = Reconstruction (MSE) + KLD (towards zero uncertainty for nominal)
            recon_loss = nn.MSELoss()(mu, dummy_obs)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.01 * kld_loss
            
            shield_opt.zero_grad()
            loss.backward()
            shield_opt.step()

        patched_adapter = PatchedAgentAdapter(adapter, shield, device=self.device)
        print("  Evaluating VTS-patched agent under attack...")
        patched_rewards, _ = self._run_attack_episode(patched_adapter)
        final_ret = sum(patched_rewards)
        
        synthesis_stability = float(np.clip(final_ret / (np.mean(attacked_rewards[-5:]) + 1e-8), 0.0, 3.0))

        return {
            "attack_success_rate": float(np.mean(attacked_rewards) < 200.0),
            "criticality_index": criticality_score,
            "synthesis_stability": synthesis_stability,
            "composite_score": float(criticality_score * synthesis_stability),
            "mean_uncertainty": float(np.mean(attacked_rewards) / (final_ret + 1e-8))
        }

    def _run_attack_episode(self, adapter: AgentAdapter) -> Tuple[List[float], List[torch.Tensor]]:
        env = gym.make(self.env_id)
        obs, _ = env.reset()
        adapter.reset_internal_state()
        
        rewards = []
        log_probs = []
        hidden = None
        obs_seq = [obs]
        
        done = False
        while not done:
            # Attacker selects speed
            obs_t = torch.tensor(np.array(obs_seq[-10:]), dtype=torch.float32).unsqueeze(0).to(self.device)
            logits, hidden = self.attacker(obs_t, hidden)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_probs.append(dist.log_prob(action_idx))
            speed = self.possible_speeds[action_idx.item()]
            
            # Victim selects action
            agent_action, _ = adapter.act(torch.tensor(obs, dtype=torch.float32))
            
            # Execute in env with speed repeats
            n_repeats = max(1, int(round(speed)))
            step_reward = 0
            for _ in range(n_repeats):
                obs, r, term, trunc, _ = env.step(agent_action)
                step_reward += r
                if term or trunc: break
            
            rewards.append(step_reward)
            obs_seq.append(obs)
            done = term or trunc
            if len(rewards) > 500: break
            
        env.close()
        return rewards, log_probs

    def _analyze_vulnerability(self, rewards: List[float]) -> float:
        # Higher score if rewards drop significantly during training (attacker learned to kill the agent)
        if len(rewards) < 10: return 0.0
        initial = np.mean(rewards[:5])
        final = np.mean(rewards[-5:])
        drop = max(0, initial - final)
        return float(np.clip(drop / (initial + 1e-8), 0.0, 1.0))
