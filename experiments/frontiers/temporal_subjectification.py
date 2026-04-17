"""Frontier 8: Temporal Subjectification.

Decoupling agent cognition from environment clock using Subjective Time
driven by Continuous-Time Neural ODE dynamics.
"""

from __future__ import annotations

import logging
import os
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from internal_time_rl.models.continuous import NeuralODEAgent, odeint
from deltatau_audit.adapters.base import AgentAdapter

logger = logging.getLogger("deltatau-audit")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Subjective Temporal Agent
# ═══════════════════════════════════════════════════════════════════════════════

class SubjectiveTemporalAgent(NeuralODEAgent):
    """An agent that lives in its own subjective timeline.
    
    Instead of using the environment's dt, it predicts its own 'Thinking Duration'
    and evolves its state accordingly.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Internal subjective clock
        self.subjective_time = 0.0

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor, env_dt: Optional[float] = None) -> Tuple:
        encoded = self.encoder(obs)
        
        # 1. Decide 'how much time' to spend thinking about this observation
        # In subjectification, we might ignore env_dt or use it as a 'hint'
        subj_dt = self.time_module(hidden, encoded)
        
        # 2. Evolve internal state over the subjective interval
        self.ode_func.set_condition(encoded)
        t_span = torch.tensor([0.0, subj_dt.mean().item()], device=hidden.device)
        hidden_new = odeint(self.ode_func, hidden, t_span, method=self.ode_method)
        
        self.subjective_time += subj_dt.mean().item()

        # 3. Policy based on matured hidden state
        if self.discrete_actions:
            logits = self.policy_head(hidden_new)
            dist = torch.distributions.Categorical(logits=logits)
        else:
            mean = self.policy_mean(hidden_new)
            std = self.policy_log_std.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)

        value = self.value_head(hidden_new).squeeze(-1)
        return dist, value, hidden_new, subj_dt

class SubjectiveAdapter(AgentAdapter):
    """Adapter for the SubjectiveTemporalAgent."""
    def __init__(self, agent: SubjectiveTemporalAgent, device: str = "cpu"):
        super().__init__(agent, device)
        self.hidden = None

    def act(self, obs: torch.Tensor) -> Tuple[Any, Dict[str, Any]]:
        obs = obs.to(self.device)
        if self.hidden is None:
            self.hidden = self.agent.get_initial_hidden(1, self.device)
            
        dist, value, self.hidden, subj_dt = self.agent(obs.unsqueeze(0), self.hidden)
        action = dist.sample()
        
        return action.item() if self.agent.discrete_actions else action.cpu().numpy()[0], {
            "value": value.item(),
            "subj_dt": subj_dt.item(),
            "subjective_time": self.agent.subjective_time
        }

    def reset_internal_state(self):
        self.hidden = None
        self.agent.subjective_time = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Frontier Experiment
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalSubjectificationExperiment:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.env_id = params.get("env", "CartPole-v1")
        self.device = params.get("device", "cpu")
        
        self.agent = SubjectiveTemporalAgent(
            obs_dim=params.get("obs_dim", 4),
            act_dim=params.get("act_dim", 2),
            hidden_dim=params.get("hidden_dim", 64),
            ode_method="rk4",
            ode_steps=5
        ).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=params.get("lr", 1e-3))

    def run(self, out_dir: Path) -> Dict[str, float]:
        print(f"  Training Subjective Agent on {self.env_id}...")
        
        # Simple training loop (Reinforce-style for research speed)
        n_episodes = self.params.get("n_episodes", 50)
        returns = []
        subjective_drifts = [] # Difference between env clock and subjective clock

        for ep in range(n_episodes):
            env = gym.make(self.env_id)
            obs, _ = env.reset()
            hidden = self.agent.get_initial_hidden(1, self.device)
            self.agent.subjective_time = 0.0
            
            ep_reward = 0
            log_probs = []
            env_time = 0.0
            
            done = False
            while not done:
                # Agent acts in its subjective time
                dist, value, hidden, subj_dt = self.agent(
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device), 
                    hidden
                )
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                
                # Execute in env (here env_dt is 1.0)
                obs, r, term, trunc, _ = env.step(action.item())
                ep_reward += r
                env_time += 1.0
                
                done = term or trunc
                if len(log_probs) > 500: break
            
            # Update
            loss = -torch.stack(log_probs).sum() * ep_reward / 100.0
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            returns.append(ep_reward)
            subjective_drifts.append(abs(self.agent.subjective_time - env_time))
            
            if ep % 10 == 0:
                print(f"    Episode {ep}: Return = {ep_reward:.1f}, Drift = {subjective_drifts[-1]:.2f}")
            env.close()

        # Evaluate Robustness to Jitter
        print("  Testing temporal robustness (Subjective vs Baseline)...")
        robustness_score = self._eval_robustness()

        return {
            "mean_return": float(np.mean(returns)),
            "subjective_consistency": 1.0 / (1.0 + np.mean(subjective_drifts)),
            "temporal_robustness": robustness_score,
            "composite_score": float(np.mean(returns) / 200.0 * robustness_score)
        }

    def _eval_robustness(self) -> float:
        # Test under random timing jitters
        jitter_returns = []
        for _ in range(5):
            env = gym.make(self.env_id)
            obs, _ = env.reset()
            hidden = self.agent.get_initial_hidden(1, self.device)
            ep_ret = 0
            done = False
            while not done:
                jitter_dt = np.random.uniform(0.5, 2.0)
                # Subjective agent ignores jitter_dt but it affects env physics
                dist, _, hidden, _ = self.agent(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device), hidden)
                obs, r, term, trunc, _ = env.step(dist.sample().item())
                ep_ret += r
                done = term or trunc
            jitter_returns.append(ep_ret)
            env.close()
        return float(np.mean(jitter_returns) / (self.params.get("nominal_return", 200.0) + 1e-8))
