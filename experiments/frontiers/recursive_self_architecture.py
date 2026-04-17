"""Frontier 9: Recursive Self-Architecture (Adaptive Complexity).

Agents that dynamically reconfigure their neural topology and 
computational depth (ODE steps, active layers) based on perceived
environmental difficulty and adversarial pressure.
"""

from __future__ import annotations

import logging
import os
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
# 1. Recursive Self-Architecting Agent
# ═══════════════════════════════════════════════════════════════════════════════

class RecursiveSelfAgent(NeuralODEAgent):
    """An agent that scales its own computational complexity.
    
    Dynamically adjusts:
    - ODE integration steps (Temporal Resolution)
    - Residual block depth (Model Capacity)
    - Pondering gate (Early Exit)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Complexity controller: predicts optimal (ode_steps, active_layers)
        self.meta_controller = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2), # [resolution_gate, depth_gate]
            nn.Sigmoid()
        )
        
    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> Tuple:
        encoded = self.encoder(obs)
        
        # 1. Sense complexity needs from current state
        meta_params = self.meta_controller(hidden)
        res_gate = meta_params[:, 0]
        depth_gate = meta_params[:, 1]
        
        # 2. Scale ODE steps dynamically (e.g. between 2 and 20)
        dynamic_steps = int(2 + 18 * res_gate.item())
        
        # 3. Decision-aware integration
        self.ode_func.set_condition(encoded)
        dt = self.time_module(hidden, encoded)
        t_span = torch.tensor([0.0, dt.mean().item()], device=hidden.device)
        
        # High resolution integration if needed
        hidden_new = odeint(self.ode_func, hidden, t_span, method="rk4", num_steps=dynamic_steps)
        
        # 4. Optional 'Deep Thinking' layers
        if depth_gate.item() > 0.7:
            # Simulated extra depth: a second pass through dynamics to refine
            hidden_new = odeint(self.ode_func, hidden_new, t_span, method="euler", num_steps=5)

        # 5. Output
        if self.discrete_actions:
            logits = self.policy_head(hidden_new)
            dist = torch.distributions.Categorical(logits=logits)
        else:
            mean = self.policy_mean(hidden_new)
            std = self.policy_log_std.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)

        value = self.value_head(hidden_new).squeeze(-1)
        return dist, value, hidden_new, dynamic_steps, depth_gate.item()

class RecursiveAdapter(AgentAdapter):
    def __init__(self, agent: RecursiveSelfAgent, device: str = "cpu"):
        super().__init__(agent, device)
        self.hidden = None

    def act(self, obs: torch.Tensor) -> Tuple[Any, Dict[str, Any]]:
        if self.hidden is None:
            self.hidden = self.agent.get_initial_hidden(1, self.device)
        
        dist, value, self.hidden, steps, depth = self.agent(obs.to(self.device).unsqueeze(0), self.hidden)
        action = dist.sample()
        
        return action.item() if self.agent.discrete_actions else action.cpu().numpy()[0], {
            "value": value.item(),
            "computation_steps": steps,
            "depth_activation": depth
        }

    def reset_internal_state(self):
        self.hidden = None

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Frontier Experiment: Survival vs Complexity
# ═══════════════════════════════════════════════════════════════════════════════

class RecursiveSelfArchitectureExperiment:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.env_id = params.get("env", "CartPole-v1")
        self.device = params.get("device", "cpu")
        
        self.agent = RecursiveSelfAgent(
            obs_dim=params.get("obs_dim", 4),
            act_dim=params.get("act_dim", 2),
            hidden_dim=params.get("hidden_dim", 64),
        ).to(self.device)
        
        # Meta-optimization: reward both performance AND efficiency
        self.optimizer = optim.Adam(self.agent.parameters(), lr=params.get("lr", 1e-3))

    def run(self, out_dir: Path) -> Dict[str, float]:
        print(f"  Evolving Recursive Self-Architecture on {self.env_id}...")
        
        n_episodes = self.params.get("n_episodes", 40)
        returns = []
        avg_steps = []
        
        for ep in range(n_episodes):
            env = gym.make(self.env_id)
            obs, _ = env.reset()
            hidden = self.agent.get_initial_hidden(1, self.device)
            
            ep_ret = 0
            ep_steps = []
            log_probs = []
            
            done = False
            while not done:
                dist, value, hidden, steps, depth = self.agent(
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device),
                    hidden
                )
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                ep_steps.append(steps)
                
                obs, r, term, trunc, _ = env.step(action.item())
                ep_ret += r
                done = term or trunc
                if len(log_probs) > 500: break
            
            # Loss: Performance - Efficiency Penalty (cost of computation)
            efficiency_penalty = np.mean(ep_steps) * 0.01
            loss = -torch.stack(log_probs).sum() * (ep_ret - efficiency_penalty) / 100.0
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            returns.append(ep_ret)
            avg_steps.append(np.mean(ep_steps))
            env.close()

        # Breakthrough check: Does complexity spike under simulated 'Stress'?
        print("  Evaluating complexity adaptivity under stress...")
        stress_adaptation = self._test_stress_adaptation()

        return {
            "mean_return": float(np.mean(returns)),
            "computation_efficiency": float(np.mean(returns) / (np.mean(avg_steps) + 1e-8)),
            "complexity_adaptivity": stress_adaptation,
            "composite_score": float(np.mean(returns) / 200.0 * stress_adaptation)
        }

    def _test_stress_adaptation(self) -> float:
        # Measure if ODE steps increase when observations are jittered
        normal_steps = []
        stress_steps = []
        
        env = gym.make(self.env_id)
        obs, _ = env.reset()
        hidden = self.agent.get_initial_hidden(1, self.device)
        
        for _ in range(20):
            # Normal
            _, _, hidden, steps, _ = self.agent(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device), hidden)
            normal_steps.append(steps)
            # Stress (add noise to trigger complexity)
            noisy_obs = torch.tensor(obs + np.random.normal(0, 0.5, obs.shape), dtype=torch.float32).unsqueeze(0).to(self.device)
            _, _, hidden, steps, _ = self.agent(noisy_obs, hidden)
            stress_steps.append(steps)
            
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc: obs, _ = env.reset(); hidden = self.agent.get_initial_hidden(1, self.device)
            
        env.close()
        # Adaptivity = Ratio of steps under stress vs normal (higher is more responsive)
        return float(np.mean(stress_steps) / (np.mean(normal_steps) + 1e-8))
