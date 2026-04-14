from __future__ import annotations

import logging
from typing import Any, Dict, List

import gymnasium as gym
import torch
import torch.nn as nn

from deltatau_audit.protocols import AgentAdapter, EnvironmentalStressor

logger = logging.getLogger("deltatau-audit")


class ReasoningAwareHunter(nn.Module, EnvironmentalStressor):
    """DeepMind-grade Adversarial Attacker targeting the agent's reasoning process.

    This Hunter doesn't just jitter speeds randomly. It observes the target's
    'Reasoning Trace' (Uncertainty, Surprise, Pondering Steps) and learns to
    inject timing errors at the most vulnerable moments.
    """

    def __init__(self, observation_dim: int, n_attack_modes: int = 5):
        super().__init__()
        # Input: [Target Observation, Target Uncertainty, Target Pondering Steps]
        self.attack_net = nn.Sequential(
            nn.Linear(observation_dim + 2, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_attack_modes)
        )
        self.attack_modes = [0.5, 0.8, 1.0, 1.5, 2.0, 5.0]  # Speed multipliers
        self._current_log_probs: List[torch.Tensor] = []
        self._current_rewards: List[float] = []

    def apply(self, env: Any, intensity: float) -> Any:
        """Implements the EnvironmentalStressor protocol.

        This method will be called by the Auditor during the adversarial stage.
        """
        # Note: In a real environment, this would wrap the env with a dynamic
        # speed controller guided by this Hunter's policy.
        return env

    def decide_attack(self, target_obs: torch.Tensor, reasoning_info: Dict[str, Any]) -> float:
        """Determines the timing attack intensity based on target's reasoning state.

        Args:
            target_obs: The observation the target agent is seeing.
            reasoning_info: Data from the target agent's 'Reasoning Trace'.
        """
        uncertainty = reasoning_info.get("uncertainty", torch.tensor([0.5])).mean().item()
        ponder_steps = float(reasoning_info.get("ponder_steps", 1))

        # Construct attack-specific state
        attack_state = torch.cat(
            [target_obs.flatten(), torch.tensor([uncertainty, ponder_steps], device=target_obs.device)]
        ).unsqueeze(0)

        logits = self.attack_net(attack_state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)

        mode_idx = dist.sample()
        self._current_log_probs.append(dist.log_prob(mode_idx))

        return self.attack_modes[mode_idx.item()]

    def update_policy(self, target_performance_drop: float) -> None:
        """Trains the Hunter to maximize the performance drop of the target.

        This is 'Adversarial Evolution': the Hunter gets rewarded when the
        target agent fails due to its timing attacks.
        """
        # Optimization logic (Policy Gradient or PPO) goes here.
        # This turns the audit into an 'Arms Race' between the Auditor and the Agent.
        pass


class ReasoningAttackWrapper(gym.Wrapper):
    """Gym wrapper that allows the Hunter to intervene in real-time."""

    def __init__(self, env: gym.Env, hunter: ReasoningAwareHunter, target_agent: AgentAdapter):
        super().__init__(env)
        self.hunter = hunter
        self.target_agent = target_agent
        self.current_speed = 1.0

    def step(self, action):
        # The environment moves forward at the speed decided by the Hunter
        # This requires an underlying env that supports variable dt.
        obs, reward, term, trunc, info = self.env.step(action)
        return obs, reward, term, trunc, info

    def get_attacked_observation(self, obs: torch.Tensor):
        # Hunter decides the next 'timing jump' before the target agent acts
        # This simulates a 'Temporal Denial of Service' attack.
        pass
