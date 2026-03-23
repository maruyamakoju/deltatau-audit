"""Adapter for DeliberativeInternalTimeAgent.

Wraps DeliberativeInternalTimeAgent (ACT-based) to expose:
- Standard AgentAdapter interface (act, reset_hidden)
- Deliberation-specific metrics: mean_ponder_steps, ponder_utilization,
  halt_efficiency

These metrics allow the auditor to answer:
  "Does the agent ponder MORE when timing is uncertain?"
A good deliberative agent should increase thinking steps under jitter.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from deltatau_audit.adapters.base import AgentAdapter


class DeliberativeAgentAdapter(AgentAdapter):
    """Wraps DeliberativeInternalTimeAgent for the deltatau audit framework.

    Tracks per-step ponder statistics and exposes them via get_deliberation_stats().

    Args:
        agent: A DeliberativeInternalTimeAgent instance.
        device: Torch device string.
    """

    def __init__(self, agent, device: str = "cpu"):
        self._agent = agent
        self._device = device

        # Ponder tracking across steps
        self._episode_ponder_steps: List[float] = []
        self._episode_halt_probs: List[float] = []

    def reset_hidden(self, batch: int = 1, device: str = "cpu") -> torch.Tensor:
        """Return zero initial hidden state."""
        d = device or self._device
        return torch.zeros(batch, self._agent.hidden_dim, device=d)

    def act(
        self,
        obs: torch.Tensor,
        hidden: Any,
    ) -> Tuple[int, float, torch.Tensor, Optional[float]]:
        """Single-step forward pass with ponder tracking.

        Returns:
            action: int action
            value: float value estimate
            hidden_new: updated hidden (weighted sum from ACT)
            dt: mean ponder steps (as proxy for internal time spent)
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if hidden is None:
            hidden = self.reset_hidden(obs.shape[0], self._device)

        obs = obs.to(self._device)
        hidden = hidden.to(self._device) if isinstance(hidden, torch.Tensor) else hidden

        with torch.no_grad():
            dist, value, hidden_new, cumulative_halt, ponder_cost = self._agent.forward(
                obs, hidden, deterministic=True
            )
            action = dist.sample()

        ponder_steps = float(ponder_cost.mean().item())
        halt_prob = float(cumulative_halt.mean().item())

        self._episode_ponder_steps.append(ponder_steps)
        self._episode_halt_probs.append(halt_prob)

        return (
            int(action.item()),
            float(value.mean().item()),
            hidden_new,
            ponder_steps,  # expose as dt — "time spent thinking"
        )

    def rerun_with_dt(
        self,
        obs: torch.Tensor,
        hidden: Any,
        target_dt: float,
    ) -> torch.Tensor:
        """Deliberative agent does not support dt override (ACT is self-determined)."""
        raise NotImplementedError(
            "DeliberativeAgentAdapter: deliberation time is self-determined by ACT. "
            "dt override is not supported."
        )

    def reset_episode(self) -> None:
        """Reset per-episode ponder tracking."""
        self._episode_ponder_steps = []
        self._episode_halt_probs = []

    def get_deliberation_stats(self) -> Dict[str, float]:
        """Return ponder statistics accumulated since last reset_episode().

        Returns:
            Dict with:
                mean_ponder_steps: Average number of thinking steps per action.
                ponder_utilization: Fraction of max_thinking_steps used on average.
                halt_efficiency: Average halt probability (1.0 = always fully halted).
        """
        if not self._episode_ponder_steps:
            return {
                "mean_ponder_steps": 0.0,
                "ponder_utilization": 0.0,
                "halt_efficiency": 0.0,
            }

        max_steps = float(self._agent.max_thinking_steps)
        mean_steps = float(np.mean(self._episode_ponder_steps))
        utilization = mean_steps / max_steps if max_steps > 0 else 0.0
        halt_eff = float(np.mean(self._episode_halt_probs))

        return {
            "mean_ponder_steps": mean_steps,
            "ponder_utilization": utilization,
            "halt_efficiency": halt_eff,
        }
