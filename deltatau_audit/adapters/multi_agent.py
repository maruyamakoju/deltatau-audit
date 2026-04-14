"""
Multi-Agent Adapter for Temporal Robustness.

Allows auditing a team of agents where each agent might have its own
distinct internal clock or communication latency.
"""

from typing import Any, List, Optional, Tuple

from .base import AgentAdapter


class MultiAgentAdapter(AgentAdapter):
    """
    Wraps multiple AgentAdapters into a single team.
    Useful for Cooperative RL or competitive scenarios where
    temporal desynchronization matters.
    """

    def __init__(self, adapters: List[AgentAdapter]):
        self.adapters = adapters
        self.n_agents = len(adapters)
        # Check if all agents support intervention
        self.supports_intervention = all(a.supports_intervention for a in adapters)

    def reset_hidden(self, batch: int = 1, device: str = "cpu") -> List[Any]:
        """Returns a list of hidden states, one per agent."""
        return [adapter.reset_hidden(batch, device) for adapter in self.adapters]

    def act(self, obs: Any, hidden: Any) -> Tuple[List[Any], List[float], List[Any], List[Optional[float]]]:
        """
        obs: Expected to be a list/tuple of observations, one per agent.
        hidden: List of hidden states from previous step.
        """
        actions = []
        values = []
        hiddens_new = []
        dts = []

        for i, adapter in enumerate(self.adapters):
            # Pass individual agent obs and hidden
            a, v, h, dt = adapter.act(obs[i], hidden[i])
            actions.append(a)
            values.append(v)
            hiddens_new.append(h)
            dts.append(dt)

        return actions, values, hiddens_new, dts

    def rerun_with_dt(self, obs: Any, hidden: Any, target_dts: List[float]) -> List[Any]:
        hiddens_new = []
        for i, adapter in enumerate(self.adapters):
            h = adapter.rerun_with_dt(obs[i], hidden[i], target_dts[i])
            hiddens_new.append(h)
        return hiddens_new

    def recompute_value(self, hiddens: List[Any]) -> List[float]:
        return [adapter.recompute_value(hiddens[i]) for i, adapter in enumerate(self.adapters)]
