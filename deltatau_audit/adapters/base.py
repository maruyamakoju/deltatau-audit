"""Agent Adapter ABC — the minimal interface for auditing any RL agent.

Any recurrent RL agent can be audited by implementing this interface.
The adapter wraps the agent's forward pass into a standard form that
the auditor can call regardless of the underlying framework (SB3,
CleanRL, custom, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch


class AgentAdapter(ABC):
    """Minimal interface for auditing a recurrent RL agent.

    Subclass this and implement the three methods to audit your agent.
    """

    @abstractmethod
    def reset_hidden(self, batch: int = 1,
                     device: str = "cpu") -> Any:
        """Return the initial hidden state for a new episode.

        Returns:
            Hidden state (any type — tensor, tuple, etc.)
        """

    @abstractmethod
    def act(self, obs: torch.Tensor, hidden: Any
            ) -> Tuple[int, float, Any, Optional[float]]:
        """Single-step forward pass.

        Args:
            obs: Observation tensor, shape (obs_dim,) or (1, obs_dim)
            hidden: Current hidden state

        Returns:
            action: int — selected action
            value: float — value estimate V(s)
            hidden_new: updated hidden state
            dt: float or None — learned Δτ (None if agent has no time module)
        """

    def rerun_with_dt(self, obs: torch.Tensor, hidden: Any,
                      target_dt: float) -> Any:
        """Re-run the RNN transition with a specific Δτ override.

        Optional — only needed for intervention ablation. If not implemented,
        the auditor will skip intervention tests.

        Args:
            obs: Observation tensor
            hidden: Current hidden state (BEFORE this step's update)
            target_dt: The Δτ value to force

        Returns:
            hidden_new: Hidden state computed with the overridden Δτ
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support dt intervention. "
            "Implement rerun_with_dt() to enable intervention ablation."
        )

    def recompute_value(self, hidden: Any) -> float:
        """Compute value from a (possibly intervened) hidden state.

        Optional — used after rerun_with_dt to get value under intervention.
        Default uses act() which may not separate hidden update from value.

        Returns:
            value: float — V(s) computed from the given hidden state
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support recompute_value(). "
            "Implement it to enable value-based intervention ablation."
        )

    @property
    def supports_intervention(self) -> bool:
        """Whether this adapter supports dt intervention."""
        try:
            # Check if rerun_with_dt is overridden
            return type(self).rerun_with_dt is not AgentAdapter.rerun_with_dt
        except AttributeError:
            return False

    @property
    def supports_value_recompute(self) -> bool:
        """Whether this adapter supports value recomputation."""
        try:
            return type(self).recompute_value is not AgentAdapter.recompute_value
        except AttributeError:
            return False
