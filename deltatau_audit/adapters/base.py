"""Agent Adapter ABC — the minimal interface for auditing any RL agent.

Any recurrent RL agent can be audited by implementing this interface.
The adapter wraps the agent's forward pass into a standard form that
the auditor can call regardless of the underlying framework (SB3,
CleanRL, custom, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch

from deltatau_audit.schema import TemporalCapability


class AgentAdapter(ABC):
    """Unified interface for auditing any RL agent.

    Subclasses must implement the methods below to enable auditing.
    This interface supports both legacy timing-ablation and modern
    reasoning-aware auditing.
    """

    def get_capabilities(self) -> TemporalCapability:
        """Returns metadata about what the agent can do.

        Override this to enable reasoning-aware auditing features.
        """
        return TemporalCapability()

    @abstractmethod
    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = True,
        ponder_steps: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Single-step forward pass.

        Args:
            obs: Observation tensor.
            deterministic: Whether to use deterministic action selection.
            ponder_steps: Optional override for internal reasoning steps.

        Returns:
            action: Selected action (int or array).
            info: Dict containing 'value', 'dt', 'hidden', 'reasoning_trace', etc.
        """

    @abstractmethod
    def reset_internal_state(self) -> None:
        """Resets recurrent or internal reasoning states."""

    def rerun_with_dt(self, obs: torch.Tensor, target_dt: float) -> Dict[str, Any]:
        """Re-run the transition logic with a specific Δτ override.

        Optional — only needed for intervention ablation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support dt intervention."
        )

    def recompute_value(self, info: Dict[str, Any]) -> float:
        """Compute value from a (possibly intervened) internal state info."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support recompute_value()."
        )

    @property
    def supports_intervention(self) -> bool:
        """Whether this adapter supports dt intervention."""
        # Check if the class has overridden rerun_with_dt
        return self.__class__.rerun_with_dt is not AgentAdapter.rerun_with_dt

    @property
    def supports_value_recompute(self) -> bool:
        """Whether this adapter supports value recomputation."""
        # Check if the class has overridden recompute_value
        return self.__class__.recompute_value is not AgentAdapter.recompute_value
