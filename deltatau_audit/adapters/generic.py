"""Generic adapter for any recurrent RL agent.

Wraps a user-provided callable interface. Users provide functions
rather than subclassing — lower barrier to entry.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .base import AgentAdapter


class GenericRecurrentAdapter(AgentAdapter):
    """Adapter that wraps user-provided callables."""

    def __init__(
        self,
        reset_fn: Callable[[], Any],
        act_fn: Callable[[torch.Tensor, Any], Tuple[Any, Dict[str, Any]]],
        rerun_fn: Optional[Callable[[torch.Tensor, float], Dict[str, Any]]] = None,
        value_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    ):
        self._reset_fn = reset_fn
        self._act_fn = act_fn
        self._rerun_fn = rerun_fn
        self._value_fn = value_fn
        self._hidden = None

    def reset_internal_state(self) -> None:
        self._hidden = self._reset_fn()

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = True,
        ponder_steps: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        if self._hidden is None:
            self.reset_internal_state()
        action, info = self._act_fn(obs, self._hidden)
        self._hidden = info.get("hidden")
        return action, info

    def rerun_with_dt(self, obs: torch.Tensor, target_dt: float) -> Dict[str, Any]:
        if self._rerun_fn is None:
            return super().rerun_with_dt(obs, target_dt)
        return self._rerun_fn(obs, target_dt)

    def recompute_value(self, info: Dict[str, Any]) -> float:
        if self._value_fn is None:
            return super().recompute_value(info)
        return self._value_fn(info)

    @property
    def supports_intervention(self) -> bool:
        return self._rerun_fn is not None

    @property
    def supports_value_recompute(self) -> bool:
        return self._value_fn is not None
