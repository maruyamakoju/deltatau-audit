"""Generic adapter for any recurrent RL agent.

Wraps a user-provided callable interface. Users provide functions
rather than subclassing — lower barrier to entry.
"""

from typing import Any, Callable, Optional, Tuple

import torch

from .base import AgentAdapter


class GenericRecurrentAdapter(AgentAdapter):
    """Adapter that wraps user-provided callables.

    Usage:
        adapter = GenericRecurrentAdapter(
            reset_fn=lambda batch, device: torch.zeros(1, 64),
            act_fn=lambda obs, h: (action, value, h_new, None),
        )

    For intervention support, also provide:
        rerun_fn=lambda obs, h, dt: h_new,
        value_fn=lambda h: value_scalar,
    """

    def __init__(
        self,
        reset_fn: Callable[[int, str], Any],
        act_fn: Callable[[torch.Tensor, Any], Tuple[int, float, Any, Optional[float]]],
        rerun_fn: Optional[Callable[[torch.Tensor, Any, float], Any]] = None,
        value_fn: Optional[Callable[[Any], float]] = None,
    ):
        self._reset_fn = reset_fn
        self._act_fn = act_fn
        self._rerun_fn = rerun_fn
        self._value_fn = value_fn

    def reset_hidden(self, batch: int = 1, device: str = "cpu") -> Any:
        return self._reset_fn(batch, device)

    def act(self, obs: torch.Tensor, hidden: Any
            ) -> Tuple[int, float, Any, Optional[float]]:
        return self._act_fn(obs, hidden)

    def rerun_with_dt(self, obs: torch.Tensor, hidden: Any,
                      target_dt: float) -> Any:
        if self._rerun_fn is None:
            raise NotImplementedError("No rerun_fn provided")
        return self._rerun_fn(obs, hidden, target_dt)

    def recompute_value(self, hidden: Any) -> float:
        if self._value_fn is None:
            raise NotImplementedError("No value_fn provided")
        return self._value_fn(hidden)

    @property
    def supports_intervention(self) -> bool:
        return self._rerun_fn is not None

    @property
    def supports_value_recompute(self) -> bool:
        return self._value_fn is not None
