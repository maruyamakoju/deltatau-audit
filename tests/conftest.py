"""Shared fixtures for deltatau-audit tests."""

import pytest
import torch
from typing import Any, Optional, Tuple

from deltatau_audit.adapters.base import AgentAdapter


class DummyAdapter(AgentAdapter):
    """Minimal adapter with NO intervention support."""

    def reset_hidden(self, batch: int = 1, device: str = "cpu") -> Any:
        return torch.zeros(batch, 4)

    def act(self, obs: torch.Tensor, hidden: Any
            ) -> Tuple[int, float, Any, Optional[float]]:
        return 0, 1.0, hidden, None


class InterventionAdapter(AgentAdapter):
    """Adapter WITH intervention + value recompute support."""

    def reset_hidden(self, batch: int = 1, device: str = "cpu") -> Any:
        return torch.zeros(batch, 4)

    def act(self, obs: torch.Tensor, hidden: Any
            ) -> Tuple[int, float, Any, Optional[float]]:
        return 0, 1.0, hidden + 0.1, 1.0

    def rerun_with_dt(self, obs: torch.Tensor, hidden: Any,
                      target_dt: float) -> Any:
        return hidden + target_dt

    def recompute_value(self, hidden: Any) -> float:
        return float(hidden.mean())


class ValueOnlyAdapter(AgentAdapter):
    """Adapter with recompute_value but NO rerun_with_dt."""

    def reset_hidden(self, batch: int = 1, device: str = "cpu") -> Any:
        return torch.zeros(batch, 4)

    def act(self, obs: torch.Tensor, hidden: Any
            ) -> Tuple[int, float, Any, Optional[float]]:
        return 0, 1.0, hidden, None

    def recompute_value(self, hidden: Any) -> float:
        return float(hidden.mean())


@pytest.fixture
def dummy_adapter():
    return DummyAdapter()


@pytest.fixture
def intervention_adapter():
    return InterventionAdapter()


@pytest.fixture
def value_only_adapter():
    return ValueOnlyAdapter()
