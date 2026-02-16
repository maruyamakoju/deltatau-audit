"""Tests for deltatau_audit.adapters — supports_intervention detection."""

import pytest
import torch

from deltatau_audit.adapters.base import AgentAdapter
from deltatau_audit.adapters.generic import GenericRecurrentAdapter


# ── Base adapter property detection ───────────────────────────────────

class TestAdapterProperties:
    def test_dummy_no_intervention(self, dummy_adapter):
        assert dummy_adapter.supports_intervention is False
        assert dummy_adapter.supports_value_recompute is False

    def test_intervention_adapter(self, intervention_adapter):
        assert intervention_adapter.supports_intervention is True
        assert intervention_adapter.supports_value_recompute is True

    def test_value_only_adapter(self, value_only_adapter):
        """Has recompute_value but not rerun_with_dt."""
        assert value_only_adapter.supports_intervention is False
        assert value_only_adapter.supports_value_recompute is True


# ── GenericRecurrentAdapter ───────────────────────────────────────────

class TestGenericAdapter:
    def test_no_intervention(self):
        adapter = GenericRecurrentAdapter(
            reset_fn=lambda b, d: torch.zeros(b, 4),
            act_fn=lambda obs, h: (0, 1.0, h, None),
        )
        assert adapter.supports_intervention is False
        assert adapter.supports_value_recompute is False

    def test_with_intervention(self):
        adapter = GenericRecurrentAdapter(
            reset_fn=lambda b, d: torch.zeros(b, 4),
            act_fn=lambda obs, h: (0, 1.0, h, 1.0),
            rerun_fn=lambda obs, h, dt: h,
            value_fn=lambda h: 0.5,
        )
        assert adapter.supports_intervention is True
        assert adapter.supports_value_recompute is True

    def test_rerun_only(self):
        adapter = GenericRecurrentAdapter(
            reset_fn=lambda b, d: torch.zeros(b, 4),
            act_fn=lambda obs, h: (0, 1.0, h, None),
            rerun_fn=lambda obs, h, dt: h,
        )
        assert adapter.supports_intervention is True
        assert adapter.supports_value_recompute is False


# ── Adapter contract (act returns 4-tuple) ────────────────────────────

class TestAdapterContract:
    def test_act_returns_4_tuple(self, dummy_adapter):
        obs = torch.randn(4)
        hidden = dummy_adapter.reset_hidden()
        result = dummy_adapter.act(obs, hidden)
        assert len(result) == 4
        action, value, hidden_new, dt = result
        assert isinstance(action, int)
        assert isinstance(value, float)
        assert dt is None  # dummy has no dt

    def test_rerun_raises_on_base(self, dummy_adapter):
        obs = torch.randn(4)
        hidden = dummy_adapter.reset_hidden()
        with pytest.raises(NotImplementedError):
            dummy_adapter.rerun_with_dt(obs, hidden, 1.0)

    def test_recompute_raises_on_base(self, dummy_adapter):
        hidden = dummy_adapter.reset_hidden()
        with pytest.raises(NotImplementedError):
            dummy_adapter.recompute_value(hidden)

    def test_intervention_adapter_rerun(self, intervention_adapter):
        obs = torch.randn(4)
        hidden = intervention_adapter.reset_hidden()
        hidden_new = intervention_adapter.rerun_with_dt(obs, hidden, 2.0)
        assert hidden_new is not None

    def test_intervention_adapter_recompute(self, intervention_adapter):
        hidden = intervention_adapter.reset_hidden()
        val = intervention_adapter.recompute_value(hidden)
        assert isinstance(val, float)
