"""Tests for deltatau_audit.adapters — supports_intervention detection."""

import pytest
import torch

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
            reset_fn=lambda: torch.zeros(1, 4),
            act_fn=lambda obs, h: (0, {"value": 1.0, "hidden": h, "dt": None}),
        )
        assert adapter.supports_intervention is False
        assert adapter.supports_value_recompute is False

    def test_with_intervention(self):
        adapter = GenericRecurrentAdapter(
            reset_fn=lambda: torch.zeros(1, 4),
            act_fn=lambda obs, h: (0, {"value": 1.0, "hidden": h, "dt": 1.0}),
            rerun_fn=lambda obs, dt: {"hidden": torch.zeros(1, 4), "dt": dt},
            value_fn=lambda info: 0.5,
        )
        assert adapter.supports_intervention is True
        assert adapter.supports_value_recompute is True

    def test_rerun_only(self):
        adapter = GenericRecurrentAdapter(
            reset_fn=lambda: torch.zeros(1, 4),
            act_fn=lambda obs, h: (0, {"value": 1.0, "hidden": h, "dt": None}),
            rerun_fn=lambda obs, dt: {"hidden": torch.zeros(1, 4), "dt": dt},
        )
        assert adapter.supports_intervention is True
        assert adapter.supports_value_recompute is False


# ── Adapter contract ─────────────────────────────────────────────────

class TestAdapterContract:
    def test_act_returns_info_dict(self, dummy_adapter):
        obs = torch.randn(4)
        dummy_adapter.reset_internal_state()
        result = dummy_adapter.act(obs)
        assert len(result) == 2
        action, info = result
        assert isinstance(action, int)
        assert isinstance(info, dict)
        assert "value" in info
        assert info["dt"] == 1.0  # dummy constant

    def test_rerun_raises_on_base(self, dummy_adapter):
        obs = torch.randn(4)
        dummy_adapter.reset_internal_state()
        with pytest.raises(NotImplementedError):
            dummy_adapter.rerun_with_dt(obs, 1.0)

    def test_recompute_raises_on_base(self, dummy_adapter):
        dummy_adapter.reset_internal_state()
        with pytest.raises(NotImplementedError):
            dummy_adapter.recompute_value({"hidden": None})

    def test_intervention_adapter_rerun(self, intervention_adapter):
        obs = torch.randn(4)
        intervention_adapter.reset_internal_state()
        info_new = intervention_adapter.rerun_with_dt(obs, 2.0)
        assert "hidden" in info_new
        assert info_new["dt"] == 2.0

    def test_intervention_adapter_recompute(self, intervention_adapter):
        intervention_adapter.reset_internal_state()
        val = intervention_adapter.recompute_value({"hidden": torch.zeros(1, 4)})
        assert isinstance(val, float)
