"""Tests for v0.6.1: SB3 TimingAuditCallback."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


# ── Import tests (no SB3 required) ─────────────────────────────────

def test_callback_module_importable():
    """callback.py can be imported without SB3 installed."""
    from deltatau_audit import callback
    assert hasattr(callback, "TimingAuditCallback")
    assert hasattr(callback, "create_timing_audit_callback")


def test_callback_class_instantiable_standalone():
    """TimingAuditCallback can be instantiated without SB3 (plain class)."""
    from deltatau_audit.callback import TimingAuditCallback
    cb = TimingAuditCallback(env_id="CartPole-v1", audit_freq=10_000)
    assert cb._env_id == "CartPole-v1"
    assert cb._audit_freq == 10_000
    assert cb._n_episodes == 10
    assert cb._speeds == [1, 2, 3, 5, 8]
    assert cb._output_dir is None
    assert cb.audit_history == []


def test_callback_custom_params():
    """Custom parameters propagate correctly."""
    from deltatau_audit.callback import TimingAuditCallback
    cb = TimingAuditCallback(
        env_id="HalfCheetah-v5",
        audit_freq=100_000,
        n_episodes=20,
        speeds=[1, 3, 5],
        n_workers=4,
        device="cuda",
        seed=42,
        output_dir="logs/audit",
        verbose=1,
    )
    assert cb._env_id == "HalfCheetah-v5"
    assert cb._audit_freq == 100_000
    assert cb._n_episodes == 20
    assert cb._speeds == [1, 3, 5]
    assert cb._n_workers == 4
    assert cb._device == "cuda"
    assert cb._seed == 42
    assert cb._output_dir == "logs/audit"
    assert cb._verbose_level == 1


def test_factory_raises_without_sb3():
    """create_timing_audit_callback raises ImportError without SB3."""
    # Temporarily hide SB3
    with patch.dict(sys.modules, {"stable_baselines3": None,
                                  "stable_baselines3.common": None,
                                  "stable_baselines3.common.callbacks": None}):
        # Force re-resolution
        from deltatau_audit.callback import create_timing_audit_callback
        with pytest.raises(ImportError, match="stable-baselines3"):
            create_timing_audit_callback(env_id="CartPole-v1")


def test_audit_history_property():
    """audit_history returns a copy."""
    from deltatau_audit.callback import TimingAuditCallback
    cb = TimingAuditCallback(env_id="CartPole-v1")
    cb._audit_history.append({"timestep": 100, "deployment_score": 0.8})
    hist = cb.audit_history
    assert len(hist) == 1
    assert hist[0]["timestep"] == 100
    # Modifying returned list doesn't affect internal state
    hist.clear()
    assert len(cb._audit_history) == 1


# ── _run_audit logic tests (mocked) ───────────────────────────────

def _mock_audit_result():
    return {
        "summary": {
            "deployment_score": 0.85,
            "stress_score": 0.55,
            "deployment_rating": "MILD",
            "stress_rating": "MILD",
            "robustness_score": 0.55,
            "quadrant": "deployment_ready",
        },
        "robustness": {
            "per_scenario_scores": {
                "jitter": {"return_ratio": 0.9},
                "delay": {"return_ratio": 0.85},
                "speed_5x": {"return_ratio": 0.55},
            },
        },
    }


def test_run_audit_logs_metrics():
    """_run_audit logs metrics to SB3 logger."""
    from deltatau_audit.callback import TimingAuditCallback

    cb = TimingAuditCallback(env_id="CartPole-v1")
    cb.num_timesteps = 50000
    cb.logger = MagicMock()
    cb.model = MagicMock()

    with patch("deltatau_audit.adapters.sb3.SB3Adapter"), \
         patch("deltatau_audit.auditor.run_full_audit", return_value=_mock_audit_result()):
        cb._run_audit()

    # Check logger.record was called with audit/ prefix
    calls = {c[0][0]: c[0][1] for c in cb.logger.record.call_args_list}
    assert "audit/deployment_score" in calls
    assert "audit/stress_score" in calls
    assert "audit/deployment_rating" in calls
    assert calls["audit/deployment_score"] == 0.85
    assert calls["audit/stress_score"] == 0.55


def test_run_audit_appends_history():
    """_run_audit appends to audit_history."""
    from deltatau_audit.callback import TimingAuditCallback

    cb = TimingAuditCallback(env_id="CartPole-v1")
    cb.num_timesteps = 100000
    cb.logger = MagicMock()
    cb.model = MagicMock()

    with patch("deltatau_audit.auditor.run_full_audit", return_value=_mock_audit_result()), \
         patch("deltatau_audit.adapters.sb3.SB3Adapter"):
        cb._run_audit()

    assert len(cb.audit_history) == 1
    entry = cb.audit_history[0]
    assert entry["timestep"] == 100000
    assert entry["deployment_score"] == 0.85
    assert entry["deployment_rating"] == "MILD"


def test_run_audit_saves_report(tmp_path):
    """_run_audit saves report when output_dir is set."""
    from deltatau_audit.callback import TimingAuditCallback

    out_dir = str(tmp_path / "audit_logs")
    cb = TimingAuditCallback(env_id="CartPole-v1", output_dir=out_dir)
    cb.num_timesteps = 25000
    cb.logger = MagicMock()
    cb.model = MagicMock()

    with patch("deltatau_audit.auditor.run_full_audit", return_value=_mock_audit_result()), \
         patch("deltatau_audit.adapters.sb3.SB3Adapter"), \
         patch("deltatau_audit.report.generate_report") as mock_report:
        cb._run_audit()

    mock_report.assert_called_once()
    call_args = mock_report.call_args
    assert "step_25000" in call_args[0][1]


def test_on_step_respects_frequency():
    """_on_step only triggers audit at the right frequency."""
    from deltatau_audit.callback import TimingAuditCallback

    cb = TimingAuditCallback(env_id="CartPole-v1", audit_freq=1000)
    cb.logger = MagicMock()
    cb.model = MagicMock()

    # Step 500 — too early, should not audit
    cb.num_timesteps = 500
    cb._last_audit_step = 0
    with patch.object(cb, "_run_audit") as mock_run:
        cb._on_step()
        mock_run.assert_not_called()

    # Step 1000 — should audit
    cb.num_timesteps = 1000
    with patch.object(cb, "_run_audit") as mock_run:
        cb._on_step()
        mock_run.assert_called_once()
