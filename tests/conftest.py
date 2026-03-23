"""Shared pytest fixtures and helpers for deltatau-audit test suite.

Import these in individual test files with:
    from conftest import _DummyAdapter, _ConstAdapter, make_result, make_summary
or use them as pytest fixtures.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from itertools import count
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
import torch

from deltatau_audit.adapters.base import AgentAdapter


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEST_TEMP_ROOT = _REPO_ROOT / ".tmp" / "pytest_temp_root"
_TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_TEST_TEMP_ROOT)
os.environ["TMP"] = str(_TEST_TEMP_ROOT)
os.environ["TEMP"] = str(_TEST_TEMP_ROOT)
tempfile.tempdir = str(_TEST_TEMP_ROOT)
_TEMP_COUNTER = count()


def _make_workspace_temp_path(prefix: str = "tmp") -> Path:
    for _ in range(1024):
        path = _TEST_TEMP_ROOT / f"{prefix}_{next(_TEMP_COUNTER):06d}"
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass
            if path.exists():
                continue
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return path
    raise RuntimeError(f"failed to allocate workspace temp path for prefix={prefix!r}")


class _WorkspaceTemporaryDirectory:
    """Workspace-local replacement for tempfile.TemporaryDirectory in tests."""

    def __init__(
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | os.PathLike[str] | None = None,
        ignore_cleanup_errors: bool = False,
    ):
        del suffix  # unused: tests only require a writable scratch dir
        del dir
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._path = _make_workspace_temp_path(prefix=prefix or "tmpdir")
        self.name = str(self._path)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        shutil.rmtree(self._path, ignore_errors=True)


tempfile.TemporaryDirectory = _WorkspaceTemporaryDirectory


@pytest.fixture
def tmp_path() -> Path:
    path = _make_workspace_temp_path(prefix="pytest")
    yield path
    shutil.rmtree(path, ignore_errors=True)

# ── Minimal adapter helpers ───────────────────────────────────────────────────

class _DummyAdapter:
    """Adapter that returns a fixed action and constant hidden state."""
    supports_intervention = False
    supports_value_recompute = False

    def reset(self):
        return np.zeros(1)

    def reset_hidden(self, batch: int = 1, device: str = "cpu"):
        return np.zeros(1)

    def act(self, obs, hidden=None):
        return 0, 0.0, np.zeros(1), None

    def rerun_with_dt(self, obs_seq, hidden_init, dt_seq):
        raise NotImplementedError("supports_intervention=False")

    def recompute_value(self, hidden):
        raise NotImplementedError("supports_value_recompute=False")


class _ConstAdapter:
    """Adapter returning a fixed reward and configurable hidden state."""
    supports_intervention = True
    supports_value_recompute = False

    def __init__(self, reward: float = 1.0, rmse_base: float = 0.1):
        self._reward = reward
        self._rmse_base = rmse_base

    def reset(self):
        return np.zeros(4)

    def act(self, obs, hidden=None):
        return 0, float(self._reward), np.zeros(4), None

    def rerun_with_dt(self, obs_seq, hidden_init, dt_seq):
        n = len(obs_seq)
        return {
            "values": [float(self._rmse_base)] * n,
            "returns": [0.0] * n,
        }


# ── Summary/result factories ──────────────────────────────────────────────────

def make_summary(
    deployment_score: float = 0.90,
    deployment_rating: str = "PASS",
    stress_score: float = 0.80,
    stress_rating: str = "MILD",
    reliance_score: Optional[float] = None,
    reliance_rating: str = "N/A",
    quadrant: str = "deployment_ready",
    deploy_threshold: float = 0.80,
    stress_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Create a minimal valid summary dict for testing."""
    return {
        "reliance_rating": reliance_rating,
        "reliance_score": reliance_score,
        "robustness_rating": deployment_rating,
        "robustness_score": deployment_score,
        "robustness_rmse_score": 1.0,
        "deployment_rating": deployment_rating,
        "deployment_score": deployment_score,
        "stress_rating": stress_rating,
        "stress_score": stress_score,
        "sensitivity_mean": None,
        "deploy_threshold": deploy_threshold,
        "stress_threshold": stress_threshold,
        "quadrant": quadrant,
        "prescription": "Test prescription text.",
    }


def make_robustness(
    jitter: float = 0.95,
    delay: float = 0.90,
    spike: float = 0.88,
    obs_noise: float = 0.99,
    speed_5x: float = 0.60,
) -> Dict[str, Any]:
    """Create a minimal valid robustness dict for testing."""
    nominal_return = 500.0

    def _sc(ratio: float) -> Dict[str, Any]:
        drop = (1.0 - ratio) * 100
        return {
            "return_ratio": ratio,
            "return_drop_pct": drop,
            "rmse_ratio": 1.0,
            "rmse_increase_pct": 0.0,
            "ci_lower": ratio - 0.05,
            "ci_upper": ratio + 0.05,
            "significant": ratio < 0.95,
        }

    worst_dep = min(jitter, delay, spike, obs_noise)
    worst_dep_name = min(
        [("jitter", jitter), ("delay", delay), ("spike", spike), ("obs_noise", obs_noise)],
        key=lambda x: x[1],
    )[0]

    from deltatau_audit.metrics import robustness_rating
    dep_rating = robustness_rating(worst_dep)
    str_rating = robustness_rating(speed_5x)

    return {
        "scenarios": {
            "nominal": {
                "n_episodes": 50,
                "total_reward_mean": nominal_return,
                "total_reward_std": 20.0,
                "total_reward_se": 2.0,
                "length_mean": 1000.0,
                "length_std": 0.0,
                "length_se": 0.0,
                "rmse_mean": 10.0,
                "rmse_std": 1.0,
                "rmse_se": 0.1,
                "mae_mean": 9.0,
                "mae_std": 0.9,
                "mae_se": 0.09,
                "bias_mean": 0.5,
                "bias_std": 0.2,
                "bias_se": 0.02,
            }
        },
        "per_scenario_scores": {
            "jitter": _sc(jitter),
            "delay": _sc(delay),
            "spike": _sc(spike),
            "obs_noise": _sc(obs_noise),
            "speed_5x": _sc(speed_5x),
        },
        "deployment": {
            "return_score": worst_dep,
            "rmse_score": 1.0,
            "rating": dep_rating,
            "worst_case": {"scenario": worst_dep_name, "return_ratio": worst_dep,
                           "return_drop_pct": (1 - worst_dep) * 100},
        },
        "stress": {
            "return_score": speed_5x,
            "rmse_score": 1.0,
            "rating": str_rating,
            "worst_case": {"scenario": "speed_5x", "return_ratio": speed_5x,
                           "return_drop_pct": (1 - speed_5x) * 100},
        },
        "return_score": min(worst_dep, speed_5x),
        "rmse_score": 1.0,
        "rating": robustness_rating(min(worst_dep, speed_5x)),
        "worst_case": {"scenario": "speed_5x" if speed_5x < worst_dep else worst_dep_name},
        "n_episodes_used": {sc: 50 for sc in ["nominal", "jitter", "delay", "spike", "obs_noise", "speed_5x"]},
        "adaptive": False,
    }


def make_result(
    deployment_score: float = 0.90,
    stress_score: float = 0.80,
    n_episodes: int = 50,
) -> Dict[str, Any]:
    """Create a minimal valid audit result dict for testing."""
    summary = make_summary(deployment_score=deployment_score, stress_score=stress_score)
    robustness = make_robustness()
    return {
        "speeds": [1, 2, 3, 5, 8],
        "n_episodes": n_episodes,
        "supports_intervention": False,
        "reliance": {
            "per_speed": {},
            "degradation": {},
            "score": None,
            "rating": "N/A",
            "worst_case": {"speed": None, "intervention": None, "rmse_ratio": None, "percent": None},
        },
        "robustness": robustness,
        "sensitivity": None,
        "summary": summary,
        "diagnosis": {
            "status": "pass",
            "failing_scenarios": [],
            "issues": [],
            "primary_pattern": None,
            "root_cause": None,
            "fix_recommendation": None,
            "summary_line": "No significant timing failures detected.",
        },
        "_version": "0.7.0",
        "_timestamp": "2026-02-21T00:00:00.000000+00:00",
    }


# ── Pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def dummy_adapter():
    return _DummyAdapter()


@pytest.fixture
def const_adapter():
    return _ConstAdapter()


@pytest.fixture
def passing_summary():
    return make_summary(deployment_score=0.95, deployment_rating="PASS",
                        stress_score=0.85, stress_rating="MILD")


@pytest.fixture
def failing_summary():
    return make_summary(deployment_score=0.20, deployment_rating="FAIL",
                        stress_score=-0.10, stress_rating="FAIL",
                        quadrant="deployment_fragile")


@pytest.fixture
def passing_robustness():
    return make_robustness(jitter=0.97, delay=0.96, spike=0.98, obs_noise=0.99, speed_5x=0.70)


@pytest.fixture
def failing_robustness():
    return make_robustness(jitter=0.13, delay=-0.04, spike=0.89, obs_noise=0.98, speed_5x=-0.13)


# ── Adapter fixtures for test_adapters.py ────────────────────────────────────


class _InterventionAdapter(AgentAdapter):
    """Adapter that supports both dt intervention and value recompute."""

    def reset_hidden(self, batch: int = 1, device: str = "cpu"):
        return torch.zeros(batch, 4)

    def act(self, obs, hidden=None):
        return 0, 1.0, torch.zeros(1, 4), 1.0

    def rerun_with_dt(self, obs, hidden, target_dt: float):
        # Return a dummy hidden state that differs from input
        return torch.zeros_like(hidden) if hidden is not None else torch.zeros(1, 4)

    def recompute_value(self, hidden) -> float:
        return 0.5


class _ValueOnlyAdapter(AgentAdapter):
    """Adapter that supports value recompute but NOT dt intervention."""

    def reset_hidden(self, batch: int = 1, device: str = "cpu"):
        return torch.zeros(batch, 4)

    def act(self, obs, hidden=None):
        return 0, 1.0, torch.zeros(1, 4), None

    def recompute_value(self, hidden) -> float:
        return 0.5


@pytest.fixture
def intervention_adapter():
    return _InterventionAdapter()


@pytest.fixture
def value_only_adapter():
    return _ValueOnlyAdapter()
