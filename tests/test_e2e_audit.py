"""E2E integration tests for the deltatau-audit pipeline.

These tests exercise the full pipeline (run_full_audit → report → badge)
using lightweight mock models rather than trained checkpoints.
They are distinguished from unit tests by exercising multiple modules
working together.
"""
from __future__ import annotations

import json
import pathlib
import tempfile
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn


# ── Minimal CartPole-compatible adapter ──────────────────────────────────────

class _CartPoleDummyAdapter:
    """Lightweight adapter that wraps a tiny MLP for CartPole-v1 E2E tests."""

    supports_intervention = False
    supports_value_recompute = False

    def __init__(self):
        # Tiny policy: obs(4) → hidden(8) → action logits(2) + value(1)
        self._net = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)
        )

    def reset_hidden(self, batch: int = 1, device: str = "cpu"):
        return None

    def act(self, obs, hidden=None):
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32)
            obs = obs.unsqueeze(0) if obs.dim() == 1 else obs
            out = self._net(obs)
            logits = out[:, :2]
            value = out[:, 2].item()
            action = int(torch.argmax(logits, dim=-1).item())
        return action, value, None, None


def _cartpole_factory():
    return gym.make("CartPole-v1")


# ── Test 1: SCHEMA_VERSION is importable ─────────────────────────────────────

def test_schema_version_importable():
    """SCHEMA_VERSION must be importable from deltatau_audit.schema."""
    from deltatau_audit.schema import SCHEMA_VERSION
    assert isinstance(SCHEMA_VERSION, str), "SCHEMA_VERSION must be a string"
    assert len(SCHEMA_VERSION) > 0, "SCHEMA_VERSION must not be empty"
    # Verify semver-like format: major.minor.patch
    parts = SCHEMA_VERSION.split(".")
    assert len(parts) == 3, f"Expected semver format, got: {SCHEMA_VERSION}"
    for part in parts:
        assert part.isdigit(), f"Non-numeric semver component: {part!r}"


def test_schema_version_in_package_init():
    """__schema_version__ must be accessible from the package root."""
    import deltatau_audit
    assert hasattr(deltatau_audit, "__schema_version__")
    assert deltatau_audit.__schema_version__ == "1.0.0"


# ── Test 2: Full audit pipeline ───────────────────────────────────────────────

def test_full_audit_pipeline_cartpole():
    """run_full_audit → report generation → summary.json all work end-to-end."""
    from deltatau_audit.auditor import run_full_audit
    from deltatau_audit.report import generate_report

    adapter = _CartPoleDummyAdapter()

    result = run_full_audit(
        adapter=adapter,
        env_factory=_cartpole_factory,
        speeds=[1, 2],
        n_episodes=3,
        gamma=0.99,
        device="cpu",
        verbose=False,
        seed=42,
    )

    # Structural checks on the audit result
    assert "schema_version" in result, "result must include schema_version"
    assert "summary" in result
    assert "robustness" in result
    assert "reliance" in result

    summary = result["summary"]
    assert "deployment_score" in summary
    assert "stress_score" in summary
    assert "quadrant" in summary

    # Report generation
    with tempfile.TemporaryDirectory() as tmp:
        generate_report(result, tmp, title="E2E CartPole Test")

        summary_json = pathlib.Path(tmp) / "summary.json"
        assert summary_json.exists(), "summary.json not written"

        data = json.loads(summary_json.read_text())
        assert "_version" in data, "summary.json missing _version"
        assert "_timestamp" in data, "summary.json missing _timestamp"


# ── Test 3: Fixer pipeline ───────────────────────────────────────────────────

def test_schema_version_propagates_to_audit_result():
    """SCHEMA_VERSION from schema.py should appear in run_full_audit output."""
    from deltatau_audit.schema import SCHEMA_VERSION
    from deltatau_audit.auditor import run_full_audit

    adapter = _CartPoleDummyAdapter()
    result = run_full_audit(
        adapter=adapter,
        env_factory=_cartpole_factory,
        speeds=[1],
        n_episodes=2,
        verbose=False,
        seed=0,
    )
    assert result.get("schema_version") == SCHEMA_VERSION, (
        f"schema_version in result ({result.get('schema_version')!r}) "
        f"does not match SCHEMA_VERSION ({SCHEMA_VERSION!r})"
    )


# ── Test 4: Formal verifier returns real Lipschitz (not random) ──────────────

def test_formal_verifier_not_purely_random():
    """LipschitzVerifier must return consistent results, not torch.randn output.

    We check that calling compute_temporal_lipschitz_constant twice on the
    same observation produces the same result (deterministic). A stub that
    returns torch.randn() would produce different values each call.
    """
    from deltatau_audit.verification.formal import LipschitzVerifier
    from deltatau_audit.protocols import AgentAdapter

    class _MinimalAdapter(AgentAdapter):
        """Minimal adapter exposing internal model for formal verification."""

        def __init__(self):
            import torch.nn as nn
            # Small model with explicit internal structure
            self._model = _TinyModel()

        def reset_hidden(self, batch=1, device="cpu"):
            return torch.zeros(batch, 8)

        def act(self, obs, hidden=None):
            return 0, 0.0, torch.zeros(1, 8), 1.0

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, obs):
            return self.fc(obs)

    adapter = _MinimalAdapter()
    verifier = LipschitzVerifier(agent=adapter)
    obs = torch.zeros(1, 4)

    # Run twice — results must be identical (or very close) for non-random impl
    result1 = verifier.compute_temporal_lipschitz_constant(obs, n_samples=5)
    result2 = verifier.compute_temporal_lipschitz_constant(obs, n_samples=5)

    # For a stub returning torch.randn, consecutive calls differ by O(1).
    # For a deterministic implementation, they should be identical.
    assert abs(result1.value - result2.value) < 1e-6, (
        f"LipschitzVerifier returned different values on repeated calls: "
        f"{result1.value} vs {result2.value}. "
        f"This suggests a random stub (torch.randn) is still in use."
    )


# ── Test 5: Badge generation ─────────────────────────────────────────────────

def test_badge_generation_e2e():
    """Badge SVG generation works end-to-end given a report summary."""
    from deltatau_audit.badge import generate_badge_svg

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "badge.svg"
        svg = generate_badge_svg(
            score=0.85,
            rating="PASS",
            output_path=out,
        )
        assert isinstance(svg, str)
        assert "<svg" in svg
        assert out.exists()


# ── Test 6: CI summary generation ────────────────────────────────────────────

def test_ci_summary_generation():
    """CI summary JSON is generated correctly from an audit result."""
    from deltatau_audit.ci import generate_ci_summary
    from tests.conftest import make_result

    result = make_result(deployment_score=0.90, stress_score=0.70)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = pathlib.Path(tmp)
        ci_data = generate_ci_summary(
            result,
            out_dir=out_dir,
            deploy_threshold=0.80,
            stress_threshold=0.50,
        )
        assert ci_data["deploy_pass"] is True
        assert ci_data["stress_pass"] is True

        ci_json = out_dir / "ci_summary.json"
        assert ci_json.exists()
        loaded = json.loads(ci_json.read_text())
        assert "deploy_pass" in loaded
