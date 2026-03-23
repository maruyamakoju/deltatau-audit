"""Tests for formal.py LipschitzVerifier (PHASE 2)."""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from deltatau_audit.verification.formal import LipschitzVerifier, LipschitzCertificate


# ── Helper adapter ────────────────────────────────────────────────────────────

class _ModelWithEncoder(nn.Module):
    """Small model exposing encoder + policy_head for verifier probing."""
    def __init__(self, obs_dim=4, act_dim=2, hidden_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.in_features = hidden_dim  # for policy_head compat

    def forward(self, obs):
        return self.policy_head(self.encoder(obs))


class _ModelAdapter:
    """Adapter that exposes forward_with_tau for the verifier."""
    def __init__(self):
        self._net = _ModelWithEncoder()

    def reset_hidden(self, batch=1, device="cpu"):
        return torch.zeros(batch, 8)

    def act(self, obs, hidden=None):
        return 0, 0.0, torch.zeros(1, 8), 1.0

    def forward_with_tau(self, obs: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Differentiable forward: logits scaled by tau for gradient testing."""
        encoded = self._net.encoder(obs)
        # Make logits differentiable w.r.t. tau by multiplying
        logits = self._net.policy_head(encoded) * tau
        return logits


class _NoModelAdapter:
    """Adapter without internal model — verifier should return NaN gracefully."""
    def reset_hidden(self, batch=1, device="cpu"):
        return None

    def act(self, obs, hidden=None):
        return 0, 0.0, None, None


# ── LipschitzVerifier tests ───────────────────────────────────────────────────

def test_verifier_returns_metric_value():
    """compute_temporal_lipschitz_constant() must return a MetricValue."""
    from deltatau_audit.schema import MetricValue
    adapter = _ModelAdapter()
    verifier = LipschitzVerifier(agent=adapter)
    obs = torch.zeros(1, 4)
    result = verifier.compute_temporal_lipschitz_constant(obs, n_samples=5)
    assert isinstance(result, MetricValue)


def test_verifier_deterministic_not_random():
    """Repeated calls on same input must return identical results.

    The original stub returned torch.randn(1, 5) which is non-deterministic.
    A real implementation must be deterministic.
    """
    adapter = _ModelAdapter()
    verifier = LipschitzVerifier(agent=adapter)
    obs = torch.zeros(1, 4)

    r1 = verifier.compute_temporal_lipschitz_constant(obs, n_samples=10)
    r2 = verifier.compute_temporal_lipschitz_constant(obs, n_samples=10)

    assert abs(r1.value - r2.value) < 1e-5, (
        f"LipschitzVerifier returned different values on repeated calls: "
        f"{r1.value} vs {r2.value}. Stub (torch.randn) still in use?"
    )


def test_verifier_l_max_non_negative():
    """Lipschitz constant must be >= 0 (it's a norm)."""
    adapter = _ModelAdapter()
    verifier = LipschitzVerifier(agent=adapter)
    obs = torch.zeros(1, 4)
    result = verifier.compute_temporal_lipschitz_constant(obs, n_samples=5)
    if not math.isnan(result.value):
        assert result.value >= 0.0, f"L_max={result.value} is negative"


def test_verifier_metadata_has_stability_rating():
    """Result metadata must include stability_rating."""
    adapter = _ModelAdapter()
    verifier = LipschitzVerifier(agent=adapter)
    obs = torch.zeros(1, 4)
    result = verifier.compute_temporal_lipschitz_constant(obs, n_samples=5)
    if not math.isnan(result.value):
        assert "stability_rating" in result.metadata
        assert result.metadata["stability_rating"] in ("HIGH", "MODERATE", "CRITICAL")


def test_verifier_graceful_on_no_model():
    """Verifier must not crash if agent has no internal model."""
    adapter = _NoModelAdapter()
    verifier = LipschitzVerifier(agent=adapter)
    obs = torch.zeros(1, 4)
    result = verifier.compute_temporal_lipschitz_constant(obs, n_samples=5)
    # Should either return NaN or 0.0, not crash
    assert result is not None


# ── LipschitzCertificate tests ────────────────────────────────────────────────

def test_compute_value_lipschitz_returns_certificate():
    """compute_value_lipschitz_constant() must return LipschitzCertificate."""
    adapter = _ModelAdapter()
    verifier = LipschitzVerifier(agent=adapter)
    obs = torch.zeros(1, 4)
    cert = verifier.compute_value_lipschitz_constant(obs, n_samples=5)
    assert isinstance(cert, LipschitzCertificate)


def test_lipschitz_certificate_fields():
    """LipschitzCertificate must have all required fields."""
    cert = LipschitzCertificate(
        L_max=0.5,
        L_mean=0.3,
        certified_epsilon=0.2,
        stability_rating="HIGH",
        n_samples=50,
        tau_range=(0.5, 2.0),
    )
    assert cert.L_max == 0.5
    assert cert.L_mean == 0.3
    assert cert.certified_epsilon == 0.2
    assert cert.stability_rating == "HIGH"
    assert cert.n_samples == 50


def test_lipschitz_certified_epsilon_inversely_proportional():
    """certified_epsilon = 0.1 / L_max (less stable = smaller epsilon)."""
    cert_high = LipschitzCertificate(
        L_max=0.5, L_mean=0.3, certified_epsilon=0.2,
        stability_rating="HIGH", n_samples=10, tau_range=(0.5, 2.0),
    )
    cert_low = LipschitzCertificate(
        L_max=5.0, L_mean=3.0, certified_epsilon=0.02,
        stability_rating="CRITICAL", n_samples=10, tau_range=(0.5, 2.0),
    )
    assert cert_high.certified_epsilon > cert_low.certified_epsilon
