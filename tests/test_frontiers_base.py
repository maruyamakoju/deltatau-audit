"""Tests for the shared frontier scaffolding modules.

Covers ``experiments.frontiers._base``, ``_geometry``, ``_lipschitz``,
``_metrics``. These modules are pure utilities with no env dependency, so
they can run in any environment that has torch + numpy.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

from experiments.frontiers._base import (
    FrontierConfig,
    FrontierExperiment,
    make_frontier_parser,
    save_summary,
    seed_all,
)
from experiments.frontiers._geometry import (
    exp_map0,
    log_map0,
    mobius_add,
    poincare_distance,
)
from experiments.frontiers._lipschitz import (
    empirical_lipschitz,
    spectral_norm_estimate,
)
from experiments.frontiers._metrics import (
    aggregate_returns,
    env_return_ceiling,
    normalize_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# _base
# ─────────────────────────────────────────────────────────────────────────────


class TestSeedAll:
    def test_torch_reproducible(self):
        seed_all(123)
        a = torch.randn(8)
        seed_all(123)
        b = torch.randn(8)
        assert torch.allclose(a, b)

    def test_numpy_reproducible(self):
        seed_all(7)
        a = np.random.rand(8)
        seed_all(7)
        b = np.random.rand(8)
        np.testing.assert_array_equal(a, b)

    def test_random_reproducible(self):
        seed_all(99)
        a = [random.random() for _ in range(8)]
        seed_all(99)
        b = [random.random() for _ in range(8)]
        assert a == b


class TestFrontierConfig:
    def test_defaults(self):
        cfg = FrontierConfig()
        assert cfg.env_id == "CartPole-v1"
        assert cfg.device == "cpu"
        assert cfg.seed == 42
        assert cfg.extra == {}

    def test_from_params_translates_env_alias(self):
        cfg = FrontierConfig.from_params({"env": "Acrobot-v1", "seed": 5})
        assert cfg.env_id == "Acrobot-v1"
        assert cfg.seed == 5

    def test_from_params_collects_unknown_into_extra(self):
        cfg = FrontierConfig.from_params({"hidden_dim": 64, "seed": 1})
        assert cfg.seed == 1
        assert cfg.extra == {"hidden_dim": 64}

    def test_from_params_passes_known_fields(self):
        cfg = FrontierConfig.from_params({"env_id": "CartPole-v1", "device": "cpu", "n_episodes": 5, "max_steps": 50})
        assert cfg.env_id == "CartPole-v1"
        assert cfg.n_episodes == 5
        assert cfg.max_steps == 50


class TestSaveSummary:
    def test_writes_indented_json(self, tmp_path: Path):
        summary = {"composite_score": 0.5, "mean_return": 100.0}
        path = save_summary(tmp_path, summary)
        assert path == tmp_path / "results.json"
        loaded = json.loads(path.read_text())
        assert loaded["composite_score"] == 0.5

    def test_creates_missing_parents(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        save_summary(nested, {"x": 1})
        assert (nested / "results.json").exists()

    def test_extras_merged_under_extras_key(self, tmp_path: Path):
        save_summary(tmp_path, {"composite_score": 0.5}, extras={"meta": "data"})
        loaded = json.loads((tmp_path / "results.json").read_text())
        assert loaded["composite_score"] == 0.5
        assert loaded["extras"] == {"meta": "data"}

    def test_handles_non_serialisable_with_default_str(self, tmp_path: Path):
        # Path objects aren't natively JSON-serialisable; default=str should kick in
        save_summary(tmp_path, {"out_dir": Path("/tmp/x")})
        loaded = json.loads((tmp_path / "results.json").read_text())
        assert "out_dir" in loaded


class _ToyExperiment(FrontierExperiment):
    """Minimal subclass for ABC contract testing."""

    def train(self) -> Dict[str, float]:
        return {"mean_return": 10.0}

    def evaluate(self) -> Dict[str, float]:
        return {"robustness": 0.8}

    def compute_composite(self, metrics: Dict[str, float]) -> float:
        return metrics["mean_return"] * metrics["robustness"]


class TestFrontierExperiment:
    def test_run_orchestrates_lifecycle(self, tmp_path: Path):
        exp = _ToyExperiment(FrontierConfig(seed=1))
        result = exp.run(tmp_path)
        assert result["mean_return"] == 10.0
        assert result["robustness"] == 0.8
        assert result["composite_score"] == 8.0
        assert (tmp_path / "results.json").exists()

    def test_run_seeds_torch(self, tmp_path: Path):
        exp1 = _ToyExperiment(FrontierConfig(seed=42))
        exp1.run(tmp_path / "a")
        a = torch.randn(4)
        exp2 = _ToyExperiment(FrontierConfig(seed=42))
        exp2.run(tmp_path / "b")
        b = torch.randn(4)
        assert torch.allclose(a, b)

    def test_setup_hook_called(self, tmp_path: Path):
        calls = []

        class WithSetup(_ToyExperiment):
            def setup(self):
                calls.append("setup")

        WithSetup(FrontierConfig()).run(tmp_path)
        assert calls == ["setup"]


class TestMakeFrontierParser:
    def test_default_args(self):
        parser = make_frontier_parser(name="test", description="x")
        ns = parser.parse_args([])
        assert ns.seed == 42
        assert ns.device == "cpu"
        assert ns.env_id == "CartPole-v1"
        assert ns.n_episodes == 30

    def test_extra_args_callback_runs(self):
        def add_lr(p):
            p.add_argument("--lr", type=float, default=1e-3)

        parser = make_frontier_parser(name="t", description="x", extra_args=[add_lr])
        ns = parser.parse_args(["--lr", "5e-4"])
        assert ns.lr == 5e-4


# ─────────────────────────────────────────────────────────────────────────────
# _geometry
# ─────────────────────────────────────────────────────────────────────────────


class TestGeometry:
    def test_mobius_add_zero_identity(self):
        # mobius_add(u, 0) ≈ u (within epsilon noise from the +1e-8 in v)
        u = torch.tensor([0.1, 0.2, 0.0])
        out = mobius_add(u, torch.zeros(3))
        assert torch.allclose(out, u, atol=1e-5)

    def test_exp_map0_zero_is_zero(self):
        out = exp_map0(torch.zeros(3))
        assert torch.allclose(out, torch.zeros(3), atol=1e-7)

    def test_exp_map0_inside_ball(self):
        # For c=1, the image of exp_map0 lies in the closed unit ball.
        # tanh saturates to exactly 1.0 in float for large inputs, which is
        # numerically OK — what matters is that nothing exceeds the boundary.
        for _ in range(10):
            u = torch.randn(8) * 5.0
            x = exp_map0(u)
            assert torch.norm(x).item() <= 1.0 + 1e-6

    def test_log_map_inverts_exp_map(self):
        u = torch.tensor([0.3, -0.4, 0.1])
        x = exp_map0(u)
        recovered = log_map0(x)
        assert torch.allclose(u, recovered, atol=1e-4)

    def test_poincare_distance_self_is_zero(self):
        x = torch.tensor([0.1, -0.2, 0.05])
        d = poincare_distance(x, x)
        assert d.item() < 1e-3

    def test_poincare_distance_symmetric(self):
        x = torch.tensor([0.1, 0.2, 0.0])
        y = torch.tensor([-0.1, 0.0, 0.3])
        assert math.isclose(
            poincare_distance(x, y).item(),
            poincare_distance(y, x).item(),
            rel_tol=1e-4,
        )


# ─────────────────────────────────────────────────────────────────────────────
# _lipschitz
# ─────────────────────────────────────────────────────────────────────────────


class TestSpectralNormEstimate:
    def test_identity_matrix_has_unit_norm(self):
        W = torch.eye(8)
        sigma = spectral_norm_estimate(W, n_iter=20)
        assert math.isclose(sigma, 1.0, abs_tol=1e-4)

    def test_diagonal_matrix_max_singular(self):
        W = torch.diag(torch.tensor([3.0, 1.0, -2.0, 0.5]))
        sigma = spectral_norm_estimate(W, n_iter=30)
        assert math.isclose(sigma, 3.0, abs_tol=1e-3)

    def test_zero_matrix(self):
        W = torch.zeros(4, 4)
        # Power iteration may produce numerical noise but should be tiny
        sigma = spectral_norm_estimate(W, n_iter=5)
        assert sigma < 1e-5

    def test_one_dim_returns_abs_max(self):
        W = torch.tensor([1.0, -3.0, 2.0])
        sigma = spectral_norm_estimate(W, n_iter=5)
        assert sigma == 3.0

    def test_empty_weight_returns_zero(self):
        W = torch.zeros(0, 4)
        assert spectral_norm_estimate(W) == 0.0

    def test_higher_dim_flattened(self):
        # Conv-like weight (out, in, k, k) — should flatten the last 3 dims
        W = torch.randn(4, 3, 2, 2)
        sigma = spectral_norm_estimate(W, n_iter=10)
        # Reference: matrix view's largest singular value
        flat = W.reshape(4, -1)
        ref = torch.linalg.svdvals(flat)[0].item()
        assert math.isclose(sigma, ref, rel_tol=0.05)


class TestEmpiricalLipschitz:
    def test_linear_function_recovers_slope(self):
        # f(x) = 2x → Lipschitz constant = 2
        W = torch.tensor([[2.0]])
        fn = lambda x: x @ W
        samples = torch.zeros(8, 1)
        L = empirical_lipschitz(fn, samples, eps=1e-3, n_perturbations=8)
        assert math.isclose(L, 2.0, rel_tol=0.05)

    def test_constant_function_is_zero_lipschitz(self):
        fn = lambda x: torch.zeros_like(x)
        samples = torch.randn(4, 3)
        L = empirical_lipschitz(fn, samples, eps=1e-3, n_perturbations=4)
        assert L < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# _metrics
# ─────────────────────────────────────────────────────────────────────────────


class TestAggregateReturns:
    def test_basic_stats(self):
        out = aggregate_returns([1.0, 2.0, 3.0, 4.0])
        assert out["mean_return"] == 2.5
        assert out["min_return"] == 1.0
        assert out["max_return"] == 4.0
        assert out["median_return"] == 2.5
        assert out["n_episodes"] == 4

    def test_empty_returns_zero_dict(self):
        out = aggregate_returns([])
        assert out["mean_return"] == 0.0
        assert out["std_return"] == 0.0
        assert out["n_episodes"] == 0

    def test_single_value(self):
        out = aggregate_returns([5.0])
        assert out["mean_return"] == 5.0
        assert out["std_return"] == 0.0
        assert out["n_episodes"] == 1


class TestNormalizeScore:
    def test_default_unit_range(self):
        assert normalize_score(0.5) == 0.5

    def test_clipped_below_baseline(self):
        assert normalize_score(-1.0, baseline=0.0, ceiling=1.0) == 0.0

    def test_clipped_above_ceiling(self):
        assert normalize_score(2.0, baseline=0.0, ceiling=1.0) == 1.0

    def test_custom_range(self):
        assert normalize_score(150.0, baseline=100.0, ceiling=200.0) == 0.5

    def test_degenerate_range_returns_zero(self):
        assert normalize_score(50.0, baseline=100.0, ceiling=100.0) == 0.0
        assert normalize_score(50.0, baseline=200.0, ceiling=100.0) == 0.0


class TestEnvReturnCeiling:
    def test_known_env(self):
        assert env_return_ceiling("CartPole-v1") == 200.0

    def test_unknown_env_uses_default(self):
        assert env_return_ceiling("UnknownEnv-v0", default=42.0) == 42.0

    def test_negative_return_envs_force_explicit_override(self):
        # Acrobot/MountainCar return zero from the table to flag the issue
        assert env_return_ceiling("Acrobot-v1") == 0.0
        assert env_return_ceiling("MountainCar-v0") == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Integration: confirm migrated frontiers expose the contract
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("experiments.frontiers.causal_temporal_reasoning", "CausalTemporalReasoningExperiment"),
        ("experiments.frontiers.frontier_14", "QTRWMExperiment"),
        ("experiments.frontiers.causal_relativistic_world_model", "CRWMExperiment"),
        ("experiments.frontiers.meta_temporal_evolution", "MTEExperiment"),
    ],
)
def test_migrated_frontiers_satisfy_run_contract(module_name, class_name):
    """Each migrated frontier still constructs from a params dict and exposes run()."""
    import importlib

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    # We don't call run() here (would be slow + non-deterministic); just
    # confirm the constructor signature still accepts a params dict.
    instance = cls(
        {
            "env": "CartPole-v1",
            "device": "cpu",
            "seed": 0,
            "n_episodes": 1,
            "obs_dim": 4,
            "act_dim": 2,
            "hidden_dim": 16,
        }
    )
    assert callable(getattr(instance, "run", None)), f"{class_name} missing run()"
