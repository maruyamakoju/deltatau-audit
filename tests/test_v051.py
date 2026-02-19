"""Tests for v0.5.1: --deploy-threshold / --stress-threshold flags."""

import argparse
import pytest


# ─────────────────────────────────────────────────────────────────────
# 1. _add_threshold_args helper
# ─────────────────────────────────────────────────────────────────────

def test_threshold_args_defaults():
    from deltatau_audit.cli import _add_threshold_args
    p = argparse.ArgumentParser()
    _add_threshold_args(p)
    args = p.parse_args([])
    assert args.deploy_threshold == 0.80
    assert args.stress_threshold == 0.50


def test_threshold_args_custom_deploy():
    from deltatau_audit.cli import _add_threshold_args
    p = argparse.ArgumentParser()
    _add_threshold_args(p)
    args = p.parse_args(["--deploy-threshold", "0.85"])
    assert args.deploy_threshold == pytest.approx(0.85)


def test_threshold_args_custom_stress():
    from deltatau_audit.cli import _add_threshold_args
    p = argparse.ArgumentParser()
    _add_threshold_args(p)
    args = p.parse_args(["--stress-threshold", "0.60"])
    assert args.stress_threshold == pytest.approx(0.60)


def test_threshold_args_both_custom():
    from deltatau_audit.cli import _add_threshold_args
    p = argparse.ArgumentParser()
    _add_threshold_args(p)
    args = p.parse_args([
        "--deploy-threshold", "0.90",
        "--stress-threshold", "0.70",
    ])
    assert args.deploy_threshold == pytest.approx(0.90)
    assert args.stress_threshold == pytest.approx(0.70)


# ─────────────────────────────────────────────────────────────────────
# 2. audit-sb3 parser accepts thresholds
# ─────────────────────────────────────────────────────────────────────

def _make_sb3_parser():
    from deltatau_audit.cli import (
        _add_ci_args, _add_seed_arg, _add_workers_arg,
        _add_compare_arg, _add_format_arg, _add_quiet_arg,
        _add_threshold_args,
    )
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    sb3 = sub.add_parser("audit-sb3")
    sb3.add_argument("--model", required=True)
    sb3.add_argument("--algo", required=True)
    sb3.add_argument("--env", required=True)
    sb3.add_argument("--episodes", type=int, default=30)
    sb3.add_argument("--speeds", type=int, nargs="+", default=[1, 2])
    sb3.add_argument("--device", type=str, default="cpu")
    sb3.add_argument("--out", type=str, default="out")
    sb3.add_argument("--title", type=str, default=None)
    _add_ci_args(sb3)
    _add_seed_arg(sb3)
    _add_workers_arg(sb3)
    _add_compare_arg(sb3)
    _add_format_arg(sb3)
    _add_quiet_arg(sb3)
    _add_threshold_args(sb3)
    return p


def test_sb3_threshold_defaults():
    p = _make_sb3_parser()
    args = p.parse_args([
        "audit-sb3", "--model", "m.zip", "--algo", "ppo", "--env", "CartPole-v1",
    ])
    assert args.deploy_threshold == 0.80
    assert args.stress_threshold == 0.50


def test_sb3_threshold_custom():
    p = _make_sb3_parser()
    args = p.parse_args([
        "audit-sb3", "--model", "m.zip", "--algo", "ppo", "--env", "CartPole-v1",
        "--deploy-threshold", "0.90",
        "--stress-threshold", "0.65",
    ])
    assert args.deploy_threshold == pytest.approx(0.90)
    assert args.stress_threshold == pytest.approx(0.65)


# ─────────────────────────────────────────────────────────────────────
# 3. run_full_audit respects deploy_threshold
# ─────────────────────────────────────────────────────────────────────

def _make_cartpole_audit(deploy_threshold=0.80, stress_threshold=0.50):
    import gymnasium as gym
    from deltatau_audit.auditor import run_full_audit
    from deltatau_audit.adapters.simple_gru import SimpleGRUAdapter, SimpleGRUPolicy

    env_factory = lambda: gym.make("CartPole-v1")
    sample_env = env_factory()
    obs_dim = sample_env.observation_space.shape[0]
    act_dim = sample_env.action_space.n
    sample_env.close()

    model = SimpleGRUPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=32)
    adapter = SimpleGRUAdapter(model=model)

    return run_full_audit(
        adapter, env_factory,
        speeds=[1],
        n_episodes=10,
        sensitivity_episodes=0,
        verbose=False,
        seed=42,
        deploy_threshold=deploy_threshold,
        stress_threshold=stress_threshold,
    )


def test_run_full_audit_threshold_stored_in_summary():
    result = _make_cartpole_audit(deploy_threshold=0.85, stress_threshold=0.60)
    assert result["summary"]["deploy_threshold"] == pytest.approx(0.85)
    assert result["summary"]["stress_threshold"] == pytest.approx(0.60)


def test_run_full_audit_strict_threshold_changes_quadrant():
    """Setting deploy_threshold=0.99 should force fragile quadrant."""
    result = _make_cartpole_audit(deploy_threshold=0.99)
    quadrant = result["summary"]["quadrant"]
    # With a random agent and threshold=0.99 (unreachable), must be fragile
    assert "fragile" in quadrant


def test_run_full_audit_zero_threshold_passes():
    """Setting deploy_threshold=0.0 should always yield non-fragile quadrant."""
    result = _make_cartpole_audit(deploy_threshold=0.0)
    quadrant = result["summary"]["quadrant"]
    assert "fragile" not in quadrant
