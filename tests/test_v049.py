"""Tests for v0.4.9: --quiet flag on audit-sb3, audit-cleanrl, audit."""

import argparse
import pytest


# ─────────────────────────────────────────────────────────────────────
# 1. _add_quiet_arg helper
# ─────────────────────────────────────────────────────────────────────

def test_add_quiet_arg_default_false():
    from deltatau_audit.cli import _add_quiet_arg
    p = argparse.ArgumentParser()
    _add_quiet_arg(p)
    args = p.parse_args([])
    assert args.quiet is False


def test_add_quiet_arg_flag():
    from deltatau_audit.cli import _add_quiet_arg
    p = argparse.ArgumentParser()
    _add_quiet_arg(p)
    args = p.parse_args(["--quiet"])
    assert args.quiet is True


def test_add_quiet_arg_short_flag():
    from deltatau_audit.cli import _add_quiet_arg
    p = argparse.ArgumentParser()
    _add_quiet_arg(p)
    args = p.parse_args(["-q"])
    assert args.quiet is True


# ─────────────────────────────────────────────────────────────────────
# 2. audit-sb3 parser accepts --quiet
# ─────────────────────────────────────────────────────────────────────

def _make_sb3_parser():
    from deltatau_audit.cli import (
        _add_ci_args, _add_seed_arg, _add_workers_arg,
        _add_compare_arg, _add_format_arg, _add_quiet_arg,
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
    return p


def test_audit_sb3_quiet_default_false():
    p = _make_sb3_parser()
    args = p.parse_args([
        "audit-sb3", "--model", "m.zip", "--algo", "ppo", "--env", "CartPole-v1",
    ])
    assert args.quiet is False


def test_audit_sb3_quiet_flag():
    p = _make_sb3_parser()
    args = p.parse_args([
        "audit-sb3", "--model", "m.zip", "--algo", "ppo", "--env", "CartPole-v1",
        "--quiet",
    ])
    assert args.quiet is True


def test_audit_sb3_quiet_short_flag():
    p = _make_sb3_parser()
    args = p.parse_args([
        "audit-sb3", "--model", "m.zip", "--algo", "ppo", "--env", "CartPole-v1",
        "-q",
    ])
    assert args.quiet is True


# ─────────────────────────────────────────────────────────────────────
# 3. audit-cleanrl parser accepts --quiet
# ─────────────────────────────────────────────────────────────────────

def _make_cleanrl_parser():
    from deltatau_audit.cli import (
        _add_ci_args, _add_seed_arg, _add_workers_arg,
        _add_compare_arg, _add_format_arg, _add_quiet_arg,
    )
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    cleanrl = sub.add_parser("audit-cleanrl")
    cleanrl.add_argument("--checkpoint", required=True)
    cleanrl.add_argument("--agent-module", required=True)
    cleanrl.add_argument("--env", required=True)
    cleanrl.add_argument("--episodes", type=int, default=30)
    cleanrl.add_argument("--speeds", type=int, nargs="+", default=[1, 2])
    cleanrl.add_argument("--device", type=str, default="cpu")
    cleanrl.add_argument("--out", type=str, default="out")
    cleanrl.add_argument("--title", type=str, default=None)
    _add_ci_args(cleanrl)
    _add_seed_arg(cleanrl)
    _add_workers_arg(cleanrl)
    _add_compare_arg(cleanrl)
    _add_format_arg(cleanrl)
    _add_quiet_arg(cleanrl)
    return p


def test_audit_cleanrl_quiet_default_false():
    p = _make_cleanrl_parser()
    args = p.parse_args([
        "audit-cleanrl",
        "--checkpoint", "a.pt",
        "--agent-module", "m.py",
        "--env", "CartPole-v1",
    ])
    assert args.quiet is False


def test_audit_cleanrl_quiet_flag():
    p = _make_cleanrl_parser()
    args = p.parse_args([
        "audit-cleanrl",
        "--checkpoint", "a.pt",
        "--agent-module", "m.py",
        "--env", "CartPole-v1",
        "--quiet",
    ])
    assert args.quiet is True


# ─────────────────────────────────────────────────────────────────────
# 4. verbose=False path: run_full_audit called with verbose=False
# ─────────────────────────────────────────────────────────────────────

def test_run_full_audit_verbose_false_no_output(capsys):
    """run_full_audit with verbose=False should produce no stdout."""
    import torch
    import gymnasium as gym
    from deltatau_audit.auditor import run_full_audit
    from deltatau_audit.adapters.simple_gru import SimpleGRUAdapter

    env_factory = lambda: gym.make("CartPole-v1")
    sample_env = env_factory()
    obs_dim = sample_env.observation_space.shape[0]
    act_dim = sample_env.action_space.n
    sample_env.close()

    from deltatau_audit.adapters.simple_gru import SimpleGRUPolicy
    model = SimpleGRUPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=32)
    adapter = SimpleGRUAdapter(model=model)

    result = run_full_audit(
        adapter, env_factory,
        speeds=[1],
        n_episodes=2,
        sensitivity_episodes=0,
        verbose=False,
    )
    captured = capsys.readouterr()
    assert captured.out == "", (
        f"Expected no stdout with verbose=False, got: {captured.out[:200]}"
    )
    # Result should still be fully populated
    assert "summary" in result
    assert result["summary"]["deployment_rating"] in (
        "PASS", "MILD", "DEGRADED", "FAIL"
    )
