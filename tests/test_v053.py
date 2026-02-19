"""Tests for v0.5.3: Adaptive episode sampling."""

import argparse
import pytest


# ─────────────────────────────────────────────────────────────────────
# 1. CLI parser: --adaptive, --target-ci-width, --max-episodes
# ─────────────────────────────────────────────────────────────────────

def _make_sb3_parser():
    from deltatau_audit.cli import (
        _add_ci_args, _add_seed_arg, _add_workers_arg, _add_compare_arg,
        _add_format_arg, _add_quiet_arg, _add_threshold_args, _add_adaptive_args,
    )
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    sb3 = sub.add_parser("audit-sb3")
    sb3.add_argument("--model", required=True)
    sb3.add_argument("--algo", required=True)
    sb3.add_argument("--env", required=True)
    sb3.add_argument("--episodes", type=int, default=30)
    sb3.add_argument("--speeds", type=int, nargs="+", default=[1, 2])
    sb3.add_argument("--device", default="cpu")
    sb3.add_argument("--out", default="out")
    sb3.add_argument("--title", default=None)
    _add_ci_args(sb3)
    _add_seed_arg(sb3)
    _add_workers_arg(sb3)
    _add_compare_arg(sb3)
    _add_format_arg(sb3)
    _add_quiet_arg(sb3)
    _add_threshold_args(sb3)
    _add_adaptive_args(sb3)
    return p


def test_adaptive_default_false():
    p = _make_sb3_parser()
    args = p.parse_args(["audit-sb3", "--model", "m.zip", "--algo", "ppo", "--env", "CartPole-v1"])
    assert args.adaptive is False


def test_adaptive_flag_sets_true():
    p = _make_sb3_parser()
    args = p.parse_args(["audit-sb3", "--model", "m.zip", "--algo", "ppo", "--env", "CartPole-v1",
                          "--adaptive"])
    assert args.adaptive is True


def test_target_ci_width_default():
    p = _make_sb3_parser()
    args = p.parse_args(["audit-sb3", "--model", "m.zip", "--algo", "ppo", "--env", "CartPole-v1"])
    assert args.target_ci_width == 0.10


def test_target_ci_width_custom():
    p = _make_sb3_parser()
    args = p.parse_args(["audit-sb3", "--model", "m.zip", "--algo", "ppo", "--env", "CartPole-v1",
                          "--target-ci-width", "0.05"])
    assert abs(args.target_ci_width - 0.05) < 1e-9


def test_max_episodes_default():
    p = _make_sb3_parser()
    args = p.parse_args(["audit-sb3", "--model", "m.zip", "--algo", "ppo", "--env", "CartPole-v1"])
    assert args.max_episodes == 500


def test_max_episodes_custom():
    p = _make_sb3_parser()
    args = p.parse_args(["audit-sb3", "--model", "m.zip", "--algo", "ppo", "--env", "CartPole-v1",
                          "--max-episodes", "100"])
    assert args.max_episodes == 100


# ─────────────────────────────────────────────────────────────────────
# 2. run_robustness_audit — adaptive=False (non-regression)
# ─────────────────────────────────────────────────────────────────────

def _make_simple_adapter_and_factory():
    import gymnasium as gym
    from deltatau_audit.adapters.base import AgentAdapter

    class _ConstAdapter(AgentAdapter):
        supports_intervention = False

        def reset_hidden(self, batch_size=1, device="cpu"):
            return None

        def act(self, obs, hidden, dt=1.0, intervention=None):
            action = self._env_action_space.sample()
            return action, 0.0, None, None

        def rerun_with_dt(self, obs_seq, dt_seq, hidden):
            return [0.0] * len(obs_seq)

    env = gym.make("CartPole-v1")
    adapter = _ConstAdapter()
    adapter._env_action_space = env.action_space
    env.close()

    env_factory = lambda: gym.make("CartPole-v1")
    return adapter, env_factory


def test_non_adaptive_result_structure():
    from deltatau_audit.auditor import run_robustness_audit
    adapter, env_factory = _make_simple_adapter_and_factory()
    result = run_robustness_audit(
        adapter, env_factory,
        scenarios=["nominal", "jitter"],
        n_episodes=3, verbose=False,
    )
    assert "per_scenario_scores" in result
    assert "jitter" in result["per_scenario_scores"]
    assert "adaptive" not in result  # not in non-adaptive mode


# ─────────────────────────────────────────────────────────────────────
# 3. run_robustness_audit — adaptive=True, max_episodes cap
# ─────────────────────────────────────────────────────────────────────

def test_adaptive_result_has_n_episodes_used():
    from deltatau_audit.auditor import run_robustness_audit
    adapter, env_factory = _make_simple_adapter_and_factory()
    result = run_robustness_audit(
        adapter, env_factory,
        scenarios=["nominal", "jitter"],
        n_episodes=3,
        adaptive=True,
        target_ci_width=0.01,   # very tight — will hit max_episodes
        max_episodes=6,          # cap at 6 episodes
        verbose=False,
    )
    assert "n_episodes_used" in result
    assert "adaptive" in result
    assert result["adaptive"] is True
    # Each scenario capped at max_episodes
    for sc, n in result["n_episodes_used"].items():
        assert n <= 6, f"{sc} used {n} episodes, expected <= 6"


def test_adaptive_result_still_has_standard_keys():
    from deltatau_audit.auditor import run_robustness_audit
    adapter, env_factory = _make_simple_adapter_and_factory()
    result = run_robustness_audit(
        adapter, env_factory,
        scenarios=["nominal", "jitter"],
        n_episodes=3,
        adaptive=True,
        target_ci_width=0.50,   # wide target → converges quickly
        max_episodes=10,
        verbose=False,
    )
    assert "per_scenario_scores" in result
    assert "deployment" in result
    assert "stress" in result
    assert "rating" in result


def test_adaptive_max_episodes_respected():
    """When CI never converges, n_episodes_used should not exceed max_episodes."""
    from deltatau_audit.auditor import run_robustness_audit
    adapter, env_factory = _make_simple_adapter_and_factory()
    result = run_robustness_audit(
        adapter, env_factory,
        scenarios=["nominal", "jitter"],
        n_episodes=5,
        adaptive=True,
        target_ci_width=1e-10,  # impossible target
        max_episodes=5,
        verbose=False,
    )
    for sc, n in result["n_episodes_used"].items():
        assert n <= 5, f"{sc}: n_episodes_used={n} exceeds max_episodes=5"


# ─────────────────────────────────────────────────────────────────────
# 4. run_full_audit — adaptive params flow through
# ─────────────────────────────────────────────────────────────────────

def test_run_full_audit_adaptive_in_result():
    from deltatau_audit.auditor import run_full_audit
    adapter, env_factory = _make_simple_adapter_and_factory()
    result = run_full_audit(
        adapter, env_factory,
        speeds=[1],
        n_episodes=3,
        adaptive=True,
        target_ci_width=0.50,
        max_episodes=6,
        verbose=False,
    )
    # robustness sub-dict should have adaptive flag when adaptive=True
    assert result["robustness"].get("adaptive") is True
    assert "n_episodes_used" in result["robustness"]
