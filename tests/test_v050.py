"""Tests for v0.5.0: audit-hf command and SB3Adapter.from_hub()."""

import argparse
import pytest


# ─────────────────────────────────────────────────────────────────────
# 1. SB3Adapter.from_hub() — missing huggingface_hub raises ImportError
# ─────────────────────────────────────────────────────────────────────

def test_from_hub_raises_import_error_without_hf(monkeypatch):
    """from_hub() should raise ImportError when huggingface_hub is absent."""
    import sys
    # Temporarily hide huggingface_hub
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    from deltatau_audit.adapters.sb3 import SB3Adapter
    with pytest.raises((ImportError, TypeError)):
        SB3Adapter.from_hub("sb3/ppo-CartPole-v1", algo="ppo")


# ─────────────────────────────────────────────────────────────────────
# 2. SB3Adapter.from_hub() — hf_hub_download called with correct args
# ─────────────────────────────────────────────────────────────────────

def test_from_hub_calls_hf_hub_download(monkeypatch, tmp_path):
    """from_hub() should call hf_hub_download with the correct repo_id."""
    import sys
    import types

    # Create a fake CartPole PPO model .zip
    import stable_baselines3 as sb3
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    model = sb3.PPO("MlpPolicy", env, n_steps=32, batch_size=16)
    model_path = str(tmp_path / "ppo-CartPole-v1.zip")
    model.save(model_path)
    env.close()

    calls = []

    def fake_hf_hub_download(repo_id, filename, token=None):
        calls.append({"repo_id": repo_id, "filename": filename})
        return model_path

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    from importlib import reload
    from deltatau_audit.adapters import sb3 as sb3_mod
    reload(sb3_mod)

    adapter = sb3_mod.SB3Adapter.from_hub(
        "sb3/ppo-CartPole-v1", algo="ppo"
    )
    assert len(calls) >= 1
    assert calls[0]["repo_id"] == "sb3/ppo-CartPole-v1"
    assert adapter is not None

    reload(sb3_mod)  # restore


def test_from_hub_uses_explicit_filename(monkeypatch, tmp_path):
    """from_hub() with --filename should try that filename directly."""
    import sys
    import types

    import stable_baselines3 as sb3
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    model = sb3.PPO("MlpPolicy", env, n_steps=32, batch_size=16)
    model_path = str(tmp_path / "custom_model.zip")
    model.save(model_path)
    env.close()

    calls = []

    def fake_hf_hub_download(repo_id, filename, token=None):
        calls.append(filename)
        return model_path

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    from importlib import reload
    from deltatau_audit.adapters import sb3 as sb3_mod
    reload(sb3_mod)

    sb3_mod.SB3Adapter.from_hub(
        "myorg/my-model", algo="ppo", filename="custom_model.zip"
    )
    assert calls == ["custom_model.zip"]
    reload(sb3_mod)


def test_from_hub_raises_file_not_found_on_all_404(monkeypatch):
    """from_hub() raises FileNotFoundError when all candidates fail."""
    import sys
    import types

    def fake_hf_hub_download(repo_id, filename, token=None):
        raise Exception("404: Not Found")

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    from importlib import reload
    from deltatau_audit.adapters import sb3 as sb3_mod
    reload(sb3_mod)

    with pytest.raises(FileNotFoundError, match="Could not find model"):
        sb3_mod.SB3Adapter.from_hub("myorg/nonexistent", algo="ppo")

    reload(sb3_mod)


# ─────────────────────────────────────────────────────────────────────
# 3. audit-hf CLI parser
# ─────────────────────────────────────────────────────────────────────

def _make_hf_parser():
    from deltatau_audit.cli import (
        _add_ci_args, _add_seed_arg, _add_workers_arg,
        _add_compare_arg, _add_format_arg, _add_quiet_arg,
    )
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    hf = sub.add_parser("audit-hf")
    hf.add_argument("--repo", required=True)
    hf.add_argument("--algo", required=True)
    hf.add_argument("--env", required=True)
    hf.add_argument("--filename", default=None)
    hf.add_argument("--hf-token", default=None)
    hf.add_argument("--episodes", type=int, default=30)
    hf.add_argument("--speeds", type=int, nargs="+", default=[1, 2])
    hf.add_argument("--device", type=str, default="cpu")
    hf.add_argument("--out", type=str, default="out")
    hf.add_argument("--title", type=str, default=None)
    _add_ci_args(hf)
    _add_seed_arg(hf)
    _add_workers_arg(hf)
    _add_compare_arg(hf)
    _add_format_arg(hf)
    _add_quiet_arg(hf)
    return p


def test_audit_hf_parser_repo_required():
    p = _make_hf_parser()
    args = p.parse_args([
        "audit-hf",
        "--repo", "sb3/ppo-CartPole-v1",
        "--algo", "ppo",
        "--env", "CartPole-v1",
    ])
    assert args.repo == "sb3/ppo-CartPole-v1"
    assert args.algo == "ppo"
    assert args.env == "CartPole-v1"


def test_audit_hf_parser_filename_optional():
    p = _make_hf_parser()
    args = p.parse_args([
        "audit-hf", "--repo", "r", "--algo", "ppo", "--env", "CartPole-v1",
    ])
    assert args.filename is None


def test_audit_hf_parser_filename_explicit():
    p = _make_hf_parser()
    args = p.parse_args([
        "audit-hf", "--repo", "r", "--algo", "ppo", "--env", "CartPole-v1",
        "--filename", "model.zip",
    ])
    assert args.filename == "model.zip"


def test_audit_hf_parser_quiet_flag():
    p = _make_hf_parser()
    args = p.parse_args([
        "audit-hf", "--repo", "r", "--algo", "ppo", "--env", "CartPole-v1",
        "--quiet",
    ])
    assert args.quiet is True


def test_audit_hf_parser_hf_token():
    p = _make_hf_parser()
    args = p.parse_args([
        "audit-hf", "--repo", "r", "--algo", "ppo", "--env", "CartPole-v1",
        "--hf-token", "hf_abc123",
    ])
    assert args.hf_token == "hf_abc123"


def test_audit_hf_parser_format_markdown():
    p = _make_hf_parser()
    args = p.parse_args([
        "audit-hf", "--repo", "r", "--algo", "ppo", "--env", "CartPole-v1",
        "--format", "markdown",
    ])
    assert args.output_format == "markdown"
