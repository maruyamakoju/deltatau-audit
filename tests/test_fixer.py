"""Tests for fix-sb3 command and fixer module."""

import subprocess
import sys

import pytest


# ── CLI argument parsing tests ───────────────────────────────────

def test_help_shows_fix_sb3():
    """CLI help includes fix-sb3 subcommand."""
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "--help"],
        capture_output=True, text=True,
    )
    assert "fix-sb3" in result.stdout


def test_fix_sb3_missing_args():
    """fix-sb3 exits with error when required args missing."""
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "fix-sb3"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "required" in result.stderr.lower() or "error" in result.stderr.lower()


def test_fix_sb3_bad_algo():
    """fix-sb3 rejects unknown algorithm."""
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "fix-sb3",
         "--algo", "dqn", "--model", "x.zip", "--env", "CartPole-v1"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_fix_sb3_missing_model():
    """fix-sb3 shows helpful message for non-existent model file."""
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "fix-sb3",
         "--algo", "ppo", "--model", "nonexistent.zip",
         "--env", "CartPole-v1"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "not found" in output.lower()


# ── Fixer module unit tests ──────────────────────────────────────

def test_estimate_timesteps():
    """Auto-estimation gives reasonable timesteps per env."""
    from deltatau_audit.fixer import _estimate_timesteps

    assert _estimate_timesteps("CartPole-v1", "ppo") == 100_000
    assert _estimate_timesteps("HalfCheetah-v5", "ppo") == 500_000
    assert _estimate_timesteps("LunarLander-v3", "ppo") == 100_000
    assert _estimate_timesteps("SomeCustomEnv-v0", "ppo") == 200_000


def test_make_robust_env():
    """Robust env wraps with JitterWrapper."""
    from deltatau_audit.fixer import _make_robust_env
    from deltatau_audit.wrappers.speed import JitterWrapper

    env = _make_robust_env("CartPole-v1", base_speed=3, jitter=2)
    assert isinstance(env, JitterWrapper)
    assert env.base_speed == 3
    assert env.jitter == 2
    env.close()


@pytest.mark.slow
@pytest.mark.xfail(
    reason="CartPole with 1k training steps + 5 eval episodes is nondeterministic: "
           "the model sometimes scores >= 0.95 and the fixer skips retraining",
    strict=False,
)
def test_fix_sb3_model_cartpole(tmp_path):
    """Full fix pipeline on CartPole (integration test)."""
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    from stable_baselines3 import PPO

    # Train a minimal model
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=0, device="cpu",
                n_steps=128, batch_size=64)
    model.learn(total_timesteps=1000)
    model_path = str(tmp_path / "test_model")
    model.save(model_path)
    env.close()

    # Run fix pipeline
    from deltatau_audit.fixer import fix_sb3_model

    result = fix_sb3_model(
        model_path=model_path + ".zip",
        algo="ppo",
        env_id="CartPole-v1",
        output_dir=str(tmp_path / "fix_output"),
        timesteps=1000,
        n_audit_episodes=5,
        verbose=False,
    )

    # Verify outputs
    assert not result["skipped"]
    assert result["fixed_model_path"] is not None
    assert result["before"] is not None
    assert result["after"] is not None
    assert (tmp_path / "fix_output" / "before" / "index.html").exists()
    assert (tmp_path / "fix_output" / "after" / "index.html").exists()
    assert (tmp_path / "fix_output" / "comparison.md").exists()
