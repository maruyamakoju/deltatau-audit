"""Tests for CLI subcommands."""

import subprocess
import sys


def test_help_shows_audit_sb3():
    """CLI help includes audit-sb3 subcommand."""
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "--help"],
        capture_output=True, text=True,
    )
    assert "audit-sb3" in result.stdout
    assert "bench" in result.stdout
    assert "stress" in result.stdout


def test_audit_sb3_help_includes_seed_sweep_flags():
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "audit-sb3", "--help"],
        capture_output=True, text=True,
    )
    assert "--seeds" in result.stdout
    assert "--ci-min-deployment-pass-rate" in result.stdout
    assert "--protocol" in result.stdout
    assert "--ci-gate-mode" in result.stdout
    assert "--explain-fail" in result.stdout
    assert "--env-wrap-time-feature" in result.stdout


def test_bench_run_help():
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "bench", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--manifest" in result.stdout
    assert "--no-resume" in result.stdout
    assert "--protocol" in result.stdout


def test_bench_table_help():
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "bench", "table", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--summary" in result.stdout


def test_stress_analyze_help():
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "stress", "analyze", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--summary" in result.stdout
    assert "--stress-threshold" in result.stdout


def test_stress_train_sb3_help():
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "stress", "train-sb3", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--out-root" in result.stdout
    assert "--variants" in result.stdout


def test_audit_sb3_missing_args():
    """audit-sb3 exits with error when required args missing."""
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "audit-sb3"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "required" in result.stderr.lower() or "error" in result.stderr.lower()


def test_audit_sb3_bad_algo():
    """audit-sb3 rejects unknown algorithm."""
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "audit-sb3",
         "--algo", "dqn", "--model", "x.zip", "--env", "CartPole-v1"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_audit_sb3_missing_sb3(monkeypatch):
    """audit-sb3 shows helpful error when SB3 not installed."""
    # This test runs in a subprocess with SB3 hidden
    code = (
        "import sys; "
        "sys.modules['stable_baselines3'] = None; "
        "from deltatau_audit.cli import main; "
        "sys.argv = ['prog', 'audit-sb3', '--algo', 'ppo', "
        "'--model', 'x.zip', '--env', 'CartPole-v1']; "
        "main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
    )
    # Should fail with import error message
    assert result.returncode != 0


def test_audit_sb3_missing_model():
    """audit-sb3 shows helpful message for non-existent model file."""
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "audit-sb3",
         "--algo", "ppo", "--model", "nonexistent.zip",
         "--env", "CartPole-v1"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "not found" in output.lower()


def test_audit_sb3_model_not_zip():
    """audit-sb3 hints about .zip extension when missing."""
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "audit-sb3",
         "--algo", "ppo", "--model", "nonexistent",
         "--env", "CartPole-v1"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert ".zip" in output


def test_audit_sb3_sample_model_hint():
    """audit-sb3 suggests sample model download on missing file."""
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "audit-sb3",
         "--algo", "ppo", "--model", "nonexistent.zip",
         "--env", "CartPole-v1"],
        capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    assert "cartpole_ppo_sb3.zip" in output


def test_audit_sb3_bad_env():
    """audit-sb3 fails gracefully with invalid environment."""
    result = subprocess.run(
        [sys.executable, "-m", "deltatau_audit", "audit-sb3",
         "--algo", "ppo", "--model", __file__,  # use this file as dummy
         "--env", "NonExistentEnv-v99"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "cannot create" in output.lower() or "error" in output.lower()
