#!/usr/bin/env python3
"""dm_control Timing Robustness Audit Example.

Demonstrates the deltatau-audit workflow on dm_control Walker-walk:
1. Check for pre-trained checkpoint (or train for 100k steps)
2. Audit the standard model -- likely FAIL/DEGRADED
3. Audit the speed-randomized model -- improvement expected
4. Generate comparison HTML report

Requirements:
    pip install "deltatau-audit[dm_control]"

Usage:
    python examples/audit_dm_control.py
"""

import json
import os
import sys
from pathlib import Path

try:
    import shimmy  # noqa: F401
    import dm_control  # noqa: F401
    HAS_DM_CONTROL = True
except ImportError:
    HAS_DM_CONTROL = False

try:
    import stable_baselines3  # noqa: F401
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

if not HAS_DM_CONTROL:
    print("dm_control + shimmy are required for this example.")
    print("Install with: pip install shimmy[dm-control] dm-control")
    sys.exit(1)

if not HAS_SB3:
    print("stable-baselines3 is required.")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)

import gymnasium as gym
import shimmy  # noqa: F401 -- registers dm_control envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from deltatau_audit.adapters.dm_control import DMControlSB3Adapter, make_dm_control_env
from deltatau_audit.auditor import run_full_audit
from deltatau_audit.report import generate_report
from deltatau_audit.ci import write_ci_summary
from deltatau_audit.diff import generate_comparison

ENV_ID = "dm_control/walker-walk-v0"
TRAIN_STEPS = 100_000
N_EPISODES = 20
SPEEDS = [1, 2, 3, 5, 8]
SEED = 42

CHECKPOINT_DIR = Path("checkpoints/dm_control")
STANDARD_CKPT = CHECKPOINT_DIR / "walker_walk_standard.zip"
ROBUST_CKPT = CHECKPOINT_DIR / "walker_walk_robust.zip"
REPORT_DIR = Path("examples/dm_control_report")


def train_standard(output_path: Path) -> None:
    """Train a standard PPO agent on Walker-walk with fixed speed=1.0."""
    print(f"Training standard agent for {TRAIN_STEPS:,} steps...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _make_env():
        return gym.make(ENV_ID)

    vec_env = DummyVecEnv([_make_env])
    model = PPO("MlpPolicy", vec_env, verbose=1, seed=SEED,
                n_steps=2048, batch_size=64, n_epochs=10,
                learning_rate=3e-4, device="cpu")
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(str(output_path))
    vec_env.close()
    print(f"Saved standard model to {output_path}")


def train_robust(output_path: Path) -> None:
    """Train a speed-randomized PPO agent on Walker-walk.

    Each episode uses a random speed in [0.5, 2.0]. This is the simplest
    form of domain randomization for timing robustness.
    """
    print(f"Training robust agent for {TRAIN_STEPS:,} steps (speed randomized)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    class SpeedRandomizedWrapper(gym.Wrapper):
        def __init__(self, env, speed_low=0.5, speed_high=2.0, seed=None):
            super().__init__(env)
            self.speed_low = speed_low
            self.speed_high = speed_high
            self.rng = np.random.default_rng(seed)

        def reset(self, **kwargs):
            speed = float(self.rng.uniform(self.speed_low, self.speed_high))
            obs, info = self.env.reset(**kwargs)
            info["speed"] = speed
            return obs, info

    def _make_robust_env():
        return SpeedRandomizedWrapper(gym.make(ENV_ID), seed=SEED)

    vec_env = DummyVecEnv([_make_robust_env])
    model = PPO("MlpPolicy", vec_env, verbose=1, seed=SEED,
                n_steps=2048, batch_size=64, n_epochs=10,
                learning_rate=3e-4, device="cpu")
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(str(output_path))
    vec_env.close()
    print(f"Saved robust model to {output_path}")


def audit_model(model_path: Path, title: str, report_subdir: str) -> dict:
    """Load model, run full audit, generate report, return result dict."""
    sep = "=" * 60
    print()
    print(sep)
    print(f"AUDIT: {title}")
    print(sep)
    print()

    model = PPO.load(str(model_path), device="cpu")
    adapter = DMControlSB3Adapter(model, env_id=ENV_ID, seed=SEED)
    env_factory = lambda: make_dm_control_env(ENV_ID, seed=SEED)

    result = run_full_audit(
        adapter,
        env_factory=env_factory,
        speeds=SPEEDS,
        n_episodes=N_EPISODES,
        seed=SEED,
    )

    out_dir = REPORT_DIR / report_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_report(result, str(out_dir), title=title)
    write_ci_summary(result["summary"], result["robustness"], str(out_dir))

    summary = result["summary"]
    dep_score = summary["deployment_score"]
    str_score = summary["stress_score"]
    quadrant = summary.get("quadrant", "N/A")
    report_path = out_dir / "report.html"
    print("  Deployment score : {:.3f}".format(dep_score))
    print("  Stress score     : {:.3f}".format(str_score))
    print("  Quadrant         : " + quadrant)
    print("  Report           : " + str(report_path))
    return result


def print_comparison(before: dict, after: dict) -> None:
    """Print side-by-side before/after comparison table."""
    sep = "=" * 60
    print()
    print(sep)
    print("BEFORE vs AFTER COMPARISON")
    print(sep)
    print()
    b_rob = before["robustness"]["per_scenario_scores"]
    a_rob = after["robustness"]["per_scenario_scores"]
    hdr = "  {:16}  {:>10}  {:>10}  {:>10}".format("Scenario","Before","After","Change")
    print(hdr)
    for sc in sorted(b_rob):
        b_pct = b_rob[sc]["return_ratio"] * 100
        a_data = a_rob.get(sc, {})
        a_ratio = a_data.get("return_ratio", float("nan"))
        a_pct = a_ratio * 100
        delta = a_pct - b_pct
        sign = "+" if delta >= 0 else ""
        print(f"  {sc:<16}  {b_pct:9.1f}%  {a_pct:9.1f}%  {sign}{delta:8.1f}%")
    b_sum = before["summary"]
    a_sum = after["summary"]
    print()
    b_dep = b_sum["deployment_score"]
    a_dep = a_sum["deployment_score"]
    b_str = b_sum["stress_score"]
    a_str = a_sum["stress_score"]
    print(f"  Deployment: {b_dep:.3f} -> {a_dep:.3f}")
    print(f"  Stress:     {b_str:.3f} -> {a_str:.3f}")


def main() -> None:
    """Run the full before/after audit pipeline."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not STANDARD_CKPT.exists():
        print(f"No pre-trained checkpoint at {STANDARD_CKPT}.")
        print(f"Training for {TRAIN_STEPS:,} steps...")
        train_standard(STANDARD_CKPT)
    else:
        print(f"Using pre-trained standard model: {STANDARD_CKPT}")

    if not ROBUST_CKPT.exists():
        print(f"No pre-trained robust checkpoint at {ROBUST_CKPT}.")
        print(f"Training speed-randomized agent for {TRAIN_STEPS:,} steps...")
        train_robust(ROBUST_CKPT)
    else:
        print(f"Using pre-trained robust model: {ROBUST_CKPT}")

    result_before = audit_model(
        STANDARD_CKPT,
        title="Walker-walk Standard PPO (Before)",
        report_subdir="before",
    )

    result_after = audit_model(
        ROBUST_CKPT,
        title="Walker-walk Speed-Randomized PPO (After)",
        report_subdir="after",
    )

    print_comparison(result_before, result_after)

    try:
        comp_path = generate_comparison(result_before, result_after, str(REPORT_DIR))
        print(f"Comparison report: {comp_path}")
    except Exception as exc:
        print(f"Could not generate comparison (non-fatal): {exc}")

    print()
    print("All reports saved to: " + str(REPORT_DIR.resolve()))


if __name__ == "__main__":
    main()
