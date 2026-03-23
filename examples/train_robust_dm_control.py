#!/usr/bin/env python3
"""Train Standard and Speed-Randomized PPO Agents on dm_control Walker-walk.

This script:
1. Trains a standard PPO agent on Walker-walk for 1M steps (fixed speed=1.0)
2. Trains a speed-randomized PPO agent for 1M steps (speed in [0.5, 2.0])
3. Saves both checkpoints
4. Audits both and prints comparison

The speed-randomized agent is expected to be more robust to timing perturbations
because it has seen many different control frequencies during training.

Requirements:
    pip install "deltatau-audit[dm_control]"

Usage:
    python examples/train_robust_dm_control.py
    python examples/train_robust_dm_control.py --timesteps 500000

Output:
    checkpoints/dm_control/walker_walk_standard.zip
    checkpoints/dm_control/walker_walk_robust.zip
    reports/dm_control_comparison/
"""

import argparse
import json
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
    print("ERROR: dm_control + shimmy required.")
    print("Install: pip install shimmy[dm-control] dm-control")
    sys.exit(1)

if not HAS_SB3:
    print("ERROR: stable-baselines3 required.")
    print("Install: pip install stable-baselines3")
    sys.exit(1)

import numpy as np
import gymnasium as gym
import shimmy  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from deltatau_audit.adapters.dm_control import DMControlSB3Adapter, make_dm_control_env
from deltatau_audit.auditor import run_full_audit
from deltatau_audit.report import generate_report
from deltatau_audit.ci import write_ci_summary
from deltatau_audit.diff import generate_comparison

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_ID = "dm_control/walker-walk-v0"
DEFAULT_TIMESTEPS = 1_000_000
N_EPISODES = 50
SPEEDS = [1, 2, 3, 5, 8]
SEED = 0
SPEED_LOW = 0.5
SPEED_HIGH = 2.0

CHECKPOINT_DIR = Path("checkpoints/dm_control")
REPORT_DIR = Path("reports/dm_control_comparison")

PPO_HYPERPARAMS = {
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "learning_rate": 3e-4,
    "device": "cpu",
    "verbose": 1,
}


class SpeedRandomizedWrapper(gym.Wrapper):
    """Gymnasium wrapper that randomizes episode speed at each reset.

    This is the core training augmentation for timing robustness:
    the agent must learn to act correctly regardless of how fast or
    slow the underlying physics is ticking.
    """

    def __init__(self, env, speed_low: float = 0.5, speed_high: float = 2.0, seed=None):
        super().__init__(env)
        self.speed_low = speed_low
        self.speed_high = speed_high
        self.rng = np.random.default_rng(seed)
        self._current_speed = 1.0

    def reset(self, **kwargs):
        self._current_speed = float(self.rng.uniform(self.speed_low, self.speed_high))
        obs, info = self.env.reset(**kwargs)
        info["speed"] = self._current_speed
        return obs, info


def train_standard_agent(timesteps: int, output_path: Path) -> PPO:
    """Train a standard PPO agent on Walker-walk at fixed speed=1.0.

    This is the baseline. It is trained exclusively at the native
    simulation speed and is therefore expected to be brittle under
    timing perturbations.

    Args:
        timesteps:   Total training timesteps.
        output_path: Where to save the .zip checkpoint.

    Returns:
        Trained PPO model.
    """
    print("[Standard] Training for {:,} timesteps on {}".format(timesteps, ENV_ID))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _make():
        return gym.make(ENV_ID)

    vec_env = DummyVecEnv([_make])
    model = PPO("MlpPolicy", vec_env, seed=SEED, **PPO_HYPERPARAMS)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(str(output_path))
    vec_env.close()
    print(f"[Standard] Saved to {output_path}")
    return model


def train_robust_agent(timesteps: int, output_path: Path) -> PPO:
    """Train a speed-randomized PPO agent on Walker-walk.

    Speed is sampled uniformly from [SPEED_LOW, SPEED_HIGH] at the start
    of each episode. The agent must learn speed-invariant policies.

    Args:
        timesteps:   Total training timesteps.
        output_path: Where to save the .zip checkpoint.

    Returns:
        Trained PPO model.
    """
    msg = "[Robust] Training {:,} timesteps (speed [{}, {}])".format(timesteps, SPEED_LOW, SPEED_HIGH)
    print(msg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _make():
        base = gym.make(ENV_ID)
        return SpeedRandomizedWrapper(base, speed_low=SPEED_LOW,
                                      speed_high=SPEED_HIGH, seed=SEED)

    vec_env = DummyVecEnv([_make])
    model = PPO("MlpPolicy", vec_env, seed=SEED, **PPO_HYPERPARAMS)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(str(output_path))
    vec_env.close()
    print(f"[Robust] Saved to {output_path}")
    return model


def audit_agent(model_path: Path, label: str, report_subdir: str) -> dict:
    """Load a checkpoint and run the full timing robustness audit.

    Args:
        model_path:    Path to SB3 .zip checkpoint.
        label:         Human-readable label for report title.
        report_subdir: Subdirectory inside REPORT_DIR for output.

    Returns:
        Full audit result dict as returned by run_full_audit().
    """
    sep = "=" * 60
    print()
    print(sep)
    print(f"AUDIT: {label}")
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
    generate_report(result, str(out_dir), title=label)
    write_ci_summary(result["summary"], result["robustness"], str(out_dir))

    summary = result["summary"]
    dep = summary["deployment_score"]
    stress = summary["stress_score"]
    quadrant = summary.get("quadrant", "N/A")
    report_path = out_dir / "report.html"
    print(f"  Deployment score : {dep:.3f}")
    print(f"  Stress score     : {stress:.3f}")
    print(f"  Quadrant         : {quadrant}")
    print(f"  Report           : {report_path}")
    return result


def print_summary(std_result: dict, rob_result: dict) -> None:
    """Print a formatted side-by-side comparison of audit results."""
    sep = "=" * 70
    print()
    print(sep)
    print("TIMING ROBUSTNESS COMPARISON: Standard vs Speed-Randomized")
    print(sep)
    print()
    std_s = std_result["summary"]
    rob_s = rob_result["summary"]
    std_dep = std_s["deployment_score"]
    rob_dep = rob_s["deployment_score"]
    std_str = std_s["stress_score"]
    rob_str = rob_s["stress_score"]
    dep_delta = rob_dep - std_dep
    str_delta = rob_str - std_str
    dep_sign = "+" if dep_delta >= 0 else ""
    str_sign = "+" if str_delta >= 0 else ""
    print(hdr1)
    print("  " + "-" * 62)
    row1 = "  {:22} {:>10.3f}  {:>10.3f}  {}{:>9.3f}"
    print(row1.format("Deployment Score", std_dep, rob_dep, dep_sign, dep_delta))
    print(row1.format("Stress Score", std_str, rob_str, str_sign, str_delta))
    print()
    std_rob = std_result["robustness"]["per_scenario_scores"]
    rob_rob = rob_result["robustness"]["per_scenario_scores"]
    hdr2 = "  {:22} {:>10}  {:>10}  {:>10}".format("Scenario","Standard","Robust","Delta")
    print(hdr2)
    print("  " + "-" * 62)
    for sc in sorted(std_rob):
        sv = std_rob[sc]["return_ratio"] * 100
        rv_data = rob_rob.get(sc, {})
        rv = rv_data.get("return_ratio", float("nan")) * 100
        d = rv - sv
        sign = "+" if d >= 0 else ""
        print("  {:<22} {:>9.1f}%  {:>9.1f}%  {}{:>8.1f}%".format(sc, sv, rv, sign, d))



def main() -> None:
    parser = argparse.ArgumentParser(description="Train+audit Standard vs Robust PPO.")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    std_path = CHECKPOINT_DIR / "walker_walk_standard.zip"
    rob_path = CHECKPOINT_DIR / "walker_walk_robust.zip"
    if not args.skip_training:
        train_standard_agent(args.timesteps, std_path)
        train_robust_agent(args.timesteps, rob_path)
    else:
        print("Skipping training -- using existing checkpoints.")
        for p in [std_path, rob_path]:
            if not p.exists():
                print(f"ERROR: not found: {p}")
                sys.exit(1)
    if not args.skip_audit:
        std_result = audit_agent(std_path, "Standard PPO", "standard")
        rob_result = audit_agent(rob_path, "Robust PPO", "robust")
        print_summary(std_result, rob_result)
        try:
            comp = generate_comparison(std_result, rob_result, str(REPORT_DIR))
            print(f"Comparison report: {comp}")
        except Exception as exc:
            print(f"Comparison skipped: {exc}")
    else:
        print("Skipping audit.")
    print()
    print("Done. Checkpoints: " + str(CHECKPOINT_DIR.resolve()))


if __name__ == "__main__":
    main()
