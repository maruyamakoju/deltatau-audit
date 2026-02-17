#!/usr/bin/env python3
"""HalfCheetah Timing Robustness Audit — end-to-end example.

Trains a standard PPO agent on HalfCheetah-v5, then audits it for
timing robustness. Demonstrates that a normally-trained MuJoCo agent
is vulnerable to timing perturbations.

Requirements:
    pip install deltatau-audit stable-baselines3 gymnasium[mujoco]

Usage:
    python examples/audit_halfcheetah.py

This script:
1. Trains PPO on HalfCheetah-v5 (500K steps, ~5-10 min)
2. Runs a full timing robustness audit
3. Generates an HTML report showing timing vulnerabilities
4. Outputs CI-compatible exit codes
"""

import gymnasium as gym
from pathlib import Path

from stable_baselines3 import PPO

from deltatau_audit.adapters.sb3 import SB3Adapter
from deltatau_audit.auditor import run_full_audit
from deltatau_audit.report import generate_report
from deltatau_audit.ci import write_ci_summary


def main():
    output_dir = Path("halfcheetah_audit")
    model_path = Path("runs/halfcheetah_ppo_500k.zip")

    # ── Step 1: Train (or load) ──────────────────────────────────
    if model_path.exists():
        print(f"Loading pre-trained model from {model_path}")
        model = PPO.load(str(model_path), device="cpu")
    else:
        print("Training PPO on HalfCheetah-v5 (500K steps)...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        env = gym.make("HalfCheetah-v5")
        model = PPO("MlpPolicy", env, verbose=1,
                     n_steps=2048, batch_size=64, n_epochs=10,
                     learning_rate=3e-4, device="cpu")
        model.learn(total_timesteps=500_000)
        model.save(str(model_path).replace(".zip", ""))
        env.close()
        print(f"Saved to {model_path}")

    # ── Step 2: Audit ────────────────────────────────────────────
    adapter = SB3Adapter(model)
    env_factory = lambda: gym.make("HalfCheetah-v5")

    print("\n" + "=" * 60)
    print("TIMING ROBUSTNESS AUDIT: HalfCheetah PPO")
    print("=" * 60 + "\n")

    result = run_full_audit(
        adapter,
        env_factory,
        speeds=[1, 2, 3, 5, 8],
        n_episodes=30,
    )

    # ── Step 3: Report ───────────────────────────────────────────
    out = str(output_dir)
    generate_report(result, out, title="HalfCheetah PPO — Timing Audit")

    exit_code = write_ci_summary(
        result["summary"], result["robustness"], out,
    )

    print(f"\nReport: {output_dir}/index.html")
    print(f"CI exit code: {exit_code} "
          f"({'pass' if exit_code == 0 else 'warn' if exit_code == 1 else 'FAIL'})")

    # ── Print key findings ───────────────────────────────────────
    rob = result["robustness"]
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    for sc_name, sc in rob["per_scenario_scores"].items():
        ratio = sc["return_ratio"]
        ci_lo = sc.get("ci_lower", ratio)
        ci_hi = sc.get("ci_upper", ratio)
        sig = sc.get("significant", False)
        star = " ***" if sig else ""
        pct = ratio * 100
        print(f"  {sc_name:12s}: {pct:5.1f}% "
              f"[{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]{star}")

    sig_count = sum(1 for s in rob["per_scenario_scores"].values()
                    if s.get("significant"))
    total = len(rob["per_scenario_scores"])
    print(f"\n  {sig_count}/{total} scenarios show statistically "
          f"significant performance drop (95% CI)")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
