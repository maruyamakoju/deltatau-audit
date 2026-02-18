#!/usr/bin/env python3
"""Quickstart: Train a CartPole agent and fix its timing failures.

This script demonstrates the full deltatau-audit workflow:
1. Train a standard PPO agent on CartPole
2. Run fix-sb3 to diagnose, fix, and verify

No pre-trained model needed — trains from scratch in ~30 seconds.

Usage:
    pip install "deltatau-audit[sb3,demo]"
    python examples/fix_cartpole.py
"""

import gymnasium as gym
from pathlib import Path

from stable_baselines3 import PPO

from deltatau_audit.fixer import fix_sb3_model


def main():
    model_path = Path("cartpole_standard_ppo.zip")

    # Step 1: Train a standard PPO (intentionally timing-naive)
    if not model_path.exists():
        print("Training standard PPO on CartPole-v1...")
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0, device="cpu")
        model.learn(total_timesteps=50_000)
        model.save(str(model_path).replace(".zip", ""))
        env.close()
        print(f"Saved: {model_path}\n")

    # Step 2: Fix it (diagnose -> retrain -> verify)
    result = fix_sb3_model(
        model_path=str(model_path),
        algo="ppo",
        env_id="CartPole-v1",
        output_dir="cartpole_fix_output",
        timesteps=100_000,
    )

    if result["skipped"]:
        print("\nModel already robust — no fix needed!")
    else:
        print(f"\nFixed model: {result['fixed_model_path']}")
        print("Open cartpole_fix_output/before/index.html to see the Before report")
        print("Open cartpole_fix_output/after/index.html to see the After report")


if __name__ == "__main__":
    main()
