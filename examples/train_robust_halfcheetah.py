#!/usr/bin/env python3
"""Train a speed-randomized PPO on HalfCheetah-v5.

Uses JitterWrapper during training so the agent experiences variable
frame-skip speeds (1-5). This makes the agent robust to timing
perturbations that destroy a standard PPO.

Usage:
    python examples/train_robust_halfcheetah.py
"""

import gymnasium as gym
from pathlib import Path

from stable_baselines3 import PPO

from deltatau_audit.wrappers.speed import JitterWrapper


def make_robust_env():
    """Create HalfCheetah with speed randomization (1-5 steps/action)."""
    env = gym.make("HalfCheetah-v5")
    # base_speed=3, jitter=2 → actual speed in {1, 2, 3, 4, 5}
    env = JitterWrapper(env, base_speed=3, jitter=2)
    return env


def main():
    model_path = Path("runs/halfcheetah_ppo_robust_500k.zip")

    if model_path.exists():
        print(f"Model already exists: {model_path}")
        return

    print("Training PPO on HalfCheetah-v5 with speed randomization...")
    print("  Speed range: 1-5 (JitterWrapper base=3, jitter=2)")
    print("  Steps: 500K")
    print()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    env = make_robust_env()

    model = PPO(
        "MlpPolicy", env, verbose=1,
        n_steps=2048, batch_size=64, n_epochs=10,
        learning_rate=3e-4, device="cpu",
    )
    model.learn(total_timesteps=500_000)
    model.save(str(model_path).replace(".zip", ""))
    env.close()

    print(f"\nDONE: saved to {model_path}")


if __name__ == "__main__":
    main()
