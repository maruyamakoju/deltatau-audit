#!/usr/bin/env python3
"""Train a standard PPO agent on HalfCheetah-v5.

This produces a baseline model that has never seen timing perturbations.
It will serve as the "Before" model in the case study.

Usage:
    python train.py --device cuda --seed 42
    python train.py --timesteps 500000 --device cpu
    python train.py --force  # retrain even if model exists

Requirements:
    pip install "deltatau-audit[sb3,mujoco]"
"""

import argparse
import sys
import time
from pathlib import Path


def _require(pkg: str, extra: str = ""):
    try:
        __import__(pkg)
    except ImportError:
        tip = f'pip install "deltatau-audit[{extra}]"' if extra else f"pip install {pkg}"
        print(f"ERROR: {pkg} is required.\n  {tip}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Train a standard PPO on HalfCheetah-v5")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps (default: 1M)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="outputs/models",
                        help="Directory to save the model (default: outputs/models)")
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if model exists")
    args = parser.parse_args()

    _require("stable_baselines3", "sb3,mujoco")
    _require("gymnasium")

    import gymnasium as gym

    # Verify MuJoCo env is available
    try:
        test_env = gym.make("HalfCheetah-v5")
        test_env.close()
    except Exception as e:
        print(f"ERROR: Cannot create HalfCheetah-v5: {e}")
        print('  pip install "deltatau-audit[sb3,mujoco]"')
        sys.exit(1)

    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy

    model_dir = Path(args.output)
    model_path = model_dir / "halfcheetah_ppo.zip"

    # Skip if model already exists (unless --force)
    if model_path.exists() and not args.force:
        print(f"Model already exists: {model_path}")
        print("  Use --force to retrain.")

        # Show evaluation of existing model
        model = PPO.load(str(model_path), device=args.device)
        env = gym.make("HalfCheetah-v5")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        env.close()
        print(f"  Eval: {mean_reward:.1f} +/- {std_reward:.1f} (10 episodes)")
        return

    # Train
    print(f"\n{'=' * 60}")
    print(f"Training PPO on HalfCheetah-v5 ({args.timesteps:,} steps)")
    print(f"  Device: {args.device}")
    print(f"  Seed:   {args.seed}")
    print(f"{'=' * 60}\n")

    model_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("HalfCheetah-v5")
    model = PPO(
        "MlpPolicy", env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    elapsed = time.time() - t0

    model.save(str(model_path))
    env.close()

    print(f"\n  Model saved to: {model_path}")
    print(f"  Training time:  {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Evaluate
    eval_env = gym.make("HalfCheetah-v5")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    eval_env.close()
    print(f"  Eval reward:    {mean_reward:.1f} +/- {std_reward:.1f} (10 episodes)")


if __name__ == "__main__":
    main()
