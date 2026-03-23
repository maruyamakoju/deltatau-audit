#!/usr/bin/env python3
"""Example: Monitor timing robustness DURING SB3 training with TimingAuditCallback.

Installs: pip install "deltatau-audit[demo,sb3]"

This shows how to audit your agent every N timesteps during training,
so you can see whether robustness improves as training progresses.
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from deltatau_audit import create_timing_audit_callback


def main():
    # 1. Create training environment
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

    # 2. Create SB3 model
    model = PPO("MlpPolicy", env, verbose=0, seed=42)

    # 3. Create timing audit callback
    #    - audit_freq: audit every N timesteps
    #    - n_episodes: episodes per condition during audit
    #    - out_dir: where to save JSON snapshots
    callback = create_timing_audit_callback(
        env_id="CartPole-v1",
        audit_freq=10_000,        # Audit every 10k steps
        n_episodes=20,            # 20 episodes per condition (fast)
        out_dir="outputs/callback_demo",
        verbose=True,
    )

    print("Training CartPole-v1 for 50k steps with timing audit callback...")
    print("  Audit will run every 10,000 timesteps")
    print()

    # 4. Train with the callback
    model.learn(total_timesteps=50_000, callback=callback, progress_bar=False)

    # 5. Check the audit snapshots
    import os
    out_dir = "outputs/callback_demo"
    if os.path.exists(out_dir):
        snapshots = sorted([d for d in os.listdir(out_dir)
                           if os.path.isdir(os.path.join(out_dir, d))])
        print(f"\nAudit snapshots saved in '{out_dir}/':")
        for snap in snapshots:
            snap_path = os.path.join(out_dir, snap, "summary.json")
            if os.path.exists(snap_path):
                import json
                with open(snap_path) as f:
                    data = json.load(f)
                s = data.get("summary", {})
                print(f"  {snap}: deployment={s.get('deployment_rating', '?')} "
                      f"({s.get('deployment_score', 0):.2f}), "
                      f"stress={s.get('stress_rating', '?')} "
                      f"({s.get('stress_score', 0):.2f})")

    print("\nDone! Use these snapshots to plot training robustness over time.")
    print(f"Load each '{out_dir}/<step>/summary.json' for time-series analysis.")


if __name__ == "__main__":
    main()
