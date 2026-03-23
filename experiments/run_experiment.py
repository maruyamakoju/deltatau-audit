"""Run baseline vs internal time comparison experiments.

This script runs the core ablation study:
1. Baseline: Standard GRU (delta_tau fixed at 1.0)
2. Internal Time GRU: Learnable delta_tau
3. Internal Time Neural ODE: Neural ODE with learnable delta_tau

Each experiment runs across multiple seeds for statistical significance.

Usage:
    python run_experiment.py                    # Full comparison
    python run_experiment.py --quick            # Quick test (fewer timesteps)
    python run_experiment.py --seeds 3          # Number of seeds
    python run_experiment.py --delay 20         # Test with longer delay
"""

import argparse
import copy
import json
import os

import numpy as np

from train import train


BASE_CONFIG = {
    "env": {
        "name": "delayed_chain",
        "length": 20,
        "delay": 10,
        "max_steps": 200,
        "noise": 0.0,
    },
    "model": {
        "hidden_dim": 128,
        "latent_dim": 64,
        "time_hidden_dim": 32,
        "transition_type": "gru",
    },
    "algorithm": {
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "time_var_coef": 0.01,
        "time_mean_coef": 0.001,
        "max_grad_norm": 0.5,
        "num_epochs": 4,
        "num_minibatches": 4,
        "num_steps": 128,
        "num_envs": 8,
        "total_timesteps": 500_000,
    },
    "logging": {
        "log_interval": 10,
        "save_interval": 100,
    },
}


EXPERIMENTS = {
    "baseline": {
        "model": {"use_internal_time": False, "transition_type": "gru"},
    },
    "internal_time_gru": {
        "model": {"use_internal_time": True, "transition_type": "gru"},
    },
    "internal_time_ode": {
        "model": {"use_internal_time": True, "transition_type": "ode"},
    },
}


def merge_config(base, override):
    """Deep merge override into base config."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = v
    return result


def run_experiments(
    num_seeds=3,
    total_timesteps=None,
    delay=None,
    chain_length=None,
    output_dir="runs",
    experiments=None,
):
    """Run all experiments across multiple seeds."""
    if experiments is None:
        experiments = EXPERIMENTS

    base = copy.deepcopy(BASE_CONFIG)
    if total_timesteps is not None:
        base["algorithm"]["total_timesteps"] = total_timesteps
    if delay is not None:
        base["env"]["delay"] = delay
    if chain_length is not None:
        base["env"]["length"] = chain_length

    all_results = {}

    for exp_name, exp_override in experiments.items():
        print("\n" + "=" * 70)
        print(f"EXPERIMENT: {exp_name}")
        print("=" * 70)

        seed_histories = []

        for seed in range(num_seeds):
            print(f"\n--- Seed {seed} ---")
            config = merge_config(base, exp_override)
            config["seed"] = seed * 1000 + 42
            config["log_dir"] = os.path.join(output_dir, exp_name, f"seed_{seed}")

            history = train(config)
            seed_histories.append(history)

        all_results[exp_name] = seed_histories

    # Save aggregated results
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    summary = {}
    for exp_name, histories in all_results.items():
        final_rewards = []
        for h in histories:
            if h["episode_rewards"]:
                final_rewards.append(np.mean(h["episode_rewards"][-20:]))
        summary[exp_name] = {
            "mean_final_reward": float(np.mean(final_rewards)) if final_rewards else 0,
            "std_final_reward": float(np.std(final_rewards)) if final_rewards else 0,
            "num_seeds": len(histories),
        }
        if any(h.get("delta_tau_means") for h in histories):
            dt_means = [
                np.mean(h["delta_tau_means"][-20:])
                for h in histories
                if h.get("delta_tau_means")
            ]
            summary[exp_name]["mean_final_delta_tau"] = float(np.mean(dt_means))

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for exp_name, s in summary.items():
        line = f"  {exp_name:25s}: reward = {s['mean_final_reward']:.3f} +/- {s['std_final_reward']:.3f}"
        if "mean_final_delta_tau" in s:
            line += f" | delta_tau = {s['mean_final_delta_tau']:.3f}"
        print(line)

    print(f"\nResults saved to {output_dir}/")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--delay", type=int, default=None)
    parser.add_argument("--chain-length", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with fewer timesteps"
    )
    args = parser.parse_args()

    timesteps = args.total_timesteps
    if args.quick and timesteps is None:
        timesteps = 50_000

    run_experiments(
        num_seeds=args.seeds,
        total_timesteps=timesteps,
        delay=args.delay,
        chain_length=args.chain_length,
        output_dir=args.output_dir,
    )
