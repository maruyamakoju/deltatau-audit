"""Speed generalization analysis and visualization.

Creates the key paper figures:
1. Δτ vs Speed (showing internal time tracks env speed)
2. Reward vs Speed (showing generalization)
3. Episode-level Δτ dynamics at different speeds
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_dt_vs_speed(results, test_speeds, train_speeds, save_path=None):
    """Plot Δτ as a function of environment speed for each model.

    This is THE key figure: internal time tracks speed, others don't.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    models_to_plot = [
        ("baseline", "Baseline GRU", "gray", "s"),
        ("internal_time", "Internal Time (ours)", "red", "o"),
        ("internal_time_discount", "Internal Time + Discount", "darkred", "^"),
        ("skip_rnn", "Skip-RNN (ACT)", "blue", "D"),
        ("external_dt", "External dt (ODE-RNN)", "green", "v"),
    ]

    speeds = np.array(test_speeds)

    for model_key, label, color, marker in models_to_plot:
        if model_key not in results:
            continue
        dts = results[model_key]["speed_dts"]
        means = [dts[str(s)]["mean"] for s in test_speeds]
        stds = [dts[str(s)].get("std", 0) for s in test_speeds]
        ax.errorbar(
            speeds, means, yerr=stds, label=label, color=color,
            marker=marker, markersize=8, linewidth=2, capsize=4,
        )

    # Mark unseen speeds
    for s in test_speeds:
        if s not in train_speeds:
            ax.axvline(x=s, color="lightgray", linestyle="--", alpha=0.3)

    ax.axhline(y=1.0, color="black", linestyle=":", alpha=0.3, label="dt=1 (baseline)")
    ax.set_xlabel("Environment Speed (action repeat)", fontsize=12)
    ax.set_ylabel("Internal Time Δτ", fontsize=12)
    ax.set_title("Internal Time Adapts to Environment Speed", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")

    # Annotate unseen region
    ax.axvspan(3.5, 8.5, alpha=0.05, color="red")
    ax.text(5.5, ax.get_ylim()[0] + 0.05, "unseen speeds →", fontsize=9,
            color="red", ha="center", alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_reward_vs_speed(results, test_speeds, train_speeds, save_path=None):
    """Plot reward as a function of speed for each model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models_to_plot = [
        ("baseline", "Baseline GRU", "gray", "s"),
        ("internal_time", "Internal Time (ours)", "red", "o"),
        ("internal_time_discount", "Internal Time + Discount", "darkred", "^"),
        ("skip_rnn", "Skip-RNN (ACT)", "blue", "D"),
    ]

    speeds = np.array(test_speeds)

    for model_key, label, color, marker in models_to_plot:
        if model_key not in results:
            continue
        rews = results[model_key]["speed_rewards"]
        means = [rews[str(s)]["mean"] for s in test_speeds]
        stds = [rews[str(s)].get("std", 0) for s in test_speeds]
        ax.errorbar(
            speeds, means, yerr=stds, label=label, color=color,
            marker=marker, markersize=8, linewidth=2, capsize=4,
        )

    ax.axvspan(3.5, 8.5, alpha=0.05, color="red")
    ax.text(5.5, ax.get_ylim()[0] + 0.02, "unseen speeds →", fontsize=9,
            color="red", ha="center", alpha=0.7)

    ax.set_xlabel("Environment Speed (action repeat)", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Reward Generalization Across Speeds", fontsize=14)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_combined_speed_analysis(results, test_speeds, train_speeds, save_path=None):
    """Combined figure: reward + Δτ side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    models = [
        ("baseline", "Baseline GRU", "gray", "s"),
        ("internal_time", "Internal Time (ours)", "red", "o"),
        ("skip_rnn", "Skip-RNN (ACT)", "blue", "D"),
    ]

    speeds = np.array(test_speeds)

    for model_key, label, color, marker in models:
        if model_key not in results:
            continue

        # Rewards
        rews = results[model_key]["speed_rewards"]
        r_means = [rews[str(s)]["mean"] for s in test_speeds]
        r_stds = [rews[str(s)].get("std", 0) for s in test_speeds]
        ax1.errorbar(speeds, r_means, yerr=r_stds, label=label, color=color,
                     marker=marker, markersize=8, linewidth=2, capsize=4)

        # Delta tau
        dts = results[model_key]["speed_dts"]
        d_means = [dts[str(s)]["mean"] for s in test_speeds]
        d_stds = [dts[str(s)].get("std", 0) for s in test_speeds]
        ax2.errorbar(speeds, d_means, yerr=d_stds, label=label, color=color,
                     marker=marker, markersize=8, linewidth=2, capsize=4)

    for ax in [ax1, ax2]:
        ax.axvspan(3.5, 8.5, alpha=0.05, color="red")

    ax1.set_xlabel("Environment Speed", fontsize=12)
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.set_title("(a) Reward vs Speed", fontsize=13)
    ax1.legend(fontsize=9)

    ax2.axhline(y=1.0, color="black", linestyle=":", alpha=0.3)
    ax2.set_xlabel("Environment Speed", fontsize=12)
    ax2.set_ylabel("Internal Time Δτ", fontsize=12)
    ax2.set_title("(b) Learned Δτ vs Speed", fontsize=13)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def generate_all_figures(results_path, output_dir=None, train_speeds=(1, 2, 3),
                         test_speeds=(1, 2, 3, 5, 8)):
    """Generate all analysis figures from a results.json file."""
    with open(results_path) as f:
        results = json.load(f)

    if output_dir is None:
        output_dir = os.path.dirname(results_path)

    os.makedirs(output_dir, exist_ok=True)

    plot_dt_vs_speed(
        results, test_speeds, train_speeds,
        save_path=os.path.join(output_dir, "dt_vs_speed.png"),
    )
    plot_reward_vs_speed(
        results, test_speeds, train_speeds,
        save_path=os.path.join(output_dir, "reward_vs_speed.png"),
    )
    plot_combined_speed_analysis(
        results, test_speeds, train_speeds,
        save_path=os.path.join(output_dir, "combined_analysis.png"),
    )

    print(f"All figures saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generate_all_figures(sys.argv[1])
    else:
        print("Usage: python -m internal_time_rl.analysis.speed_analysis <results.json>")
