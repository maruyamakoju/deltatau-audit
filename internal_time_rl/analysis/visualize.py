"""Visualization and analysis tools for Internal Time RL experiments."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np


def smooth(data, window=10):
    """Simple moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_training_curves(history, title="Training Curves", save_path=None):
    """Plot comprehensive training metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16)

    # Episode rewards
    if history.get("episode_rewards"):
        rewards = np.array(history["episode_rewards"])
        axes[0, 0].plot(rewards, alpha=0.3, color="blue")
        if len(rewards) > 10:
            axes[0, 0].plot(smooth(rewards), color="blue", linewidth=2)
        axes[0, 0].set_title("Episode Reward")
        axes[0, 0].set_xlabel("Update")
        axes[0, 0].set_ylabel("Reward")

    # Episode lengths
    if history.get("episode_lengths"):
        lengths = np.array(history["episode_lengths"])
        axes[0, 1].plot(lengths, alpha=0.3, color="green")
        if len(lengths) > 10:
            axes[0, 1].plot(smooth(lengths), color="green", linewidth=2)
        axes[0, 1].set_title("Episode Length")
        axes[0, 1].set_xlabel("Update")

    # Internal time dynamics
    if history.get("delta_tau_means"):
        means = np.array(history["delta_tau_means"])
        stds = np.array(history["delta_tau_stds"])
        x = np.arange(len(means))
        axes[0, 2].plot(x, means, color="red", label="Mean")
        axes[0, 2].fill_between(
            x, means - stds, means + stds, alpha=0.3, color="red"
        )
        axes[0, 2].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (1.0)")
        axes[0, 2].set_title("Internal Time delta_tau")
        axes[0, 2].set_xlabel("Update")
        axes[0, 2].legend()

    # Policy loss
    if history.get("policy_losses"):
        axes[1, 0].plot(history["policy_losses"])
        axes[1, 0].set_title("Policy Loss")
        axes[1, 0].set_xlabel("Update")

    # Value loss
    if history.get("value_losses"):
        axes[1, 1].plot(history["value_losses"])
        axes[1, 1].set_title("Value Loss")
        axes[1, 1].set_xlabel("Update")

    # Entropy
    if history.get("entropies"):
        axes[1, 2].plot(history["entropies"])
        axes[1, 2].set_title("Policy Entropy")
        axes[1, 2].set_xlabel("Update")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def compare_experiments(paths, labels, save_path=None):
    """Compare training curves across multiple experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["blue", "red", "green", "orange", "purple"]

    for i, (path, label) in enumerate(zip(paths, labels)):
        history_file = os.path.join(path, "history.json")
        if not os.path.exists(history_file):
            print(f"Warning: {history_file} not found, skipping")
            continue
        with open(history_file) as f:
            history = json.load(f)

        color = colors[i % len(colors)]

        # Rewards
        if history.get("episode_rewards"):
            rewards = np.array(history["episode_rewards"])
            if len(rewards) > 10:
                axes[0].plot(smooth(rewards, 20), label=label, color=color, linewidth=2)
            else:
                axes[0].plot(rewards, label=label, color=color)

        # Internal time
        if history.get("delta_tau_means"):
            means = np.array(history["delta_tau_means"])
            stds = np.array(history["delta_tau_stds"])
            x = np.arange(len(means))
            axes[1].plot(x, means, label=label, color=color)
            axes[1].fill_between(x, means - stds, means + stds, alpha=0.2, color=color)

        # Entropy
        if history.get("entropies"):
            axes[2].plot(history["entropies"], label=label, color=color)

    axes[0].set_title("Episode Reward (smoothed)")
    axes[0].set_xlabel("Update")
    axes[0].legend()

    axes[1].set_title("Internal Time delta_tau")
    axes[1].set_xlabel("Update")
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].legend()

    axes[2].set_title("Policy Entropy")
    axes[2].set_xlabel("Update")
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_episode_time_dynamics(delta_taus, rewards, positions, save_path=None):
    """Plot internal time dynamics over a single episode.

    Shows how delta_tau changes as the agent navigates and
    approaches/receives rewards.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    steps = np.arange(len(delta_taus))

    axes[0].plot(steps, delta_taus, color="red", linewidth=2)
    axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("delta_tau")
    axes[0].set_title("Internal Time Dynamics During Episode")

    axes[1].plot(steps, positions, color="blue", linewidth=2)
    axes[1].set_ylabel("Position")

    axes[2].bar(steps, rewards, color="green", alpha=0.7)
    axes[2].set_ylabel("Reward")
    axes[2].set_xlabel("Environment Step")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            history = json.load(f)
        plot_training_curves(history, save_path="training_curves.png")
    else:
        print("Usage: python -m internal_time_rl.analysis.visualize <history.json>")
