"""Training curve visualization across all experiments.

Shows:
1. Reward curves during training for each model
2. Δτ evolution during training
3. Speed-conditioned Δτ evolution (if speed_dt_pairs available)

Usage:
    python -m internal_time_rl.analysis.training_curves runs/speed_gen_hidden
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 200,
})

MODEL_COLORS = {
    "baseline": "#7f7f7f",
    "internal_time": "#d62728",
    "internal_time_discount": "#8c1515",
    "skip_rnn": "#1f77b4",
    "external_dt": "#2ca02c",
}

MODEL_LABELS = {
    "baseline": "Baseline GRU",
    "internal_time": "Internal Time (ours)",
    "internal_time_discount": r"IT + $\gamma^{\Delta\tau}$",
    "skip_rnn": "Skip-RNN",
    "external_dt": "External dt",
}


def load_histories(experiment_dir):
    """Load training histories for all models and seeds."""
    histories = {}  # model_name -> [history_per_seed]

    for model_dir in sorted(os.listdir(experiment_dir)):
        model_path = os.path.join(experiment_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        if model_dir in ("figures", "paper_figures", "episode_dynamics"):
            continue

        model_histories = []
        for seed_dir in sorted(os.listdir(model_path)):
            hist_path = os.path.join(model_path, seed_dir, "history.json")
            if os.path.exists(hist_path):
                with open(hist_path) as f:
                    model_histories.append(json.load(f))

        if model_histories:
            histories[model_dir] = model_histories

    return histories


def smooth(data, window=10):
    """Simple moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_training_curves(experiment_dir, save_path=None):
    """Plot reward and Δτ curves during training."""
    histories = load_histories(experiment_dir)
    if not histories:
        print(f"No histories found in {experiment_dir}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for model_name, hist_list in histories.items():
        color = MODEL_COLORS.get(model_name, "black")
        label = MODEL_LABELS.get(model_name, model_name)

        # Collect reward curves from all seeds
        all_rewards = []
        all_dts = []
        for h in hist_list:
            r = h.get("episode_rewards", [])
            d = h.get("delta_tau_means", [])
            if r:
                all_rewards.append(smooth(np.array(r), window=5))
            if d:
                all_dts.append(smooth(np.array(d), window=5))

        # Plot mean ± std across seeds
        if all_rewards:
            min_len = min(len(r) for r in all_rewards)
            rewards_arr = np.array([r[:min_len] for r in all_rewards])
            mean_r = rewards_arr.mean(axis=0)
            std_r = rewards_arr.std(axis=0)
            x = np.arange(len(mean_r))
            ax1.plot(x, mean_r, color=color, label=label, linewidth=1.5)
            ax1.fill_between(x, mean_r - std_r, mean_r + std_r, alpha=0.15, color=color)

        if all_dts:
            min_len = min(len(d) for d in all_dts)
            dts_arr = np.array([d[:min_len] for d in all_dts])
            mean_d = dts_arr.mean(axis=0)
            std_d = dts_arr.std(axis=0)
            x = np.arange(len(mean_d))
            ax2.plot(x, mean_d, color=color, label=label, linewidth=1.5)
            ax2.fill_between(x, mean_d - std_d, mean_d + std_d, alpha=0.15, color=color)

    ax1.set_ylabel("Episode Reward")
    ax1.set_title("(a) Training Reward", fontsize=13)
    ax1.legend(fontsize=9)

    ax2.axhline(y=1.0, color="black", linestyle=":", alpha=0.3)
    ax2.set_ylabel(r"$\Delta\tau$ (mean)")
    ax2.set_xlabel("PPO Update")
    ax2.set_title(r"(b) $\Delta\tau$ During Training", fontsize=13)
    ax2.legend(fontsize=9)

    exp_name = os.path.basename(experiment_dir)
    fig.suptitle(f"Training Curves — {exp_name}", fontsize=14)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(experiment_dir, "paper_figures", "training_curves.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for d in sys.argv[1:]:
            plot_training_curves(d)
    else:
        print("Usage: python -m internal_time_rl.analysis.training_curves <experiment_dir>")
