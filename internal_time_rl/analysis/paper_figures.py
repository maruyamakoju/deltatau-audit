"""Publication-quality figure generator for the Internal Time paper.

Generates:
1. Figure 1 (Main Result): Multi-panel showing Δτ vs speed + reward vs speed + gen gap
2. Figure 2 (Episode Dynamics): Step-by-step Δτ at different speeds
3. Figure 3 (Generalization Gap): Bar chart comparing all models
4. Figure 4 (Training Curves): Δτ evolution during training
5. Table 1: Full results with generalization gap metric

Usage:
    python -m internal_time_rl.analysis.paper_figures --results-dir runs/speed_gen_hidden
    python -m internal_time_rl.analysis.paper_figures --all  # All experiments
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Consistent color scheme across all figures
MODEL_STYLES = {
    "baseline": {
        "label": "Baseline GRU",
        "color": "#7f7f7f",
        "marker": "s",
        "linestyle": "--",
    },
    "internal_time": {
        "label": "Internal Time (ours)",
        "color": "#d62728",
        "marker": "o",
        "linestyle": "-",
    },
    "internal_time_discount": {
        "label": r"Internal Time + $\gamma^{\Delta\tau}$",
        "color": "#8c1515",
        "marker": "^",
        "linestyle": "-",
    },
    "skip_rnn": {
        "label": "Skip-RNN (ACT)",
        "color": "#1f77b4",
        "marker": "D",
        "linestyle": "-.",
    },
    "external_dt": {
        "label": "External dt (ODE-RNN)",
        "color": "#2ca02c",
        "marker": "v",
        "linestyle": ":",
    },
}

TRAIN_SPEEDS = (1, 2, 3)
TEST_SPEEDS = (1, 2, 3, 5, 8)


def load_results(results_path):
    with open(results_path) as f:
        return json.load(f)


def compute_generalization_gap(results, train_speeds=TRAIN_SPEEDS, test_speeds=TEST_SPEEDS):
    """Compute generalization gap: mean(seen reward) - mean(unseen reward).

    Lower gap = better generalization.
    """
    gaps = {}
    for model_name, model_data in results.items():
        rews = model_data["speed_rewards"]
        seen = [rews[str(s)]["mean"] for s in test_speeds if s in train_speeds]
        unseen = [rews[str(s)]["mean"] for s in test_speeds if s not in train_speeds]
        if seen and unseen:
            gaps[model_name] = np.mean(seen) - np.mean(unseen)
        else:
            gaps[model_name] = float("nan")
    return gaps


def compute_dt_slope(results, test_speeds=TEST_SPEEDS):
    """Compute Δτ slope: linear regression of Δτ on speed.

    Higher slope = better speed tracking.
    """
    slopes = {}
    for model_name, model_data in results.items():
        dts = model_data["speed_dts"]
        speeds = np.array([float(s) for s in test_speeds])
        means = np.array([dts[str(s)]["mean"] for s in test_speeds])
        # Linear regression
        slope = np.polyfit(speeds, means, 1)[0]
        slopes[model_name] = slope
    return slopes


def figure_main_result(results, save_path, title_suffix=""):
    """Figure 1: Main result - 3-panel (Δτ vs speed, reward vs speed, gen gap).

    This is THE key figure for the paper.
    """
    fig = plt.figure(figsize=(16, 4.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.7], wspace=0.35)

    ax_dt = fig.add_subplot(gs[0])
    ax_rew = fig.add_subplot(gs[1])
    ax_gap = fig.add_subplot(gs[2])

    speeds = np.array(TEST_SPEEDS)

    # Panel (a): Δτ vs Speed
    for model_key, style in MODEL_STYLES.items():
        if model_key not in results:
            continue
        dts = results[model_key]["speed_dts"]
        means = [dts[str(s)]["mean"] for s in TEST_SPEEDS]
        stds = [dts[str(s)].get("std", 0) for s in TEST_SPEEDS]
        ax_dt.errorbar(
            speeds, means, yerr=stds, label=style["label"],
            color=style["color"], marker=style["marker"],
            markersize=7, linewidth=2, capsize=3,
            linestyle=style["linestyle"],
        )

    ax_dt.axhline(y=1.0, color="black", linestyle=":", alpha=0.3, linewidth=0.8)
    ax_dt.axvspan(3.5, 8.5, alpha=0.06, color="red")
    ax_dt.set_xlabel("Environment Speed (action repeat)")
    ax_dt.set_ylabel(r"Internal Time $\Delta\tau$")
    ax_dt.set_title(r"(a) Learned $\Delta\tau$ vs Speed")
    ax_dt.set_xticks(TEST_SPEEDS)
    ax_dt.legend(loc="upper left", framealpha=0.9)

    # Add "unseen" annotation
    ymin, ymax = ax_dt.get_ylim()
    ax_dt.annotate("unseen", xy=(5, ymin), xytext=(6.5, ymin + 0.03 * (ymax - ymin)),
                    fontsize=8, color="red", alpha=0.7, ha="center")

    # Panel (b): Reward vs Speed
    plot_models_reward = ["baseline", "internal_time", "internal_time_discount", "skip_rnn"]
    for model_key in plot_models_reward:
        if model_key not in results:
            continue
        style = MODEL_STYLES[model_key]
        rews = results[model_key]["speed_rewards"]
        means = [rews[str(s)]["mean"] for s in TEST_SPEEDS]
        stds = [rews[str(s)].get("std", 0) for s in TEST_SPEEDS]
        ax_rew.errorbar(
            speeds, means, yerr=stds, label=style["label"],
            color=style["color"], marker=style["marker"],
            markersize=7, linewidth=2, capsize=3,
            linestyle=style["linestyle"],
        )

    ax_rew.axvspan(3.5, 8.5, alpha=0.06, color="red")
    ax_rew.set_xlabel("Environment Speed (action repeat)")
    ax_rew.set_ylabel("Episode Reward")
    ax_rew.set_title("(b) Reward vs Speed")
    ax_rew.set_xticks(TEST_SPEEDS)
    ax_rew.legend(loc="lower right", framealpha=0.9)

    # Panel (c): Generalization Gap bar chart
    gaps = compute_generalization_gap(results)
    # Sort by gap (ascending = better)
    sorted_models = sorted(gaps.keys(), key=lambda k: gaps[k])
    # Only plot models that have finite gap
    sorted_models = [m for m in sorted_models if not np.isnan(gaps[m])]

    bar_colors = [MODEL_STYLES[m]["color"] for m in sorted_models]
    bar_labels = [MODEL_STYLES[m]["label"].replace(r"$\gamma^{\Delta\tau}$", "disc")
                  for m in sorted_models]
    bar_values = [gaps[m] for m in sorted_models]

    bars = ax_gap.barh(range(len(sorted_models)), bar_values, color=bar_colors, alpha=0.8)
    ax_gap.set_yticks(range(len(sorted_models)))
    ax_gap.set_yticklabels(bar_labels, fontsize=9)
    ax_gap.set_xlabel("Generalization Gap\n(seen - unseen reward)")
    ax_gap.set_title("(c) Speed Generalization")
    ax_gap.axvline(x=0, color="black", linewidth=0.5)

    # Annotate bars with values
    for i, (val, bar) in enumerate(zip(bar_values, bars)):
        ax_gap.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=8)

    title = "Internal Time Adapts to Environment Speed"
    if title_suffix:
        title += f" ({title_suffix})"
    fig.suptitle(title, fontsize=14, y=1.02)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def figure_dt_tracking_detail(results, save_path):
    """Figure 2: Detailed Δτ tracking - just Internal Time variants vs speed.

    Clean figure focusing on the key mechanism: Δτ linearly tracks speed.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    speeds = np.array(TEST_SPEEDS, dtype=float)

    focus_models = ["internal_time", "internal_time_discount"]
    for model_key in focus_models:
        if model_key not in results:
            continue
        style = MODEL_STYLES[model_key]
        dts = results[model_key]["speed_dts"]
        means = np.array([dts[str(s)]["mean"] for s in TEST_SPEEDS])
        stds = np.array([dts[str(s)].get("std", 0) for s in TEST_SPEEDS])

        ax.errorbar(
            speeds, means, yerr=stds, label=style["label"],
            color=style["color"], marker=style["marker"],
            markersize=9, linewidth=2.5, capsize=4,
            linestyle=style["linestyle"],
        )

        # Add linear fit
        coeffs = np.polyfit(speeds, means, 1)
        fit_x = np.linspace(0.5, 9, 100)
        fit_y = np.polyval(coeffs, fit_x)
        ax.plot(fit_x, fit_y, color=style["color"], alpha=0.3, linewidth=1)
        ax.text(7.5, fit_y[-1] + 0.01, f"slope={coeffs[0]:.4f}",
                fontsize=8, color=style["color"], alpha=0.7)

    # Reference lines
    ax.axhline(y=1.0, color="black", linestyle=":", alpha=0.3, linewidth=0.8,
               label=r"$\Delta\tau = 1$ (standard)")

    # Mark train vs test regions
    for s in TEST_SPEEDS:
        if s not in TRAIN_SPEEDS:
            ax.axvline(x=s, color="red", linestyle="--", alpha=0.1)

    ax.fill_betweenx([ax.get_ylim()[0], 3], 3.5, 8.5, alpha=0.04, color="red")

    ax.set_xlabel("Environment Speed (action repeat)", fontsize=13)
    ax.set_ylabel(r"Learned Internal Time $\Delta\tau$", fontsize=13)
    ax.set_title(r"$\Delta\tau$ Linearly Tracks Environment Speed", fontsize=14)
    ax.set_xticks(TEST_SPEEDS)
    ax.legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def figure_generalization_gap_comparison(results_dict, save_path):
    """Figure 3: Compare generalization gaps across experiments.

    results_dict: {experiment_name: results_data}
    """
    fig, axes = plt.subplots(1, len(results_dict), figsize=(6 * len(results_dict), 5),
                              squeeze=False)

    for idx, (exp_name, results) in enumerate(results_dict.items()):
        ax = axes[0, idx]
        gaps = compute_generalization_gap(results)
        sorted_models = sorted(
            [m for m in gaps if not np.isnan(gaps[m])],
            key=lambda k: gaps[k]
        )

        bar_colors = [MODEL_STYLES[m]["color"] for m in sorted_models]
        bar_labels = [MODEL_STYLES[m]["label"].split("(")[0].strip() for m in sorted_models]
        bar_values = [gaps[m] for m in sorted_models]

        bars = ax.barh(range(len(sorted_models)), bar_values, color=bar_colors, alpha=0.85)
        ax.set_yticks(range(len(sorted_models)))
        ax.set_yticklabels(bar_labels, fontsize=9)
        ax.set_xlabel("Generalization Gap")
        ax.set_title(exp_name, fontsize=12)
        ax.axvline(x=0, color="black", linewidth=0.5)

        for i, val in enumerate(bar_values):
            ax.text(val + 0.003, i, f"{val:.3f}", va="center", fontsize=8)

    fig.suptitle("Generalization Gap: Seen Speed Reward - Unseen Speed Reward\n(lower = better)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def figure_episode_dynamics(agent, env_fn, speeds, device, save_path, agent_type="internal_time"):
    """Figure 4: Episode-level Δτ dynamics at different speeds.

    Shows step-by-step Δτ values during single episodes at various speeds.
    This visualizes HOW the agent adapts its internal clock.
    """
    import torch

    fig, axes = plt.subplots(len(speeds), 1, figsize=(10, 2.5 * len(speeds)),
                              sharex=False)
    if len(speeds) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(speeds)))

    agent.eval()
    for idx, speed in enumerate(speeds):
        ax = axes[idx]
        env = env_fn(speed)

        obs, info = env.reset()
        hidden = agent.get_initial_hidden(1, device)

        step_dts = []
        step_positions = []
        step_rewards = []
        done = False
        step_count = 0

        while not done and step_count < 150:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            if agent_type == "external_dt":
                ext_dt = torch.tensor([[speed / 3.0]], dtype=torch.float32, device=device)
                agent.set_external_dt(ext_dt)

            with torch.no_grad():
                action, _, _, _, hidden, dt = agent.get_action_and_value(obs_t, hidden)

            step_dts.append(dt.item())
            obs, reward, term, trunc, info = env.step(action.item())
            step_positions.append(info.get("position", 0))
            step_rewards.append(reward)
            done = term or trunc
            step_count += 1

        steps = np.arange(len(step_dts))

        # Plot Δτ
        ax.plot(steps, step_dts, color=colors[idx], linewidth=1.5, label=r"$\Delta\tau$")
        ax.axhline(y=1.0, color="black", linestyle=":", alpha=0.3, linewidth=0.8)

        # Shade regions where position changes rapidly (high dt makes sense)
        ax.fill_between(steps, 0, step_dts, alpha=0.15, color=colors[idx])

        # Add position as secondary axis
        ax2 = ax.twinx()
        pos_norm = np.array(step_positions) / max(max(step_positions), 1)
        ax2.plot(steps, pos_norm, color="gray", linewidth=0.8, alpha=0.5, linestyle="--")
        ax2.set_ylabel("Position (norm)", fontsize=8, color="gray", alpha=0.7)
        ax2.set_ylim(-0.1, 1.2)
        ax2.tick_params(axis="y", labelsize=8, colors="gray")

        marker = " *" if speed not in TRAIN_SPEEDS else ""
        ax.set_ylabel(r"$\Delta\tau$")
        ax.set_title(f"Speed = {speed}{marker}", fontsize=11,
                      color="red" if speed not in TRAIN_SPEEDS else "black")
        ax.set_ylim(0, max(max(step_dts) * 1.2, 1.5))

    axes[-1].set_xlabel("Agent Step")
    fig.suptitle(r"Episode-Level $\Delta\tau$ Dynamics at Different Speeds", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def figure_comparison_two_experiments(results_visible, results_hidden, save_path):
    """Figure comparing speed-visible vs speed-hidden results side by side.

    Shows that Internal Time tracks speed EVEN without speed in observation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    speeds = np.array(TEST_SPEEDS)

    titles = [
        "(a) Speed in Observation",
        "(b) Speed Hidden from Observation",
    ]
    results_list = [results_visible, results_hidden]

    for ax, results, title in zip(axes, results_list, titles):
        for model_key in ["baseline", "internal_time", "internal_time_discount", "skip_rnn"]:
            if model_key not in results:
                continue
            style = MODEL_STYLES[model_key]
            dts = results[model_key]["speed_dts"]
            means = [dts[str(s)]["mean"] for s in TEST_SPEEDS]
            stds = [dts[str(s)].get("std", 0) for s in TEST_SPEEDS]
            ax.errorbar(
                speeds, means, yerr=stds, label=style["label"],
                color=style["color"], marker=style["marker"],
                markersize=7, linewidth=2, capsize=3,
                linestyle=style["linestyle"],
            )

        ax.axhline(y=1.0, color="black", linestyle=":", alpha=0.3, linewidth=0.8)
        ax.axvspan(3.5, 8.5, alpha=0.06, color="red")
        ax.set_xlabel("Environment Speed")
        ax.set_ylabel(r"$\Delta\tau$")
        ax.set_title(title, fontsize=13)
        ax.set_xticks(TEST_SPEEDS)
        ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(r"$\Delta\tau$ Tracks Speed Even Without Explicit Speed Information",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_results_table(results, label=""):
    """Generate a formatted results table as text."""
    lines = []
    lines.append(f"{'='*80}")
    if label:
        lines.append(f"Results: {label}")
    lines.append(f"{'='*80}")

    # Header
    header = f"{'Model':30s}"
    for s in TEST_SPEEDS:
        marker = "" if s in TRAIN_SPEEDS else "*"
        header += f"  S={s}{marker:3s}"
    header += "  | Gen Gap | Δτ Slope"
    lines.append(header)
    lines.append("-" * 80)

    gaps = compute_generalization_gap(results)
    slopes = compute_dt_slope(results)

    # Reward rows
    lines.append("REWARD:")
    for model_name in MODEL_STYLES:
        if model_name not in results:
            continue
        rews = results[model_name]["speed_rewards"]
        row = f"  {MODEL_STYLES[model_name]['label']:28s}"
        for s in TEST_SPEEDS:
            r = rews[str(s)]
            row += f"  {r['mean']:+.3f}"
        gap = gaps.get(model_name, float("nan"))
        slope = slopes.get(model_name, float("nan"))
        row += f"  | {gap:+.3f}  | {slope:+.5f}"
        lines.append(row)

    # Δτ rows
    lines.append("\nΔτ:")
    for model_name in MODEL_STYLES:
        if model_name not in results:
            continue
        dts = results[model_name]["speed_dts"]
        row = f"  {MODEL_STYLES[model_name]['label']:28s}"
        for s in TEST_SPEEDS:
            d = dts[str(s)]
            row += f"  {d['mean']:.3f} "
        lines.append(row)

    lines.append(f"\n* = unseen during training")
    lines.append(f"Gen Gap = mean(seen reward) - mean(unseen reward), lower = better")
    lines.append(f"Δτ Slope = linear regression of Δτ on speed, higher = better tracking")

    return "\n".join(lines)


def generate_all_paper_figures(results_dir, output_dir=None):
    """Generate all figures from a single experiment results.json."""
    results_path = os.path.join(results_dir, "results.json")
    if not os.path.exists(results_path):
        print(f"No results.json found in {results_dir}")
        return

    results = load_results(results_path)

    if output_dir is None:
        output_dir = os.path.join(results_dir, "paper_figures")
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Main result
    figure_main_result(results, os.path.join(output_dir, "fig1_main_result.png"))

    # Figure 2: Δτ tracking detail
    figure_dt_tracking_detail(results, os.path.join(output_dir, "fig2_dt_tracking.png"))

    # Results table
    table = generate_results_table(results, label=os.path.basename(results_dir))
    print(table)
    with open(os.path.join(output_dir, "results_table.txt"), "w") as f:
        f.write(table)
    print(f"Saved: {os.path.join(output_dir, 'results_table.txt')}")


def generate_cross_experiment_figures(results_dirs, output_dir):
    """Generate comparison figures across multiple experiments."""
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    for rdir in results_dirs:
        results_path = os.path.join(rdir, "results.json")
        if os.path.exists(results_path):
            name = os.path.basename(rdir)
            all_results[name] = load_results(results_path)

    if len(all_results) < 2:
        print("Need at least 2 experiments for cross-comparison")
        return

    # Generalization gap comparison
    figure_generalization_gap_comparison(
        all_results,
        os.path.join(output_dir, "gen_gap_comparison.png")
    )

    # If we have visible + hidden, make the comparison figure
    visible_key = None
    hidden_key = None
    for k in all_results:
        if "hidden" in k:
            hidden_key = k
        elif "chain" in k:
            visible_key = k

    if visible_key and hidden_key:
        figure_comparison_two_experiments(
            all_results[visible_key],
            all_results[hidden_key],
            os.path.join(output_dir, "fig_visible_vs_hidden.png"),
        )

    # Print all tables
    for name, results in all_results.items():
        print()
        print(generate_results_table(results, label=name))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", type=str, help="Single results directory")
    parser.add_argument("--all", action="store_true", help="Process all experiments in runs/")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.all:
        runs_dir = "runs"
        result_dirs = []
        for d in sorted(os.listdir(runs_dir)):
            full = os.path.join(runs_dir, d)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, "results.json")):
                result_dirs.append(full)
                generate_all_paper_figures(full)

        if len(result_dirs) >= 2:
            out = args.output_dir or os.path.join(runs_dir, "paper_figures")
            generate_cross_experiment_figures(result_dirs, out)

    elif args.results_dir:
        generate_all_paper_figures(args.results_dir, args.output_dir)
    else:
        parser.print_help()
