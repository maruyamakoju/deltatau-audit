"""Generate all paper materials from completed experiments.

Collects all results, generates figures, and creates summary statistics.
Includes statistical metrics for reviewer defense:
- Spearman ρ (Δτ-speed rank correlation)
- Monotonicity rate (% of speed pairs where Δτ is correctly ordered)
- Bootstrap CI on Δτ slope
- Return variance across seeds
- Lag metric from switching experiments

Run this after all experiments are complete.

Usage:
    python generate_paper.py
"""

import json
import os
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from internal_time_rl.analysis.paper_figures import (
    load_results,
    compute_generalization_gap,
    compute_dt_slope,
    MODEL_STYLES,
    TRAIN_SPEEDS,
    TEST_SPEEDS,
    figure_main_result,
    figure_dt_tracking_detail,
    figure_comparison_two_experiments,
    generate_results_table,
)
from internal_time_rl.analysis.training_curves import plot_training_curves

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def find_experiments(runs_dir="runs"):
    """Find all completed experiments."""
    experiments = {}
    for d in sorted(os.listdir(runs_dir)):
        results_path = os.path.join(runs_dir, d, "results.json")
        if os.path.exists(results_path):
            experiments[d] = load_results(results_path)
    return experiments


###############################################################################
# Statistical metrics for reviewer defense
###############################################################################


def compute_spearman_rho(results, test_speeds=TEST_SPEEDS):
    """Compute Spearman rank correlation between Δτ and speed for each model.

    ρ = 1.0 means perfect monotonic positive relationship.
    Baseline (Δτ=1 always) gets ρ = NaN.
    """
    rho_dict = {}
    for model_name, model_data in results.items():
        dts = model_data["speed_dts"]
        speeds = np.array([float(s) for s in test_speeds])
        means = np.array([dts[str(s)]["mean"] for s in test_speeds])

        # Check if all values are identical (baseline case)
        if np.std(means) < 1e-8:
            rho_dict[model_name] = {"rho": float("nan"), "p": float("nan")}
        else:
            rho, p = scipy_stats.spearmanr(speeds, means)
            rho_dict[model_name] = {"rho": rho, "p": p}
    return rho_dict


def compute_monotonicity_rate(results, test_speeds=TEST_SPEEDS):
    """Compute fraction of speed pairs where Δτ is correctly ordered.

    For speeds s_i < s_j, we check if Δτ(s_i) < Δτ(s_j).
    Rate of 1.0 means perfect monotonic increase.
    """
    mono_dict = {}
    for model_name, model_data in results.items():
        dts = model_data["speed_dts"]
        means = np.array([dts[str(s)]["mean"] for s in test_speeds])

        correct = 0
        total = 0
        for i in range(len(test_speeds)):
            for j in range(i + 1, len(test_speeds)):
                total += 1
                if means[j] > means[i]:
                    correct += 1
        mono_dict[model_name] = correct / total if total > 0 else 0
    return mono_dict


def bootstrap_dt_slope(results, test_speeds=TEST_SPEEDS, n_bootstrap=10000,
                       ci=0.95):
    """Bootstrap confidence interval for Δτ slope.

    Uses the per-speed mean±std to generate bootstrap samples, then fits
    a linear regression to each. Returns the CI for the slope.
    """
    slope_ci = {}
    for model_name, model_data in results.items():
        dts = model_data["speed_dts"]
        speeds = np.array([float(s) for s in test_speeds])
        means = np.array([dts[str(s)]["mean"] for s in test_speeds])
        stds = np.array([dts[str(s)].get("std", 0) for s in test_speeds])

        # Bootstrap: resample Δτ from N(mean, std) at each speed
        bootstrap_slopes = []
        for _ in range(n_bootstrap):
            sampled = means + np.random.randn(len(means)) * stds
            slope = np.polyfit(speeds, sampled, 1)[0]
            bootstrap_slopes.append(slope)

        bootstrap_slopes = np.array(bootstrap_slopes)
        alpha = (1 - ci) / 2
        lo = np.percentile(bootstrap_slopes, alpha * 100)
        hi = np.percentile(bootstrap_slopes, (1 - alpha) * 100)
        slope_ci[model_name] = {
            "mean": float(np.mean(bootstrap_slopes)),
            "lo": float(lo),
            "hi": float(hi),
            "ci": ci,
        }
    return slope_ci


def compute_return_variance(results, test_speeds=TEST_SPEEDS):
    """Compute return variance across seeds for each model and speed."""
    var_dict = {}
    for model_name, model_data in results.items():
        rews = model_data["speed_rewards"]
        total_var = 0
        count = 0
        per_speed = {}
        for s in test_speeds:
            std = rews[str(s)].get("std", 0)
            per_speed[str(s)] = std ** 2
            total_var += std ** 2
            count += 1
        var_dict[model_name] = {
            "mean_variance": total_var / count if count > 0 else 0,
            "per_speed": per_speed,
        }
    return var_dict


def generate_statistical_summary(results, label=""):
    """Generate a comprehensive statistical summary table."""
    lines = []
    lines.append("=" * 90)
    lines.append(f"STATISTICAL METRICS{f' — {label}' if label else ''}")
    lines.append("=" * 90)

    # Spearman ρ
    rho_dict = compute_spearman_rho(results)
    lines.append("\n1. Spearman ρ (Δτ vs Speed rank correlation):")
    lines.append(f"   {'Model':35s} {'ρ':>8s} {'p-value':>10s}")
    lines.append("   " + "-" * 55)
    for m in MODEL_STYLES:
        if m not in rho_dict:
            continue
        r = rho_dict[m]
        rho_str = f"{r['rho']:.4f}" if not np.isnan(r["rho"]) else "  N/A"
        p_str = f"{r['p']:.2e}" if not np.isnan(r["p"]) else "  N/A"
        lines.append(f"   {MODEL_STYLES[m]['label']:35s} {rho_str:>8s} {p_str:>10s}")

    # Monotonicity rate
    mono_dict = compute_monotonicity_rate(results)
    lines.append("\n2. Monotonicity Rate (fraction of correct Δτ orderings):")
    lines.append(f"   {'Model':35s} {'Rate':>8s}")
    lines.append("   " + "-" * 45)
    for m in MODEL_STYLES:
        if m not in mono_dict:
            continue
        lines.append(f"   {MODEL_STYLES[m]['label']:35s} {mono_dict[m]:>8.1%}")

    # Bootstrap CI
    slope_ci = bootstrap_dt_slope(results)
    lines.append(f"\n3. Δτ Slope Bootstrap 95% CI:")
    lines.append(f"   {'Model':35s} {'Slope':>8s} {'95% CI':>20s}")
    lines.append("   " + "-" * 65)
    for m in MODEL_STYLES:
        if m not in slope_ci:
            continue
        ci = slope_ci[m]
        lines.append(f"   {MODEL_STYLES[m]['label']:35s} {ci['mean']:>+8.5f} "
                      f"[{ci['lo']:+.5f}, {ci['hi']:+.5f}]")

    # Return variance
    var_dict = compute_return_variance(results)
    lines.append(f"\n4. Return Variance Across Seeds (mean across speeds):")
    lines.append(f"   {'Model':35s} {'Mean Var':>10s}")
    lines.append("   " + "-" * 47)
    for m in MODEL_STYLES:
        if m not in var_dict:
            continue
        v = var_dict[m]
        lines.append(f"   {MODEL_STYLES[m]['label']:35s} {v['mean_variance']:>10.2e}")

    lines.append("\n" + "=" * 90)
    return "\n".join(lines)


def generate_statistical_json(results, label=""):
    """Generate structured JSON with all statistical metrics."""
    return {
        "label": label,
        "spearman_rho": compute_spearman_rho(results),
        "monotonicity_rate": compute_monotonicity_rate(results),
        "dt_slope_bootstrap_ci": bootstrap_dt_slope(results),
        "return_variance": compute_return_variance(results),
        "generalization_gap": compute_generalization_gap(results),
        "dt_slope": compute_dt_slope(results),
    }


def figure_killer(hidden_results, switching_data, save_path):
    """THE killer figure: 3-panel combining strongest evidence.

    (a) Δτ vs Speed with statistical annotations (ρ, CI)
    (b) Mid-episode switching response (honest)
    (c) Statistical metrics comparison (slope bar + monotonicity)
    """
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.0, 0.8], wspace=0.35)

    speeds = np.array(TEST_SPEEDS)

    # ── Panel (a): Δτ vs Speed with annotations ──
    ax_a = fig.add_subplot(gs[0])
    for model_key in ["baseline", "internal_time", "internal_time_discount", "skip_rnn"]:
        if model_key not in hidden_results:
            continue
        s = MODEL_STYLES[model_key]
        dts = hidden_results[model_key]["speed_dts"]
        means = [dts[str(sp)]["mean"] for sp in TEST_SPEEDS]
        stds = [dts[str(sp)].get("std", 0) for sp in TEST_SPEEDS]
        ax_a.errorbar(speeds, means, yerr=stds, label=s["label"],
                      color=s["color"], marker=s["marker"], markersize=7,
                      linewidth=2, capsize=3, linestyle=s["linestyle"])

    # Linear fit for internal_time
    if "internal_time" in hidden_results:
        dts = hidden_results["internal_time"]["speed_dts"]
        means_arr = np.array([dts[str(sp)]["mean"] for sp in TEST_SPEEDS])
        coeffs = np.polyfit(speeds, means_arr, 1)
        fit_x = np.linspace(0.5, 9, 50)
        ax_a.plot(fit_x, np.polyval(coeffs, fit_x), color="#d62728", alpha=0.2, linewidth=1)

    ax_a.axhline(y=1.0, color="black", linestyle=":", alpha=0.2)
    ax_a.axvspan(3.5, 8.5, alpha=0.05, color="red")
    ax_a.set_xlabel("Environment Speed")
    ax_a.set_ylabel(r"$\Delta\tau$")
    ax_a.set_title(r"(a) Learned $\Delta\tau$ vs Speed (speed hidden)", fontsize=12)
    ax_a.set_xticks(TEST_SPEEDS)
    ax_a.legend(fontsize=8, loc="upper left")

    # Statistical annotations
    rho_dict = compute_spearman_rho(hidden_results)
    slope_ci = bootstrap_dt_slope(hidden_results)
    if "internal_time" in rho_dict:
        rho = rho_dict["internal_time"]["rho"]
        ci = slope_ci["internal_time"]
        ax_a.text(0.97, 0.15, f"Internal Time:\n"
                  f"Spearman ρ = {rho:.2f}\n"
                  f"slope = {ci['mean']:.4f}\n"
                  f"95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}]",
                  transform=ax_a.transAxes, fontsize=8, va="bottom", ha="right",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffe0e0",
                            edgecolor="#d62728", alpha=0.9))

    # ── Panel (b): Switching response ──
    ax_b = fig.add_subplot(gs[1])
    if switching_data:
        agent_colors = {
            "internal_time": "#d62728",
            "internal_time_discount": "#8c1515",
            "skip_rnn": "#1f77b4",
            "baseline": "#7f7f7f",
        }
        agent_labels = {
            "internal_time": "Internal Time",
            "internal_time_discount": r"IT + $\gamma^{\Delta\tau}$",
            "skip_rnn": "Skip-RNN",
            "baseline": "Baseline GRU",
        }

        any_data = next(iter(switching_data.values()))
        switch_step = any_data["switch_step"]

        # Speed background
        ax_b2 = ax_b.twinx()
        steps_bg = np.array(any_data["steps"])
        speeds_bg = np.array(any_data["speeds"])
        ax_b2.fill_between(steps_bg, 0, speeds_bg, alpha=0.06, color="orange", step="post")
        ax_b2.step(steps_bg, speeds_bg, color="orange", linewidth=1.2, alpha=0.4,
                   where="post")
        ax_b2.set_ylabel("Speed", color="orange", fontsize=10)
        ax_b2.set_ylim(0, 12)
        ax_b2.tick_params(axis="y", labelcolor="orange")

        for agent_name in ["baseline", "skip_rnn", "internal_time_discount", "internal_time"]:
            if agent_name not in switching_data:
                continue
            data = switching_data[agent_name]
            steps = np.array(data["steps"])
            dts = np.array(data["delta_taus"])
            color = agent_colors.get(agent_name, "black")
            label = agent_labels.get(agent_name, agent_name)
            lw = 2.5 if agent_name == "internal_time" else 1.5
            alpha = 1.0 if "internal_time" in agent_name else 0.6
            ax_b.plot(steps, dts, color=color, linewidth=lw, alpha=alpha, label=label)

        ax_b.axvline(x=switch_step, color="black", linestyle="--", alpha=0.5, linewidth=1.5)
        ax_b.axhline(y=1.0, color="black", linestyle=":", alpha=0.2)
        ax_b.set_xlabel("Agent Step")
        ax_b.set_ylabel(r"$\Delta\tau$")
        ax_b.set_title(r"(b) Response to Speed 1$\rightarrow$8 Switch", fontsize=12)
        ax_b.legend(fontsize=7, loc="upper right")
        ax_b.set_ylim(0.5, 1.5)

    # ── Panel (c): Statistical metrics bar chart ──
    ax_c = fig.add_subplot(gs[2])
    mono_dict = compute_monotonicity_rate(hidden_results)
    slope_dict = compute_dt_slope(hidden_results)

    models = ["baseline", "skip_rnn", "internal_time", "internal_time_discount"]
    models = [m for m in models if m in hidden_results]
    x = np.arange(len(models))

    # Monotonicity rate as bars
    mono_vals = [mono_dict.get(m, 0) * 100 for m in models]
    colors = [MODEL_STYLES[m]["color"] for m in models]
    labels = [MODEL_STYLES[m]["label"].split("(")[0].strip()[:12] for m in models]

    bars = ax_c.bar(x, mono_vals, color=colors, alpha=0.85)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, fontsize=8, rotation=15)
    ax_c.set_ylabel("Monotonicity Rate (%)", fontsize=10)
    ax_c.set_title("(c) Δτ Monotonicity", fontsize=12)
    ax_c.set_ylim(0, 115)

    for i, val in enumerate(mono_vals):
        ax_c.text(i, val + 2, f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(r"Internal Time $\Delta\tau$ Tracks Latent Environment Speed",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def figure_ablation_combined(hidden_results, value_ablation, early_inference,
                              save_path, value_multiseed=None):
    """Combined ablation figure: Δτ tracking + value function causal test.

    3-panel figure:
    (a) Δτ vs Speed — the core finding (monotonic tracking)
    (b) Value RMSE under intervention — causal evidence (multi-seed if available)
    (c) Early inference — amortized speed estimation

    If value_multiseed is provided, panel (b) uses multi-seed aggregate
    with SE error bars instead of single-seed data.
    """
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.35)

    speeds = np.array(TEST_SPEEDS)

    # ── Panel (a): Δτ vs Speed with statistical annotations ──
    ax_a = fig.add_subplot(gs[0])
    for model_key in ["baseline", "internal_time", "internal_time_discount", "skip_rnn"]:
        if model_key not in hidden_results:
            continue
        s = MODEL_STYLES[model_key]
        dts = hidden_results[model_key]["speed_dts"]
        means = [dts[str(sp)]["mean"] for sp in TEST_SPEEDS]
        stds = [dts[str(sp)].get("std", 0) for sp in TEST_SPEEDS]
        ax_a.errorbar(speeds, means, yerr=stds, label=s["label"],
                      color=s["color"], marker=s["marker"], markersize=7,
                      linewidth=2, capsize=3, linestyle=s["linestyle"])

    # Linear fit
    if "internal_time" in hidden_results:
        dts = hidden_results["internal_time"]["speed_dts"]
        means_arr = np.array([dts[str(sp)]["mean"] for sp in TEST_SPEEDS])
        coeffs = np.polyfit(speeds, means_arr, 1)
        fit_x = np.linspace(0.5, 9, 50)
        ax_a.plot(fit_x, np.polyval(coeffs, fit_x), color="#d62728",
                  alpha=0.2, linewidth=1)

    ax_a.axhline(y=1.0, color="black", linestyle=":", alpha=0.2)
    ax_a.axvspan(3.5, 8.5, alpha=0.05, color="red")
    ax_a.set_xlabel("Environment Speed")
    ax_a.set_ylabel(r"$\Delta\tau$")
    ax_a.set_title(r"(a) Learned $\Delta\tau$ vs Speed", fontsize=12)
    ax_a.set_xticks(TEST_SPEEDS)
    ax_a.legend(fontsize=7, loc="upper left")

    # Stats annotation
    rho_dict = compute_spearman_rho(hidden_results)
    slope_ci = bootstrap_dt_slope(hidden_results)
    if "internal_time" in rho_dict:
        rho = rho_dict["internal_time"]["rho"]
        ci = slope_ci["internal_time"]
        ax_a.text(0.97, 0.05, f"Spearman ρ = {rho:.2f}\n"
                  f"slope CI [{ci['lo']:.4f}, {ci['hi']:.4f}]",
                  transform=ax_a.transAxes, fontsize=8, va="bottom", ha="right",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffe0e0",
                            edgecolor="#d62728", alpha=0.9))

    # ── Panel (b): Value function accuracy under intervention ──
    ax_b = fig.add_subplot(gs[1])
    intervention_styles = {
        "none": {"label": r"Normal $\Delta\tau$", "color": "#d62728",
                 "marker": "o", "ls": "-"},
        "clamp_1": {"label": r"$\Delta\tau = 1.0$", "color": "#7f7f7f",
                    "marker": "s", "ls": "--"},
        "reverse": {"label": r"$\Delta\tau$ reversed", "color": "#2ca02c",
                    "marker": "^", "ls": "-."},
        "random": {"label": r"$\Delta\tau$ random", "color": "#ff7f0e",
                   "marker": "D", "ls": ":"},
    }

    # Use multi-seed if available, otherwise fall back to single-seed
    val_data = None
    n_seeds_label = ""
    if value_multiseed and "internal_time" in value_multiseed:
        val_data = value_multiseed["internal_time"]
        n_seeds = val_data["none"][str(TEST_SPEEDS[0])].get("n_seeds", "?")
        n_seeds_label = f" (n={n_seeds} seeds)"
        use_se = True
    elif value_ablation:
        val_data = value_ablation
        use_se = False

    if val_data:
        for interv, style in intervention_styles.items():
            if interv not in val_data:
                continue
            rmse = [val_data[interv][str(s)]["rmse_mean"] for s in TEST_SPEEDS]
            if use_se:
                yerr = [val_data[interv][str(s)].get("rmse_se", 0) for s in TEST_SPEEDS]
            else:
                yerr = None
            ax_b.errorbar(speeds, rmse, yerr=yerr, label=style["label"],
                          color=style["color"], marker=style["marker"],
                          markersize=7, linewidth=2, capsize=4,
                          linestyle=style["ls"])

        ax_b.axvspan(3.5, 8.5, alpha=0.05, color="red")

        # Annotate the degradation at S=8
        none_8 = val_data["none"]["8"]["rmse_mean"]
        rev_8 = val_data["reverse"]["8"]["rmse_mean"]
        pct = (rev_8 / none_8 - 1) * 100
        ax_b.annotate(f"+{pct:.0f}%{n_seeds_label}",
                      xy=(8, rev_8), xytext=(6.5, rev_8 + 0.02),
                      fontsize=9, fontweight="bold", color="#2ca02c",
                      arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5))

    ax_b.set_xlabel("Environment Speed")
    ax_b.set_ylabel("Value RMSE")
    ax_b.set_title(r"(b) Value Accuracy Under $\Delta\tau$ Intervention", fontsize=12)
    ax_b.set_xticks(TEST_SPEEDS)
    ax_b.legend(fontsize=7, loc="upper left")

    # ── Panel (c): Early inference / amortized speed estimation ──
    ax_c = fig.add_subplot(gs[2])
    if early_inference and "internal_time" in early_inference:
        inf = early_inference["internal_time"]
        means = inf["per_step_means"]
        min_len = inf["min_len"]
        steps = np.arange(min(min_len, 15))  # Show first 15 steps

        speed_colors = {
            "1": "#2166ac", "2": "#67a9cf", "3": "#d1e5f0",
            "5": "#ef8a62", "8": "#b2182b",
        }
        for s_str in ["1", "3", "8"]:  # Show 3 speeds for clarity
            m = np.array(means[s_str])[:len(steps)]
            unseen = " *" if int(s_str) not in TRAIN_SPEEDS else ""
            ax_c.plot(steps, m, color=speed_colors[s_str], linewidth=2,
                      label=f"Speed {s_str}{unseen}", marker="o", markersize=3)

        ax_c.axhline(y=1.0, color="black", linestyle=":", alpha=0.2)
        sep = inf["separation_step"]
        if sep is not None and sep < len(steps):
            ax_c.axvline(x=sep, color="green", linestyle="--", alpha=0.5)
            ax_c.text(sep + 0.3, ax_c.get_ylim()[1] * 0.95,
                      f"p<0.05\nstep {sep}", fontsize=8, color="green", va="top")

    ax_c.set_xlabel("Agent Step")
    ax_c.set_ylabel(r"$\Delta\tau$")
    ax_c.set_title(r"(c) Amortized Speed Inference", fontsize=12)
    ax_c.legend(fontsize=8, loc="best")

    fig.suptitle(r"$\Delta\tau$ as Learned Internal Clock: Tracking, Causality, and Inference",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def figure_hero(visible_results, hidden_results, save_path):
    """THE hero figure: 4-panel showing everything.

    (a) Δτ vs Speed (hidden)  - Key mechanism result
    (b) Visible vs Hidden     - Robustness
    (c) Reward vs Speed       - Performance
    (d) Gen gap bars          - Summary metric
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    speeds = np.array(TEST_SPEEDS)

    # (a) Δτ vs Speed - Hidden speed (KEY figure)
    ax_a = fig.add_subplot(gs[0, 0])
    for model_key in ["baseline", "internal_time", "internal_time_discount", "skip_rnn"]:
        if model_key not in hidden_results:
            continue
        s = MODEL_STYLES[model_key]
        dts = hidden_results[model_key]["speed_dts"]
        means = [dts[str(sp)]["mean"] for sp in TEST_SPEEDS]
        stds = [dts[str(sp)].get("std", 0) for sp in TEST_SPEEDS]
        ax_a.errorbar(speeds, means, yerr=stds, label=s["label"],
                      color=s["color"], marker=s["marker"], markersize=7,
                      linewidth=2, capsize=3, linestyle=s["linestyle"])

    ax_a.axhline(y=1.0, color="black", linestyle=":", alpha=0.3)
    ax_a.axvspan(3.5, 8.5, alpha=0.06, color="red")
    ax_a.set_xlabel("Environment Speed")
    ax_a.set_ylabel(r"$\Delta\tau$")
    ax_a.set_title(r"(a) Learned $\Delta\tau$ vs Speed (speed hidden)", fontsize=12)
    ax_a.set_xticks(TEST_SPEEDS)
    ax_a.legend(fontsize=8, loc="upper left")

    # Add linear fit for internal_time
    if "internal_time" in hidden_results:
        dts = hidden_results["internal_time"]["speed_dts"]
        means = [dts[str(sp)]["mean"] for sp in TEST_SPEEDS]
        coeffs = np.polyfit(speeds, means, 1)
        fit_x = np.linspace(0.5, 9, 50)
        ax_a.plot(fit_x, np.polyval(coeffs, fit_x), color="#d62728", alpha=0.2, linewidth=1)

    # (b) Visible vs Hidden comparison
    ax_b = fig.add_subplot(gs[0, 1])
    if visible_results and "internal_time" in visible_results and "internal_time" in hidden_results:
        vis_dts = visible_results["internal_time"]["speed_dts"]
        hid_dts = hidden_results["internal_time"]["speed_dts"]
        vis_means = [vis_dts[str(sp)]["mean"] for sp in TEST_SPEEDS]
        hid_means = [hid_dts[str(sp)]["mean"] for sp in TEST_SPEEDS]
        vis_stds = [vis_dts[str(sp)].get("std", 0) for sp in TEST_SPEEDS]
        hid_stds = [hid_dts[str(sp)].get("std", 0) for sp in TEST_SPEEDS]

        ax_b.errorbar(speeds, vis_means, yerr=vis_stds,
                      label="Speed visible", color="#d62728",
                      marker="o", markersize=7, linewidth=2, capsize=3)
        ax_b.errorbar(speeds, hid_means, yerr=hid_stds,
                      label="Speed hidden", color="#d62728",
                      marker="s", markersize=7, linewidth=2, capsize=3,
                      linestyle="--")

    ax_b.axhline(y=1.0, color="black", linestyle=":", alpha=0.3)
    ax_b.axvspan(3.5, 8.5, alpha=0.06, color="red")
    ax_b.set_xlabel("Environment Speed")
    ax_b.set_ylabel(r"$\Delta\tau$")
    ax_b.set_title(r"(b) $\Delta\tau$ Tracks Speed Even When Hidden", fontsize=12)
    ax_b.set_xticks(TEST_SPEEDS)
    ax_b.legend(fontsize=9)

    # (c) Reward vs speed for hidden
    ax_c = fig.add_subplot(gs[1, 0])
    for model_key in ["baseline", "internal_time", "internal_time_discount", "skip_rnn"]:
        if model_key not in hidden_results:
            continue
        s = MODEL_STYLES[model_key]
        rews = hidden_results[model_key]["speed_rewards"]
        means = [rews[str(sp)]["mean"] for sp in TEST_SPEEDS]
        stds = [rews[str(sp)].get("std", 0) for sp in TEST_SPEEDS]
        ax_c.errorbar(speeds, means, yerr=stds, label=s["label"],
                      color=s["color"], marker=s["marker"], markersize=7,
                      linewidth=2, capsize=3, linestyle=s["linestyle"])

    ax_c.axvspan(3.5, 8.5, alpha=0.06, color="red")
    ax_c.set_xlabel("Environment Speed")
    ax_c.set_ylabel("Episode Reward")
    ax_c.set_title("(c) Reward vs Speed (speed hidden)", fontsize=12)
    ax_c.set_xticks(TEST_SPEEDS)
    ax_c.legend(fontsize=8, loc="lower right")

    # (d) Δτ slope comparison across models
    ax_d = fig.add_subplot(gs[1, 1])
    slopes = compute_dt_slope(hidden_results)
    models = ["baseline", "skip_rnn", "internal_time", "internal_time_discount"]
    models = [m for m in models if m in slopes]
    bar_colors = [MODEL_STYLES[m]["color"] for m in models]
    bar_labels = [MODEL_STYLES[m]["label"].split("(")[0].strip() for m in models]
    bar_values = [slopes[m] for m in models]

    bars = ax_d.bar(range(len(models)), bar_values, color=bar_colors, alpha=0.85)
    ax_d.set_xticks(range(len(models)))
    ax_d.set_xticklabels(bar_labels, fontsize=9, rotation=15)
    ax_d.set_ylabel(r"$\Delta\tau$ Slope (linear fit)")
    ax_d.set_title(r"(d) Speed Tracking: $d(\Delta\tau)/d(\text{speed})$", fontsize=12)
    ax_d.axhline(y=0, color="black", linewidth=0.5)

    for i, val in enumerate(bar_values):
        ax_d.text(i, val + 0.001, f"{val:.4f}", ha="center", fontsize=8)

    fig.suptitle("Learning Internal Time: Adaptive Temporal Reparameterization",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_latex_table(experiments):
    """Generate LaTeX-formatted results table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Speed Generalization Results}")
    lines.append(r"\begin{tabular}{l" + "c" * len(TEST_SPEEDS) + r"cc}")
    lines.append(r"\toprule")

    header = r"Model"
    for s in TEST_SPEEDS:
        marker = "" if s in TRAIN_SPEEDS else r"$^*$"
        header += f" & S={s}{marker}"
    header += r" & Gen Gap & $\Delta\tau$ Slope \\"
    lines.append(header)
    lines.append(r"\midrule")

    for exp_name, results in experiments.items():
        lines.append(f"\\multicolumn{{{len(TEST_SPEEDS)+3}}}{{l}}"
                     f"{{\\textit{{{exp_name}}}}} \\\\")

        gaps = compute_generalization_gap(results)
        slopes = compute_dt_slope(results)

        for model_key in ["baseline", "internal_time", "internal_time_discount", "skip_rnn"]:
            if model_key not in results:
                continue
            label = MODEL_STYLES[model_key]["label"].replace(r"$\gamma^{\Delta\tau}$",
                                                              r"$\gamma^{\Delta\tau}$")
            rews = results[model_key]["speed_rewards"]
            row = f"  {label}"
            for s in TEST_SPEEDS:
                r = rews[str(s)]
                row += f" & {r['mean']:.3f}"
            gap = gaps.get(model_key, float("nan"))
            slope = slopes.get(model_key, float("nan"))
            row += f" & {gap:+.3f} & {slope:+.4f} \\\\"
            lines.append(row)

        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    output_dir = "runs/paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    experiments = find_experiments()
    print(f"Found experiments: {list(experiments.keys())}")

    # Get key experiments
    chain_visible = experiments.get("speed_gen_chain")
    chain_hidden = experiments.get("speed_gen_hidden")
    chain_hidden_5seed = experiments.get("speed_gen_hidden_5seed")
    hard_chain = experiments.get("hard_chain")

    # Use best available hidden data (5-seed = main, single seed = supplementary)
    best_hidden = chain_hidden_5seed or chain_hidden

    if best_hidden:
        # ── FIGURE 1: Hero (main result overview) ──
        figure_hero(chain_visible, best_hidden,
                    os.path.join(output_dir, "fig_hero.png"))

        # ── FIGURE 2: Δτ tracking detail ──
        figure_main_result(best_hidden,
                          os.path.join(output_dir, "fig_main_result.png"),
                          title_suffix="Speed Hidden")
        figure_dt_tracking_detail(best_hidden,
                                 os.path.join(output_dir, "fig_dt_tracking_detail.png"))

        # ── FIGURE 3: Ablation combined (THE key figure for reviewers) ──
        # Load value ablation data (prefer multi-seed)
        value_ablation = None
        value_multiseed = None
        early_inference = None
        for exp_name in ["speed_gen_hidden_5seed", "speed_gen_hidden"]:
            ms_path = os.path.join("runs", exp_name, "ablation",
                                   "value_ablation_multiseed.json")
            val_path = os.path.join("runs", exp_name, "ablation", "value_ablation.json")
            ei_path = os.path.join("runs", exp_name, "ablation", "early_inference.json")
            if os.path.exists(ms_path) and value_multiseed is None:
                with open(ms_path) as f:
                    value_multiseed = json.load(f)
                print(f"Loaded multi-seed value ablation from {ms_path}")
            if os.path.exists(val_path) and value_ablation is None:
                with open(val_path) as f:
                    value_ablation = json.load(f)
                print(f"Loaded value ablation from {val_path}")
            if os.path.exists(ei_path) and early_inference is None:
                with open(ei_path) as f:
                    early_inference = json.load(f)
                print(f"Loaded early inference from {ei_path}")

        if value_ablation or value_multiseed or early_inference:
            figure_ablation_combined(
                best_hidden, value_ablation, early_inference,
                os.path.join(output_dir, "fig_ablation.png"),
                value_multiseed=value_multiseed,
            )

        # ── Killer figure (legacy, with switching) ──
        switching_data = None
        for exp_name in ["speed_gen_hidden_5seed", "speed_gen_hidden"]:
            switching_path = os.path.join("runs", exp_name,
                                         "switching_dynamics", "switching_data.json")
            if os.path.exists(switching_path):
                with open(switching_path) as f:
                    switching_data = json.load(f)
                print(f"Loaded switching data from {switching_path}")
                break

        figure_killer(best_hidden, switching_data,
                      os.path.join(output_dir, "fig_killer.png"))

    if chain_visible and best_hidden:
        figure_comparison_two_experiments(
            chain_visible, best_hidden,
            os.path.join(output_dir, "fig_visible_vs_hidden.png")
        )

    # Generate all individual experiment figures
    for exp_name, results in experiments.items():
        exp_dir = os.path.join("runs", exp_name)
        fig_dir = os.path.join(exp_dir, "paper_figures")
        os.makedirs(fig_dir, exist_ok=True)
        figure_main_result(results,
                          os.path.join(fig_dir, "fig1_main_result.png"),
                          title_suffix=exp_name)

    # Training curves for key experiments
    for exp_name in ["speed_gen_hidden", "speed_gen_chain", "speed_gen_hidden_5seed", "hard_chain"]:
        exp_dir = os.path.join("runs", exp_name)
        if os.path.isdir(exp_dir):
            plot_training_curves(exp_dir)

    # ── RESULTS TABLES ──
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)

    for exp_name, results in experiments.items():
        table = generate_results_table(results, label=exp_name)
        print(f"\n{table}\n")

    with open(os.path.join(output_dir, "all_results.txt"), "w") as f:
        for exp_name, results in experiments.items():
            f.write(generate_results_table(results, label=exp_name))
            f.write("\n\n")
    print(f"All results saved to {output_dir}/all_results.txt")

    # LaTeX table
    key_experiments = {}
    if chain_hidden:
        key_experiments["Chain (hidden speed)"] = chain_hidden
    if chain_hidden_5seed:
        key_experiments["Chain (hidden, 5 seeds)"] = chain_hidden_5seed
    if hard_chain:
        key_experiments["Hard Chain (flickering)"] = hard_chain

    if key_experiments:
        latex = generate_latex_table(key_experiments)
        with open(os.path.join(output_dir, "results_table.tex"), "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to {output_dir}/results_table.tex")

    # ── STATISTICAL METRICS ──
    print("\n" + "=" * 90)
    print("STATISTICAL METRICS FOR REVIEWER DEFENSE")
    print("=" * 90)

    all_stats = {}
    for exp_name, results in experiments.items():
        summary = generate_statistical_summary(results, label=exp_name)
        print(f"\n{summary}")
        all_stats[exp_name] = generate_statistical_json(results, label=exp_name)

    # Add value ablation to stats
    if value_ablation:
        all_stats["value_ablation"] = value_ablation
        # Print value ablation summary
        print("\n" + "=" * 70)
        print("VALUE FUNCTION ABLATION (causal test)")
        print("=" * 70)
        print(f"  {'Intervention':20s} {'RMSE(S=1)':>10s} {'RMSE(S=8)':>10s} "
              f"{'Δ at S=8':>10s} {'Bias(S=8)':>10s}")
        print("  " + "-" * 65)
        none_8 = value_ablation["none"]["8"]["rmse_mean"]
        for interv in ["none", "clamp_1", "reverse", "random"]:
            r1 = value_ablation[interv]["1"]["rmse_mean"]
            r8 = value_ablation[interv]["8"]["rmse_mean"]
            delta_pct = (r8 / none_8 - 1) * 100 if interv != "none" else 0
            b8 = value_ablation[interv]["8"]["bias_mean"]
            pct_str = f"+{delta_pct:.0f}%" if interv != "none" else "baseline"
            print(f"  {interv:20s} {r1:>10.4f} {r8:>10.4f} {pct_str:>10s} {b8:>+10.4f}")

    # Save stats
    with open(os.path.join(output_dir, "statistical_metrics.json"), "w") as f:
        def clean(obj):
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean(v) for v in obj]
            return obj
        json.dump(clean(all_stats), f, indent=2)
    print(f"\nStatistical metrics saved to {output_dir}/statistical_metrics.json")

    with open(os.path.join(output_dir, "statistical_summary.txt"), "w") as f:
        for exp_name, results in experiments.items():
            f.write(generate_statistical_summary(results, label=exp_name))
            f.write("\n\n")
    print(f"Statistical summary saved to {output_dir}/statistical_summary.txt")

    # Switching experiment lag metrics
    for exp_name in experiments:
        lag_path = os.path.join("runs", exp_name, "switching_dynamics", "lag_metrics.json")
        if os.path.exists(lag_path):
            with open(lag_path) as f:
                lag_data = json.load(f)
            print(f"\nSwitching Lag Metrics ({exp_name}):")
            for agent, metrics in lag_data.items():
                lag_str = f"{metrics['lag']:.0f}" if metrics['lag'] is not None else "N/A"
                shift_str = f"{metrics['shift']:+.3f}" if metrics['shift'] is not None else "N/A"
                print(f"  {agent:30s}  lag={lag_str:>5s}  shift={shift_str}")

    # ── FINAL SUMMARY ──
    print("\n" + "=" * 70)
    print("PAPER FIGURE INVENTORY")
    print("=" * 70)
    figures = [
        ("fig_hero.png", "Fig 1: Main result overview (4-panel)"),
        ("fig_ablation.png", "Fig 2: Ablation (Δτ tracking + value causal test + early inference)"),
        ("fig_dt_tracking_detail.png", "Fig 3: Δτ tracking detail"),
        ("fig_killer.png", "Fig S1: Supplementary (switching response)"),
        ("fig_visible_vs_hidden.png", "Fig S2: Visible vs hidden speed comparison"),
        ("results_table.tex", "Table 1: Full results"),
    ]
    for fname, desc in figures:
        path = os.path.join(output_dir, fname)
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{status:7s}] {fname:40s} — {desc}")

    print(f"\nAll paper materials saved to {output_dir}/")


if __name__ == "__main__":
    main()
