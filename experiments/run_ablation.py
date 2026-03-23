"""Ablation studies and early inference analysis.

Two key analyses:
1. EARLY INFERENCE: How quickly does Δτ diverge across speeds?
   Runs many episodes at each speed, records step-level Δτ,
   measures when speed=1 vs speed=8 become statistically separable.

2. INTERVENTION ABLATION: What happens if we clamp/reverse/randomize Δτ?
   Uses trained internal_time model, but at test time overrides Δτ.
   Measures reward degradation → proves Δτ is functionally useful.

Usage:
    python run_ablation.py --results-dir runs/speed_gen_hidden_5seed --speed-hidden
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from scipy import stats as scipy_stats

from internal_time_rl.models.policy import InternalTimeAgent
from internal_time_rl.models.baselines import SkipRNNAgent
from internal_time_rl.envs.variable_frequency import VariableFrequencyChainEnv

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 200,
    "savefig.dpi": 300,
})

TEST_SPEEDS = [1, 2, 3, 5, 8]
TRAIN_SPEEDS = [1, 2, 3]


###############################################################################
# Agent loading
###############################################################################

def load_agent(model_dir, obs_dim, act_dim, agent_type, device="cpu"):
    ckpt_path = os.path.join(model_dir, "final.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if agent_type == "baseline":
        agent = InternalTimeAgent(obs_dim, act_dim, use_internal_time=False)
    elif agent_type in ("internal_time", "internal_time_discount"):
        agent = InternalTimeAgent(obs_dim, act_dim, use_internal_time=True)
    elif agent_type == "skip_rnn":
        agent = SkipRNNAgent(obs_dim, act_dim)
    else:
        raise ValueError(f"Unknown: {agent_type}")

    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    return agent


###############################################################################
# Part 1: Early Inference Metric
###############################################################################

def record_episodes_at_speed(agent, speed, n_episodes=50, device="cpu",
                              speed_in_obs=False, total_steps=30):
    """Record step-level Δτ for many episodes at a given constant speed.

    Runs for `total_steps` regardless of episode termination (resets env
    but keeps hidden state, like the switching experiment). This ensures
    all speeds produce the same number of steps for fair comparison.

    Uses noise=0.05 to get stochastic episodes and non-zero variance.
    """
    env = VariableFrequencyChainEnv(
        chain_length=20, delay=10, max_agent_steps=100,
        train_speeds=(speed,), speed_in_obs=speed_in_obs,
        fixed_speed=speed, noise=0.05,
    )

    all_dts = []  # list of lists

    for _ in range(n_episodes):
        obs, info = env.reset()
        hidden = agent.get_initial_hidden(1, device)
        ep_dts = []

        for step in range(total_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _, hidden, dt = agent.get_action_and_value(obs_t, hidden)
            ep_dts.append(dt.item())
            obs, reward, term, trunc, info = env.step(action.item())
            if term or trunc:
                obs, _ = env.reset()
                # Keep hidden state — agent retains memory across resets

        all_dts.append(ep_dts)

    return all_dts


def compute_early_inference_metrics(agent, speeds=TEST_SPEEDS, n_episodes=100,
                                      device="cpu", speed_in_obs=False):
    """Compute how quickly Δτ diverges across speeds.

    Returns:
    - per_step_means: {speed: [mean_dt_at_step0, mean_dt_at_step1, ...]}
    - per_step_stds: same but std
    - separation_step: step at which speed=1 and speed=8 become
      statistically separable (p < 0.05, Mann-Whitney U)
    - divergence_at_k: mean absolute separation at step K for each pair
    """
    speed_dts = {}

    for speed in speeds:
        episodes = record_episodes_at_speed(
            agent, speed, n_episodes=n_episodes, device=device,
            speed_in_obs=speed_in_obs,
        )
        speed_dts[speed] = episodes

    # Compute per-step statistics (all episodes now have same length)
    min_len = min(
        min(len(ep) for ep in speed_dts[s]) for s in speeds
    )

    per_step_means = {}
    per_step_stds = {}
    for speed in speeds:
        padded = np.array([ep[:min_len] for ep in speed_dts[speed] if len(ep) >= min_len])
        per_step_means[speed] = padded.mean(axis=0).tolist()
        per_step_stds[speed] = padded.std(axis=0).tolist()

    # Separation step: when does speed=1 vs speed=8 diverge significantly?
    s1_eps = [ep[:min_len] for ep in speed_dts[1] if len(ep) >= min_len]
    s8_eps = [ep[:min_len] for ep in speed_dts[8] if len(ep) >= min_len]
    s1_arr = np.array(s1_eps)
    s8_arr = np.array(s8_eps)

    separation_step = None
    p_values = []
    for step in range(min_len):
        stat, p = scipy_stats.mannwhitneyu(
            s1_arr[:, step], s8_arr[:, step], alternative="two-sided"
        )
        p_values.append(float(p))
        if p < 0.05 and separation_step is None:
            separation_step = step

    # Cohen's d at each step (effect size)
    cohens_d = []
    for step in range(min_len):
        m1, m8 = s1_arr[:, step].mean(), s8_arr[:, step].mean()
        v1, v8 = s1_arr[:, step].var(), s8_arr[:, step].var()
        s_pooled = np.sqrt((v1 + v8) / 2)
        if s_pooled > 1e-8:
            d = (m8 - m1) / s_pooled
        else:
            # Zero variance → use raw difference (infinite effect if nonzero)
            d = float(np.sign(m8 - m1)) * min(abs(m8 - m1) * 1000, 10.0)
        cohens_d.append(float(d))

    # Separation at K=3 (early) and K=full
    early_sep = abs(np.mean(s8_arr[:, :3]) - np.mean(s1_arr[:, :3]))
    late_sep = abs(np.mean(s8_arr[:, -3:]) - np.mean(s1_arr[:, -3:]))

    return {
        "per_step_means": {str(s): v for s, v in per_step_means.items()},
        "per_step_stds": {str(s): v for s, v in per_step_stds.items()},
        "separation_step": separation_step,
        "p_values": p_values,
        "cohens_d": cohens_d,
        "early_separation_k3": float(early_sep),
        "late_separation": float(late_sep),
        "min_len": min_len,
    }


def plot_early_inference(inference_data, save_path, agent_label="Internal Time"):
    """Plot the early inference figure: per-step Δτ divergence across speeds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    means = inference_data["per_step_means"]
    stds = inference_data["per_step_stds"]
    min_len = inference_data["min_len"]
    steps = np.arange(min_len)

    speed_colors = {
        "1": "#2166ac", "2": "#67a9cf", "3": "#d1e5f0",
        "5": "#ef8a62", "8": "#b2182b",
    }

    # Panel (a): Δτ trajectory per speed (mean ± std)
    for s_str in ["1", "2", "3", "5", "8"]:
        m = np.array(means[s_str])
        sd = np.array(stds[s_str])
        color = speed_colors[s_str]
        unseen = " *" if int(s_str) not in TRAIN_SPEEDS else ""
        ax1.plot(steps, m, color=color, linewidth=2, label=f"Speed {s_str}{unseen}")
        ax1.fill_between(steps, m - sd, m + sd, alpha=0.15, color=color)

    ax1.axhline(y=1.0, color="black", linestyle=":", alpha=0.2)

    # Mark separation step
    sep = inference_data["separation_step"]
    if sep is not None:
        ax1.axvline(x=sep, color="green", linestyle="--", alpha=0.5, linewidth=1.5)
        ax1.text(sep + 0.3, ax1.get_ylim()[1] * 0.95,
                 f"p < 0.05\nat step {sep}", fontsize=8, color="green", va="top")

    ax1.set_xlabel("Agent Step", fontsize=12)
    ax1.set_ylabel(r"$\Delta\tau$", fontsize=12)
    ax1.set_title(f"(a) {agent_label}: Speed-Dependent " + r"$\Delta\tau$ Trajectories",
                  fontsize=12)
    ax1.legend(fontsize=9, loc="best")

    # Panel (b): Cohen's d (effect size) and p-values
    d_vals = inference_data["cohens_d"]
    p_vals = inference_data["p_values"]

    ax2.bar(steps, d_vals, color="#d62728", alpha=0.7, label="Cohen's d (S=1 vs S=8)")
    ax2.set_xlabel("Agent Step", fontsize=12)
    ax2.set_ylabel("Cohen's d (effect size)", fontsize=12, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # p-values on secondary axis
    ax2b = ax2.twinx()
    ax2b.plot(steps, [-np.log10(max(p, 1e-50)) for p in p_vals],
              color="#1f77b4", marker="o", markersize=4, linewidth=1.5,
              label=r"$-\log_{10}(p)$")
    ax2b.axhline(y=-np.log10(0.05), color="#1f77b4", linestyle="--",
                 alpha=0.3, linewidth=1)
    ax2b.text(min_len - 0.5, -np.log10(0.05) + 0.5, "p=0.05",
              fontsize=8, color="#1f77b4", ha="right")
    ax2b.set_ylabel(r"$-\log_{10}(p)$", fontsize=12, color="#1f77b4")
    ax2b.tick_params(axis="y", labelcolor="#1f77b4")

    ax2.set_title("(b) Separation Speed: S=1 vs S=8", fontsize=12)

    # Annotate key metrics
    info = (f"Separation step: {sep if sep is not None else 'N/A'}\n"
            f"Early sep (K=3): {inference_data['early_separation_k3']:.4f}\n"
            f"Late sep: {inference_data['late_separation']:.4f}")
    ax2.text(0.97, 0.97, info, transform=ax2.transAxes, fontsize=9,
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                       edgecolor="gray", alpha=0.9))

    fig.suptitle(r"Amortized Speed Inference: $\Delta\tau$ Diverges Within First Few Steps",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


###############################################################################
# Part 2: Intervention Ablation
###############################################################################

def evaluate_with_intervention(agent, speed, intervention="none",
                                n_episodes=100, device="cpu",
                                speed_in_obs=False,
                                max_agent_steps=35, noise=0.1):
    """Evaluate agent at a speed with Δτ intervention.

    Uses a HARDER evaluation setting (tight step budget + noise) to make
    Δτ functionally important. Standard chain has generous step budget
    where any reasonable policy succeeds.

    Interventions:
    - "none": normal agent (baseline)
    - "clamp_1": force Δτ=1.0 at every step
    - "reverse": force Δτ = 2.0 - normal_dt (invert the mapping)
    - "random": Δτ ~ Uniform(0.5, 1.5) at every step
    """
    env = VariableFrequencyChainEnv(
        chain_length=20, delay=10, max_agent_steps=max_agent_steps,
        train_speeds=(speed,), speed_in_obs=speed_in_obs,
        fixed_speed=speed, noise=noise,
    )

    rewards = []
    ep_lengths = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        hidden = agent.get_initial_hidden(1, device)
        total_reward = 0
        done = False
        steps = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, _, _, _, hidden_new, dt = agent.get_action_and_value(obs_t, hidden)

            # Apply intervention to hidden state update
            # The intervention modifies the NEXT hidden state by re-running
            # the time module with a different Δτ
            if intervention == "clamp_1":
                # Re-run forward pass with Δτ=1.0
                hidden_new = _rerun_with_dt(agent, obs_t, hidden, 1.0, device)
            elif intervention == "reverse":
                # Reverse: if normal Δτ > 1, make it < 1 and vice versa
                reversed_dt = 2.0 - dt.item()
                reversed_dt = max(0.3, min(2.5, reversed_dt))
                hidden_new = _rerun_with_dt(agent, obs_t, hidden, reversed_dt, device)
            elif intervention == "random":
                random_dt = np.random.uniform(0.5, 1.5)
                hidden_new = _rerun_with_dt(agent, obs_t, hidden, random_dt, device)

            hidden = hidden_new
            obs, reward, term, trunc, info = env.step(action.item())
            total_reward += reward
            done = term or trunc
            steps += 1

        rewards.append(total_reward)
        ep_lengths.append(steps)

    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "median": float(np.median(rewards)),
        "mean_length": float(np.mean(ep_lengths)),
    }


def _rerun_with_dt(agent, obs_t, hidden, target_dt, device):
    """Re-run the GRU cell with a specific Δτ value."""
    with torch.no_grad():
        encoded = agent.encoder(obs_t)
        dt_tensor = torch.tensor([[target_dt]], dtype=torch.float32, device=device)
        hidden_new = agent.rnn(encoded, hidden, dt_tensor)
    return hidden_new


def run_intervention_ablation(agent, speeds=TEST_SPEEDS, n_episodes=100,
                               device="cpu", speed_in_obs=False):
    """Run all intervention conditions at all speeds.

    Tests two regimes:
    - "hard": tight step budget (35 steps) + noise (0.1) — where Δτ should matter
    - "easy": generous budget (100 steps) + no noise — baseline comparison
    """
    interventions = ["none", "clamp_1", "reverse", "random"]
    conditions = {
        "hard": {"max_agent_steps": 35, "noise": 0.1},
        "easy": {"max_agent_steps": 100, "noise": 0.0},
    }

    results = {}

    for cond_name, cond_params in conditions.items():
        print(f"\n  Condition: {cond_name} (steps={cond_params['max_agent_steps']}, "
              f"noise={cond_params['noise']})")
        results[cond_name] = {}

        for intervention in interventions:
            print(f"    Intervention: {intervention}")
            results[cond_name][intervention] = {}
            for speed in speeds:
                res = evaluate_with_intervention(
                    agent, speed, intervention=intervention,
                    n_episodes=n_episodes, device=device,
                    speed_in_obs=speed_in_obs,
                    **cond_params,
                )
                results[cond_name][intervention][str(speed)] = res
            # Summary line
            mean_r = np.mean([results[cond_name][intervention][str(s)]["mean"]
                              for s in speeds])
            mean_std = np.mean([results[cond_name][intervention][str(s)]["std"]
                                for s in speeds])
            print(f"      avg R={mean_r:.3f}±{mean_std:.3f}")

    return results


def plot_intervention_ablation(ablation_results, save_path):
    """Plot intervention ablation results for both easy and hard conditions."""
    conditions = [c for c in ["hard", "easy"] if c in ablation_results]

    fig, axes = plt.subplots(1, len(conditions), figsize=(7 * len(conditions), 5))
    if len(conditions) == 1:
        axes = [axes]

    speeds = np.array(TEST_SPEEDS)
    intervention_styles = {
        "none": {"label": r"Normal $\Delta\tau$ (learned)", "color": "#d62728",
                 "marker": "o", "linestyle": "-"},
        "clamp_1": {"label": r"$\Delta\tau = 1.0$ (clamped)", "color": "#7f7f7f",
                    "marker": "s", "linestyle": "--"},
        "reverse": {"label": r"$\Delta\tau$ reversed", "color": "#2ca02c",
                    "marker": "^", "linestyle": "-."},
        "random": {"label": r"$\Delta\tau$ random", "color": "#ff7f0e",
                   "marker": "D", "linestyle": ":"},
    }

    for idx, cond_name in enumerate(conditions):
        ax = axes[idx]
        cond_data = ablation_results[cond_name]

        for interv, style in intervention_styles.items():
            if interv not in cond_data:
                continue
            means = [cond_data[interv][str(s)]["mean"] for s in TEST_SPEEDS]
            stds = [cond_data[interv][str(s)]["std"] for s in TEST_SPEEDS]
            ax.errorbar(speeds, means, yerr=stds, label=style["label"],
                        color=style["color"], marker=style["marker"],
                        markersize=7, linewidth=2, capsize=3,
                        linestyle=style["linestyle"])

        ax.axvspan(3.5, 8.5, alpha=0.05, color="red")
        ax.set_xlabel("Environment Speed", fontsize=12)
        ax.set_ylabel("Episode Reward", fontsize=12)
        title_suffix = "(tight budget + noise)" if cond_name == "hard" else "(standard)"
        ax.set_title(f"({chr(97+idx)}) {cond_name.title()} {title_suffix}", fontsize=12)
        ax.set_xticks(TEST_SPEEDS)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(r"Intervention Ablation: Does Learned $\Delta\tau$ Aid Performance?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


###############################################################################
# Part 3: Value Function Accuracy Under Intervention
###############################################################################

def evaluate_value_accuracy(agent, speed, intervention="none",
                             n_episodes=50, device="cpu",
                             speed_in_obs=False, gamma=0.99):
    """Evaluate value prediction accuracy under Δτ intervention.

    Key insight: on the chain task, the optimal policy is trivially "go right"
    regardless of Δτ, so reward-based ablation shows no effect. But the VALUE
    FUNCTION depends on knowing the speed (it determines remaining steps →
    discounting). If Δτ encodes speed information into the hidden state,
    clamping/reversing Δτ should degrade value prediction accuracy.

    Returns per-episode value RMSE and mean absolute error.
    """
    env = VariableFrequencyChainEnv(
        chain_length=20, delay=10, max_agent_steps=100,
        train_speeds=(speed,), speed_in_obs=speed_in_obs,
        fixed_speed=speed, noise=0.0,
    )

    all_rmse = []
    all_mae = []
    all_value_bias = []  # positive = overestimates, negative = underestimates

    for _ in range(n_episodes):
        obs, info = env.reset()
        hidden = agent.get_initial_hidden(1, device)
        done = False

        step_rewards = []
        step_values = []

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                # Get action and value from normal forward pass
                action, _, _, value, hidden_new, dt = agent.get_action_and_value(obs_t, hidden)

            # Apply intervention to hidden state (affects NEXT step's value)
            if intervention == "clamp_1":
                hidden_new = _rerun_with_dt(agent, obs_t, hidden, 1.0, device)
            elif intervention == "reverse":
                reversed_dt = 2.0 - dt.item()
                reversed_dt = max(0.3, min(2.5, reversed_dt))
                hidden_new = _rerun_with_dt(agent, obs_t, hidden, reversed_dt, device)
            elif intervention == "random":
                random_dt = np.random.uniform(0.5, 1.5)
                hidden_new = _rerun_with_dt(agent, obs_t, hidden, random_dt, device)

            # For intervention, also recompute VALUE from intervened hidden state
            if intervention != "none":
                with torch.no_grad():
                    encoded = agent.encoder(obs_t)
                    # Use the intervened hidden for value computation
                    value = agent.value_head(hidden_new).squeeze(-1)

            step_values.append(value.item())
            hidden = hidden_new

            obs, reward, term, trunc, info = env.step(action.item())
            step_rewards.append(reward)
            done = term or trunc

        # Compute actual discounted returns from each step
        T = len(step_rewards)
        returns = np.zeros(T)
        G = 0
        for t in reversed(range(T)):
            G = step_rewards[t] + gamma * G
            returns[t] = G

        values_arr = np.array(step_values)
        returns_arr = returns

        # Value prediction error metrics
        errors = values_arr - returns_arr
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        bias = np.mean(errors)  # systematic over/under-estimation

        all_rmse.append(rmse)
        all_mae.append(mae)
        all_value_bias.append(bias)

    return {
        "rmse_mean": float(np.mean(all_rmse)),
        "rmse_std": float(np.std(all_rmse)),
        "mae_mean": float(np.mean(all_mae)),
        "mae_std": float(np.std(all_mae)),
        "bias_mean": float(np.mean(all_value_bias)),
        "bias_std": float(np.std(all_value_bias)),
    }


def run_value_ablation(agent, speeds=TEST_SPEEDS, n_episodes=50,
                        device="cpu", speed_in_obs=False):
    """Run value function accuracy ablation across all speeds and interventions."""
    interventions = ["none", "clamp_1", "reverse", "random"]
    results = {}

    for intervention in interventions:
        print(f"    Intervention: {intervention}")
        results[intervention] = {}
        for speed in speeds:
            res = evaluate_value_accuracy(
                agent, speed, intervention=intervention,
                n_episodes=n_episodes, device=device,
                speed_in_obs=speed_in_obs,
            )
            results[intervention][str(speed)] = res
        # Summary
        mean_rmse = np.mean([results[intervention][str(s)]["rmse_mean"] for s in speeds])
        mean_bias = np.mean([results[intervention][str(s)]["bias_mean"] for s in speeds])
        print(f"      avg RMSE={mean_rmse:.4f}, avg bias={mean_bias:+.4f}")

    return results


def plot_value_ablation(value_results, save_path):
    """Plot value function accuracy under Δτ intervention.

    2-panel figure:
    (a) Value RMSE by speed for each intervention
    (b) Value bias by speed — shows systematic over/under-estimation
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    speeds = np.array(TEST_SPEEDS)
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

    for interv, style in intervention_styles.items():
        if interv not in value_results:
            continue
        rmse = [value_results[interv][str(s)]["rmse_mean"] for s in TEST_SPEEDS]
        rmse_std = [value_results[interv][str(s)]["rmse_std"] for s in TEST_SPEEDS]
        ax1.errorbar(speeds, rmse, yerr=rmse_std, label=style["label"],
                     color=style["color"], marker=style["marker"],
                     markersize=7, linewidth=2, capsize=3, linestyle=style["ls"])

        bias = [value_results[interv][str(s)]["bias_mean"] for s in TEST_SPEEDS]
        bias_std = [value_results[interv][str(s)]["bias_std"] for s in TEST_SPEEDS]
        ax2.errorbar(speeds, bias, yerr=bias_std, label=style["label"],
                     color=style["color"], marker=style["marker"],
                     markersize=7, linewidth=2, capsize=3, linestyle=style["ls"])

    ax1.axvspan(3.5, 8.5, alpha=0.05, color="red")
    ax1.set_xlabel("Environment Speed", fontsize=12)
    ax1.set_ylabel("Value RMSE", fontsize=12)
    ax1.set_title("(a) Value Prediction Error", fontsize=12)
    ax1.set_xticks(TEST_SPEEDS)
    ax1.legend(fontsize=9)

    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axvspan(3.5, 8.5, alpha=0.05, color="red")
    ax2.set_xlabel("Environment Speed", fontsize=12)
    ax2.set_ylabel("Value Bias (V - G)", fontsize=12)
    ax2.set_title("(b) Systematic Value Bias", fontsize=12)
    ax2.set_xticks(TEST_SPEEDS)
    ax2.legend(fontsize=9)

    fig.suptitle(r"Value Function Accuracy Under $\Delta\tau$ Intervention",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


###############################################################################
# Part 4: Multi-seed Value Ablation (statistical hardening)
###############################################################################

def run_multiseed_value_ablation(results_dir, agent_type, n_seeds=5,
                                  speeds=TEST_SPEEDS, n_episodes=50,
                                  device="cpu", speed_in_obs=False):
    """Run value ablation across multiple seeds for statistical robustness.

    Returns per-seed results and aggregate statistics with SE.
    """
    sample_env = VariableFrequencyChainEnv(
        chain_length=20, delay=10, max_agent_steps=100,
        train_speeds=(1, 2, 3), speed_in_obs=speed_in_obs,
    )
    obs_dim = sample_env.observation_space.shape[0]
    act_dim = sample_env.action_space.n

    interventions = ["none", "clamp_1", "reverse", "random"]
    per_seed = []

    for seed in range(n_seeds):
        model_dir = os.path.join(results_dir, agent_type, f"seed_{seed}")
        ckpt_path = os.path.join(model_dir, "final.pt")
        if not os.path.exists(ckpt_path):
            print(f"    Seed {seed}: no checkpoint, skipping")
            continue

        print(f"    Seed {seed}...")
        agent = load_agent(model_dir, obs_dim, act_dim, agent_type, device)

        seed_results = run_value_ablation(
            agent, speeds=speeds, n_episodes=n_episodes,
            device=device, speed_in_obs=speed_in_obs,
        )
        per_seed.append(seed_results)

    # Aggregate across seeds
    n = len(per_seed)
    aggregate = {}
    for interv in interventions:
        aggregate[interv] = {}
        for s in speeds:
            s_str = str(s)
            rmses = [ps[interv][s_str]["rmse_mean"] for ps in per_seed]
            biases = [ps[interv][s_str]["bias_mean"] for ps in per_seed]
            aggregate[interv][s_str] = {
                "rmse_mean": float(np.mean(rmses)),
                "rmse_se": float(np.std(rmses) / np.sqrt(n)) if n > 1 else 0.0,
                "rmse_std": float(np.std(rmses)),
                "rmse_per_seed": [float(r) for r in rmses],
                "bias_mean": float(np.mean(biases)),
                "bias_se": float(np.std(biases) / np.sqrt(n)) if n > 1 else 0.0,
                "bias_per_seed": [float(b) for b in biases],
                "n_seeds": n,
            }

    return {"per_seed": per_seed, "aggregate": aggregate}


def plot_multiseed_value_ablation(results_by_model, save_path):
    """Plot multi-seed value ablation for one or more models.

    Shows RMSE with SE error bars. Annotates % degradation.
    """
    n_models = len(results_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5.5),
                             squeeze=False)

    speeds = np.array(TEST_SPEEDS)
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

    model_labels = {
        "internal_time": "Internal Time",
        "internal_time_discount": r"Internal Time + $\gamma^{\Delta\tau}$",
    }

    for idx, (model_name, data) in enumerate(results_by_model.items()):
        ax = axes[0, idx]
        agg = data["aggregate"]
        n_seeds = agg["none"][str(TEST_SPEEDS[0])]["n_seeds"]

        for interv, style in intervention_styles.items():
            if interv not in agg:
                continue
            rmse = [agg[interv][str(s)]["rmse_mean"] for s in TEST_SPEEDS]
            rmse_se = [agg[interv][str(s)]["rmse_se"] for s in TEST_SPEEDS]
            ax.errorbar(speeds, rmse, yerr=rmse_se, label=style["label"],
                        color=style["color"], marker=style["marker"],
                        markersize=7, linewidth=2, capsize=4,
                        linestyle=style["ls"])

        ax.axvspan(3.5, 8.5, alpha=0.05, color="red")

        # Annotate degradation at S=8
        none_8 = agg["none"]["8"]["rmse_mean"]
        for interv in ["reverse", "clamp_1", "random"]:
            r8 = agg[interv]["8"]["rmse_mean"]
            pct = (r8 / none_8 - 1) * 100
            if interv == "reverse":
                ax.annotate(f"+{pct:.0f}%", xy=(8, r8),
                            xytext=(6.8, r8 + 0.015),
                            fontsize=9, fontweight="bold",
                            color=intervention_styles[interv]["color"],
                            arrowprops=dict(arrowstyle="->",
                                            color=intervention_styles[interv]["color"],
                                            lw=1.5))

        label = model_labels.get(model_name, model_name)
        ax.set_xlabel("Environment Speed", fontsize=12)
        ax.set_ylabel("Value RMSE", fontsize=12)
        ax.set_title(f"({chr(97 + idx)}) {label} (n={n_seeds} seeds)", fontsize=12)
        ax.set_xticks(TEST_SPEEDS)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(r"Value Function Accuracy Under $\Delta\tau$ Intervention (Multi-Seed)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--speed-hidden", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--skip-reward-ablation", action="store_true",
                        help="Skip the reward-based ablation (already shown to have no effect)")
    parser.add_argument("--value-only", action="store_true",
                        help="Only run value function ablation (single seed)")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run value ablation across all seeds (statistical hardening)")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    speed_in_obs = not args.speed_hidden
    device = args.device

    sample_env = VariableFrequencyChainEnv(
        chain_length=20, delay=10, max_agent_steps=100,
        train_speeds=(1, 2, 3), speed_in_obs=speed_in_obs,
    )
    obs_dim = sample_env.observation_space.shape[0]
    act_dim = sample_env.action_space.n

    output_dir = os.path.join(args.results_dir, "ablation")
    os.makedirs(output_dir, exist_ok=True)

    # ── Multi-seed mode: fast path for statistical hardening ──
    if args.multi_seed:
        print("=" * 60)
        print("MULTI-SEED VALUE ABLATION")
        print("=" * 60)

        results_by_model = {}
        for agent_type in ["internal_time", "internal_time_discount"]:
            print(f"\n  Model: {agent_type}")
            res = run_multiseed_value_ablation(
                args.results_dir, agent_type, n_seeds=args.n_seeds,
                n_episodes=args.n_episodes, device=device,
                speed_in_obs=speed_in_obs,
            )
            results_by_model[agent_type] = res

            # Print summary
            agg = res["aggregate"]
            n = agg["none"][str(TEST_SPEEDS[0])]["n_seeds"]
            print(f"\n  {agent_type} (n={n} seeds):")
            print(f"  {'Intervention':20s} {'RMSE(S=1)':>12s} {'RMSE(S=8)':>12s} "
                  f"{'Δ%':>8s} {'Bias(S=8)':>12s}")
            print("  " + "-" * 70)
            none_8 = agg["none"]["8"]["rmse_mean"]
            for interv in ["none", "clamp_1", "reverse", "random"]:
                r1 = agg[interv]["1"]["rmse_mean"]
                r1_se = agg[interv]["1"]["rmse_se"]
                r8 = agg[interv]["8"]["rmse_mean"]
                r8_se = agg[interv]["8"]["rmse_se"]
                pct = (r8 / none_8 - 1) * 100 if interv != "none" else 0
                b8 = agg[interv]["8"]["bias_mean"]
                b8_se = agg[interv]["8"]["bias_se"]
                pct_str = f"+{pct:.0f}%" if interv != "none" else "base"
                print(f"  {interv:20s} {r1:.4f}±{r1_se:.4f} {r8:.4f}±{r8_se:.4f} "
                      f"{pct_str:>8s} {b8:+.4f}±{b8_se:.4f}")

        # Save
        save_data = {}
        for model_name, res in results_by_model.items():
            save_data[model_name] = res["aggregate"]
        with open(os.path.join(output_dir, "value_ablation_multiseed.json"), "w") as f:
            json.dump(save_data, f, indent=2)

        # Plot
        plot_multiseed_value_ablation(
            results_by_model,
            os.path.join(output_dir, "value_ablation_multiseed.png"),
        )

        print(f"\n  Multi-seed results saved to {output_dir}/")
        return

    if not args.value_only:
        # ── Part 1: Early Inference ──
        print("=" * 60)
        print("PART 1: EARLY INFERENCE ANALYSIS")
        print("=" * 60)

        agent_types_to_analyze = ["internal_time", "internal_time_discount", "skip_rnn"]
        all_inference = {}

        for agent_type in agent_types_to_analyze:
            model_dir = os.path.join(args.results_dir, agent_type, "seed_0")
            if not os.path.exists(os.path.join(model_dir, "final.pt")):
                print(f"  Skipping {agent_type} (no checkpoint)")
                continue

            print(f"\n  Analyzing {agent_type}...")
            agent = load_agent(model_dir, obs_dim, act_dim, agent_type, device)

            inference = compute_early_inference_metrics(
                agent, n_episodes=args.n_episodes, device=device,
                speed_in_obs=speed_in_obs,
            )
            all_inference[agent_type] = inference

            print(f"    Separation step (S=1 vs S=8): {inference['separation_step']}")
            print(f"    Early sep (K=3): {inference['early_separation_k3']:.4f}")
            print(f"    Late sep: {inference['late_separation']:.4f}")
            print(f"    Cohen's d: {[f'{d:.2f}' for d in inference['cohens_d']]}")

            plot_early_inference(
                inference,
                os.path.join(output_dir, f"early_inference_{agent_type}.png"),
                agent_label=agent_type.replace("_", " ").title(),
            )

        with open(os.path.join(output_dir, "early_inference.json"), "w") as f:
            json.dump(all_inference, f, indent=2)
        print(f"\n  Early inference data saved to {output_dir}/early_inference.json")

        # ── Part 2: Intervention Ablation (reward-based) ──
        if not args.skip_reward_ablation:
            print("\n" + "=" * 60)
            print("PART 2: INTERVENTION ABLATION (reward)")
            print("=" * 60)

            it_dir = os.path.join(args.results_dir, "internal_time", "seed_0")
            if os.path.exists(os.path.join(it_dir, "final.pt")):
                print("\n  Loading internal_time agent...")
                agent = load_agent(it_dir, obs_dim, act_dim, "internal_time", device)

                ablation_results = run_intervention_ablation(
                    agent, n_episodes=args.n_episodes, device=device,
                    speed_in_obs=speed_in_obs,
                )

                with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
                    json.dump(ablation_results, f, indent=2)

                plot_intervention_ablation(
                    ablation_results,
                    os.path.join(output_dir, "intervention_ablation.png"),
                )

    # ── Part 3: Value Function Accuracy Ablation ──
    print("\n" + "=" * 60)
    print("PART 3: VALUE FUNCTION ACCURACY ABLATION")
    print("=" * 60)

    it_dir = os.path.join(args.results_dir, "internal_time", "seed_0")
    if os.path.exists(os.path.join(it_dir, "final.pt")):
        print("\n  Loading internal_time agent for value ablation...")
        agent = load_agent(it_dir, obs_dim, act_dim, "internal_time", device)

        value_results = run_value_ablation(
            agent, n_episodes=args.n_episodes, device=device,
            speed_in_obs=speed_in_obs,
        )

        with open(os.path.join(output_dir, "value_ablation.json"), "w") as f:
            json.dump(value_results, f, indent=2)

        plot_value_ablation(
            value_results,
            os.path.join(output_dir, "value_ablation.png"),
        )

        print(f"\n  VALUE ABLATION SUMMARY:")
        print(f"  {'Intervention':20s} {'RMSE(S=1)':>10s} {'RMSE(S=8)':>10s} "
              f"{'Δ RMSE':>8s} {'Bias(S=8)':>10s}")
        print("  " + "-" * 62)
        none_rmse_8 = value_results["none"]["8"]["rmse_mean"]
        for interv in ["none", "clamp_1", "reverse", "random"]:
            r1 = value_results[interv]["1"]["rmse_mean"]
            r8 = value_results[interv]["8"]["rmse_mean"]
            delta = r8 - none_rmse_8
            b8 = value_results[interv]["8"]["bias_mean"]
            print(f"  {interv:20s} {r1:>10.4f} {r8:>10.4f} {delta:>+8.4f} {b8:>+10.4f}")

    print(f"\n  All results saved to {output_dir}/")


if __name__ == "__main__":
    main()
