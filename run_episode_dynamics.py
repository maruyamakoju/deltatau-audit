"""Record and visualize episode-level Δτ dynamics at different speeds.

Loads a trained agent and runs individual episodes at each test speed,
recording step-by-step Δτ, position, reward to create the episode
dynamics figure (A2 from the plan).

Also supports --switch mode for the "killer figure": mid-episode speed
switching (e.g., speed 1→8) to demonstrate dynamic Δτ adaptation.

Usage:
    python run_episode_dynamics.py --results-dir runs/speed_gen_hidden --speed-hidden
    python run_episode_dynamics.py --results-dir runs/speed_gen_hidden_5seed --switch --speed-hidden
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from internal_time_rl.models.policy import InternalTimeAgent
from internal_time_rl.models.baselines import SkipRNNAgent, ExternalDtAgent
from internal_time_rl.envs.variable_frequency import VariableFrequencyChainEnv


# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 200,
})

TEST_SPEEDS = [1, 2, 3, 5, 8]
TRAIN_SPEEDS = [1, 2, 3]


def load_agent(model_dir, obs_dim, act_dim, agent_type, device="cpu"):
    """Load a trained agent from checkpoint."""
    ckpt_path = os.path.join(model_dir, "final.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if agent_type == "baseline":
        agent = InternalTimeAgent(obs_dim, act_dim, use_internal_time=False)
    elif agent_type in ("internal_time", "internal_time_discount"):
        agent = InternalTimeAgent(obs_dim, act_dim, use_internal_time=True)
    elif agent_type == "skip_rnn":
        agent = SkipRNNAgent(obs_dim, act_dim)
    elif agent_type == "external_dt":
        agent = ExternalDtAgent(obs_dim, act_dim)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    return agent


def record_episode(agent, speed, agent_type, device="cpu",
                   speed_in_obs=True, chain_length=20, delay=10):
    """Run one episode and record step-level data."""
    env = VariableFrequencyChainEnv(
        chain_length=chain_length,
        delay=delay,
        max_agent_steps=100,
        train_speeds=(speed,),
        speed_in_obs=speed_in_obs,
        noise=0.0,
        fixed_speed=speed,
    )

    obs, info = env.reset()
    hidden = agent.get_initial_hidden(1, device)

    episode_data = {
        "speed": speed,
        "steps": [],
        "delta_taus": [],
        "positions": [],
        "rewards": [],
        "actions": [],
    }

    done = False
    step = 0

    while not done and step < 150:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if agent_type == "external_dt":
            ext_dt = torch.tensor([[speed / 3.0]], dtype=torch.float32, device=device)
            agent.set_external_dt(ext_dt)

        with torch.no_grad():
            action, _, _, _, hidden, dt = agent.get_action_and_value(obs_t, hidden)

        episode_data["steps"].append(step)
        episode_data["delta_taus"].append(dt.item())

        a = action.item()
        obs, reward, term, trunc, info = env.step(a)
        episode_data["positions"].append(info.get("position", 0))
        episode_data["rewards"].append(reward)
        episode_data["actions"].append(a)

        done = term or trunc
        step += 1

    episode_data["total_reward"] = sum(episode_data["rewards"])
    episode_data["num_steps"] = step
    return episode_data


def record_switching_episode(agent, agent_type, switch_speeds=(1, 8),
                             switch_step=40, device="cpu",
                             speed_in_obs=False, chain_length=20, delay=10,
                             max_agent_steps=100, total_record_steps=80):
    """Run one extended episode with mid-episode speed switch.

    This is the KEY experiment: speed changes from switch_speeds[0] to
    switch_speeds[1] at step `switch_step`. We record Δτ to see if the
    agent dynamically adapts.

    The episode runs for `total_record_steps` regardless of termination.
    If the env terminates (goal reached), we reset the env but keep the
    agent's hidden state and the new speed — this gives us a long
    trajectory to observe Δτ dynamics across the speed transition.
    """
    env = VariableFrequencyChainEnv(
        chain_length=chain_length,
        delay=delay,
        max_agent_steps=max_agent_steps,
        train_speeds=(1, 2, 3),
        speed_in_obs=speed_in_obs,
        noise=0.0,
        speed_schedule="switch",
        switch_speeds=switch_speeds,
        switch_step=switch_step,
    )

    obs, info = env.reset()
    hidden = agent.get_initial_hidden(1, device)

    episode_data = {
        "switch_speeds": list(switch_speeds),
        "switch_step": switch_step,
        "steps": [],
        "delta_taus": [],
        "positions": [],
        "rewards": [],
        "actions": [],
        "speeds": [],  # Track actual speed at each step
    }

    step = 0

    while step < total_record_steps:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if agent_type == "external_dt":
            ext_dt = torch.tensor([[env.current_speed / 3.0]],
                                  dtype=torch.float32, device=device)
            agent.set_external_dt(ext_dt)

        with torch.no_grad():
            action, _, _, _, hidden, dt = agent.get_action_and_value(obs_t, hidden)

        # Record the speed BEFORE the step (to get the speed at this step)
        current_speed = env.current_speed
        # But we need to manually track the schedule since env may reset
        if step < switch_step:
            current_speed = switch_speeds[0]
        else:
            current_speed = switch_speeds[1]

        episode_data["steps"].append(step)
        episode_data["delta_taus"].append(dt.item())
        episode_data["speeds"].append(current_speed)

        a = action.item()
        obs, reward, term, trunc, info = env.step(a)
        episode_data["positions"].append(info.get("position", 0))
        episode_data["rewards"].append(reward)
        episode_data["actions"].append(a)

        if term or trunc:
            # Reset env but keep hidden state (agent's "memory" persists)
            obs, _ = env.reset()
            # Override speed to match our schedule
            if step < switch_step:
                env.current_speed = switch_speeds[0]
            else:
                env.current_speed = switch_speeds[1]

        step += 1

    episode_data["total_reward"] = sum(episode_data["rewards"])
    episode_data["num_steps"] = step
    return episode_data


def compute_lag_metric(delta_taus, speeds, switch_step, window=5):
    """Compute adaptation lag: steps after switch to reach 50% of final Δτ shift.

    Returns lag in steps. Lower = faster adaptation.
    Also returns the Δτ shift magnitude.
    """
    dts = np.array(delta_taus)

    pre_end = switch_step
    post_start = switch_step
    post_end = len(dts)

    if pre_end < 3 or post_end - post_start < 3:
        return float("nan"), float("nan")

    # Use last portion of each phase for stable estimates
    pre_window = min(15, pre_end)
    post_window = min(15, post_end - post_start)
    pre_mean = np.mean(dts[pre_end - pre_window:pre_end])
    post_mean = np.mean(dts[post_end - post_window:post_end])

    shift = post_mean - pre_mean
    if abs(shift) < 1e-4:
        return float("nan"), shift

    # Target: 50% of shift from pre_mean
    target = pre_mean + 0.5 * shift

    # Use a smoothed signal to find crossing (rolling average)
    lag = float("nan")
    for i in range(switch_step, min(switch_step + 40, len(dts) - window + 1)):
        smooth_dt = np.mean(dts[i:i + window])
        if shift > 0 and smooth_dt >= target:
            lag = i - switch_step
            break
        elif shift < 0 and smooth_dt <= target:
            lag = i - switch_step
            break

    return lag, shift


def plot_switching_figure(all_agent_data, save_path, title=""):
    """THE killer figure: speed(t) and Δτ(t) overlay for all agents.

    Shows mid-episode speed switch and how each agent's Δτ responds.
    This proves dynamic adaptation, not just per-episode constant estimation.
    """
    agents = list(all_agent_data.keys())
    n_agents = len(agents)

    fig, axes = plt.subplots(n_agents, 1, figsize=(12, 3.0 * n_agents),
                              sharex=True)
    if n_agents == 1:
        axes = [axes]

    agent_colors = {
        "internal_time": "#d62728",
        "internal_time_discount": "#8c1515",
        "skip_rnn": "#1f77b4",
        "baseline": "#7f7f7f",
        "external_dt": "#2ca02c",
    }
    agent_labels = {
        "internal_time": "Internal Time (ours)",
        "internal_time_discount": r"IT + $\gamma^{\Delta\tau}$",
        "skip_rnn": "Skip-RNN (ACT)",
        "baseline": "Baseline GRU",
        "external_dt": "External dt (ODE-RNN)",
    }

    for idx, agent_name in enumerate(agents):
        ax = axes[idx]
        data = all_agent_data[agent_name]
        color = agent_colors.get(agent_name, "black")
        label = agent_labels.get(agent_name, agent_name)

        steps = np.array(data["steps"])
        dts = np.array(data["delta_taus"])
        speeds = np.array(data["speeds"])
        switch_step = data["switch_step"]

        # Δτ line
        ax.plot(steps, dts, color=color, linewidth=2.5, label=r"$\Delta\tau$", zorder=3)
        ax.fill_between(steps, 0, dts, alpha=0.1, color=color)
        ax.axhline(y=1.0, color="black", linestyle=":", alpha=0.2, linewidth=0.8)

        # Speed on secondary axis (shaded background)
        ax2 = ax.twinx()
        # Shade pre/post switch regions
        ax2.fill_between(steps, 0, speeds, alpha=0.08, color="orange", step="post")
        ax2.step(steps, speeds, color="orange", linewidth=1.5, alpha=0.6,
                 where="post", label="Speed")
        ax2.set_ylabel("Speed", fontsize=10, color="orange")
        ax2.set_ylim(0, 12)
        ax2.tick_params(axis="y", labelcolor="orange")

        # Switch line
        ax.axvline(x=switch_step, color="black", linestyle="--", alpha=0.5,
                   linewidth=1.5, zorder=2)
        ymin, ymax = 0.2, max(np.max(dts) * 1.15, 1.8)
        ax.text(switch_step + 1, ymax * 0.95, "speed\nswitch",
                fontsize=8, va="top", ha="left", color="black", alpha=0.6)

        # Compute and annotate lag
        lag, shift = compute_lag_metric(dts, speeds, switch_step)
        pre_mean = np.mean(dts[max(0, switch_step - 10):switch_step])
        post_mean = np.mean(dts[min(switch_step + 10, len(dts)):])

        info_text = f"{label}\n"
        info_text += f"pre: {pre_mean:.3f}  post: {post_mean:.3f}\n"
        if not np.isnan(lag):
            info_text += f"lag: {lag:.0f} steps  shift: {shift:+.3f}"
        else:
            info_text += f"shift: {shift:+.3f}" if not np.isnan(shift) else "no shift"

        ax.text(0.02, 0.95, info_text, transform=ax.transAxes,
                fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.9))

        ax.set_ylabel(r"$\Delta\tau$", fontsize=12)
        ax.set_ylim(ymin, ymax)

        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                  fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("Agent Step", fontsize=12)

    suptitle = r"Dynamic $\Delta\tau$ Adaptation During Mid-Episode Speed Switch"
    if title:
        suptitle += f"\n{title}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_switching_hero(all_agent_data, save_path):
    """Compact 2-panel hero version of the switching figure.

    Panel (a): Internal Time agent (focus on the response)
    Panel (b): All agents overlaid for comparison
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 4.5))

    agent_colors = {
        "internal_time": "#d62728",
        "internal_time_discount": "#8c1515",
        "skip_rnn": "#1f77b4",
        "baseline": "#7f7f7f",
    }
    agent_labels = {
        "internal_time": "Internal Time (ours)",
        "internal_time_discount": r"IT + $\gamma^{\Delta\tau}$",
        "skip_rnn": "Skip-RNN",
        "baseline": "Baseline GRU",
    }

    # Get switch_step from any agent
    any_data = next(iter(all_agent_data.values()))
    switch_step = any_data["switch_step"]

    # Panel (a): Focus on Internal Time
    focus = "internal_time"
    if focus in all_agent_data:
        data = all_agent_data[focus]
        steps = np.array(data["steps"])
        dts = np.array(data["delta_taus"])
        speeds = np.array(data["speeds"])

        # Speed background
        ax_a2 = ax_a.twinx()
        ax_a2.fill_between(steps, 0, speeds, alpha=0.08, color="orange", step="post")
        ax_a2.step(steps, speeds, color="orange", linewidth=1.5, alpha=0.5,
                   where="post", label="Env Speed")
        ax_a2.set_ylabel("Speed", color="orange", fontsize=11)
        ax_a2.set_ylim(0, 12)
        ax_a2.tick_params(axis="y", labelcolor="orange")

        ax_a.plot(steps, dts, color="#d62728", linewidth=2.5,
                  label=r"$\Delta\tau$ (Internal Time)")
        ax_a.fill_between(steps, 0, dts, alpha=0.12, color="#d62728")
        ax_a.axhline(y=1.0, color="black", linestyle=":", alpha=0.2)
        ax_a.axvline(x=switch_step, color="black", linestyle="--", alpha=0.5, linewidth=1.5)

        # Annotate lag
        lag, shift = compute_lag_metric(dts, speeds, switch_step)
        if not np.isnan(lag):
            ax_a.annotate(
                f"lag = {lag:.0f} steps",
                xy=(switch_step + lag, dts[switch_step + int(lag)] if switch_step + int(lag) < len(dts) else dts[-1]),
                xytext=(switch_step + lag + 8, 1.4),
                fontsize=9, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
            )

        ax_a.set_xlabel("Agent Step", fontsize=12)
        ax_a.set_ylabel(r"$\Delta\tau$", fontsize=12)
        ax_a.set_title(r"(a) Internal Time $\Delta\tau$ Tracks Speed Switch", fontsize=12)
        ax_a.set_ylim(0.2, max(np.max(dts) * 1.15, 1.8))

        lines1, labels1 = ax_a.get_legend_handles_labels()
        lines2, labels2 = ax_a2.get_legend_handles_labels()
        ax_a.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # Panel (b): All agents overlaid
    for agent_name in ["baseline", "skip_rnn", "internal_time_discount", "internal_time"]:
        if agent_name not in all_agent_data:
            continue
        data = all_agent_data[agent_name]
        steps = np.array(data["steps"])
        dts = np.array(data["delta_taus"])
        color = agent_colors.get(agent_name, "black")
        label = agent_labels.get(agent_name, agent_name)
        lw = 2.5 if agent_name == "internal_time" else 1.5
        alpha = 1.0 if "internal_time" in agent_name else 0.7
        ax_b.plot(steps[:min(len(steps), len(dts))],
                  dts[:min(len(steps), len(dts))],
                  color=color, linewidth=lw, alpha=alpha, label=label)

    # Speed background on panel b
    if any_data:
        steps_bg = np.array(any_data["steps"])
        speeds_bg = np.array(any_data["speeds"])
        ax_b2 = ax_b.twinx()
        ax_b2.fill_between(steps_bg, 0, speeds_bg, alpha=0.06, color="orange", step="post")
        ax_b2.step(steps_bg, speeds_bg, color="orange", linewidth=1, alpha=0.4,
                   where="post")
        ax_b2.set_ylabel("Speed", color="orange", fontsize=11)
        ax_b2.set_ylim(0, 12)
        ax_b2.tick_params(axis="y", labelcolor="orange")

    ax_b.axvline(x=switch_step, color="black", linestyle="--", alpha=0.5, linewidth=1.5)
    ax_b.axhline(y=1.0, color="black", linestyle=":", alpha=0.2)
    ax_b.set_xlabel("Agent Step", fontsize=12)
    ax_b.set_ylabel(r"$\Delta\tau$", fontsize=12)
    ax_b.set_title("(b) All Models: Response to Speed Switch", fontsize=12)
    ax_b.legend(loc="upper left", fontsize=9)
    ax_b.set_ylim(0.2, 2.0)

    fig.suptitle(r"Dynamic $\Delta\tau$ Adaptation: Speed $1 \rightarrow 8$ Mid-Episode",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def run_switching_experiment(results_dir, speed_in_obs=False, device="cpu",
                             switch_speeds=(1, 8), switch_step=40,
                             n_episodes=5):
    """Run switching experiment for all agents in results_dir.

    Runs multiple episodes and averages to get smooth Δτ trajectories.
    """
    sample_env = VariableFrequencyChainEnv(
        chain_length=20, delay=10, max_agent_steps=100,
        train_speeds=(1, 2, 3), speed_in_obs=speed_in_obs,
    )
    obs_dim = sample_env.observation_space.shape[0]
    act_dim = sample_env.action_space.n

    output_dir = os.path.join(results_dir, "switching_dynamics")
    os.makedirs(output_dir, exist_ok=True)

    agent_types = {
        "baseline": "baseline",
        "internal_time": "internal_time",
        "internal_time_discount": "internal_time_discount",
        "skip_rnn": "skip_rnn",
    }

    all_agent_data = {}
    all_agent_multi = {}  # Multiple episodes per agent

    for agent_name, agent_type in agent_types.items():
        # Try seed_0 first, fall back to any available seed
        model_dir = os.path.join(results_dir, agent_name, "seed_0")
        if not os.path.exists(os.path.join(model_dir, "final.pt")):
            print(f"  Skipping {agent_name} (no checkpoint)")
            continue

        print(f"  Recording switching episodes for {agent_name}...")
        agent = load_agent(model_dir, obs_dim, act_dim, agent_type, device)

        episodes = []
        for ep_idx in range(n_episodes):
            ep = record_switching_episode(
                agent, agent_type,
                switch_speeds=switch_speeds,
                switch_step=switch_step,
                device=device,
                speed_in_obs=speed_in_obs,
            )
            episodes.append(ep)

        all_agent_multi[agent_name] = episodes

        # Average Δτ across episodes (align by step)
        max_steps = max(len(ep["delta_taus"]) for ep in episodes)
        avg_dts = []
        avg_speeds = []
        for s in range(max_steps):
            dt_vals = [ep["delta_taus"][s] for ep in episodes if s < len(ep["delta_taus"])]
            sp_vals = [ep["speeds"][s] for ep in episodes if s < len(ep["speeds"])]
            avg_dts.append(np.mean(dt_vals))
            avg_speeds.append(np.mean(sp_vals) if sp_vals else 0)

        # Use averaged data for plotting
        avg_data = {
            "switch_speeds": list(switch_speeds),
            "switch_step": switch_step,
            "steps": list(range(max_steps)),
            "delta_taus": avg_dts,
            "speeds": avg_speeds,
            "positions": episodes[0]["positions"],  # from first episode
            "total_reward": np.mean([ep["total_reward"] for ep in episodes]),
            "num_steps": max_steps,
        }
        all_agent_data[agent_name] = avg_data

        # Report
        pre_dts = avg_dts[:switch_step]
        post_dts = avg_dts[switch_step:]
        pre_mean = np.mean(pre_dts) if pre_dts else 0
        post_mean = np.mean(post_dts) if post_dts else 0
        lag, shift = compute_lag_metric(avg_dts, avg_speeds, switch_step)
        print(f"    pre_dt={pre_mean:.3f}  post_dt={post_mean:.3f}  "
              f"shift={shift:+.3f}  lag={lag}")

    # Plot the killer figure
    if len(all_agent_data) >= 2:
        plot_switching_figure(
            all_agent_data,
            os.path.join(output_dir, "switching_all_agents.png"),
            title=f"Speed {switch_speeds[0]}→{switch_speeds[1]} at step {switch_step} "
                  f"({n_episodes} episodes averaged)"
        )
        plot_switching_hero(
            all_agent_data,
            os.path.join(output_dir, "switching_hero.png"),
        )

    # Save data
    serializable = {}
    for agent_name, data in all_agent_data.items():
        serializable[agent_name] = {
            k: v if not isinstance(v, (np.ndarray, np.floating)) else (
                v.tolist() if isinstance(v, np.ndarray) else float(v)
            )
            for k, v in data.items()
        }
    with open(os.path.join(output_dir, "switching_data.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    # Compute lag metrics summary
    lag_summary = {}
    for agent_name, data in all_agent_data.items():
        lag, shift = compute_lag_metric(
            data["delta_taus"], data["speeds"], data["switch_step"]
        )
        lag_summary[agent_name] = {"lag": lag if not np.isnan(lag) else None,
                                   "shift": float(shift) if not np.isnan(shift) else None}
    with open(os.path.join(output_dir, "lag_metrics.json"), "w") as f:
        json.dump(lag_summary, f, indent=2)
    print(f"\n  Lag metrics: {lag_summary}")
    print(f"  All switching data saved to {output_dir}/")

    return all_agent_data


def plot_episode_dynamics(episodes_by_speed, save_path, agent_label="Internal Time"):
    """Plot episode-level Δτ dynamics for one agent across speeds."""
    n_speeds = len(episodes_by_speed)
    fig, axes = plt.subplots(n_speeds, 1, figsize=(12, 2.5 * n_speeds), sharex=False)
    if n_speeds == 1:
        axes = [axes]

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, n_speeds))

    for idx, (speed, ep_data) in enumerate(sorted(episodes_by_speed.items())):
        ax = axes[idx]
        steps = np.array(ep_data["steps"])
        dts = np.array(ep_data["delta_taus"])
        positions = np.array(ep_data["positions"])

        # Δτ as filled area
        ax.fill_between(steps, 0, dts, alpha=0.2, color=colors[idx])
        ax.plot(steps, dts, color=colors[idx], linewidth=2, label=r"$\Delta\tau$")
        ax.axhline(y=1.0, color="black", linestyle=":", alpha=0.3, linewidth=0.8)

        # Mean Δτ annotation
        mean_dt = np.mean(dts)
        ax.axhline(y=mean_dt, color=colors[idx], linestyle="--", alpha=0.4, linewidth=1)
        ax.text(len(steps) + 1, mean_dt, f"mean={mean_dt:.3f}",
                fontsize=8, va="center", color=colors[idx])

        # Position on secondary axis
        ax2 = ax.twinx()
        ax2.plot(steps, positions, color="gray", linewidth=0.8, alpha=0.4,
                 linestyle="--", label="position")
        ax2.set_ylabel("Position", fontsize=8, color="gray")
        ax2.set_ylim(-1, 22)
        ax2.tick_params(axis="y", labelsize=7, colors="gray")

        # Speed label
        is_unseen = speed not in TRAIN_SPEEDS
        title_color = "#d62728" if is_unseen else "black"
        unseen_marker = " (unseen)" if is_unseen else ""
        ax.set_title(f"Speed = {speed}{unseen_marker}  |  "
                     f"R = {ep_data['total_reward']:.2f}  |  "
                     f"Steps = {ep_data['num_steps']}",
                     fontsize=11, color=title_color)
        ax.set_ylabel(r"$\Delta\tau$")

        # Set consistent y-limits
        ax.set_ylim(0.2, max(max(dts) * 1.15, 1.5))

    axes[-1].set_xlabel("Agent Step")
    fig.suptitle(f"Episode-Level $\\Delta\\tau$ Dynamics — {agent_label}",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_multi_agent_dynamics(all_agent_episodes, save_path):
    """Plot Δτ dynamics for multiple agents side by side.

    all_agent_episodes: {agent_name: {speed: episode_data}}
    """
    agents = list(all_agent_episodes.keys())
    n_agents = len(agents)
    speeds = sorted(list(all_agent_episodes[agents[0]].keys()))
    n_speeds = len(speeds)

    fig, axes = plt.subplots(n_speeds, n_agents, figsize=(5 * n_agents, 2.2 * n_speeds),
                              sharex=False, sharey="row")

    agent_colors = {
        "internal_time": "#d62728",
        "internal_time_discount": "#8c1515",
        "skip_rnn": "#1f77b4",
        "baseline": "#7f7f7f",
        "external_dt": "#2ca02c",
    }

    agent_labels = {
        "internal_time": "Internal Time (ours)",
        "internal_time_discount": "IT + Discount",
        "skip_rnn": "Skip-RNN",
        "baseline": "Baseline GRU",
        "external_dt": "External dt",
    }

    for col, agent_name in enumerate(agents):
        color = agent_colors.get(agent_name, "black")
        for row, speed in enumerate(speeds):
            ax = axes[row, col] if n_agents > 1 else axes[row]
            ep_data = all_agent_episodes[agent_name].get(speed)
            if ep_data is None:
                continue

            steps = np.array(ep_data["steps"])
            dts = np.array(ep_data["delta_taus"])

            ax.fill_between(steps, 0, dts, alpha=0.15, color=color)
            ax.plot(steps, dts, color=color, linewidth=1.5)
            ax.axhline(y=1.0, color="black", linestyle=":", alpha=0.2, linewidth=0.5)

            mean_dt = np.mean(dts)
            ax.axhline(y=mean_dt, color=color, linestyle="--", alpha=0.3)

            if col == 0:
                is_unseen = speed not in TRAIN_SPEEDS
                label = f"Speed={speed}"
                if is_unseen:
                    label += " *"
                ax.set_ylabel(label, fontsize=10,
                              color="#d62728" if is_unseen else "black")
            if row == 0:
                ax.set_title(agent_labels.get(agent_name, agent_name), fontsize=11,
                             color=color, fontweight="bold")
            if row == n_speeds - 1:
                ax.set_xlabel("Step")

            ax.set_ylim(0.2, 2.0)
            ax.text(0.95, 0.95, f"dt={mean_dt:.2f}",
                    transform=ax.transAxes, fontsize=7, va="top", ha="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle(r"Episode-Level $\Delta\tau$ Dynamics: All Models $\times$ All Speeds",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def record_multi_episode_dynamics(agent, speed, agent_type, n_episodes=20,
                                   device="cpu", speed_in_obs=False,
                                   chain_length=20, delay=10):
    """Record multiple episodes at a given speed and return averaged Δτ trajectory."""
    all_dts = []
    for _ in range(n_episodes):
        ep = record_episode(agent, speed, agent_type, device,
                            speed_in_obs=speed_in_obs,
                            chain_length=chain_length, delay=delay)
        all_dts.append(ep["delta_taus"])

    # Align by padding to max length
    max_len = max(len(d) for d in all_dts)
    padded = np.full((n_episodes, max_len), np.nan)
    for i, d in enumerate(all_dts):
        padded[i, :len(d)] = d

    mean_dts = np.nanmean(padded, axis=0)
    std_dts = np.nanstd(padded, axis=0)

    return mean_dts, std_dts, max_len


def plot_speed_inference_figure(agent_data, save_path):
    """THE revised killer figure: Δτ trajectory at each constant speed, overlaid.

    Shows that the agent rapidly infers speed from first few observations
    and adjusts Δτ to different levels for different speeds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    speed_colors = {1: "#2166ac", 2: "#67a9cf", 3: "#fddbc7",
                    5: "#ef8a62", 8: "#b2182b"}

    # Panel (a): Internal Time agent - Δτ trajectories per speed
    ax = axes[0]
    agent_name = "internal_time"
    if agent_name in agent_data:
        for speed in TEST_SPEEDS:
            if speed not in agent_data[agent_name]:
                continue
            mean_dt, std_dt, _ = agent_data[agent_name][speed]
            steps = np.arange(len(mean_dt))
            color = speed_colors[speed]
            unseen = " *" if speed not in TRAIN_SPEEDS else ""
            ax.plot(steps, mean_dt, color=color, linewidth=2,
                    label=f"Speed {speed}{unseen}")
            ax.fill_between(steps, mean_dt - std_dt, mean_dt + std_dt,
                            alpha=0.15, color=color)

    ax.axhline(y=1.0, color="black", linestyle=":", alpha=0.2, linewidth=0.8)
    ax.set_xlabel("Agent Step", fontsize=12)
    ax.set_ylabel(r"$\Delta\tau$", fontsize=12)
    ax.set_title(r"(a) Internal Time: $\Delta\tau$ Trajectory by Speed", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0.5, 2.0)

    # Panel (b): Mean Δτ per speed (bar chart with CI) — all models
    ax2 = axes[1]
    model_order = ["baseline", "internal_time", "internal_time_discount", "skip_rnn"]
    x = np.arange(len(TEST_SPEEDS))
    width = 0.18

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
        "baseline": "Baseline",
    }

    for i, model_name in enumerate(model_order):
        if model_name not in agent_data:
            continue
        means = []
        for speed in TEST_SPEEDS:
            if speed in agent_data[model_name]:
                mean_dt, _, _ = agent_data[model_name][speed]
                means.append(np.mean(mean_dt))
            else:
                means.append(1.0)
        color = agent_colors.get(model_name, "gray")
        label = agent_labels.get(model_name, model_name)
        ax2.bar(x + i * width, means, width, label=label, color=color, alpha=0.85)

    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([f"S={s}{'*' if s not in TRAIN_SPEEDS else ''}" for s in TEST_SPEEDS])
    ax2.axhline(y=1.0, color="black", linestyle=":", alpha=0.3)
    ax2.set_xlabel("Environment Speed", fontsize=12)
    ax2.set_ylabel(r"Mean $\Delta\tau$", fontsize=12)
    ax2.set_title(r"(b) Episode-Average $\Delta\tau$ by Model and Speed", fontsize=12)
    ax2.legend(fontsize=9, loc="upper left")

    fig.suptitle(r"Adaptive $\Delta\tau$: Speed Inference from Observations",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def run_for_experiment(results_dir, speed_in_obs=True, device="cpu"):
    """Record and plot episode dynamics for all agents in an experiment."""
    # Get obs dim from env
    sample_env = VariableFrequencyChainEnv(
        chain_length=20, delay=10, max_agent_steps=100,
        train_speeds=(1, 2, 3), speed_in_obs=speed_in_obs,
    )
    obs_dim = sample_env.observation_space.shape[0]
    act_dim = sample_env.action_space.n

    output_dir = os.path.join(results_dir, "episode_dynamics")
    os.makedirs(output_dir, exist_ok=True)

    agent_types = {
        "baseline": "baseline",
        "internal_time": "internal_time",
        "internal_time_discount": "internal_time_discount",
        "skip_rnn": "skip_rnn",
        "external_dt": "external_dt",
    }

    all_agent_episodes = {}
    all_agent_multi = {}  # For multi-episode averaged data

    for agent_name, agent_type in agent_types.items():
        model_dir = os.path.join(results_dir, agent_name, "seed_0")
        if not os.path.exists(os.path.join(model_dir, "final.pt")):
            print(f"  Skipping {agent_name} (no checkpoint)")
            continue

        print(f"  Recording episodes for {agent_name}...")
        agent = load_agent(model_dir, obs_dim, act_dim, agent_type, device)

        episodes = {}
        multi_episodes = {}
        for speed in TEST_SPEEDS:
            # Single episode for backwards compatibility
            ep = record_episode(agent, speed, agent_type, device,
                                speed_in_obs=speed_in_obs)
            episodes[speed] = ep
            # Multiple episodes for averaged dynamics
            mean_dt, std_dt, max_len = record_multi_episode_dynamics(
                agent, speed, agent_type, n_episodes=20, device=device,
                speed_in_obs=speed_in_obs,
            )
            multi_episodes[speed] = (mean_dt, std_dt, max_len)
            print(f"    Speed {speed}: dt_mean={np.mean(mean_dt):.3f}, "
                  f"R={ep['total_reward']:.2f}, steps={ep['num_steps']}")

        all_agent_episodes[agent_name] = episodes
        all_agent_multi[agent_name] = multi_episodes

        # Individual agent figure
        plot_episode_dynamics(
            episodes,
            os.path.join(output_dir, f"dynamics_{agent_name}.png"),
            agent_label=agent_name.replace("_", " ").title(),
        )

    # Multi-agent comparison
    if len(all_agent_episodes) >= 2:
        plot_multi_agent_dynamics(
            all_agent_episodes,
            os.path.join(output_dir, "dynamics_all_agents.png"),
        )

    # Speed inference figure (the NEW killer figure)
    if len(all_agent_multi) >= 2:
        plot_speed_inference_figure(
            all_agent_multi,
            os.path.join(output_dir, "speed_inference_figure.png"),
        )

    # Save episode data
    serializable = {}
    for agent_name, episodes in all_agent_episodes.items():
        serializable[agent_name] = {
            str(speed): {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in ep_data.items()
            }
            for speed, ep_data in episodes.items()
        }
    with open(os.path.join(output_dir, "episode_data.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Episode data saved to {output_dir}/episode_data.json")

    return all_agent_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--speed-hidden", action="store_true")
    parser.add_argument("--switch", action="store_true",
                        help="Run mid-episode speed switching experiment")
    parser.add_argument("--switch-from", type=int, default=1,
                        help="Speed before switch (default: 1)")
    parser.add_argument("--switch-to", type=int, default=8,
                        help="Speed after switch (default: 8)")
    parser.add_argument("--switch-step", type=int, default=40,
                        help="Step at which speed switches (default: 40)")
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Number of episodes to average (default: 10)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.switch:
        run_switching_experiment(
            args.results_dir,
            speed_in_obs=not args.speed_hidden,
            device=args.device,
            switch_speeds=(args.switch_from, args.switch_to),
            switch_step=args.switch_step,
            n_episodes=args.n_episodes,
        )
    else:
        run_for_experiment(
            args.results_dir,
            speed_in_obs=not args.speed_hidden,
            device=args.device,
        )
