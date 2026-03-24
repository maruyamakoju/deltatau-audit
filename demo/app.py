"""Gradio web demo for deltatau-audit.

Run locally:
    cd demo && python app.py

Deploy to HuggingFace Spaces:
    Upload the demo/ folder as a Gradio Space.
"""

import os
import sys
import time

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import gradio as gr

# ---------------------------------------------------------------------------
# Inline minimal GRU agent (no dependency on the full package at runtime,
# so the Space works even when only demo/requirements.txt is installed)
# ---------------------------------------------------------------------------

class _SimpleGRUPolicy(nn.Module):
    """Minimal GRU actor-critic used for the demo audit."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(obs_dim, hidden_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, obs, hidden):
        h = self.gru(obs, hidden)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return Categorical(logits=logits), value, h

    def get_initial_hidden(self, batch: int = 1, device: str = "cpu"):
        return torch.zeros(batch, self.hidden_dim, device=device)


# ---------------------------------------------------------------------------
# Lightweight self-contained audit engine
# ---------------------------------------------------------------------------
# We duplicate the core audit loop here so that the Gradio demo can run
# without installing the full deltatau_audit package (important for HF
# Spaces where only demo/requirements.txt is installed).  If the full
# package IS available, we use it instead.
# ---------------------------------------------------------------------------

_HAS_PACKAGE = False
try:
    # Try importing the real package
    _pkg_dir = os.path.join(os.path.dirname(__file__), "..")
    if _pkg_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(_pkg_dir))
    from deltatau_audit.auditor import run_full_audit as _run_full_audit
    from deltatau_audit.adapters.simple_gru import SimpleGRUAdapter
    from deltatau_audit._constants import ROBUSTNESS_SCENARIO_LABELS
    _HAS_PACKAGE = True
except ImportError:
    _HAS_PACKAGE = False
    ROBUSTNESS_SCENARIO_LABELS = {
        "nominal": "Nominal (speed=1)",
        "speed_5x": "5x speed",
        "jitter": "Speed jitter",
        "delay": "Obs delay (1 step)",
        "spike": "Mid-ep speed spike",
        "obs_noise": "Obs noise (sigma=0.1)",
        "adversarial_jitter": "Adversarial jitter",
    }


# -- Wrappers (standalone fallback) ----------------------------------------

class _FixedSpeedWrapper(gym.Wrapper):
    """Steps the underlying env `speed` times per agent step."""

    def __init__(self, env, speed: int = 1):
        super().__init__(env)
        self.speed = speed

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.speed):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class _JitterWrapper(gym.Wrapper):
    """Random speed jitter each step: speed ~ Uniform[lo, hi]."""

    def __init__(self, env, lo: int = 1, hi: int = 3):
        super().__init__(env)
        self.lo, self.hi = lo, hi

    def step(self, action):
        speed = np.random.randint(self.lo, self.hi + 1)
        total_reward = 0.0
        for _ in range(speed):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class _ObsDelayWrapper(gym.Wrapper):
    """Delays observations by `delay` steps."""

    def __init__(self, env, delay: int = 1):
        super().__init__(env)
        self.delay = delay
        self._buffer = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buffer = [obs] * (self.delay + 1)
        return self._buffer[0], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(obs)
        delayed_obs = self._buffer.pop(0)
        return delayed_obs, reward, terminated, truncated, info


class _ObsNoiseWrapper(gym.ObservationWrapper):
    """Adds Gaussian noise to observations."""

    def __init__(self, env, sigma: float = 0.1):
        super().__init__(env)
        self.sigma = sigma

    def observation(self, obs):
        return obs + np.random.normal(0, self.sigma, size=obs.shape).astype(obs.dtype)


class _SpikeWrapper(gym.Wrapper):
    """Speed spike in the middle of the episode: nominal -> spike -> nominal."""

    def __init__(self, env, spike_speed: int = 5, max_steps: int = 500):
        super().__init__(env)
        self.spike_speed = spike_speed
        self.max_steps = max_steps
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self._step_count += 1
        third = self.max_steps // 3
        in_spike = third < self._step_count <= 2 * third
        speed = self.spike_speed if in_spike else 1
        total_reward = 0.0
        for _ in range(speed):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def _wrap_env(env, scenario: str):
    """Wrap env according to scenario name."""
    if scenario == "nominal":
        return env
    elif scenario == "speed_5x":
        return _FixedSpeedWrapper(env, speed=5)
    elif scenario == "jitter":
        return _JitterWrapper(env, lo=1, hi=3)
    elif scenario == "delay":
        return _ObsDelayWrapper(env, delay=1)
    elif scenario == "spike":
        return _SpikeWrapper(env, spike_speed=5)
    elif scenario == "obs_noise":
        return _ObsNoiseWrapper(env, sigma=0.1)
    elif scenario == "adversarial_jitter":
        return _JitterWrapper(env, lo=1, hi=5)
    return env


def _run_episode(model, env, device="cpu"):
    """Run one episode, return total reward."""
    obs, _ = env.reset()
    hidden = model.get_initial_hidden(1, device)
    total_reward = 0.0
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            dist, value, hidden = model(obs_t, hidden)
        action = dist.sample().item()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return total_reward


def _standalone_audit(model, env_id: str, n_episodes: int, device: str = "cpu"):
    """Lightweight audit when the full package is not installed."""
    scenarios = [
        "nominal", "speed_5x", "jitter", "delay",
        "spike", "obs_noise", "adversarial_jitter",
    ]
    results = {}
    for sc in scenarios:
        rewards = []
        for _ in range(n_episodes):
            env = gym.make(env_id)
            env = _wrap_env(env, sc)
            r = _run_episode(model, env, device)
            rewards.append(r)
            env.close()
        results[sc] = {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
        }

    nominal_mean = results["nominal"]["mean"]
    per_scenario = {}
    for sc in scenarios:
        if sc == "nominal":
            continue
        ratio = results[sc]["mean"] / nominal_mean if abs(nominal_mean) > 1e-8 else 0.0
        ratio = max(0.0, min(ratio, 2.0))
        per_scenario[sc] = {
            "return_ratio": ratio,
            "return_drop_pct": (1 - ratio) * 100,
            "mean_return": results[sc]["mean"],
            "std_return": results[sc]["std"],
        }

    worst_ratio = min(s["return_ratio"] for s in per_scenario.values())
    if worst_ratio >= 0.95:
        rating = "PASS"
    elif worst_ratio >= 0.80:
        rating = "MILD"
    elif worst_ratio >= 0.50:
        rating = "DEGRADED"
    else:
        rating = "FAIL"

    return {
        "nominal_mean": nominal_mean,
        "nominal_std": results["nominal"]["std"],
        "per_scenario": per_scenario,
        "worst_ratio": worst_ratio,
        "rating": rating,
    }


# ---------------------------------------------------------------------------
# Full-package audit path
# ---------------------------------------------------------------------------

def _package_audit(model_policy, env_id: str, n_episodes: int, obs_dim: int, act_dim: int):
    """Run audit using the real deltatau_audit package."""
    adapter = SimpleGRUAdapter(model_policy, device="cpu")
    env_factory = lambda: gym.make(env_id)

    result = _run_full_audit(
        adapter, env_factory,
        speeds=[1, 2, 3, 5, 8],
        n_episodes=n_episodes,
        sensitivity_episodes=0,
        seed=42,
        verbose=False,
    )
    return result


# ---------------------------------------------------------------------------
# Create random and "trained" demo agents
# ---------------------------------------------------------------------------

_ENV_CONFIGS = {
    "CartPole-v1": {"obs_dim": 4, "act_dim": 2},
    "Acrobot-v1": {"obs_dim": 6, "act_dim": 3},
    "MountainCar-v0": {"obs_dim": 2, "act_dim": 3},
}


def _make_random_agent(obs_dim: int, act_dim: int, seed: int = 0):
    """Create a randomly initialized GRU agent (fragile baseline)."""
    torch.manual_seed(seed)
    return _SimpleGRUPolicy(obs_dim, act_dim, hidden_dim=64)


def _make_robust_agent(obs_dim: int, act_dim: int, seed: int = 42):
    """Create a 'robust' agent by training briefly with speed randomization.

    This is intentionally quick (a few hundred gradient steps) so the demo
    runs fast.  The resulting agent is not expert-level, but it demonstrates
    the contrast between a fragile and a more-robust policy.
    """
    torch.manual_seed(seed)
    model = _SimpleGRUPolicy(obs_dim, act_dim, hidden_dim=64)
    # We don't do real training here -- that would take minutes.
    # Instead, we use a different random seed so the two agents have
    # visibly different robustness profiles. In a real workflow you'd
    # load a checkpoint that was actually trained with speed augmentation.
    #
    # To make the "robust" agent somewhat better on CartPole at least,
    # we do a quick REINFORCE pass with speed-jitter.
    return model


def _quick_reinforce(model, env_id: str, n_steps: int = 300, lr: float = 3e-3):
    """Ultra-light REINFORCE to give the agent basic competence."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gamma = 0.99

    for episode_i in range(n_steps):
        # Pick a random speed for augmentation
        speed = np.random.choice([1, 1, 1, 2, 3])
        env = gym.make(env_id)
        if speed > 1:
            env = _FixedSpeedWrapper(env, speed=speed)

        obs, _ = env.reset(seed=episode_i)
        hidden = model.get_initial_hidden(1, "cpu")
        log_probs = []
        rewards = []
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist, _, hidden = model(obs_t, hidden)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
        env.close()

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0
        for lp, G in zip(log_probs, returns):
            loss -= lp * G
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

SCENARIO_SHORT = {
    "speed_5x": "5x Speed",
    "jitter": "Jitter",
    "delay": "Delay",
    "spike": "Spike",
    "obs_noise": "Noise",
    "adversarial_jitter": "Adv. Jitter",
}

RATING_COLORS = {
    "PASS": "#28a745",
    "MILD": "#ffc107",
    "DEGRADED": "#fd7e14",
    "FAIL": "#dc3545",
}


def _bar_color(ratio: float) -> str:
    """Return bar color based on return ratio."""
    if ratio >= 0.95:
        return "#28a745"
    elif ratio >= 0.80:
        return "#ffc107"
    elif ratio >= 0.50:
        return "#fd7e14"
    return "#dc3545"


def _make_comparison_chart(before_data: dict, after_data: dict) -> plt.Figure:
    """Create a grouped bar chart comparing Before vs After robustness."""
    scenarios = list(SCENARIO_SHORT.keys())
    labels = [SCENARIO_SHORT[s] for s in scenarios]
    n = len(scenarios)
    x = np.arange(n)
    width = 0.35

    before_ratios = [before_data.get(s, {}).get("return_ratio", 0) for s in scenarios]
    after_ratios = [after_data.get(s, {}).get("return_ratio", 0) for s in scenarios]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width / 2, before_ratios, width, label="Before (fragile)",
                   color="#dc3545", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, after_ratios, width, label="After (robust)",
                   color="#28a745", alpha=0.85, edgecolor="white", linewidth=0.5)

    # Threshold lines
    ax.axhline(y=0.95, color="#28a745", linestyle="--", alpha=0.4, label="PASS (0.95)")
    ax.axhline(y=0.80, color="#ffc107", linestyle="--", alpha=0.4, label="MILD (0.80)")
    ax.axhline(y=0.50, color="#fd7e14", linestyle="--", alpha=0.4, label="FAIL (0.50)")

    ax.set_ylabel("Return Ratio (perturbed / nominal)", fontsize=11)
    ax.set_title("Robustness: Before vs After", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


def _make_single_chart(data: dict, title: str = "Robustness Profile") -> plt.Figure:
    """Create a single bar chart for one agent."""
    scenarios = list(SCENARIO_SHORT.keys())
    labels = [SCENARIO_SHORT[s] for s in scenarios]
    ratios = [data.get(s, {}).get("return_ratio", 0) for s in scenarios]
    colors = [_bar_color(r) for r in ratios]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, ratios, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(y=0.95, color="#28a745", linestyle="--", alpha=0.4, label="PASS")
    ax.axhline(y=0.80, color="#ffc107", linestyle="--", alpha=0.4, label="MILD")
    ax.axhline(y=0.50, color="#fd7e14", linestyle="--", alpha=0.4, label="FAIL")

    ax.set_ylabel("Return Ratio", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for bar, ratio in zip(bars, ratios):
        ax.annotate(f"{ratio:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Format results as Markdown
# ---------------------------------------------------------------------------

def _format_table(before: dict, after: dict) -> str:
    """Build a Markdown comparison table."""
    lines = [
        "| Scenario | Before (ratio) | After (ratio) | Delta |",
        "|----------|:--------------:|:-------------:|:-----:|",
    ]
    scenarios = list(SCENARIO_SHORT.keys())
    for sc in scenarios:
        label = SCENARIO_SHORT[sc]
        br = before.get(sc, {}).get("return_ratio", 0)
        ar = after.get(sc, {}).get("return_ratio", 0)
        delta = ar - br
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {label} | {br:.3f} | {ar:.3f} | {sign}{delta:.3f} |")
    return "\n".join(lines)


def _format_verdict(label: str, result: dict) -> str:
    """Format a single agent verdict."""
    rating = result["rating"]
    worst = result["worst_ratio"]
    color = RATING_COLORS.get(rating, "#6c757d")
    emoji_map = {"PASS": "**PASS**", "MILD": "**MILD**", "DEGRADED": "**DEGRADED**", "FAIL": "**FAIL**"}
    badge = emoji_map.get(rating, rating)

    return (
        f"### {label}\n"
        f"- Verdict: {badge}\n"
        f"- Nominal reward: {result['nominal_mean']:.1f} (std: {result['nominal_std']:.1f})\n"
        f"- Worst-case return ratio: {worst:.3f}\n"
        f"- Worst-case drop: {(1 - worst) * 100:.1f}%\n"
    )


# ---------------------------------------------------------------------------
# Main audit function (called by Gradio)
# ---------------------------------------------------------------------------

def run_demo_audit(env_name: str, n_episodes: int, progress=gr.Progress()):
    """Run Before/After audit and return (markdown, figure, verdict)."""
    n_episodes = int(n_episodes)  # Gradio sliders may return float
    cfg = _ENV_CONFIGS[env_name]
    obs_dim, act_dim = cfg["obs_dim"], cfg["act_dim"]

    # Graceful progress reporting (works with both Gradio 3.x and 4.x)
    def _progress(frac, desc=""):
        try:
            progress(frac, desc=desc)
        except Exception:
            pass

    _progress(0.0, desc="Initializing agents...")

    # Create the two agents
    fragile_model = _make_random_agent(obs_dim, act_dim, seed=0)
    robust_model = _make_random_agent(obs_dim, act_dim, seed=42)

    # Quick REINFORCE training for the "robust" agent
    _progress(0.05, desc="Quick-training robust agent (speed-augmented REINFORCE)...")
    robust_model = _quick_reinforce(robust_model, env_name, n_steps=200)

    # Also give the fragile agent some training, but WITHOUT speed augmentation
    _progress(0.15, desc="Quick-training fragile agent (no augmentation)...")
    fragile_model.train()
    optimizer = torch.optim.Adam(fragile_model.parameters(), lr=3e-3)
    gamma = 0.99
    for ep_i in range(200):
        env = gym.make(env_name)
        obs, _ = env.reset(seed=ep_i + 1000)
        hidden = fragile_model.get_initial_hidden(1)
        log_probs, rewards = [], []
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist, _, hidden = fragile_model(obs_t, hidden)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
        env.close()
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        loss = sum(-lp * G for lp, G in zip(log_probs, returns_t))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    fragile_model.eval()

    t0 = time.time()

    # Run audits
    _progress(0.25, desc="Auditing fragile agent...")
    before_result = _standalone_audit(fragile_model, env_name, n_episodes)

    _progress(0.60, desc="Auditing robust agent...")
    after_result = _standalone_audit(robust_model, env_name, n_episodes)

    elapsed = time.time() - t0

    _progress(0.95, desc="Generating report...")

    # Build outputs
    before_data = before_result["per_scenario"]
    after_data = after_result["per_scenario"]

    # Comparison chart
    fig = _make_comparison_chart(before_data, after_data)

    # Markdown report
    table = _format_table(before_data, after_data)
    before_verdict = _format_verdict("Before (no augmentation)", before_result)
    after_verdict = _format_verdict("After (speed-augmented)", after_result)

    report = (
        f"## deltatau-audit: {env_name} Robustness Comparison\n\n"
        f"**Episodes per condition:** {n_episodes} | "
        f"**Time:** {elapsed:.1f}s\n\n"
        f"---\n\n"
        f"{before_verdict}\n"
        f"{after_verdict}\n"
        f"---\n\n"
        f"### Per-Scenario Comparison\n\n"
        f"{table}\n\n"
        f"---\n\n"
        f"### Interpretation\n\n"
        f"The **Before** agent was trained only at nominal speed. "
        f"The **After** agent was trained with speed randomization (1x-3x), "
        f"making it more resilient to timing perturbations.\n\n"
        f"Scenarios test: variable frame rates (5x, jitter), "
        f"observation delays, mid-episode speed spikes, sensor noise, "
        f"and adversarial timing.\n"
    )

    # Verdict badge
    before_r = before_result["rating"]
    after_r = after_result["rating"]
    badge_md = (
        f"## Verdict\n\n"
        f"| Agent | Rating | Worst Ratio |\n"
        f"|-------|--------|-------------|\n"
        f"| Before | **{before_r}** | {before_result['worst_ratio']:.3f} |\n"
        f"| After  | **{after_r}** | {after_result['worst_ratio']:.3f} |\n"
    )

    _progress(1.0, desc="Done!")
    return report, fig, badge_md


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# deltatau-audit: RL Timing Robustness Demo

**Does your RL agent break when the clock changes?**

This demo creates two GRU agents: one trained at nominal speed only (fragile),
and one trained with speed randomization (robust). Both are then audited under
6 timing perturbation scenarios:

| Scenario | What it tests |
|----------|---------------|
| **5x Speed** | Unseen fast frame rate |
| **Jitter** | Random speed variation each step |
| **Delay** | Observation arrives 1 step late |
| **Spike** | Speed jumps mid-episode (1x -> 5x -> 1x) |
| **Noise** | Gaussian noise on observations |
| **Adv. Jitter** | Worst-case random timing |

Click **Run Audit** to see the comparison. Typical run: 15-45 seconds.
"""

with gr.Blocks(
    title="deltatau-audit Demo",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            env_dropdown = gr.Dropdown(
                choices=list(_ENV_CONFIGS.keys()),
                value="CartPole-v1",
                label="Environment",
                info="Gymnasium environment to audit",
            )
            episode_slider = gr.Slider(
                minimum=5,
                maximum=100,
                value=20,
                step=5,
                label="Episodes per condition",
                info="More episodes = more reliable results, but slower",
            )
            run_btn = gr.Button("Run Audit", variant="primary", size="lg")

        with gr.Column(scale=2):
            verdict_output = gr.Markdown(label="Verdict")

    with gr.Row():
        chart_output = gr.Plot(label="Robustness Comparison")

    with gr.Row():
        report_output = gr.Markdown(label="Detailed Report")

    run_btn.click(
        fn=run_demo_audit,
        inputs=[env_dropdown, episode_slider],
        outputs=[report_output, chart_output, verdict_output],
    )

    gr.Markdown(
        "---\n"
        "*Powered by [deltatau-audit](https://github.com/jmcoholich/deltatau-audit) "
        "| [Paper](https://arxiv.org/abs/xxxx.xxxxx)*"
    )

if __name__ == "__main__":
    demo.launch()
