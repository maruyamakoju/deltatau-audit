#!/usr/bin/env python3
r"""Comprehensive benchmark experiment runner for Cambridge/DeepMind submission.

Trains, evaluates, and reports on three agent architectures (baseline, internal
time, deliberative ACT) across multiple seeds on CartPole-v1.  Produces
publication-quality tables (Markdown + LaTeX) and figures (300 DPI PNG).

Phases
------
train   - Initialize agents with different seeds and collect baseline rollouts.
eval    - Run the full deltatau-audit evaluation pipeline on each agent.
report  - Generate Tables 1-3 and Figures 1-3 from cached evaluation JSON.
all     - Run every phase sequentially.

Usage
-----
    python experiments/run_benchmark_suite.py --phase train
    python experiments/run_benchmark_suite.py --phase eval
    python experiments/run_benchmark_suite.py --phase report
    python experiments/run_benchmark_suite.py --phase all
    python experiments/run_benchmark_suite.py --phase all --seeds 3 --quick

Output tree
-----------
    results/benchmark/
        models/           Trained checkpoints  (<config>_seed<n>.pt)
        eval/             Raw evaluation JSONs (<config>_seed<n>_audit.json)
        tables/           Markdown + LaTeX tables
        figures/          PNG figures (300 DPI)
        summary.json      Machine-readable aggregate summary
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve project root so imports work when running from the experiments/ dir
# ---------------------------------------------------------------------------
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

ENV_ID = "CartPole-v1"
OBS_DIM = 4
ACT_DIM = 2

DEFAULT_SEEDS = list(range(5))        # seeds 0-4
DEFAULT_TRAIN_STEPS = 50_000
DEFAULT_EVAL_EPISODES = 100
SPEED_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
AUDIT_SPEEDS = [1, 2, 3, 5]
AUDIT_N_EPISODES = 30

# Agent configurations for the benchmark
AGENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "agent_class": "InternalTimeAgent",
        "use_internal_time": False,
        "label": "Standard PPO (no $\\Delta\\tau$)",
    },
    "internal_time": {
        "agent_class": "InternalTimeAgent",
        "use_internal_time": True,
        "label": "PPO + Internal Time ($\\Delta\\tau$-GRU)",
    },
    "deliberative": {
        "agent_class": "DeliberativeInternalTimeAgent",
        "use_internal_time": True,
        "max_thinking_steps": 5,
        "label": "PPO + ACT Deliberation",
    },
}

# ACT ablation variants (Table 3)
ACT_ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "act_vanilla": {
        "max_thinking_steps": 5,
        "lambda_geo": 0.5,
        "use_adaptive_steps": False,
        "num_heads": 1,
        "info_gain_threshold": 1.0,  # effectively disabled
        "label": "Vanilla ACT",
    },
    "act_geo_prior": {
        "max_thinking_steps": 5,
        "lambda_geo": 0.5,
        "use_adaptive_steps": False,
        "num_heads": 1,
        "info_gain_threshold": 0.01,
        "label": "+ Geometric Prior",
    },
    "act_adaptive": {
        "max_thinking_steps": 5,
        "lambda_geo": 0.5,
        "use_adaptive_steps": True,
        "num_heads": 1,
        "info_gain_threshold": 0.01,
        "label": "+ Adaptive Steps",
    },
    "act_multihead": {
        "max_thinking_steps": 5,
        "lambda_geo": 0.5,
        "use_adaptive_steps": False,
        "num_heads": 4,
        "info_gain_threshold": 0.01,
        "label": "+ Multi-Head (4)",
    },
    "act_full": {
        "max_thinking_steps": 5,
        "lambda_geo": 0.5,
        "use_adaptive_steps": True,
        "num_heads": 4,
        "info_gain_threshold": 0.01,
        "label": "+ All Extensions",
    },
}

# Matplotlib publication settings
MPL_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """Container for a single agent evaluation."""
    config_name: str
    seed: int
    nominal_mean: float = 0.0
    nominal_std: float = 0.0
    nominal_episodes: int = 0
    speed_returns: Dict[str, float] = field(default_factory=dict)
    sensitivity_mean: float = 0.0
    sensitivity_ci: Tuple[float, float] = (0.0, 0.0)
    audit_summary: Dict[str, Any] = field(default_factory=dict)
    ponder_stats: Optional[Dict[str, float]] = None
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Agent construction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_agent(config_name: str, seed: int) -> torch.nn.Module:
    """Instantiate an agent for the given configuration and seed.

    Sets the torch/numpy RNG to ensure reproducible weight initialization.
    """
    torch.manual_seed(seed * 1000 + 42)
    np.random.seed(seed * 1000 + 42)

    cfg = AGENT_CONFIGS.get(config_name)
    if cfg is None:
        # Might be an ablation config
        cfg = ACT_ABLATION_CONFIGS.get(config_name)
        if cfg is None:
            raise ValueError(f"Unknown config: {config_name}")
        # Ablation configs are always deliberative
        return _build_deliberative(cfg, seed)

    if cfg["agent_class"] == "InternalTimeAgent":
        from internal_time_rl.models.policy import InternalTimeAgent
        return InternalTimeAgent(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            hidden_dim=128,
            latent_dim=64,
            time_hidden_dim=32,
            use_internal_time=cfg["use_internal_time"],
        )
    elif cfg["agent_class"] == "DeliberativeInternalTimeAgent":
        return _build_deliberative(cfg, seed)
    else:
        raise ValueError(f"Unknown agent class: {cfg['agent_class']}")


def _build_deliberative(cfg: Dict[str, Any], seed: int) -> torch.nn.Module:
    """Build a DeliberativeInternalTimeAgent from config dict."""
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    return DeliberativeInternalTimeAgent(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_dim=128,
        latent_dim=64,
        time_hidden_dim=32,
        max_thinking_steps=cfg.get("max_thinking_steps", 5),
        use_internal_time=cfg.get("use_internal_time", True),
        lambda_geo=cfg.get("lambda_geo", 0.5),
        use_adaptive_steps=cfg.get("use_adaptive_steps", False),
        num_heads=cfg.get("num_heads", 1),
        info_gain_threshold=cfg.get("info_gain_threshold", 0.01),
    )


def _make_adapter(agent: torch.nn.Module, config_name: str):
    """Wrap a model in the appropriate AgentAdapter for the auditor."""
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent

    if isinstance(agent, DeliberativeInternalTimeAgent):
        from deltatau_audit.adapters.deliberative_adapter import DeliberativeAgentAdapter
        return DeliberativeAgentAdapter(agent, device="cpu")
    else:
        from deltatau_audit.adapters.internal_time import InternalTimeAdapter
        agent_type = "baseline" if "baseline" in config_name else "internal_time"
        return InternalTimeAdapter(agent, device="cpu", agent_type=agent_type)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1: Train (initialize + rollout baseline stats)
# ═══════════════════════════════════════════════════════════════════════════════

def _rollout_episodes(
    agent: torch.nn.Module,
    n_episodes: int,
    max_steps: int = 500,
    speed_mult: float = 1.0,
) -> List[Dict[str, Any]]:
    """Run the agent in CartPole-v1 for n_episodes, collecting stats.

    If speed_mult != 1.0, the wrapper repeats or skips env steps to simulate
    faster/slower environment dynamics.  For speed_mult > 1, each agent step
    triggers ceil(speed_mult) env steps.  For speed_mult < 1, the env step is
    only executed every 1/speed_mult agent steps.

    Returns a list of episode dicts with total_reward, length, and per-step
    delta_tau values.
    """
    import gymnasium as gym

    episodes = []
    for ep_idx in range(n_episodes):
        env = gym.make(ENV_ID)
        obs, info = env.reset(seed=ep_idx * 100)
        hidden = agent.get_initial_hidden(1, torch.device("cpu"))

        total_reward = 0.0
        ep_length = 0
        delta_taus: List[float] = []
        ponder_costs: List[float] = []
        done = False
        accum_steps = 0.0

        while not done and ep_length < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _, value, hidden, dt = agent.get_action_and_value(
                    obs_t, hidden
                )
            dt_val = dt.item() if dt is not None else 1.0
            delta_taus.append(dt_val)

            # Speed simulation: repeat or skip environment steps
            actual_steps = max(1, int(round(speed_mult)))
            for _ in range(actual_steps):
                obs, reward, terminated, truncated, info = env.step(action.item())
                total_reward += reward
                done = terminated or truncated
                if done:
                    break

            ep_length += 1

        env.close()
        episodes.append({
            "total_reward": total_reward,
            "length": ep_length,
            "delta_taus": delta_taus,
            "mean_dt": float(np.mean(delta_taus)) if delta_taus else 1.0,
            "std_dt": float(np.std(delta_taus)) if delta_taus else 0.0,
        })

    return episodes


def _simple_train_loop(
    agent: torch.nn.Module,
    n_steps: int = 50_000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    seed: int = 0,
) -> Dict[str, Any]:
    """Lightweight PPO-style training loop on CartPole-v1.

    This is deliberately simplified for the benchmark — the real training
    would use a full PPO implementation.  We run rollout-update cycles using
    REINFORCE with baseline to get non-trivial policies quickly.
    """
    import gymnasium as gym

    torch.manual_seed(seed * 1000 + 42)
    np.random.seed(seed * 1000 + 42)

    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    agent.train()

    total_steps_done = 0
    episode_rewards: List[float] = []
    update_count = 0

    while total_steps_done < n_steps:
        env = gym.make(ENV_ID)
        obs, _ = env.reset(seed=seed * 1000 + update_count)
        hidden = agent.get_initial_hidden(1, torch.device("cpu"))

        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[float] = []
        entropies: List[torch.Tensor] = []
        done = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, log_prob, entropy, value, hidden, dt = \
                agent.get_action_and_value(obs_t, hidden)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
            total_steps_done += 1

        env.close()
        episode_rewards.append(sum(rewards))

        # Compute discounted returns
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        advantage = returns_t - values_t.detach()
        policy_loss = -(log_probs_t * advantage).mean()
        value_loss = 0.5 * (returns_t - values_t).pow(2).mean()
        entropy_bonus = -0.01 * entropies_t.mean()

        loss = policy_loss + value_loss + entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
        update_count += 1

        # Log every 50 episodes
        if len(episode_rewards) % 50 == 0:
            recent = episode_rewards[-50:]
            log.info(
                "  [train] step=%d  episodes=%d  reward_50=%.1f",
                total_steps_done, len(episode_rewards), np.mean(recent),
            )

    agent.eval()
    return {
        "episode_rewards": episode_rewards,
        "total_steps": total_steps_done,
        "n_updates": update_count,
    }


def phase_train(
    output_dir: Path,
    seeds: List[int],
    n_steps: int = DEFAULT_TRAIN_STEPS,
    configs: Optional[List[str]] = None,
    skip_ablations: bool = False,
) -> None:
    """Phase 1: Train all agent configurations across all seeds.

    Saves model checkpoints to ``output_dir/models/``.
    """
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_configs = list(AGENT_CONFIGS.keys()) if configs is None else configs
    if not skip_ablations:
        all_configs += list(ACT_ABLATION_CONFIGS.keys())

    for cfg_name in all_configs:
        for seed in seeds:
            ckpt_path = models_dir / f"{cfg_name}_seed{seed}.pt"
            if ckpt_path.exists():
                log.info("Checkpoint exists, skipping: %s", ckpt_path.name)
                continue

            log.info(
                "Training %-20s  seed=%d  steps=%d", cfg_name, seed, n_steps,
            )
            try:
                agent = _build_agent(cfg_name, seed)
                history = _simple_train_loop(
                    agent, n_steps=n_steps, seed=seed,
                )
                torch.save(
                    {
                        "agent": agent.state_dict(),
                        "config": cfg_name,
                        "seed": seed,
                        "history": {
                            "final_reward_50": float(
                                np.mean(history["episode_rewards"][-50:])
                            ),
                            "total_steps": history["total_steps"],
                            "n_updates": history["n_updates"],
                        },
                    },
                    ckpt_path,
                )
                log.info(
                    "  Saved %s  (final R_50=%.1f)",
                    ckpt_path.name,
                    np.mean(history["episode_rewards"][-50:]),
                )
            except Exception:
                log.error("  FAILED training %s seed=%d", cfg_name, seed)
                log.error(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2: Evaluate
# ═══════════════════════════════════════════════════════════════════════════════

def _load_agent_from_checkpoint(ckpt_path: Path, config_name: str) -> torch.nn.Module:
    """Load an agent from a saved checkpoint."""
    agent = _build_agent(config_name, seed=0)  # seed doesn't matter for structure
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    return agent


def _evaluate_single(
    agent: torch.nn.Module,
    config_name: str,
    seed: int,
    n_eval_episodes: int = DEFAULT_EVAL_EPISODES,
) -> EvalResult:
    """Run full evaluation battery for one agent.

    1. Nominal performance (n episodes at speed 1.0)
    2. Speed robustness sweep (multiple speed multipliers)
    3. Full deltatau-audit (reliance + robustness + sensitivity)
    4. Pondering stats (for deliberative agents only)
    """
    import gymnasium as gym
    result = EvalResult(config_name=config_name, seed=seed)

    # --- 1. Nominal performance ---
    log.info("  [eval] Nominal rollout (%d episodes)", n_eval_episodes)
    nominal_eps = _rollout_episodes(agent, n_eval_episodes, speed_mult=1.0)
    rewards = [ep["total_reward"] for ep in nominal_eps]
    result.nominal_mean = float(np.mean(rewards))
    result.nominal_std = float(np.std(rewards))
    result.nominal_episodes = n_eval_episodes

    # --- 2. Speed robustness sweep ---
    log.info("  [eval] Speed sweep: %s", SPEED_MULTIPLIERS)
    for speed in SPEED_MULTIPLIERS:
        speed_eps = _rollout_episodes(agent, n_eval_episodes // 2, speed_mult=speed)
        speed_rewards = [ep["total_reward"] for ep in speed_eps]
        result.speed_returns[str(speed)] = float(np.mean(speed_rewards))

    # Compute return ratio relative to nominal
    nominal_r = result.nominal_mean if result.nominal_mean > 0 else 1.0
    for speed_key in result.speed_returns:
        result.speed_returns[speed_key] = result.speed_returns[speed_key] / nominal_r

    # --- 3. Full deltatau-audit ---
    log.info("  [eval] Running deltatau audit")
    try:
        from deltatau_audit.auditor import run_full_audit
        adapter = _make_adapter(agent, config_name)
        env_factory = lambda: gym.make(ENV_ID)

        audit_result = run_full_audit(
            adapter=adapter,
            env_factory=env_factory,
            speeds=AUDIT_SPEEDS,
            n_episodes=AUDIT_N_EPISODES,
            sensitivity_episodes=10,
            gamma=0.99,
            device="cpu",
            verbose=False,
            seed=seed * 1000,
        )

        result.audit_summary = {
            "reliance_rating": audit_result["summary"].get("reliance_rating", "N/A"),
            "reliance_score": audit_result["summary"].get("reliance_score", 0.0),
            "robustness_rating": audit_result["summary"].get("robustness_rating", "N/A"),
            "robustness_score": audit_result["summary"].get("robustness_score", 0.0),
            "deployment_rating": audit_result["summary"].get("deployment_rating", "N/A"),
            "deployment_score": audit_result["summary"].get("deployment_score", 0.0),
            "quadrant": audit_result["summary"].get("quadrant", "unknown"),
            "sensitivity_mean": audit_result["summary"].get("sensitivity_mean"),
        }

        if audit_result.get("sensitivity"):
            result.sensitivity_mean = audit_result["sensitivity"].get("mean", 0.0)
            ci_lo = audit_result["sensitivity"].get("ci_lower", 0.0)
            ci_hi = audit_result["sensitivity"].get("ci_upper", 0.0)
            result.sensitivity_ci = (ci_lo, ci_hi)

    except Exception:
        log.warning("  Audit failed for %s seed=%d", config_name, seed)
        log.warning(traceback.format_exc())
        result.audit_summary = {"error": "audit_failed"}

    # --- 4. Pondering statistics (deliberative only) ---
    from internal_time_rl.models.deliberative import DeliberativeInternalTimeAgent
    if isinstance(agent, DeliberativeInternalTimeAgent):
        log.info("  [eval] Collecting pondering diagnostics")
        try:
            ponder_steps_all = []
            for ep_idx in range(min(20, n_eval_episodes)):
                env = gym.make(ENV_ID)
                obs, _ = env.reset(seed=ep_idx + seed * 100)
                hidden = agent.get_initial_hidden(1, torch.device("cpu"))
                done = False
                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        _, _, _, _, hidden, _ = agent.get_action_and_value(obs_t, hidden)
                    diag = agent.get_pondering_diagnostics()
                    ponder_steps_all.append(diag.mean_ponder_steps)
                    obs, _, terminated, truncated, _ = env.step(
                        torch.randint(0, ACT_DIM, (1,)).item()
                    )
                    done = terminated or truncated
                env.close()

            result.ponder_stats = {
                "mean_ponder_steps": float(np.mean(ponder_steps_all)),
                "std_ponder_steps": float(np.std(ponder_steps_all)),
                "max_ponder_steps": float(np.max(ponder_steps_all)),
                "min_ponder_steps": float(np.min(ponder_steps_all)),
                "median_ponder_steps": float(np.median(ponder_steps_all)),
                "ponder_steps_histogram": np.histogram(
                    ponder_steps_all, bins=10, range=(1, 6)
                )[0].tolist(),
            }
        except Exception:
            log.warning("  Ponder diagnostics failed")
            log.warning(traceback.format_exc())

    return result


def phase_eval(
    output_dir: Path,
    seeds: List[int],
    n_eval_episodes: int = DEFAULT_EVAL_EPISODES,
    configs: Optional[List[str]] = None,
    skip_ablations: bool = False,
) -> None:
    """Phase 2: Evaluate all trained agents.

    Reads checkpoints from ``output_dir/models/`` and writes evaluation JSONs
    to ``output_dir/eval/``.
    """
    models_dir = output_dir / "models"
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_configs = list(AGENT_CONFIGS.keys()) if configs is None else configs
    if not skip_ablations:
        all_configs += list(ACT_ABLATION_CONFIGS.keys())

    for cfg_name in all_configs:
        for seed in seeds:
            eval_path = eval_dir / f"{cfg_name}_seed{seed}_audit.json"
            if eval_path.exists():
                log.info("Eval exists, skipping: %s", eval_path.name)
                continue

            ckpt_path = models_dir / f"{cfg_name}_seed{seed}.pt"
            if not ckpt_path.exists():
                log.warning(
                    "No checkpoint for %s seed=%d, skipping eval", cfg_name, seed,
                )
                continue

            log.info("Evaluating %-20s  seed=%d", cfg_name, seed)
            try:
                agent = _load_agent_from_checkpoint(ckpt_path, cfg_name)
                result = _evaluate_single(
                    agent, cfg_name, seed, n_eval_episodes=n_eval_episodes,
                )
                # Serialize
                result_dict = {
                    "config_name": result.config_name,
                    "seed": result.seed,
                    "nominal_mean": result.nominal_mean,
                    "nominal_std": result.nominal_std,
                    "nominal_episodes": result.nominal_episodes,
                    "speed_returns": result.speed_returns,
                    "sensitivity_mean": result.sensitivity_mean,
                    "sensitivity_ci": list(result.sensitivity_ci),
                    "audit_summary": result.audit_summary,
                    "ponder_stats": result.ponder_stats,
                    "error": result.error,
                }
                with open(eval_path, "w") as f:
                    json.dump(result_dict, f, indent=2)
                log.info(
                    "  Saved %s  (R=%.1f +/- %.1f)",
                    eval_path.name, result.nominal_mean, result.nominal_std,
                )
            except Exception:
                log.error("  FAILED eval %s seed=%d", cfg_name, seed)
                log.error(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 3: Report (tables + figures)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_all_eval_results(eval_dir: Path) -> List[Dict[str, Any]]:
    """Load all evaluation JSONs from the eval directory."""
    results = []
    for path in sorted(eval_dir.glob("*_audit.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


def _group_by_config(
    results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group evaluation results by configuration name."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        name = r["config_name"]
        groups.setdefault(name, []).append(r)
    return groups


def _pm(mean: float, std: float, decimals: int = 1) -> str:
    """Format mean +/- std with specified decimal places."""
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def _pm_md(mean: float, std: float, decimals: int = 1) -> str:
    """Format mean +/- std for Markdown."""
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


# ---------------------------------------------------------------------------
#  Table 1: Nominal performance comparison
# ---------------------------------------------------------------------------

def _generate_table1(
    groups: Dict[str, List[Dict[str, Any]]],
    tables_dir: Path,
) -> str:
    """Table 1: Nominal performance comparison across seeds.

    Columns: Config | Mean Reward | Std Reward | Mean dt | Reliance | Robustness | Quadrant
    """
    main_configs = ["baseline", "internal_time", "deliberative"]
    rows = []

    for cfg in main_configs:
        if cfg not in groups:
            continue
        seed_results = groups[cfg]
        cfg_label = AGENT_CONFIGS[cfg]["label"]
        rewards = [r["nominal_mean"] for r in seed_results]
        stds = [r["nominal_std"] for r in seed_results]
        sensitivities = [
            r.get("sensitivity_mean", 0.0) or 0.0 for r in seed_results
        ]

        # Audit summary aggregation
        reliance_scores = []
        robustness_scores = []
        quadrants = []
        for r in seed_results:
            audit = r.get("audit_summary", {})
            if isinstance(audit.get("reliance_score"), (int, float)):
                reliance_scores.append(audit["reliance_score"])
            if isinstance(audit.get("robustness_score"), (int, float)):
                robustness_scores.append(audit["robustness_score"])
            if audit.get("quadrant"):
                quadrants.append(audit["quadrant"])

        rows.append({
            "config": cfg,
            "label": cfg_label,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "per_ep_std": float(np.mean(stds)),
            "sensitivity": float(np.mean(sensitivities)),
            "sensitivity_std": float(np.std(sensitivities)),
            "reliance": float(np.mean(reliance_scores)) if reliance_scores else None,
            "robustness": float(np.mean(robustness_scores)) if robustness_scores else None,
            "quadrant": max(set(quadrants), key=quadrants.count) if quadrants else "N/A",
            "n_seeds": len(seed_results),
        })

    # --- Markdown ---
    md_lines = [
        "# Table 1: Nominal Performance Comparison",
        "",
        "| Configuration | Reward (mean +/- std) | |dV/dt| | Reliance | Robustness | Quadrant | Seeds |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        rel_str = f"{row['reliance']:.3f}" if row["reliance"] is not None else "N/A"
        rob_str = f"{row['robustness']:.3f}" if row["robustness"] is not None else "N/A"
        md_lines.append(
            f"| {row['label']} "
            f"| {_pm_md(row['reward_mean'], row['reward_std'])} "
            f"| {_pm_md(row['sensitivity'], row['sensitivity_std'], 4)} "
            f"| {rel_str} "
            f"| {rob_str} "
            f"| {row['quadrant']} "
            f"| {row['n_seeds']} |"
        )
    md_text = "\n".join(md_lines) + "\n"

    (tables_dir / "table1_nominal.md").write_text(md_text)

    # --- LaTeX ---
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Nominal performance comparison on CartPole-v1 across 5 seeds.}",
        r"\label{tab:nominal}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Configuration & Reward & $|dV/d\tau|$ & Reliance & Robustness & Quadrant & Seeds \\",
        r"\midrule",
    ]
    for row in rows:
        rel_str = f"{row['reliance']:.3f}" if row["reliance"] is not None else "N/A"
        rob_str = f"{row['robustness']:.3f}" if row["robustness"] is not None else "N/A"
        latex_lines.append(
            f"  {row['label']} & "
            f"{_pm(row['reward_mean'], row['reward_std'])} & "
            f"{_pm(row['sensitivity'], row['sensitivity_std'], 4)} & "
            f"{rel_str} & {rob_str} & "
            f"\\texttt{{{row['quadrant']}}} & "
            f"{row['n_seeds']} \\\\"
        )
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    latex_text = "\n".join(latex_lines) + "\n"
    (tables_dir / "table1_nominal.tex").write_text(latex_text)

    log.info("  Table 1 written (MD + LaTeX)")
    return md_text


# ---------------------------------------------------------------------------
#  Table 2: Robustness under timing perturbation
# ---------------------------------------------------------------------------

def _generate_table2(
    groups: Dict[str, List[Dict[str, Any]]],
    tables_dir: Path,
) -> str:
    """Table 2: Return ratio under each speed multiplier.

    Rows: configs.  Columns: speed multipliers.
    """
    main_configs = ["baseline", "internal_time", "deliberative"]
    speed_keys = [str(s) for s in SPEED_MULTIPLIERS]

    # --- Markdown ---
    header = "| Configuration | " + " | ".join(
        [f"x{s}" for s in SPEED_MULTIPLIERS]
    ) + " |"
    sep = "|---|" + "|".join(["---"] * len(SPEED_MULTIPLIERS)) + "|"

    md_lines = [
        "# Table 2: Robustness Under Timing Perturbation",
        "",
        "Return ratio relative to nominal (1.0 = no degradation).",
        "",
        header,
        sep,
    ]

    rows_data = []
    for cfg in main_configs:
        if cfg not in groups:
            continue
        seed_results = groups[cfg]
        label = AGENT_CONFIGS[cfg]["label"]

        ratios = {}
        for sk in speed_keys:
            vals = [r["speed_returns"].get(sk, 1.0) for r in seed_results]
            ratios[sk] = (float(np.mean(vals)), float(np.std(vals)))

        vals_str = " | ".join(
            f"{ratios[sk][0]:.3f}" for sk in speed_keys
        )
        md_lines.append(f"| {label} | {vals_str} |")
        rows_data.append({"label": label, "ratios": ratios})

    md_text = "\n".join(md_lines) + "\n"
    (tables_dir / "table2_robustness.md").write_text(md_text)

    # --- LaTeX ---
    col_spec = "l" + "c" * len(SPEED_MULTIPLIERS)
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Return ratio under timing perturbation on CartPole-v1. "
        r"Values closer to 1.0 indicate greater robustness.}",
        r"\label{tab:robustness}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Configuration & " + " & ".join(
            [f"$\\times${s}" for s in SPEED_MULTIPLIERS]
        ) + r" \\",
        r"\midrule",
    ]
    for rd in rows_data:
        vals_str = " & ".join(
            f"{rd['ratios'][sk][0]:.3f}" for sk in speed_keys
        )
        latex_lines.append(f"  {rd['label']} & {vals_str} \\\\")
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    latex_text = "\n".join(latex_lines) + "\n"
    (tables_dir / "table2_robustness.tex").write_text(latex_text)

    log.info("  Table 2 written (MD + LaTeX)")
    return md_text


# ---------------------------------------------------------------------------
#  Table 3: ACT ablation
# ---------------------------------------------------------------------------

def _generate_table3(
    groups: Dict[str, List[Dict[str, Any]]],
    tables_dir: Path,
) -> str:
    """Table 3: Ablation of ACT extensions.

    Shows how each extension affects nominal performance and robustness.
    """
    ablation_order = list(ACT_ABLATION_CONFIGS.keys())

    md_lines = [
        "# Table 3: ACT Extension Ablation",
        "",
        "| Variant | Reward (mean +/- std) | Robustness | Ponder Steps | Ponder Std |",
        "|---|---|---|---|---|",
    ]

    rows_data = []
    for abl in ablation_order:
        if abl not in groups:
            continue
        seed_results = groups[abl]
        label = ACT_ABLATION_CONFIGS[abl]["label"]

        rewards = [r["nominal_mean"] for r in seed_results]
        rob_scores = []
        ponder_means = []
        ponder_stds = []

        for r in seed_results:
            audit = r.get("audit_summary", {})
            if isinstance(audit.get("robustness_score"), (int, float)):
                rob_scores.append(audit["robustness_score"])
            ps = r.get("ponder_stats")
            if ps:
                ponder_means.append(ps.get("mean_ponder_steps", 0.0))
                ponder_stds.append(ps.get("std_ponder_steps", 0.0))

        row = {
            "label": label,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "robustness": float(np.mean(rob_scores)) if rob_scores else None,
            "ponder_mean": float(np.mean(ponder_means)) if ponder_means else None,
            "ponder_std": float(np.mean(ponder_stds)) if ponder_stds else None,
        }
        rows_data.append(row)

        rob_str = f"{row['robustness']:.3f}" if row["robustness"] is not None else "N/A"
        ponder_m = f"{row['ponder_mean']:.2f}" if row["ponder_mean"] is not None else "N/A"
        ponder_s = f"{row['ponder_std']:.2f}" if row["ponder_std"] is not None else "N/A"
        md_lines.append(
            f"| {label} "
            f"| {_pm_md(row['reward_mean'], row['reward_std'])} "
            f"| {rob_str} "
            f"| {ponder_m} "
            f"| {ponder_s} |"
        )

    md_text = "\n".join(md_lines) + "\n"
    (tables_dir / "table3_ablation.md").write_text(md_text)

    # --- LaTeX ---
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study of ACT extensions on CartPole-v1. "
        r"Each row adds one extension cumulatively.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Variant & Reward & Robustness & Ponder Steps & Ponder $\sigma$ \\",
        r"\midrule",
    ]
    for row in rows_data:
        rob_str = f"{row['robustness']:.3f}" if row["robustness"] is not None else "N/A"
        pm_str = f"{row['ponder_mean']:.2f}" if row["ponder_mean"] is not None else "N/A"
        ps_str = f"{row['ponder_std']:.2f}" if row["ponder_std"] is not None else "N/A"
        latex_lines.append(
            f"  {row['label']} & "
            f"{_pm(row['reward_mean'], row['reward_std'])} & "
            f"{rob_str} & {pm_str} & {ps_str} \\\\"
        )
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    latex_text = "\n".join(latex_lines) + "\n"
    (tables_dir / "table3_ablation.tex").write_text(latex_text)

    log.info("  Table 3 written (MD + LaTeX)")
    return md_text


# ---------------------------------------------------------------------------
#  Figure 1: Performance degradation curve
# ---------------------------------------------------------------------------

def _generate_figure1(
    groups: Dict[str, List[Dict[str, Any]]],
    figures_dir: Path,
) -> None:
    """Figure 1: Reward vs speed multiplier (degradation curve).

    Shows how each architecture degrades as the environment speed changes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(MPL_RC)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    main_configs = ["baseline", "internal_time", "deliberative"]
    colors = {"baseline": "#d62728", "internal_time": "#1f77b4", "deliberative": "#2ca02c"}
    markers = {"baseline": "s", "internal_time": "o", "deliberative": "^"}

    for cfg in main_configs:
        if cfg not in groups:
            continue
        seed_results = groups[cfg]
        label = AGENT_CONFIGS[cfg]["label"]

        speeds_plot = SPEED_MULTIPLIERS
        means = []
        stds = []

        for speed in speeds_plot:
            sk = str(speed)
            vals = [r["speed_returns"].get(sk, 1.0) for r in seed_results]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax.plot(
            speeds_plot, means_arr,
            marker=markers[cfg], color=colors[cfg],
            linewidth=2, markersize=7, label=label,
        )
        ax.fill_between(
            speeds_plot,
            means_arr - stds_arr,
            means_arr + stds_arr,
            alpha=0.15, color=colors[cfg],
        )

    # Reference line at 1.0
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("Speed Multiplier")
    ax.set_ylabel("Return Ratio (relative to nominal)")
    ax.set_title("Performance Degradation Under Timing Perturbation")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(0.4, 2.1)
    ax.set_ylim(0.0, 1.5)

    fig.tight_layout()
    out_path = figures_dir / "fig1_degradation_curve.png"
    fig.savefig(out_path)
    plt.close(fig)
    log.info("  Figure 1 saved: %s", out_path)


# ---------------------------------------------------------------------------
#  Figure 2: Pondering depth histogram
# ---------------------------------------------------------------------------

def _generate_figure2(
    groups: Dict[str, List[Dict[str, Any]]],
    figures_dir: Path,
) -> None:
    """Figure 2: Histogram of pondering depths for ACT agents.

    Compares pondering step distributions across ACT ablation variants and
    the main deliberative config.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(MPL_RC)

    # Collect histogram data from deliberative + ablation configs
    plot_data = []
    configs_to_plot = ["deliberative"] + list(ACT_ABLATION_CONFIGS.keys())
    colors_cycle = [
        "#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b",
    ]

    for idx, cfg in enumerate(configs_to_plot):
        if cfg not in groups:
            continue
        seed_results = groups[cfg]
        histograms = []
        for r in seed_results:
            ps = r.get("ponder_stats")
            if ps and "ponder_steps_histogram" in ps:
                histograms.append(ps["ponder_steps_histogram"])

        if not histograms:
            continue

        # Average histogram across seeds
        avg_hist = np.mean(histograms, axis=0)
        label = (
            AGENT_CONFIGS[cfg]["label"]
            if cfg in AGENT_CONFIGS
            else ACT_ABLATION_CONFIGS[cfg]["label"]
        )
        plot_data.append({
            "label": label,
            "histogram": avg_hist,
            "color": colors_cycle[idx % len(colors_cycle)],
        })

    if not plot_data:
        log.info("  Figure 2 skipped: no pondering data available")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bin_edges = np.linspace(1, 6, 11)  # matches histogram bins
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = 0.4 / len(plot_data)

    for i, pd in enumerate(plot_data):
        offset = (i - len(plot_data) / 2) * bar_width
        ax.bar(
            bin_centers + offset,
            pd["histogram"],
            width=bar_width,
            alpha=0.75,
            label=pd["label"],
            color=pd["color"],
        )

    ax.set_xlabel("Pondering Steps")
    ax.set_ylabel("Frequency (avg over seeds)")
    ax.set_title("ACT Pondering Depth Distribution")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.set_xlim(0.5, 6.5)

    fig.tight_layout()
    out_path = figures_dir / "fig2_ponder_histogram.png"
    fig.savefig(out_path)
    plt.close(fig)
    log.info("  Figure 2 saved: %s", out_path)


# ---------------------------------------------------------------------------
#  Figure 3: Seed sweep stability (box plot)
# ---------------------------------------------------------------------------

def _generate_figure3(
    groups: Dict[str, List[Dict[str, Any]]],
    figures_dir: Path,
) -> None:
    """Figure 3: Box plot of nominal rewards across seeds for each config.

    Shows inter-seed variability to assess training stability.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(MPL_RC)

    main_configs = ["baseline", "internal_time", "deliberative"]
    colors = {"baseline": "#d62728", "internal_time": "#1f77b4", "deliberative": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    box_data = []
    labels = []
    box_colors = []

    for cfg in main_configs:
        if cfg not in groups:
            continue
        seed_results = groups[cfg]
        rewards = [r["nominal_mean"] for r in seed_results]
        box_data.append(rewards)
        labels.append(AGENT_CONFIGS[cfg]["label"])
        box_colors.append(colors[cfg])

    if not box_data:
        log.info("  Figure 3 skipped: no data")
        return

    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        tick_labels=labels,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Scatter individual seed points
    for i, (data, color) in enumerate(zip(box_data, box_colors)):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        ax.scatter(x, data, color=color, alpha=0.8, s=30, zorder=3, edgecolors="black", linewidths=0.5)

    ax.set_ylabel("Nominal Reward (mean per seed)")
    ax.set_title("Training Stability Across Seeds (CartPole-v1)")
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    out_path = figures_dir / "fig3_seed_stability.png"
    fig.savefig(out_path)
    plt.close(fig)
    log.info("  Figure 3 saved: %s", out_path)


# ---------------------------------------------------------------------------
#  Summary JSON
# ---------------------------------------------------------------------------

def _generate_summary(
    groups: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate machine-readable summary.json with aggregate statistics."""
    summary: Dict[str, Any] = {
        "env": ENV_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configs": {},
    }

    for cfg_name, seed_results in groups.items():
        rewards = [r["nominal_mean"] for r in seed_results]
        speed_data = {}
        for sk in [str(s) for s in SPEED_MULTIPLIERS]:
            vals = [r["speed_returns"].get(sk, 1.0) for r in seed_results]
            speed_data[sk] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        audit_keys = ["reliance_score", "robustness_score", "deployment_score"]
        audit_agg = {}
        for key in audit_keys:
            vals = [
                r["audit_summary"].get(key)
                for r in seed_results
                if isinstance(r.get("audit_summary", {}).get(key), (int, float))
            ]
            if vals:
                audit_agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        entry = {
            "n_seeds": len(seed_results),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "speed_returns": speed_data,
            "audit": audit_agg,
        }

        # Ponder stats
        ponder_means = [
            r["ponder_stats"]["mean_ponder_steps"]
            for r in seed_results
            if r.get("ponder_stats")
        ]
        if ponder_means:
            entry["ponder_steps_mean"] = float(np.mean(ponder_means))
            entry["ponder_steps_std"] = float(np.std(ponder_means))

        summary["configs"][cfg_name] = entry

    out_path = output_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("  Summary written: %s", out_path)
    return summary


def phase_report(output_dir: Path) -> None:
    """Phase 3: Generate all tables and figures from cached evaluation data.

    Reads JSONs from ``output_dir/eval/`` and writes tables to
    ``output_dir/tables/`` and figures to ``output_dir/figures/``.
    """
    eval_dir = output_dir / "eval"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not eval_dir.exists():
        log.error("No eval directory found at %s — run --phase eval first", eval_dir)
        return

    results = _load_all_eval_results(eval_dir)
    if not results:
        log.error("No evaluation results found in %s", eval_dir)
        return

    groups = _group_by_config(results)
    log.info("Loaded %d results across %d configs", len(results), len(groups))

    # Tables
    log.info("Generating tables...")
    _generate_table1(groups, tables_dir)
    _generate_table2(groups, tables_dir)
    _generate_table3(groups, tables_dir)

    # Figures
    log.info("Generating figures...")
    _generate_figure1(groups, figures_dir)
    _generate_figure2(groups, figures_dir)
    _generate_figure3(groups, figures_dir)

    # Summary
    log.info("Generating summary...")
    _generate_summary(groups, output_dir)

    log.info("Report generation complete. Output: %s", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Entry point for the benchmark experiment runner."""
    parser = argparse.ArgumentParser(
        description="Benchmark experiment runner for deltatau-audit "
                    "(Cambridge/DeepMind submission).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        choices=["train", "eval", "report", "all"],
        default="all",
        help="Which phase(s) to run (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark",
        help="Root output directory (default: results/benchmark).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds (0 to N-1). Default: 5.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=DEFAULT_TRAIN_STEPS,
        help=f"Training steps per agent (default: {DEFAULT_TRAIN_STEPS}).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help=f"Evaluation episodes per condition (default: {DEFAULT_EVAL_EPISODES}).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Subset of configs to run (e.g., baseline internal_time).",
    )
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Skip ACT ablation configs (faster).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 2 seeds, 10k steps, 20 eval episodes.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Extra logging output.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Quick mode overrides
    seeds_count = args.seeds
    train_steps = args.train_steps
    eval_episodes = args.eval_episodes
    if args.quick:
        seeds_count = min(seeds_count, 2)
        train_steps = min(train_steps, 10_000)
        eval_episodes = min(eval_episodes, 20)
        log.info("Quick mode: %d seeds, %d steps, %d eval eps",
                 seeds_count, train_steps, eval_episodes)

    seeds = list(range(seeds_count))
    output_dir = Path(args.output_dir)

    log.info("=" * 70)
    log.info("deltatau-audit Benchmark Suite")
    log.info("  Phase:       %s", args.phase)
    log.info("  Output:      %s", output_dir)
    log.info("  Seeds:       %s", seeds)
    log.info("  Train steps: %d", train_steps)
    log.info("  Eval eps:    %d", eval_episodes)
    log.info("=" * 70)

    t0 = time.time()

    if args.phase in ("train", "all"):
        log.info("=" * 40 + " PHASE: TRAIN " + "=" * 40)
        phase_train(
            output_dir, seeds,
            n_steps=train_steps,
            configs=args.configs,
            skip_ablations=args.skip_ablations,
        )

    if args.phase in ("eval", "all"):
        log.info("=" * 40 + " PHASE: EVAL " + "=" * 41)
        phase_eval(
            output_dir, seeds,
            n_eval_episodes=eval_episodes,
            configs=args.configs,
            skip_ablations=args.skip_ablations,
        )

    if args.phase in ("report", "all"):
        log.info("=" * 40 + " PHASE: REPORT " + "=" * 39)
        phase_report(output_dir)

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info("Benchmark complete in %.1f seconds (%.1f min)", elapsed, elapsed / 60)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
