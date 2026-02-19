"""Core auditor: 2-axis evaluation engine.

Axis 1 — Timing Reliance (intervention ablation):
    Tampers with the agent's internal Δτ to measure causal dependence
    on internal time representation. High reliance = timing channel works.

Axis 2 — Timing Robustness (env wrappers):
    Wraps the environment with realistic timing perturbations (jitter,
    delay, speed changes) to measure operational resilience.

Bonus — Temporal Sensitivity:
    Finite-difference |dV/dτ| measuring value function's local
    sensitivity to internal time — the "timing Jacobian".
"""

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from typing import Any, Dict, List, Optional

import gymnasium as gym

# Optional tqdm for episode progress bars
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from .adapters.base import AgentAdapter
from .wrappers.speed import FixedSpeedWrapper, JitterWrapper, PiecewiseSwitchWrapper
from .wrappers.latency import ObservationDelayWrapper, ObsNoiseWrapper
from .metrics import (
    compute_discounted_returns,
    compute_value_rmse,
    compute_value_bias,
    compute_value_mae,
    aggregate_episode_metrics,
    compute_degradation,
    compute_return_ratio,
    bootstrap_return_ratio,
    reliance_rating,
    robustness_rating,
    severity_rating,
)


# ── Labels ────────────────────────────────────────────────────────────

INTERVENTIONS = {
    "none": "Normal (learned Δτ)",
    "clamp_1": "Δτ clamped to 1.0",
    "reverse": "Δτ reversed (2.0 − learned)",
    "random": "Δτ ~ Uniform(0.5, 1.5)",
}

ROBUSTNESS_SCENARIOS = {
    "nominal": "Nominal (speed=1, no wrapper)",
    "speed_5x": "5× speed (unseen frequency)",
    "jitter": "Speed jitter (2 ± 1)",
    "delay": "Observation delay (1 step)",
    "spike": "Mid-episode speed spike (1→5→1)",
    "obs_noise": "Observation noise (σ=0.1)",
}

# Deployment = realistic conditions an agent might face
DEPLOYMENT_SCENARIOS = ["jitter", "delay", "spike", "obs_noise"]
# Stress = extreme conditions for stress testing
STRESS_SCENARIOS = ["speed_5x"]


# ── Parallel episode runner ───────────────────────────────────────────

def _run_episodes_parallel(
    adapter: "AgentAdapter",
    env_factory: callable,
    scenario: str,
    intervention: str,
    n_episodes: int,
    gamma: float,
    device: str,
    seed: Optional[int],
    n_workers: int,
    label: str,
    verbose: bool,
    seed_offset: int = 0,
) -> List[Dict]:
    """Run n_episodes, optionally in parallel via ThreadPoolExecutor.

    Each episode gets its own env (created via env_factory) and its own
    hidden state, so there is no shared mutable state between threads.
    PyTorch forward-pass over shared read-only weights is thread-safe.

    Args:
        n_workers: Number of parallel threads. 1 = serial (default).
        seed_offset: Added to per-episode seed to keep scenarios distinct.
    """
    def _one(ep_idx: int) -> Dict:
        ep_seed = (None if seed is None
                   else seed + seed_offset + ep_idx)
        env = _make_wrapped_env(env_factory, scenario)
        ep = _run_single_episode(adapter, env, intervention,
                                 gamma, device, seed=ep_seed)
        env.close()
        return ep

    if n_workers <= 1 or n_episodes <= 1:
        # Serial path with tqdm
        bar = _episode_iter(n_episodes, label, verbose)
        results = []
        for ep_idx in bar:
            ep = _one(ep_idx)
            results.append(ep)
            if _HAS_TQDM and verbose and hasattr(bar, "set_postfix"):
                bar.set_postfix(R=f"{ep['total_reward']:.1f}")
        if not (_HAS_TQDM and verbose) and verbose:
            print()
        return results

    # Parallel path
    if _HAS_TQDM and verbose:
        bar = _tqdm(total=n_episodes,
                    desc=f"    {label:<28}", ncols=72, leave=True)
    elif verbose:
        print(f"    {label}...", end="", flush=True)

    results = [None] * n_episodes
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_one, i): i
                   for i in range(n_episodes)}
        for future in as_completed(futures):
            idx = futures[future]
            ep = future.result()
            results[idx] = ep
            if _HAS_TQDM and verbose:
                bar.update(1)
                bar.set_postfix(R=f"{ep['total_reward']:.1f}")

    if _HAS_TQDM and verbose:
        bar.close()
    elif verbose:
        print()

    return results


# ── Episode iterator (serial path helper) ─────────────────────────────

def _episode_iter(n: int, label: str, verbose: bool):
    """Return an iterator for n episodes, with tqdm bar if available."""
    if _HAS_TQDM and verbose:
        return _tqdm(range(n), desc=f"    {label:<28}", ncols=72, leave=True)
    else:
        if verbose:
            print(f"    {label}...", end="", flush=True)
        return range(n)


def _run_single_episode(
    adapter: AgentAdapter,
    env: gym.Env,
    intervention: str = "none",
    gamma: float = 0.99,
    device: str = "cpu",
    seed: Optional[int] = None,
    max_steps: int = 10_000,
) -> Dict:
    """Run one episode and collect value/return data.

    Args:
        max_steps: Hard cap on episode length to prevent infinite loops.
                   Episodes exceeding this are truncated with a warning.
        seed: If provided, passed to env.reset(seed=seed) for reproducibility.
    """
    reset_kwargs = {"seed": seed} if seed is not None else {}
    obs, info = env.reset(**reset_kwargs)
    hidden = adapter.reset_hidden(1, device)
    done = False
    n_steps = 0

    step_values = []
    step_rewards = []
    step_dts = []

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        action, value, hidden_new, dt = adapter.act(obs_t, hidden)

        # Apply intervention if supported
        if intervention != "none" and adapter.supports_intervention:
            if intervention == "clamp_1":
                target_dt = 1.0
            elif intervention == "reverse":
                target_dt = 2.0 - (dt if dt is not None else 1.0)
                target_dt = max(0.3, min(2.5, target_dt))
            elif intervention == "random":
                target_dt = float(np.random.uniform(0.5, 1.5))
            else:
                target_dt = 1.0

            hidden_new = adapter.rerun_with_dt(obs_t, hidden, target_dt)

            if adapter.supports_value_recompute:
                value = adapter.recompute_value(hidden_new)

        step_values.append(value)
        step_dts.append(dt)
        hidden = hidden_new

        obs, reward, term, trunc, info = env.step(action)
        step_rewards.append(reward)
        done = term or trunc
        n_steps += 1

        if n_steps >= max_steps and not done:
            warnings.warn(
                f"Episode exceeded max_steps={max_steps}. "
                "Truncating. Check env or wrapper for infinite loops.",
                RuntimeWarning, stacklevel=3,
            )
            done = True

    returns = compute_discounted_returns(step_rewards, gamma)

    return {
        "rmse": compute_value_rmse(step_values, returns),
        "mae": compute_value_mae(step_values, returns),
        "bias": compute_value_bias(step_values, returns),
        "total_reward": sum(step_rewards),
        "length": len(step_rewards),
        "dt_mean": float(np.mean([d for d in step_dts if d is not None]))
                   if any(d is not None for d in step_dts) else None,
    }


# ══════════════════════════════════════════════════════════════════════
# AXIS 1: RELIANCE AUDIT
# ══════════════════════════════════════════════════════════════════════

def run_reliance_audit(
    adapter: AgentAdapter,
    env_factory: callable,
    speeds: List[int] = None,
    n_episodes: int = 50,
    interventions: List[str] = None,
    gamma: float = 0.99,
    device: str = "cpu",
    verbose: bool = True,
    seed: Optional[int] = None,
    n_workers: int = 1,
) -> Dict:
    """Axis 1: Intervention ablation at multiple speeds.

    Tampers with the agent's internal Δτ to test whether the value
    function causally depends on the timing representation.

    Returns:
        Dict with per_speed results, degradation, score, rating, worst_case.
    """
    if speeds is None:
        speeds = [1, 2, 3, 5, 8]

    # No intervention support → Reliance is N/A
    if not adapter.supports_intervention:
        if verbose:
            print("  Reliance Test: N/A (no intervention support)")
        return {
            "per_speed": {},
            "degradation": {},
            "score": None,
            "rating": "N/A",
            "worst_case": {
                "speed": None,
                "intervention": None,
                "rmse_ratio": None,
                "percent": None,
            },
        }

    if interventions is None:
        interventions = ["none", "clamp_1", "reverse", "random"]

    if verbose:
        print("  Reliance Test (intervention ablation)")

    per_speed = {}
    for speed in speeds:
        results = {}
        for interv in interventions:
            if interv != "none" and not adapter.supports_intervention:
                continue

            label = f"speed={speed} [{interv}]"

            # Build a factory that also applies the speed wrapper
            if speed > 1:
                def _speed_factory(s=speed):
                    return FixedSpeedWrapper(env_factory(), speed=s)
                _factory = _speed_factory
            else:
                _factory = env_factory

            episodes = _run_episodes_parallel(
                adapter, _factory, "nominal", interv,
                n_episodes, gamma, device, seed, n_workers,
                label, verbose,
                seed_offset=speed * 1000,
            )

            agg = aggregate_episode_metrics(episodes)
            results[interv] = agg
            if not (_HAS_TQDM and verbose) and verbose:
                r = agg.get("total_reward_mean", 0)
                rmse = agg.get("rmse_mean", 0)
                print(f" R={r:.3f}, RMSE={rmse:.4f}")

        per_speed[str(speed)] = results

    # Compute reliance summary
    worst_ratio = 1.0
    worst_speed = None
    worst_interv = None
    degradation = {}

    for interv in ["clamp_1", "reverse", "random"]:
        deg_by_speed = {}
        for s in speeds:
            sr = per_speed[str(s)]
            if "none" in sr and interv in sr:
                base = sr["none"].get("rmse_mean", 0)
                test = sr[interv].get("rmse_mean", 0)
                deg = compute_degradation(base, test)
                deg["severity"] = severity_rating(deg["percent_increase"])
                deg_by_speed[str(s)] = deg

                if deg["ratio"] > worst_ratio:
                    worst_ratio = deg["ratio"]
                    worst_speed = str(s)
                    worst_interv = interv

        if deg_by_speed:
            degradation[interv] = deg_by_speed

    rating = reliance_rating(worst_ratio)

    if verbose:
        from .color import colored_rating, dim
        print(f"    -> Reliance: {colored_rating(rating)} "
              f"{dim(f'(worst RMSE ratio: {worst_ratio:.2f}x)')}")

    return {
        "per_speed": per_speed,
        "degradation": degradation,
        "score": worst_ratio,
        "rating": rating,
        "worst_case": {
            "speed": worst_speed,
            "intervention": worst_interv,
            "rmse_ratio": worst_ratio,
            "percent": (worst_ratio - 1) * 100,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# AXIS 2: ROBUSTNESS AUDIT
# ══════════════════════════════════════════════════════════════════════

def _make_wrapped_env(env_factory, scenario: str):
    """Create a wrapped env for a robustness scenario."""
    env = env_factory()
    if scenario == "nominal":
        return env
    elif scenario == "speed_5x":
        return FixedSpeedWrapper(env, speed=5)
    elif scenario == "jitter":
        return JitterWrapper(env, base_speed=2, jitter=1)
    elif scenario == "delay":
        return ObservationDelayWrapper(env, delay=1)
    elif scenario == "spike":
        return PiecewiseSwitchWrapper(
            env, schedule=[(0, 1), (20, 5), (40, 1)])
    elif scenario == "obs_noise":
        return ObsNoiseWrapper(env, std=0.1)
    else:
        raise ValueError(f"Unknown robustness scenario: {scenario}")


def run_robustness_audit(
    adapter: AgentAdapter,
    env_factory: callable,
    scenarios: List[str] = None,
    n_episodes: int = 50,
    gamma: float = 0.99,
    device: str = "cpu",
    verbose: bool = True,
    seed: Optional[int] = None,
    n_workers: int = 1,
) -> Dict:
    """Axis 2: Realistic timing perturbations via env wrappers.

    The agent runs NORMALLY (no internal intervention). Only the
    environment is perturbed. Measures whether performance holds
    under deployment-realistic timing conditions.

    Returns:
        Dict with scenarios, per_scenario_scores, return_score, rating, worst_case.
    """
    if scenarios is None:
        scenarios = list(ROBUSTNESS_SCENARIOS.keys())

    if "nominal" not in scenarios:
        scenarios = ["nominal"] + list(scenarios)

    if verbose:
        print("  Robustness Test (env wrappers)")

    scenario_results = {}
    scenario_episode_returns = {}  # raw per-episode returns for bootstrap
    for sc_idx, scenario in enumerate(scenarios):
        label = ROBUSTNESS_SCENARIOS.get(scenario, scenario)

        episodes = _run_episodes_parallel(
            adapter, env_factory, scenario, "none",
            n_episodes, gamma, device, seed, n_workers,
            label, verbose,
            seed_offset=sc_idx * 1000,
        )

        agg = aggregate_episode_metrics(episodes)
        scenario_results[scenario] = agg
        scenario_episode_returns[scenario] = [
            ep["total_reward"] for ep in episodes
        ]

        if not (_HAS_TQDM and verbose) and verbose:
            r = agg.get("total_reward_mean", 0)
            rmse = agg.get("rmse_mean", 0)
            print(f" R={r:.3f}, RMSE={rmse:.4f}")

    # Compute robustness scores
    nominal = scenario_results["nominal"]
    nominal_return = nominal.get("total_reward_mean", 0)
    nominal_rmse = nominal.get("rmse_mean", 0)

    worst_return_ratio = 1.0
    worst_rmse_ratio = 1.0
    worst_scenario = None
    per_scenario_scores = {}

    nominal_ep_returns = scenario_episode_returns.get("nominal", [])

    for scenario in scenarios:
        if scenario == "nominal":
            continue

        s_result = scenario_results[scenario]
        s_return = s_result.get("total_reward_mean", 0)
        s_rmse = s_result.get("rmse_mean", 0)

        ret_ratio = compute_return_ratio(nominal_return, s_return)
        rmse_ratio = s_rmse / nominal_rmse if nominal_rmse > 1e-10 else 1.0

        # Bootstrap CI for return ratio
        pert_ep_returns = scenario_episode_returns.get(scenario, [])
        bci = bootstrap_return_ratio(nominal_ep_returns, pert_ep_returns)

        per_scenario_scores[scenario] = {
            "return_ratio": ret_ratio,
            "return_drop_pct": (1 - ret_ratio) * 100,
            "rmse_ratio": rmse_ratio,
            "rmse_increase_pct": (rmse_ratio - 1) * 100,
            "ci_lower": bci["ci_lower"],
            "ci_upper": bci["ci_upper"],
            "significant": bci["significant"],
        }

        if ret_ratio < worst_return_ratio:
            worst_return_ratio = ret_ratio
            worst_scenario = scenario

        if rmse_ratio > worst_rmse_ratio:
            worst_rmse_ratio = rmse_ratio

    rating = robustness_rating(worst_return_ratio)

    # Compute deployment vs stress sub-scores
    def _sub_score(scenario_list):
        w_ret, w_rmse, w_sc = 1.0, 1.0, None
        for sc_name in scenario_list:
            if sc_name not in per_scenario_scores:
                continue
            sc = per_scenario_scores[sc_name]
            if sc["return_ratio"] < w_ret:
                w_ret = sc["return_ratio"]
                w_sc = sc_name
            if sc["rmse_ratio"] > w_rmse:
                w_rmse = sc["rmse_ratio"]
        return {
            "return_score": w_ret,
            "rmse_score": w_rmse,
            "rating": robustness_rating(w_ret),
            "worst_case": {
                "scenario": w_sc,
                "return_ratio": w_ret,
                "return_drop_pct": (1 - w_ret) * 100,
            },
        }

    deployment = _sub_score(DEPLOYMENT_SCENARIOS)
    stress = _sub_score(STRESS_SCENARIOS)

    if verbose:
        from .color import colored_rating, dim
        drop = (1 - worst_return_ratio) * 100
        dep_wc = deployment["worst_case"]
        str_wc = stress["worst_case"]
        print(f"    -> Overall:    {colored_rating(rating)} "
              f"{dim('(worst return drop: ' + f'{drop:.1f}%)')}")
        dep_detail = (f"(worst: {dep_wc['scenario']}, "
                      f"drop: {dep_wc['return_drop_pct']:.1f}%)")
        print(f"    -> Deployment: {colored_rating(deployment['rating'])} "
              f"{dim(dep_detail)}")
        str_detail = (f"(worst: {str_wc['scenario']}, "
                      f"drop: {str_wc['return_drop_pct']:.1f}%)")
        print(f"    -> Stress:     {colored_rating(stress['rating'])} "
              f"{dim(str_detail)}")
        # Show bootstrap CIs
        sig_count = sum(1 for s in per_scenario_scores.values()
                        if s.get("significant"))
        total = len(per_scenario_scores)
        print(f"    -> {sig_count}/{total} scenarios with "
              f"statistically significant drop (95% CI)")

    return {
        "scenarios": scenario_results,
        "per_scenario_scores": per_scenario_scores,
        "deployment": deployment,
        "stress": stress,
        "return_score": worst_return_ratio,
        "rmse_score": worst_rmse_ratio,
        "rating": rating,
        "worst_case": {
            "scenario": worst_scenario,
            "return_ratio": worst_return_ratio,
            "return_drop_pct": (1 - worst_return_ratio) * 100,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# TEMPORAL SENSITIVITY: |dV/dτ|
# ══════════════════════════════════════════════════════════════════════

def compute_temporal_sensitivity(
    adapter: AgentAdapter,
    env_factory: callable,
    speeds: List[int] = None,
    n_episodes: int = 20,
    epsilon: float = 0.1,
    gamma: float = 0.99,
    device: str = "cpu",
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Optional[Dict]:
    """Compute temporal sensitivity: |dV/dτ| via finite difference.

    S = E[ |V(τ+ε) − V(τ−ε)| / (2ε) ]

    This is the "timing Jacobian" — how sensitive the value function
    is to small perturbations in the internal time representation.
    High sensitivity at unseen speeds suggests active adaptation.

    Returns None if the agent doesn't support intervention.
    """
    if not adapter.supports_intervention or not adapter.supports_value_recompute:
        if verbose:
            print("  Temporal Sensitivity: skipped (no intervention support)")
        return None

    if speeds is None:
        speeds = [1, 3, 5]

    if verbose:
        print("  Temporal Sensitivity (|dV/dt|)")

    per_speed = {}
    all_sensitivities = []

    for speed in speeds:
        if verbose:
            print(f"    Speed {speed}...", end="", flush=True)

        speed_sensitivities = []
        for _ in range(n_episodes):
            env = env_factory()
            if speed > 1:
                env = FixedSpeedWrapper(env, speed=speed)

            obs, _ = env.reset()
            hidden = adapter.reset_hidden(1, device)
            done = False

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32)
                action, value, hidden_new, dt = adapter.act(obs_t, hidden)

                if dt is not None:
                    dt_plus = min(2.5, dt + epsilon)
                    dt_minus = max(0.3, dt - epsilon)
                    actual_2eps = dt_plus - dt_minus

                    if actual_2eps > 1e-6:
                        h_plus = adapter.rerun_with_dt(obs_t, hidden, dt_plus)
                        h_minus = adapter.rerun_with_dt(obs_t, hidden, dt_minus)
                        v_plus = adapter.recompute_value(h_plus)
                        v_minus = adapter.recompute_value(h_minus)

                        sens = abs(v_plus - v_minus) / actual_2eps
                        speed_sensitivities.append(sens)

                hidden = hidden_new
                obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
            env.close()

        if speed_sensitivities:
            mean_s = float(np.mean(speed_sensitivities))
            per_speed[str(speed)] = {
                "mean": mean_s,
                "std": float(np.std(speed_sensitivities)),
                "n_samples": len(speed_sensitivities),
            }
            all_sensitivities.extend(speed_sensitivities)
            if verbose:
                print(f" S={mean_s:.4f}")
        else:
            if verbose:
                print(" (no dt samples)")

    if not all_sensitivities:
        return None

    result = {
        "mean": float(np.mean(all_sensitivities)),
        "std": float(np.std(all_sensitivities)),
        "median": float(np.median(all_sensitivities)),
        "n_samples": len(all_sensitivities),
        "per_speed": per_speed,
    }

    if verbose:
        print(f"    -> Mean sensitivity: {result['mean']:.4f}")

    return result


# ══════════════════════════════════════════════════════════════════════
# FULL 2-AXIS AUDIT
# ══════════════════════════════════════════════════════════════════════

def run_full_audit(
    adapter: AgentAdapter,
    env_factory: callable,
    speeds: List[int] = None,
    n_episodes: int = 50,
    interventions: List[str] = None,
    robustness_scenarios: List[str] = None,
    sensitivity_episodes: int = 20,
    gamma: float = 0.99,
    device: str = "cpu",
    verbose: bool = True,
    seed: Optional[int] = None,
    n_workers: int = 1,
) -> Dict:
    """Run the complete 2-axis time robustness audit.

    Axis 1 — Reliance: intervention ablation
    Axis 2 — Robustness: env wrappers
    Bonus  — Sensitivity: |dV/dτ| finite difference

    Returns structured dict ready for report generation.
    """
    if speeds is None:
        speeds = [1, 2, 3, 5, 8]

    if verbose:
        print(f"Time Robustness Audit (2-axis)")
        print(f"  Speeds: {speeds}")
        print(f"  Episodes per condition: {n_episodes}")
        print(f"  Intervention support: {adapter.supports_intervention}")
        print()

    # Axis 1: Reliance
    reliance = run_reliance_audit(
        adapter, env_factory, speeds=speeds,
        n_episodes=n_episodes, interventions=interventions,
        gamma=gamma, device=device, verbose=verbose, seed=seed,
        n_workers=n_workers,
    )

    if verbose:
        print()

    # Axis 2: Robustness
    robustness = run_robustness_audit(
        adapter, env_factory, scenarios=robustness_scenarios,
        n_episodes=n_episodes, gamma=gamma, device=device,
        verbose=verbose, seed=seed, n_workers=n_workers,
    )

    if verbose:
        print()

    # Bonus: Temporal sensitivity
    sensitivity = compute_temporal_sensitivity(
        adapter, env_factory, speeds=[1, 3, 5],
        n_episodes=sensitivity_episodes,
        gamma=gamma, device=device, verbose=verbose, seed=seed,
    )

    # 2-axis summary with deployment/stress split
    deploy = robustness["deployment"]
    stress = robustness["stress"]

    summary = {
        "reliance_rating": reliance["rating"],
        "reliance_score": reliance["score"],
        "robustness_rating": robustness["rating"],
        "robustness_score": robustness["return_score"],
        "robustness_rmse_score": robustness["rmse_score"],
        "deployment_rating": deploy["rating"],
        "deployment_score": deploy["return_score"],
        "stress_rating": stress["rating"],
        "stress_score": stress["return_score"],
        "sensitivity_mean": sensitivity["mean"] if sensitivity else None,
    }

    # Prescription based on quadrant
    reliance_available = reliance["rating"] != "N/A"
    # Use deployment score (not overall) for quadrant classification
    good_deployment = deploy["return_score"] >= 0.80

    if reliance_available:
        # Full 2-axis quadrant (internal time agents)
        # Threshold 2.0: below = structural GRU sensitivity, above = learned timing
        high_reliance = reliance["score"] >= 2.0

        if high_reliance and good_deployment:
            summary["quadrant"] = "time_aware_robust"
            summary["prescription"] = (
                "Agent actively uses internal timing and maintains performance "
                "under deployment conditions. The timing channel is functional "
                "and well-calibrated."
            )
        elif high_reliance and not good_deployment:
            summary["quadrant"] = "time_aware_fragile"
            summary["prescription"] = (
                "Agent uses internal timing but degrades under deployment "
                "conditions. Consider: (1) calibrating the time module with "
                "speed-randomized training, (2) adding explicit frame timing "
                "to observations, (3) implementing adaptive discount correction."
            )
        elif not high_reliance and not good_deployment:
            summary["quadrant"] = "time_blind_fragile"
            summary["prescription"] = (
                "Agent ignores timing information and is vulnerable to timing "
                "changes. Add a time-aware mechanism: Dt-GRU, frame timing in "
                "observations, or adaptive discount factor."
            )
        else:
            summary["quadrant"] = "time_blind_robust"
            summary["prescription"] = (
                "Agent maintains performance without explicit timing. Consider "
                "whether a timing mechanism would improve value estimation "
                "accuracy, especially at unseen speeds."
            )
    else:
        # 1-axis classification (external models, no reliance data)
        if good_deployment:
            summary["quadrant"] = "deployment_ready"
            summary["prescription"] = (
                "Agent maintains performance under deployment timing conditions. "
                "No immediate action needed. Consider adding timing awareness "
                "for enhanced performance at extreme speeds."
            )
        else:
            summary["quadrant"] = "deployment_fragile"
            summary["prescription"] = (
                "Agent degrades under deployment timing conditions. "
                "Recommended fix: train with speed randomization "
                "(jitter/delay/spike augmentation)."
            )

    if verbose:
        print()
        _print_summary(summary)

    return {
        "speeds": speeds,
        "n_episodes": n_episodes,
        "supports_intervention": adapter.supports_intervention,
        "reliance": reliance,
        "robustness": robustness,
        "sensitivity": sensitivity,
        "summary": summary,
    }


def _print_summary(summary: Dict):
    """Print human-readable 2-axis summary."""
    from .color import colored_rating, bold, dim
    print("=" * 60)
    rel_r = summary["reliance_rating"]
    rel_s = summary.get("reliance_score")
    dep_r = summary["deployment_rating"]
    dep_s = summary["deployment_score"]
    str_r = summary["stress_rating"]
    str_s = summary["stress_score"]
    if rel_r != "N/A" and rel_s is not None:
        print(f"  Reliance:    {colored_rating(rel_r, 10)}  "
              f"{dim('(RMSE ratio: ' + f'{rel_s:.2f}x)')}")
    else:
        print(f"  Reliance:    {colored_rating('N/A', 10)}  "
              f"{dim('(no intervention support)')}")
    print(f"  Deployment:  {colored_rating(dep_r, 10)}  "
          f"{dim('(return ratio: ' + f'{dep_s:.2f})')}")
    print(f"  Stress:      {colored_rating(str_r, 10)}  "
          f"{dim('(return ratio: ' + f'{str_s:.2f})')}")
    if summary.get("sensitivity_mean") is not None:
        sens = summary["sensitivity_mean"]
        print(f"  Sensitivity:  {sens:>9.4f}  {dim('(|dV/dt|)')}")
    print(f"  Quadrant:    {bold(summary['quadrant'])}")
    print("=" * 60)
    print(f"\n  {summary['prescription']}")
