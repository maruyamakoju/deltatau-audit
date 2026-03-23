"""Core auditor: 2-axis evaluation engine.

Axis 1 — Timing Channel Analysis (intervention ablation):
    Tampers with the agent's internal Dt to measure causal dependence
    on internal time representation.  HIGH reliance is *informational*:
    it means the agent has learned timing features, which is a
    desirable property when the agent was *designed* to use timing
    information (e.g. Dt-GRU architectures).  Only when combined with
    poor robustness does high reliance indicate a vulnerability.

Axis 2 — Timing Robustness (env wrappers):
    Wraps the environment with realistic timing perturbations (jitter,
    delay, speed changes) to measure operational resilience.

Bonus — Temporal Sensitivity:
    Finite-difference |dV/dt| measuring value function's local
    sensitivity to internal time -- the "timing Jacobian".
"""

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger("deltatau-audit")

import gymnasium as gym
import numpy as np
import torch

# Optional tqdm for episode progress bars
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from ._constants import (
    DEPLOYMENT_SCENARIOS as _DEPLOYMENT_SCENARIOS,
)
from ._constants import (
    INTERVENTION_LABELS as _INTERVENTION_LABELS,
)
from ._constants import (
    ROBUSTNESS_SCENARIO_LABELS as ROBUSTNESS_SCENARIOS,
)
from ._constants import (
    STRESS_SCENARIOS as _STRESS_SCENARIOS,
)
from ._theme import (
    DEPLOY_THRESHOLD_DEFAULT,
    RELIANCE_THRESHOLD,
    STRESS_THRESHOLD_DEFAULT,
)
from .adapters.base import AgentAdapter
from .metrics import (
    aggregate_episode_metrics,
    bootstrap_return_ratio,
    compute_degradation,
    compute_discounted_returns,
    compute_return_ratio,
    compute_value_bias,
    compute_value_mae,
    compute_value_rmse,
    reliance_rating,
    robustness_rating,
    severity_rating,
)
from .schema import SCHEMA_VERSION
from .wrappers.latency import ObservationDelayWrapper, ObsNoiseWrapper
from .wrappers.speed import FixedSpeedWrapper, JitterWrapper, PiecewiseSwitchWrapper

# ── Public aliases (backward compatibility) ──────────────────────────

# Keep these names exported from auditor.py for compatibility with older code.
INTERVENTIONS = dict(_INTERVENTION_LABELS)
DEPLOYMENT_SCENARIOS = list(_DEPLOYMENT_SCENARIOS)
STRESS_SCENARIOS = list(_STRESS_SCENARIOS)


# ── Parallel episode runner ───────────────────────────────────────────

def _run_episodes_parallel(
    adapter: "AgentAdapter",
    env_factory: Callable[[], Any],
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
        env = _make_wrapped_env(env_factory, scenario, adapter=adapter)
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

    results = [None] * n_episodes  # type: ignore[list-item]
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
        # Pass dicts/tuples as is so adapters can handle them natively
        if isinstance(obs, (dict, tuple)):
            obs_input = obs
        else:
            obs_input = torch.tensor(obs, dtype=torch.float32)
            
        action, value, hidden_new, dt = adapter.act(obs_input, hidden)

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

            hidden_new = adapter.rerun_with_dt(obs_input, hidden, target_dt)

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
        "dt_trace": [float(d) if d is not None else 1.0 for d in step_dts],
    }


# ══════════════════════════════════════════════════════════════════════
# AXIS 1: RELIANCE AUDIT
# ══════════════════════════════════════════════════════════════════════

def run_reliance_audit(
    adapter: AgentAdapter,
    env_factory: Callable[[], Any],
    speeds: Optional[List[int]] = None,
    n_episodes: int = 50,
    interventions: Optional[List[str]] = None,
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
        interventions = [k for k in INTERVENTIONS if k != "none"]
        interventions = ["none", *interventions]

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
                def _speed_factory(s: int = speed) -> Any:
                    return FixedSpeedWrapper(env_factory(), speed=s)
                _factory: Callable[[], Any] = _speed_factory
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

    # Neutral framing: high reliance = "agent has learned timing features"
    # This is informational, not a warning.  It is a vulnerability only
    # when combined with poor robustness.
    reliance_interpretation = (
        "Agent has learned timing features (informational)"
        if rating in ("HIGH", "EXTREME")
        else "Agent shows minimal timing dependence"
    )

    if verbose:
        from .color import colored_rating, dim
        print(f"    -> Timing Channel Analysis: {colored_rating(rating)} "
              f"{dim(f'(worst RMSE ratio: {worst_ratio:.2f}x)')}")
        print(f"       {dim(reliance_interpretation)}")

    return {
        "per_speed": per_speed,
        "degradation": degradation,
        "score": worst_ratio,
        "rating": rating,
        "interpretation": reliance_interpretation,
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

def _make_wrapped_env(env_factory, scenario: str, adapter: Optional[AgentAdapter] = None):
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
    elif scenario == "adversarial_jitter":
        from .wrappers.adversarial import AdversarialSpeedWrapper
        return AdversarialSpeedWrapper(env, agent_adapter=adapter, possible_speeds=[1, 2, 3, 5, 8])
    else:
        raise ValueError(f"Unknown robustness scenario: {scenario}")


def run_robustness_audit(
    adapter: AgentAdapter,
    env_factory: Callable[[], Any],
    scenarios: Optional[List[str]] = None,
    n_episodes: int = 50,
    gamma: float = 0.99,
    device: str = "cpu",
    verbose: bool = True,
    seed: Optional[int] = None,
    n_workers: int = 1,
    adaptive: bool = False,
    target_ci_width: float = 0.10,
    max_episodes: int = 500,
    bootstrap_samples: int = 2000,
) -> Dict:
    """Axis 2: Realistic timing perturbations via env wrappers.

    The agent runs NORMALLY (no internal intervention). Only the
    environment is perturbed. Measures whether performance holds
    under deployment-realistic timing conditions.

    Args:
        adaptive: When True, use adaptive sampling: run episodes in batches
            (pilot size = n_episodes) and keep adding batches until the 95%
            bootstrap CI width on every scenario's return ratio is below
            target_ci_width, or until max_episodes per scenario is reached.
        target_ci_width: Target 95% CI width (ci_upper - ci_lower) for the
            return ratio. Ignored when adaptive=False. Default: 0.10.
        max_episodes: Hard cap on episodes per scenario in adaptive mode.
            Default: 500.
        bootstrap_samples: Number of bootstrap resamples used for
            return-ratio confidence intervals. Default: 2000.

    Returns:
        Dict with scenarios, per_scenario_scores, return_score, rating, worst_case.
        When adaptive=True, also includes 'n_episodes_used' (dict per scenario).
    """
    if scenarios is None:
        scenarios = list(ROBUSTNESS_SCENARIOS.keys())

    if "nominal" not in scenarios:
        scenarios = ["nominal"] + list(scenarios)

    if verbose:
        if adaptive:
            print(f"  Robustness Test (env wrappers) "
                  f"[adaptive: target CI ±{target_ci_width:.2f}, max {max_episodes} eps]")
        else:
            print("  Robustness Test (env wrappers)")

    scenario_results = {}
    scenario_episode_returns = {}  # raw per-episode returns for bootstrap
    n_episodes_used: Dict[str, int] = {}

    if not adaptive:
        # ── Fixed episode count (original behaviour) ──────────────────
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

    else:
        # ── Adaptive sampling ──────────────────────────────────────────
        # Run all scenarios in lockstep batches until every non-nominal
        # scenario's return-ratio CI is narrower than target_ci_width.
        all_ep_lists: Dict[str, List[Dict]] = {sc: [] for sc in scenarios}
        batch_num = 0
        batch_size = max(1, n_episodes)  # pilot size per round

        while True:
            # One batch per scenario
            for sc_idx, scenario in enumerate(scenarios):
                current = len(all_ep_lists[scenario])
                if current >= max_episodes:
                    continue
                remaining = max_episodes - current
                this_batch = min(batch_size, remaining)
                label = ROBUSTNESS_SCENARIOS.get(scenario, scenario)
                new_eps = _run_episodes_parallel(
                    adapter, env_factory, scenario, "none",
                    this_batch, gamma, device, seed, n_workers,
                    label, verbose if batch_num == 0 else False,
                    seed_offset=sc_idx * 1000 + batch_num * 100_000,
                )
                all_ep_lists[scenario].extend(new_eps)

            batch_num += 1
            total_eps = len(all_ep_lists.get("nominal", []))

            # Check convergence: every non-nominal scenario CI < target
            nom_rets = [ep["total_reward"] for ep in all_ep_lists.get("nominal", [])]
            all_converged = True
            for scenario in scenarios:
                if scenario == "nominal":
                    continue
                pert_rets = [ep["total_reward"] for ep in all_ep_lists[scenario]]
                if len(pert_rets) < 5:
                    all_converged = False
                    break
                bci = bootstrap_return_ratio(
                    nom_rets,
                    pert_rets,
                    n_bootstrap=bootstrap_samples,
                )
                ci_w = bci["ci_upper"] - bci["ci_lower"]
                if ci_w > target_ci_width:
                    all_converged = False
                    break

            if all_converged:
                if verbose:
                    print(f"    [adaptive: converged in {total_eps} eps/scenario]")
                break
            if total_eps >= max_episodes:
                # ── CI convergence warning ──────────────────────────
                # Compute final CI widths for each scenario so the
                # warning includes actionable numbers.
                non_converged: List[str] = []
                _nom_rets_final = [
                    ep["total_reward"]
                    for ep in all_ep_lists.get("nominal", [])
                ]
                for _sc in scenarios:
                    if _sc == "nominal":
                        continue
                    _p_rets = [ep["total_reward"]
                               for ep in all_ep_lists[_sc]]
                    if len(_p_rets) < 5:
                        non_converged.append(f"{_sc}(n<5)")
                        continue
                    _bci = bootstrap_return_ratio(
                        _nom_rets_final, _p_rets,
                        n_bootstrap=bootstrap_samples,
                    )
                    _ci_w = _bci["ci_upper"] - _bci["ci_lower"]
                    if _ci_w > target_ci_width:
                        non_converged.append(
                            f"{_sc}(CI={_ci_w:.3f}>{target_ci_width:.2f})"
                        )

                _logger.warning(
                    "Bootstrap CI did not converge within max_episodes=%d. "
                    "Non-converged scenarios: %s. "
                    "Consider increasing max_episodes or relaxing "
                    "target_ci_width (current: %.2f).",
                    max_episodes,
                    ", ".join(non_converged) if non_converged else "(none)",
                    target_ci_width,
                )

                if verbose:
                    print(f"    [adaptive: max_episodes={max_episodes} reached "
                          f"(CI may be wider than {target_ci_width:.2f})]")
                break

        # Build results from accumulated episode pools
        for sc_idx, scenario in enumerate(scenarios):
            eps = all_ep_lists[scenario]
            agg = aggregate_episode_metrics(eps)
            scenario_results[scenario] = agg
            scenario_episode_returns[scenario] = [
                ep["total_reward"] for ep in eps
            ]
            n_episodes_used[scenario] = len(eps)
            if verbose and not (_HAS_TQDM and verbose):
                r = agg.get("total_reward_mean", 0)
                rmse = agg.get("rmse_mean", 0)
                label = ROBUSTNESS_SCENARIOS.get(scenario, scenario)
                print(f"    {label}: R={r:.3f}, RMSE={rmse:.4f} "
                      f"(n={len(eps)})")

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
        bci = bootstrap_return_ratio(
            nominal_ep_returns,
            pert_ep_returns,
            n_bootstrap=bootstrap_samples,
        )

        per_scenario_scores[scenario] = {
            "return_ratio": ret_ratio,
            "return_drop_pct": (1 - ret_ratio) * 100,
            "rmse_ratio": rmse_ratio,
            "rmse_increase_pct": (rmse_ratio - 1) * 100,
            "ci_lower": bci["ci_lower"],
            "ci_upper": bci["ci_upper"],
            "significant": bci["significant"],
            "significant_change": bci.get("significant_change", bci["significant"]),
            "mean_nominal": bci.get("mean_nominal", nominal_return),
            "mean_perturbed": bci.get("mean_perturbed", s_return),
            "mean_difference": bci.get("mean_difference", s_return - nominal_return),
            "cohens_d": bci.get("cohens_d", 0.0),
            "cohens_d_magnitude": bci.get("cohens_d_magnitude", "NEGLIGIBLE"),
            "cliffs_delta": bci.get("cliffs_delta", 0.0),
            "common_language_effect": bci.get("common_language_effect", 0.5),
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

    # ── Per-scenario CI widths (for convergence diagnostics) ──────────
    ci_widths: Dict[str, float] = {}
    for sc_name, sc_scores in per_scenario_scores.items():
        ci_w = sc_scores.get("ci_upper", 0.0) - sc_scores.get("ci_lower", 0.0)
        ci_widths[sc_name] = ci_w

    result = {
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
        "ci_widths": ci_widths,
    }
    if adaptive:
        result["n_episodes_used"] = n_episodes_used
        result["adaptive"] = True

        # Record whether CI converged for each scenario
        ci_converged = {
            sc: (ci_widths.get(sc, 0.0) <= target_ci_width)
            for sc in scenarios if sc != "nominal"
        }
        result["ci_converged"] = ci_converged
        all_ci_converged = all(ci_converged.values())
        if not all_ci_converged:
            _logger.warning(
                "Robustness audit CI did not converge for all scenarios. "
                "Widths: %s. Target: %.2f.",
                {k: f"{v:.3f}" for k, v in ci_widths.items()
                 if ci_widths.get(k, 0) > target_ci_width},
                target_ci_width,
            )
    return result


# ══════════════════════════════════════════════════════════════════════
# TEMPORAL SENSITIVITY: |dV/dt|
# ══════════════════════════════════════════════════════════════════════

def compute_temporal_sensitivity(
    adapter: AgentAdapter,
    env_factory: Callable[[], Any],
    speeds: Optional[List[int]] = None,
    n_episodes: int = 20,
    epsilon: float = 0.1,
    gamma: float = 0.99,
    device: str = "cpu",
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Optional[Dict]:
    """Compute temporal sensitivity |dV/dt| via symmetric finite difference.

    The **temporal sensitivity** (a.k.a. "timing Jacobian") measures how
    the value function responds to infinitesimal perturbations of the
    agent's internal time representation:

    .. math::

        S = \\mathbb{E}\\left[
          \\frac{|V(\\tau + \\varepsilon) - V(\\tau - \\varepsilon)|}
               {2\\varepsilon}
        \\right]

    where :math:`\\tau` is the agent's learned internal timestep and
    :math:`\\varepsilon` is a small perturbation (default 0.1).

    **Interpretation** (neutral "Timing Channel Analysis" framing):

    * High sensitivity at *trained* speeds: the agent has learned to
      use timing information actively.  This is a feature, not a bug.
    * High sensitivity at *unseen* speeds: the agent generalises its
      timing representation -- strong evidence of temporal abstraction.
    * Low sensitivity everywhere: the agent ignores internal timing.

    The result includes per-speed breakdowns and a 95% bootstrap CI on
    the overall mean sensitivity.

    Parameters
    ----------
    adapter : AgentAdapter
    env_factory : callable
    speeds : list of int or None
    n_episodes : int
    epsilon : float
        Finite-difference step size (default 0.1).
    gamma : float
    device : str
    verbose : bool
    seed : int or None

    Returns
    -------
    dict or None
        ``None`` if the agent doesn't support intervention.
        Otherwise a dict with ``mean``, ``std``, ``median``,
        ``ci_lower``, ``ci_upper``, ``n_samples``, ``per_speed``.
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

    # Bootstrap 95% CI on overall mean sensitivity
    sens_arr = np.array(all_sensitivities, dtype=np.float64)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        float(np.mean(rng.choice(sens_arr, size=len(sens_arr), replace=True)))
        for _ in range(2000)
    ])
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Interpretation (neutral framing)
    mean_sens = float(np.mean(all_sensitivities))
    if mean_sens > 0.5:
        interpretation = (
            "Agent has learned strong timing features -- the value "
            "function is highly sensitive to internal time "
            "perturbations. This indicates active temporal adaptation."
        )
    elif mean_sens > 0.1:
        interpretation = (
            "Agent shows moderate timing sensitivity. The value "
            "function partially depends on internal time representation."
        )
    else:
        interpretation = (
            "Agent shows minimal timing sensitivity. The value "
            "function is largely invariant to internal time changes."
        )

    result = {
        "mean": mean_sens,
        "std": float(np.std(all_sensitivities)),
        "median": float(np.median(all_sensitivities)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_samples": len(all_sensitivities),
        "per_speed": per_speed,
        "interpretation": interpretation,
    }

    if verbose:
        print(f"    -> Mean sensitivity: {result['mean']:.4f} "
              f"[{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"       {interpretation}")

    return result


# ══════════════════════════════════════════════════════════════════════
# FULL 2-AXIS AUDIT
# ══════════════════════════════════════════════════════════════════════

def run_full_audit(
    adapter: AgentAdapter,
    env_factory: Callable[[], Any],
    speeds: Optional[List[int]] = None,
    n_episodes: int = 50,
    interventions: Optional[List[str]] = None,
    robustness_scenarios: Optional[List[str]] = None,
    sensitivity_episodes: int = 20,
    gamma: float = 0.99,
    device: str = "cpu",
    verbose: bool = True,
    seed: Optional[int] = None,
    n_workers: int = 1,
    deploy_threshold: float = DEPLOY_THRESHOLD_DEFAULT,
    stress_threshold: float = STRESS_THRESHOLD_DEFAULT,
    adaptive: bool = False,
    target_ci_width: float = 0.10,
    max_episodes: int = 500,
    bootstrap_samples: int = 2000,
) -> Dict:
    """Run the complete 2-axis time robustness audit.

    Axis 1 — Reliance: intervention ablation
    Axis 2 — Robustness: env wrappers
    Bonus  — Sensitivity: |dV/dτ| finite difference

    Args:
        deploy_threshold: Minimum deployment return ratio to classify as
            "good deployment" in the quadrant (default: 0.80).
        stress_threshold: Minimum stress return ratio for CI pass
            (default: 0.50). Stored in summary for downstream use.
        bootstrap_samples: Number of bootstrap resamples used to estimate
            per-scenario return-ratio confidence intervals (default: 2000).

    Returns structured dict ready for report generation.
    """
    if speeds is None:
        speeds = [1, 2, 3, 5, 8]

    if verbose:
        print("Time Robustness Audit (2-axis)")
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
        adaptive=adaptive, target_ci_width=target_ci_width,
        max_episodes=max_episodes,
        bootstrap_samples=bootstrap_samples,
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
        "reliance_interpretation": reliance.get(
            "interpretation",
            "Agent shows minimal timing dependence",
        ),
        "robustness_rating": robustness["rating"],
        "robustness_score": robustness["return_score"],
        "robustness_rmse_score": robustness["rmse_score"],
        "deployment_rating": deploy["rating"],
        "deployment_score": deploy["return_score"],
        "stress_rating": stress["rating"],
        "stress_score": stress["return_score"],
        "sensitivity_mean": sensitivity["mean"] if sensitivity else None,
        "sensitivity_ci_lower": sensitivity.get("ci_lower") if sensitivity else None,
        "sensitivity_ci_upper": sensitivity.get("ci_upper") if sensitivity else None,
        "sensitivity_interpretation": sensitivity.get("interpretation") if sensitivity else None,
        "ci_widths": robustness.get("ci_widths", {}),
    }

    # Prescription based on quadrant
    reliance_available = reliance["rating"] != "N/A"
    # Use deployment score (not overall) for quadrant classification
    good_deployment = deploy["return_score"] >= deploy_threshold
    summary["deploy_threshold"] = deploy_threshold
    summary["stress_threshold"] = stress_threshold

    if reliance_available:
        # Full 2-axis quadrant (internal time agents)
        # Threshold from theme constants: below = structural sensitivity,
        # above = strong learned timing reliance.
        high_reliance = reliance["score"] >= RELIANCE_THRESHOLD

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

    # Failure diagnosis: pattern → root cause → fix
    from .diagnose import generate_diagnosis
    diagnosis = generate_diagnosis(summary, robustness)

    if verbose:
        print()
        _print_summary(summary, diagnosis)

    return {
        "schema_version": SCHEMA_VERSION,
        "speeds": speeds,
        "n_episodes": n_episodes,
        "supports_intervention": adapter.supports_intervention,
        "reliance": reliance,
        "robustness": robustness,
        "sensitivity": sensitivity,
        "summary": summary,
        "diagnosis": diagnosis,
        "manifest": {},
    }


def run_deliberative_audit(
    adapter,
    env_factory: Callable[[], Any],
    speeds: Optional[List[int]] = None,
    n_episodes: int = 20,
    gamma: float = 0.99,
    device: str = "cpu",
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Dict:
    """Audit a deliberative agent: measure how ponder depth responds to timing stress.

    Key metric: does the agent ponder MORE when timing is uncertain?
    A well-calibrated deliberative agent increases thinking steps under jitter.

    The adapter must be a DeliberativeAgentAdapter (or expose
    get_deliberation_stats() and reset_episode() methods).

    Args:
        adapter: DeliberativeAgentAdapter instance.
        env_factory: Callable returning a fresh gymnasium env.
        speeds: List of speed multipliers to test (default: [1, 2, 5]).
        n_episodes: Episodes per speed condition.
        gamma: Discount factor.
        device: Torch device.
        verbose: Print progress.
        seed: Random seed.

    Returns:
        Dict with:
            ponder_vs_speed: {speed_str: mean_ponder_steps}
            deliberative_score: Spearman correlation between speed and ponder steps
                                (positive = more pondering under stress, good)
            stats_by_speed: {speed_str: deliberation stats dict}
            rating: "ADAPTIVE" | "NEUTRAL" | "ANTI_ADAPTIVE"
    """
    if speeds is None:
        speeds = [1, 2, 5]

    if verbose:
        print("  Deliberative Audit (ponder depth vs timing stress)")

    ponder_vs_speed: Dict[str, float] = {}
    stats_by_speed: Dict[str, Dict] = {}

    for speed in speeds:
        if hasattr(adapter, "reset_episode"):
            adapter.reset_episode()

        # Build env with speed wrapper
        from .wrappers.speed import FixedSpeedWrapper

        def _factory(s=speed):
            env = env_factory()
            return FixedSpeedWrapper(env, speed=s) if s > 1 else env

        total_ponder = 0.0
        for ep_idx in range(n_episodes):
            ep_seed = (None if seed is None else seed + ep_idx + speed * 1000)
            env = _factory()
            reset_kwargs = {"seed": ep_seed} if ep_seed is not None else {}
            obs, _ = env.reset(**reset_kwargs)
            hidden = adapter.reset_hidden(1, device)
            done = False

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32) if not isinstance(obs, torch.Tensor) else obs
                action, value, hidden_new, dt = adapter.act(obs_t, hidden)
                hidden = hidden_new
                obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
            env.close()

        stats = adapter.get_deliberation_stats() if hasattr(adapter, "get_deliberation_stats") else {}
        mean_ponder = stats.get("mean_ponder_steps", 0.0)
        ponder_vs_speed[str(speed)] = mean_ponder
        stats_by_speed[str(speed)] = stats

        if verbose:
            print(f"    Speed {speed}x: mean_ponder={mean_ponder:.2f} steps, "
                  f"utilization={stats.get('ponder_utilization', 0):.1%}")

        if hasattr(adapter, "reset_episode"):
            adapter.reset_episode()

    # Compute deliberative score: Spearman correlation (speed vs ponder steps)
    speed_vals = np.array([float(s) for s in speeds])
    ponder_vals = np.array([ponder_vs_speed[str(s)] for s in speeds])

    if len(speeds) >= 3 and ponder_vals.std() > 1e-6:
        # Rank correlation: positive = more ponder under higher speed = good
        from scipy.stats import spearmanr
        try:
            corr, _ = spearmanr(speed_vals, ponder_vals)
            deliberative_score = float(corr)
        except Exception:
            deliberative_score = 0.0
    elif ponder_vals.std() < 1e-6:
        deliberative_score = 0.0  # No variation = neutral
    else:
        # 2-point approximation
        deliberative_score = float(
            np.sign(ponder_vals[-1] - ponder_vals[0])
        )

    if deliberative_score > 0.3:
        rating = "ADAPTIVE"
    elif deliberative_score < -0.3:
        rating = "ANTI_ADAPTIVE"
    else:
        rating = "NEUTRAL"

    if verbose:
        print(f"    -> Deliberative score: {deliberative_score:.3f} ({rating})")

    return {
        "ponder_vs_speed": ponder_vs_speed,
        "deliberative_score": deliberative_score,
        "stats_by_speed": stats_by_speed,
        "rating": rating,
    }


def _print_summary(summary: Dict, diagnosis: Optional[Dict] = None):
    """Print human-readable 2-axis summary, with optional failure diagnosis."""
    from .color import bold, colored_rating, dim
    print("=" * 60)
    rel_r = summary["reliance_rating"]
    rel_s = summary.get("reliance_score")
    dep_r = summary["deployment_rating"]
    dep_s = summary["deployment_score"]
    str_r = summary["stress_rating"]
    str_s = summary["stress_score"]
    if rel_r != "N/A" and rel_s is not None:
        # Neutral framing: "Timing Channel Analysis" instead of "Reliance"
        interp = summary.get("reliance_interpretation", "")
        print(f"  Timing Channel: {colored_rating(rel_r, 10)}  "
              f"{dim('(RMSE ratio: ' + f'{rel_s:.2f}x)')}")
        if interp:
            print(f"                  {dim(interp)}")
    else:
        print(f"  Timing Channel: {colored_rating('N/A', 10)}  "
              f"{dim('(no intervention support)')}")
    print(f"  Deployment:  {colored_rating(dep_r, 10)}  "
          f"{dim('(return ratio: ' + f'{dep_s:.2f})')}")
    print(f"  Stress:      {colored_rating(str_r, 10)}  "
          f"{dim('(return ratio: ' + f'{str_s:.2f})')}")
    if summary.get("sensitivity_mean") is not None:
        sens = summary["sensitivity_mean"]
        ci_lo = summary.get("sensitivity_ci_lower")
        ci_hi = summary.get("sensitivity_ci_upper")
        if ci_lo is not None and ci_hi is not None:
            ci_str = f" [{ci_lo:.4f}, {ci_hi:.4f}]"
        else:
            ci_str = ""
        print(f"  Sensitivity:  {sens:>9.4f}{ci_str}  {dim('(|dV/dt|)')}")
        sens_interp = summary.get("sensitivity_interpretation")
        if sens_interp:
            print(f"                {dim(sens_interp)}")
    print(f"  Quadrant:    {bold(summary['quadrant'])}")
    print("=" * 60)
    print(f"\n  {summary['prescription']}")

    # Show failure diagnosis when there are issues
    if diagnosis and diagnosis.get("issues"):
        print()
        print(f"  Failure Analysis  ({diagnosis['summary_line']})")
        print("  " + "─" * 56)
        primary = diagnosis["issues"][0]
        print(f"  Pattern:  {bold(primary['pattern'])}  [{colored_rating(primary['rating'], 0)}]")
        print(f"  Cause:    {dim(primary['cause'])}")
        print(f"  Fix:      {primary['fix']}")
        if len(diagnosis["issues"]) > 1:
            others = [f"{i['scenario']} ({i['rating']})" for i in diagnosis["issues"][1:]]
            print(f"\n  Other issues: {', '.join(others)}")
