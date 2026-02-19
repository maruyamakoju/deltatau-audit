"""Audit metrics: 2-axis evaluation (Reliance × Robustness).

Axis 1 — Timing Reliance:
    Does the agent USE internal timing? Measured via intervention ablation.
    HIGH reliance = agent depends on Δτ = timing channel is functional.

Axis 2 — Timing Robustness:
    Does the agent SURVIVE realistic timing perturbations? Measured via env wrappers.
    PASS = performance maintained under jitter/delay/speed changes.

All functions return plain dicts/floats for easy JSON serialization.
"""

import numpy as np
from typing import Any, Dict, List


# ── Value prediction metrics ──────────────────────────────────────────

def compute_value_rmse(values: List[float], returns: List[float]) -> float:
    """RMSE between predicted values and actual discounted returns."""
    v = np.array(values)
    g = np.array(returns)
    return float(np.sqrt(np.mean((v - g) ** 2)))


def compute_value_bias(values: List[float], returns: List[float]) -> float:
    """Mean signed error (positive = overestimate)."""
    return float(np.mean(np.array(values) - np.array(returns)))


def compute_value_mae(values: List[float], returns: List[float]) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.array(values) - np.array(returns))))


def compute_discounted_returns(rewards: List[float], gamma: float = 0.99
                                ) -> List[float]:
    """Compute discounted return from each timestep."""
    T = len(rewards)
    returns = np.zeros(T)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns.tolist()


# ── Aggregation ───────────────────────────────────────────────────────

def aggregate_episode_metrics(episode_results: List[Dict]) -> Dict:
    """Aggregate metrics across multiple episodes."""
    n = len(episode_results)
    if n == 0:
        return {"n_episodes": 0}

    keys = ["rmse", "mae", "bias", "total_reward", "length"]
    agg: Dict[str, Any] = {"n_episodes": n}

    for key in keys:
        vals = [ep[key] for ep in episode_results if key in ep]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
            agg[f"{key}_se"] = float(np.std(vals) / np.sqrt(len(vals)))

    dt_means = [ep["dt_mean"] for ep in episode_results
                if ep.get("dt_mean") is not None]
    if dt_means:
        agg["dt_mean"] = float(np.mean(dt_means))
        agg["dt_std"] = float(np.std(dt_means))

    return agg


# ── Degradation & ratios ─────────────────────────────────────────────

def compute_degradation(baseline_rmse: float, intervention_rmse: float
                         ) -> Dict:
    """Compute degradation metrics for an intervention vs baseline."""
    if baseline_rmse > 1e-10:
        pct = (intervention_rmse / baseline_rmse - 1) * 100
        ratio = intervention_rmse / baseline_rmse
    else:
        pct = 0.0 if intervention_rmse < 1e-10 else float("inf")
        ratio = 1.0 if intervention_rmse < 1e-10 else float("inf")

    return {
        "baseline_rmse": baseline_rmse,
        "intervention_rmse": intervention_rmse,
        "absolute_increase": intervention_rmse - baseline_rmse,
        "percent_increase": pct,
        "ratio": ratio,
    }


def compute_return_ratio(nominal_return: float,
                          perturbed_return: float) -> float:
    """Ratio measuring perturbed performance relative to nominal.

    Semantics: 1.0 = same, < 1.0 = worse, > 1.0 = better.

    Handles negative nominal returns (e.g. penalty-heavy envs) correctly:
    - nominal=-100, perturbed=-50  → 1.5  (less penalty = improvement)
    - nominal=-100, perturbed=-150 → 0.5  (more penalty = degradation)
    - nominal=+100, perturbed=+50  → 0.5  (lower return = degradation)
    """
    if abs(nominal_return) < 1e-10:
        return 1.0 if abs(perturbed_return) < 1e-10 else 0.0
    if nominal_return > 0:
        return perturbed_return / nominal_return
    else:
        # nominal < 0: measure relative change preserving sign semantics
        # ratio = 1 + (improvement) / |nominal|
        return 1.0 + (perturbed_return - nominal_return) / abs(nominal_return)


# ── Bootstrap confidence intervals ────────────────────────────────────

def bootstrap_ci(data: List[float], n_bootstrap: int = 2000,
                 ci: float = 0.95, seed: int = 42) -> Dict:
    """Compute bootstrap confidence interval for the mean.

    Returns:
        Dict with mean, ci_lower, ci_upper, std, n.
    """
    arr = np.array(data)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "std": 0.0, "n": 0}
    if n == 1:
        v = float(arr[0])
        return {"mean": v, "ci_lower": v, "ci_upper": v,
                "std": 0.0, "n": 1}

    rng = np.random.RandomState(seed)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = arr[rng.randint(0, n, size=n)]
        means[i] = sample.mean()

    alpha = (1 - ci) / 2
    lower = float(np.percentile(means, alpha * 100))
    upper = float(np.percentile(means, (1 - alpha) * 100))

    return {
        "mean": float(arr.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "std": float(arr.std()),
        "n": n,
    }


def _safe_return_ratio(nominal_mean: float, pert_mean: float) -> float:
    """Sign-aware return ratio consistent with compute_return_ratio."""
    if abs(nominal_mean) < 1e-10:
        return 1.0 if abs(pert_mean) < 1e-10 else 0.0
    if nominal_mean > 0:
        return pert_mean / nominal_mean
    else:
        return 1.0 + (pert_mean - nominal_mean) / abs(nominal_mean)


def bootstrap_return_ratio(nominal_returns: List[float],
                           perturbed_returns: List[float],
                           n_bootstrap: int = 2000,
                           ci: float = 0.95,
                           seed: int = 42) -> Dict:
    """Bootstrap CI for the return ratio.

    Uses sign-aware ratio so negative nominal returns are handled correctly.

    Returns:
        Dict with ratio, ci_lower, ci_upper, significant (bool).
        significant=True means CI excludes 1.0 (statistically significant drop).
    """
    nom = np.array(nominal_returns)
    pert = np.array(perturbed_returns)

    if len(nom) == 0 or len(pert) == 0 or abs(nom.mean()) < 1e-10:
        return {"ratio": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "significant": False}

    rng = np.random.RandomState(seed)
    ratios = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        nom_sample = nom[rng.randint(0, len(nom), size=len(nom))]
        pert_sample = pert[rng.randint(0, len(pert), size=len(pert))]
        ratios[i] = _safe_return_ratio(nom_sample.mean(), pert_sample.mean())

    alpha = (1 - ci) / 2
    lower = float(np.percentile(ratios, alpha * 100))
    upper = float(np.percentile(ratios, (1 - alpha) * 100))
    ratio = _safe_return_ratio(float(nom.mean()), float(pert.mean()))

    return {
        "ratio": ratio,
        "ci_lower": lower,
        "ci_upper": upper,
        "significant": upper < 1.0,  # CI entirely below 1.0 = real drop
    }


# ══════════════════════════════════════════════════════════════════════
# 2-AXIS RATING SYSTEM
# ══════════════════════════════════════════════════════════════════════

# ── Axis 1: Reliance ─────────────────────────────────────────────────
# Based on RMSE ratio (intervention / none). HIGH = timing IS used.

def reliance_rating(rmse_ratio: float) -> str:
    """Rate timing reliance from RMSE ratio (intervention/baseline).

    HIGH reliance = the agent's value function depends on Δτ.
    This is INFORMATIONAL — high reliance on a time-aware agent is expected.
    """
    if rmse_ratio < 1.05:
        return "LOW"
    elif rmse_ratio < 1.20:
        return "MODERATE"
    elif rmse_ratio < 2.0:
        return "HIGH"
    else:
        return "VERY_HIGH"


def reliance_color(rating: str) -> str:
    """Color for reliance badge (informational blue spectrum)."""
    return {
        "N/A": "#BDBDBD",
        "LOW": "#9E9E9E",
        "MODERATE": "#42A5F5",
        "HIGH": "#1E88E5",
        "VERY_HIGH": "#1565C0",
    }.get(rating, "#9E9E9E")


# ── Axis 2: Robustness ───────────────────────────────────────────────
# Based on return ratio (wrapper / nominal). FAIL = agent breaks.

def robustness_rating(return_ratio: float) -> str:
    """Rate operational robustness from worst-case return ratio.

    PASS = performance maintained under realistic timing perturbations.
    FAIL = significant performance loss in deployment conditions.
    """
    if return_ratio > 0.95:
        return "PASS"
    elif return_ratio > 0.80:
        return "MILD"
    elif return_ratio > 0.50:
        return "DEGRADED"
    else:
        return "FAIL"


def robustness_color(rating: str) -> str:
    """Color for robustness badge (green=good, red=bad)."""
    return {
        "PASS": "#28a745",
        "MILD": "#ffc107",
        "DEGRADED": "#fd7e14",
        "FAIL": "#dc3545",
    }.get(rating, "#6c757d")


# ── Legacy single-axis (kept for backward compat) ────────────────────

def severity_rating(pct_increase: float) -> str:
    """Legacy single-axis severity rating."""
    if pct_increase < 5:
        return "PASS"
    elif pct_increase < 20:
        return "MILD"
    elif pct_increase < 50:
        return "MODERATE"
    elif pct_increase < 100:
        return "SEVERE"
    else:
        return "CRITICAL"


def severity_color(rating: str) -> str:
    """Legacy color for single-axis severity."""
    return {
        "PASS": "#28a745",
        "MILD": "#ffc107",
        "MODERATE": "#fd7e14",
        "SEVERE": "#dc3545",
        "CRITICAL": "#721c24",
    }.get(rating, "#6c757d")
